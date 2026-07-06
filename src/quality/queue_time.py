"""
Post-LITHO queue-time windows (M7 Stage B).

Photoresist-aging analogy (stylized, not a physical model)
------------------------------------------------------------
Real litho cells impose a hard "queue time" (Q-time) limit between coating a
wafer and exposing/developing it: once photoresist is applied, airborne
contamination and resist relaxation degrade the pattern if the wafer waits too
long before the next process step. Fabs track this as a scheduling constraint
and treat a breach as a real yield risk, not just a scheduling nuisance.

This module borrows the SHAPE of that idea (a queue-time window whose breach
is a risk signal) without modeling resist chemistry, contamination levels, or
any physical mechanism. It is a stylized analogy: "the longer a lot waits
after LITHO before its next step starts, the higher we declare its risk to
be," nothing more. Every column and flag below should be read with that
disclaimer attached.

What is measured
------------------
For each of the two LITHO visits (route positions ``step_seq`` 3 and 5, i.e.
the two re-entrant mask layers - see ``factory_generator.default_config``),
the "post-litho queue time" for a lot is:

    gap = process_start_time(next step)  -  process_complete_time(LITHO visit)

This is the wafer's dwell time between finishing exposure/develop at that
LITHO visit and starting the immediately following route step (ETCH after the
first visit, IMPLANT after the second). A violation is flagged when
``gap > W`` for a fixed window ``W`` (hours), read as "resist aged past the
tolerated window."

Window calibration
---------------------
``W`` is NOT re-derived at runtime. It was calibrated ONCE (see
``calibrate_window`` below) against the DEFAULT-SEED baseline log
(``factory_generator.default_config()`` + ``simulate(cfg)``, seed 42, no
anomalies, no tool offsets) by taking the quantile of the POOLED per-visit gap
distribution (both LITHO visits pooled into one sample of "gap" observations)
at ``TARGET_VIOLATION_QUANTILE = 0.90``. That calibration run produced:

    W = 0.410203188811795 hours  (see DEFAULT_WINDOW_HOURS below)
    baseline violation rate = 0.1001  (~10.0%, i.e. ~1 in 10 LITHO visits)

Provenance: computed by calling ``calibrate_window(log, quantile=0.90)`` on
the log from ``simulate(default_config())`` with no draws override (lazy RNG
path, seed=42, the same default every other validated module uses). Re-running
that exact call reproduces the same number bit-for-bit because the DES itself
is fully seeded. ``DEFAULT_WINDOW_HOURS`` below is the FIXED, hand-copied
result of that one-time calibration - it is a constant, not something
recomputed on every call, so that violation flags stay comparable across runs,
scenarios, and later M7 stages (deliberately mirrors how M5's control-chart
baseline is fit once and reused, not re-fit per anomaly).
"""

from __future__ import annotations

import pandas as pd

#: Quantile of the pooled (both-visit) baseline gap distribution used to pick
#: the fixed window. See module docstring for the exact calibration call.
TARGET_VIOLATION_QUANTILE = 0.90

#: Fixed post-LITHO queue-time window (hours), calibrated once against the
#: default-seed baseline log (seed=42, no anomalies, no tool offsets) at
#: TARGET_VIOLATION_QUANTILE = 0.90. Baseline violation rate at this window
#: was 0.1001 (~10.0%), which is the target this constant is locked to.
#: DO NOT recompute this per call - see "Window calibration" above.
DEFAULT_WINDOW_HOURS = 0.410203188811795


def _litho_step_positions(route: list) -> list:
    """Route positions (0-indexed step_seq values) where ``route`` visits LITHO.

    For the locked route ``CLEAN, FURNACE, DEPO, LITHO, ETCH, LITHO, IMPLANT,
    METRO`` this is ``[3, 5]`` - the two re-entrant mask-layer visits.
    """
    return [i for i, station in enumerate(route) if station == "LITHO"]


def post_litho_queue_times(log: pd.DataFrame, route: list) -> pd.DataFrame:
    """Per-lot, per-LITHO-visit post-litho queue time ("gap").

    Parameters
    ----------
    log : pd.DataFrame
        Event log with columns ``lot_id, step_seq, station, process_start_time,
        process_complete_time, tool_id`` (the Stage-A schema).
    route : list
        The route used to generate ``log`` (``cfg.route``), so LITHO's
        ``step_seq`` positions and "the immediately following step" are read
        from the actual config rather than hard-coded.

    Returns
    -------
    pd.DataFrame with one row per (lot_id, visit) that has a following step:
        lot_id     : lot identifier
        visit      : 1 or 2 (first / second LITHO visit in route order)
        litho_step_seq : the LITHO visit's step_seq (route position)
        litho_tool     : tool_id that ran that LITHO visit (e.g. "LITHO-1")
        litho_complete_time : process_complete_time of the LITHO visit
        next_start_time     : process_start_time of the immediately following
                               route step for the same lot
        gap        : next_start_time - litho_complete_time (hours); the
                     stylized "photoresist aging" dwell time (see module
                     docstring - analogy only, not a physical model)

    A lot is dropped from the output for a given visit only if the log has no
    row for the following step (e.g. an incomplete lot at the end of the
    horizon); this is a natural filter, not a modeling choice.
    """
    litho_steps = _litho_step_positions(route)
    frames = []
    for visit, step in enumerate(litho_steps, start=1):
        next_step = step + 1
        if next_step >= len(route):
            continue  # LITHO is the last route step; no "next step" to wait for
        litho_rows = log.loc[log["step_seq"] == step,
                              ["lot_id", "process_complete_time", "tool_id"]]
        litho_rows = litho_rows.rename(columns={
            "process_complete_time": "litho_complete_time",
            "tool_id": "litho_tool",
        })
        next_rows = log.loc[log["step_seq"] == next_step,
                             ["lot_id", "process_start_time"]]
        next_rows = next_rows.rename(columns={
            "process_start_time": "next_start_time",
        })
        merged = litho_rows.merge(next_rows, on="lot_id", how="inner")
        merged["visit"] = visit
        merged["litho_step_seq"] = step
        merged["gap"] = merged["next_start_time"] - merged["litho_complete_time"]
        frames.append(merged[["lot_id", "visit", "litho_step_seq", "litho_tool",
                               "litho_complete_time", "next_start_time", "gap"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["lot_id", "visit", "litho_step_seq", "litho_tool",
                 "litho_complete_time", "next_start_time", "gap"])


def calibrate_window(log: pd.DataFrame, route: list,
                      quantile: float = TARGET_VIOLATION_QUANTILE) -> tuple[float, float]:
    """Return (W, baseline_violation_rate) for ``quantile`` on ``log``.

    ``W`` is the ``quantile``-th quantile of the POOLED gap distribution (both
    LITHO visits combined into one sample) computed by
    ``post_litho_queue_times``. ``baseline_violation_rate`` is the fraction of
    pooled (lot, visit) rows with ``gap > W`` - by construction close to
    ``1 - quantile`` for a continuous distribution.

    This function is provided for reproducible re-calibration (e.g. if the
    generator or default config ever changes materially). It does NOT change
    ``DEFAULT_WINDOW_HOURS`` automatically - that constant must be updated by
    hand, deliberately, with the new provenance documented, so that violation
    flags stay stable and comparable across the M7 stages that consume them.
    """
    tidy = post_litho_queue_times(log, route)
    W = float(tidy["gap"].quantile(quantile))
    violation_rate = float((tidy["gap"] > W).mean())
    return W, violation_rate


def flag_violations(log: pd.DataFrame, route: list,
                     window_hours: float = DEFAULT_WINDOW_HOURS) -> pd.DataFrame:
    """Per-lot, per-visit queue times with a boolean ``violation`` flag.

    Same rows/columns as ``post_litho_queue_times`` plus:
        violation : bool, True when gap > window_hours

    ``window_hours`` defaults to the fixed, pre-calibrated
    ``DEFAULT_WINDOW_HOURS`` so callers get consistent flags without having to
    recalibrate; pass an explicit value only for sensitivity checks.
    """
    tidy = post_litho_queue_times(log, route)
    tidy = tidy.copy()
    tidy["violation"] = tidy["gap"] > window_hours
    return tidy
