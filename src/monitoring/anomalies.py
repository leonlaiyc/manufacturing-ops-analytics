"""
Anomaly scenario construction and paired clean/anomalous runs (M5, Step 1).

The three injection primitives live with the simulator
(``factory_generator``: BreakdownAnomaly / DegradationAnomaly / DemandSurgeAnomaly)
because ``simulate`` must interpret them. This module builds the standard M5
*scenario* — three labeled anomalies at spaced windows on the longer M5 horizon,
with a clean baseline period at the front and clean recovery gaps between — and
runs each experiment as a CRN-paired pair (clean twin + anomalous run on the same
draw table).

Every anomaly carries its own ``[t_start, t_end]`` window; ``scenario_windows``
exposes those labels as the ground truth the detectors are scored against.
"""

from __future__ import annotations

from factory_generator import (
    BreakdownAnomaly,
    DegradationAnomaly,
    DemandSurgeAnomaly,
    draw_randoms,
    simulate,
)

DAY = 24.0


def default_scenario(cfg) -> list:
    """Three labeled anomalies spaced across the M5 horizon.

    Windows are chosen (for a 160-day horizon, 6-day warm-up) to leave a clean
    baseline (~day 10-48) for leakage-free statistics and generous clean recovery
    gaps between anomalies. The gaps matter: at rho ~= 0.85 the bottleneck drains
    a backlog slowly, so each anomaly's KPI footprint outlives its injection
    window; the anomalies are kept short/mild and widely spaced so that days
    counted as "clean" for the false-alarm rate are genuinely back to baseline
    (see EVAL_GRACE_DAYS).

    Types:
      - breakdown  : S4 loses one of two tools (capacity halved) for 2 days — a
                     sudden shock the control chart should catch fast.
      - demand_surge: arrival rate +0.4 lots/h for 4 days — a sudden load jump.
      - degradation: S4 processing time ramps slowly and gently over 25 days
                     (stays near rho<1) — a slow drift the EWMA should catch
                     before the control chart does.
    """
    return [
        BreakdownAnomaly(station="S4",
                         t_start=55 * DAY, t_end=57 * DAY, tools_removed=1),
        DemandSurgeAnomaly(t_start=82 * DAY, t_end=86 * DAY,
                           extra_rate=0.4, seed=7),
        DegradationAnomaly(station="S4",
                           t_onset=110 * DAY, t_end=135 * DAY, alpha=0.00025),
    ]


# Days after each injection window before a day counts as "clean" again — long
# enough to let the bottleneck backlog drain, so the false-alarm rate is measured
# on genuinely-recovered days, not on an anomaly's aftermath.
EVAL_GRACE_DAYS = 12


def scenario_windows(anomalies: list) -> list:
    """Ground-truth labels: [{type, t_start, t_end, ...}, ...] in time order."""
    return sorted((a.label() for a in anomalies), key=lambda d: d["t_start"])


def run_clean_and_anomalous(cfg, seed: int, anomalies: list) -> dict:
    """Run the CRN-paired clean twin and anomalous run on one shared draw table.

    Returns {"clean": (log, life, meta), "anomalous": (log, life, meta),
             "windows": [...], "seed": seed}. The two runs share ``draws`` so they
    differ only by the injected anomalies; before the first ``t_start`` they are
    identical (causality).
    """
    draws = draw_randoms(cfg, seed)
    clean = simulate(cfg, draws)
    anomalous = simulate(cfg, draws, anomalies=anomalies)
    return {
        "clean": clean,
        "anomalous": anomalous,
        "windows": scenario_windows(anomalies),
        "seed": seed,
    }
