"""
M9 Stage A dispatch-policy check (run end to end).

Confirms the configurable dispatch layer added to factory_generator.py
(queue_discipline, due dates, release control) behaves correctly and never
disturbs the locked FIFO default:

  GATE 1 - FIFO byte-identity: a run with cfg.queue_discipline explicitly set
           to "fifo" (default) produces an event log and lifecycle identical
           row-for-row to a run that never touches the new M9 fields, and the
           existing golden-file regression (crn_check.py GATE 1) still passes
           untouched.
  GATE 2 - reproducibility: for every policy, two runs with the same seed
           (lazy path) produce byte-identical logs.
  GATE 3 - policy sanity: under EDD, spot-assert on the log that whenever lot
           A is dispatched at a station while lot B is already waiting in
           that station's queue, A's due date is never later than B's
           (A "jumping the queue" over a lot with an earlier due date would be
           an EDD violation). Analogous spot-assert for queue_time_aware at
           ETCH (least post-litho slack served first).
  GATE 4 - CRN pairing across policies: same seed (same RandomDraws table),
           two different queue_discipline values -> every lot's realized
           processing-time draw at each (station, visit) is identical across
           the two runs (proves the draw-indexing argument in the module
           docstring: draws are keyed by (lot_id, route_step), never by queue
           position, so reordering the queue cannot desynchronize CRN pairing).
  GATE 5 - release control: with a low LITHO WIP threshold, measured LITHO WIP
           (queued + in-process, both re-entrant visits) never exceeds
           threshold + n_tools(LITHO) at any completion event (the in-flight
           tolerance: up to n_tools additional lots can already be IN SERVICE
           at the instant the threshold was last checked, since release only
           gates NEW arrivals into CLEAN, not lots already admitted upstream
           of LITHO on their way there).

Run:  py src/generator/dispatch_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factory_generator import (
    ReleaseControlConfig,
    default_config,
    draw_randoms,
    simulate,
)

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "quality"))
from queue_time import DEFAULT_WINDOW_HOURS  # noqa: E402


# --------------------------------------------------------------------------- #
# GATE 1 - FIFO byte-identity
# --------------------------------------------------------------------------- #
def gate1_fifo_byte_identity() -> bool:
    print("=" * 64)
    print("GATE 1 - FIFO byte-identity (default vs explicit queue_discipline)")
    print("=" * 64)

    cfg_implicit = default_config()                 # never touches M9 fields
    cfg_explicit = default_config()
    cfg_explicit.queue_discipline = "fifo"           # explicit, still the default value

    log_i, life_i, _ = simulate(cfg_implicit)
    log_e, life_e, _ = simulate(cfg_explicit)

    log_same = log_i.equals(log_e)
    # due_date is a new column; compare it separately from the pre-existing
    # arrival/completion columns so a schema-shape mismatch is caught clearly.
    life_same = life_i.equals(life_e)

    print(f"  event_log identical      : {log_same}")
    print(f"  lot_lifecycle identical  : {life_same}")

    ok = log_same and life_same
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def gate1b_golden_file_regression() -> bool:
    """The pre-existing crn_check.py GATE 1 (golden-file byte comparison) must
    still pass untouched - i.e. adding M9 fields with their defaults does not
    change a single byte of the default simulate(cfg) output. Re-run it here
    directly (not as a subprocess) so a failure surfaces in this script's exit
    code too.
    """
    print("=" * 64)
    print("GATE 1b - crn_check.py golden-file regression (re-run here)")
    print("=" * 64)

    SYN = Path(__file__).resolve().parents[2] / "data" / "synthetic"
    cfg = default_config()
    log, lifecycle, _ = simulate(cfg)

    def csv_bytes(df: pd.DataFrame) -> bytes:
        import io
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    ok = True
    for name, df in [("event_log.csv", log), ("lot_lifecycle.csv", lifecycle)]:
        golden_path = SYN / name
        if not golden_path.exists():
            print(f"  [SKIP] {name}: no committed golden at {golden_path}")
            continue
        fresh = csv_bytes(df)
        golden = golden_path.read_bytes()
        same = fresh.replace(b"\r\n", b"\n") == golden.replace(b"\r\n", b"\n")
        print(f"  {name:<20} identical={same}")
        ok = ok and same

    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# GATE 2 - reproducibility per policy
# --------------------------------------------------------------------------- #
def gate2_reproducibility() -> bool:
    print("=" * 64)
    print("GATE 2 - reproducibility under fixed seed, per policy")
    print("=" * 64)

    ok = True
    for policy in ["fifo", "edd", "critical_ratio", "queue_time_aware"]:
        cfg = default_config()
        cfg.queue_discipline = policy
        log1, life1, _ = simulate(cfg)
        log2, life2, _ = simulate(cfg)
        same = log1.equals(log2) and life1.equals(life2)
        print(f"  {policy:<18} two runs identical : {same}")
        ok = ok and same

    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# GATE 3 - policy sanity (EDD, queue_time_aware)
# --------------------------------------------------------------------------- #
def _spot_assert_priority(log: pd.DataFrame, lifecycle: pd.DataFrame,
                           station: str, key_col: str, ascending: bool) -> tuple[int, int]:
    """At ``station``, for every pair of runs where a lot B was ALREADY
    WAITING (queue_entry_time < the other run's process_start_time) when a
    run led by lot A started, assert A's priority key is never worse than B's
    priority key AS IT STOOD AT THE INSTANT A WAS DISPATCHED (worse = later
    due date for EDD, more slack for queue_time_aware, since ascending=True
    means "smaller is more urgent").

    For "due_date" (EDD) the key is time-invariant, so B's own row value is
    used directly. For "slack" (queue_time_aware) the key SHRINKS as time
    passes, so B's slack must be recomputed at A's dispatch instant
    (``W - (a.process_start_time - b.queue_entry_time)``), NOT read from B's
    own eventual process_start_time - comparing two different reference
    instants would be comparing apples to oranges and produce false positives.

    Returns (violations, comparisons_checked).
    """
    ops = log[log["station"] == station].copy()
    ops = ops.merge(lifecycle[["lot_id", "due_date"]], on="lot_id", how="left")
    if key_col == "slack":
        ops = ops[ops["queue_entry_time"] > 0]  # only rows that came straight from a LITHO visit

    starts = ops[["lot_id", "process_start_time", "queue_entry_time", "due_date"]].drop_duplicates()
    violations = 0
    checked = 0
    # Compare every dispatched run's lead-lot key against every OTHER lot that
    # was already queued (qentry earlier) but started later at this station.
    for _, a in starts.iterrows():
        waiting = starts[(starts["queue_entry_time"] < a["process_start_time"])
                         & (starts["process_start_time"] > a["process_start_time"])]
        for _, b in waiting.iterrows():
            checked += 1
            if key_col == "due_date":
                a_key, b_key = a["due_date"], b["due_date"]
            else:  # "slack", evaluated at A's dispatch instant for both
                a_key = DEFAULT_WINDOW_HOURS - (a["process_start_time"] - a["queue_entry_time"])
                b_key = DEFAULT_WINDOW_HOURS - (a["process_start_time"] - b["queue_entry_time"])
            bad = (a_key > b_key) if ascending else (a_key < b_key)
            if bad:
                violations += 1
    return violations, checked


def gate3_policy_sanity() -> bool:
    print("=" * 64)
    print("GATE 3 - policy sanity (EDD, queue_time_aware spot-asserts)")
    print("=" * 64)

    # EDD: across ALL stations, a dispatched-first lot should not have a later
    # due date than a lot left waiting at the SAME station.
    cfg = default_config()
    cfg.queue_discipline = "edd"
    log, lifecycle, _ = simulate(cfg)

    edd_violations = 0
    edd_checked = 0
    for station in cfg.stations:
        v, c = _spot_assert_priority(log, lifecycle, station, "due_date", ascending=True)
        edd_violations += v
        edd_checked += c
    print(f"  EDD: comparisons checked={edd_checked}  violations={edd_violations}")
    edd_ok = edd_checked > 0 and edd_violations == 0

    # queue_time_aware at ETCH: least remaining post-litho slack served first.
    cfg2 = default_config()
    cfg2.queue_discipline = "queue_time_aware"
    log2, lifecycle2, _ = simulate(cfg2)
    qta_violations, qta_checked = _spot_assert_priority(
        log2, lifecycle2, "ETCH", "slack", ascending=True)
    print(f"  queue_time_aware @ ETCH: comparisons checked={qta_checked}  "
          f"violations={qta_violations}")
    qta_ok = qta_checked > 0 and qta_violations == 0

    ok = edd_ok and qta_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# GATE 4 - CRN pairing across policies
# --------------------------------------------------------------------------- #
def gate4_crn_pairing_across_policies() -> bool:
    print("=" * 64)
    print("GATE 4 - CRN pairing across policies (same table, different discipline)")
    print("=" * 64)

    base = default_config()
    draws = draw_randoms(base, seed=base.seed)

    cfg_fifo = copy.deepcopy(base)
    cfg_fifo.queue_discipline = "fifo"
    cfg_edd = copy.deepcopy(base)
    cfg_edd.queue_discipline = "edd"

    log_fifo, _, _ = simulate(cfg_fifo, draws)
    log_edd, _, _ = simulate(cfg_edd, draws)

    # No tool_offsets are set in either config, so at a SERIAL station
    # (batch_size 1) each run computes process_complete_time EXACTLY as
    # process_start_time + draws.proc_times[lot][step] (see simulate()'s
    # try_dispatch: push(now + pt, ...)). Asserting that identity BIT-EXACTLY
    # in BOTH logs, against the ONE shared table, proves both runs consumed
    # the identical draw for every (lot, station-visit) - the CRN pairing
    # claim - without the float-roundoff trap of comparing reconstructed
    # durations across runs: (now1 + pt) - now1 and (now2 + pt) - now2 differ
    # in the last bits when now1 != now2 even though pt is identical, so a
    # duration-vs-duration comparison would show ~1e-13 noise that has nothing
    # to do with CRN pairing.
    #
    # At the one BATCH station (FURNACE), the run's realized duration is BY
    # DESIGN the first-loaded lot's draw (see module docstring "Batch
    # semantics"), and a different queue_discipline can legitimately choose a
    # DIFFERENT lead lot for the same batch - so a member row's realized
    # duration may differ between fifo and edd even though the underlying
    # table is identical. That is unchanged batch semantics, not a CRN break.
    # There the exact assertion is: each run's batch completion time equals
    # start + SOME member's table draw (the lead's) bit-exactly, proving
    # FURNACE also reads only draws.proc_times[lot][step] values from the
    # shared table.
    serial_stations = [s for s, st in base.stations.items() if st.batch_size == 1]

    def exact_table_mismatches(log: pd.DataFrame) -> tuple[int, int]:
        """(mismatches, rows_checked): serial-station rows where
        start + table_draw != complete bit-exactly."""
        rows = log[log["station"].isin(serial_stations)]
        bad = 0
        for lot, step, start, comp in zip(rows["lot_id"], rows["step_seq"],
                                          rows["process_start_time"],
                                          rows["process_complete_time"]):
            if start + draws.proc_times[lot][step] != comp:
                bad += 1
        return bad, len(rows)

    bad_fifo, n_fifo = exact_table_mismatches(log_fifo)
    bad_edd, n_edd = exact_table_mismatches(log_edd)

    def furnace_lead_mismatches(log: pd.DataFrame) -> tuple[int, int]:
        """(mismatches, runs_checked): FURNACE runs whose completion time does
        not equal start + SOME member's table draw (the lead's) bit-exactly."""
        fur = log[log["station"] == "FURNACE"]
        bad = 0
        n_runs = 0
        runs = fur.groupby(["process_start_time", "process_complete_time", "tool_id"])
        for (start, comp, _tool), grp in runs:
            n_runs += 1
            ok_run = any(start + draws.proc_times[lot][step] == comp
                         for lot, step in zip(grp["lot_id"], grp["step_seq"]))
            if not ok_run:
                bad += 1
        return bad, n_runs

    fbad_fifo, fruns_fifo = furnace_lead_mismatches(log_fifo)
    fbad_edd, fruns_edd = furnace_lead_mismatches(log_edd)

    print(f"  serial rows checked (fifo run)   : {n_fifo}  exact mismatches: {bad_fifo}")
    print(f"  serial rows checked (edd run)    : {n_edd}  exact mismatches: {bad_edd}")
    print(f"  FURNACE runs checked (fifo run)  : {fruns_fifo}  lead-draw mismatches: {fbad_fifo}")
    print(f"  FURNACE runs checked (edd run)   : {fruns_edd}  lead-draw mismatches: {fbad_edd}")
    print("  (both runs share ONE RandomDraws table; bit-exact start+draw==complete")
    print("   in both logs proves each (lot, visit) consumed the identical draw)")

    ok = (bad_fifo == 0 and bad_edd == 0 and n_fifo > 0 and n_edd > 0
          and fbad_fifo == 0 and fbad_edd == 0 and fruns_fifo > 0 and fruns_edd > 0)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# GATE 5 - release control (bottleneck-WIP)
# --------------------------------------------------------------------------- #
def gate5_release_control() -> bool:
    print("=" * 64)
    print("GATE 5 - release control (LITHO WIP bound)")
    print("=" * 64)

    threshold = 3
    cfg = default_config()
    cfg.release_control = ReleaseControlConfig(litho_wip_threshold=threshold)
    n_tools_litho = cfg.stations["LITHO"].n_tools

    log, lifecycle, _ = simulate(cfg)

    # Reconstruct LITHO WIP (queued + in-process, both re-entrant visits) at
    # every event boundary in the log: a lot is "at LITHO" for
    # [queue_entry_time, process_complete_time) on either visit.
    litho_rows = log[log["station"] == "LITHO"]
    boundaries = sorted(set(litho_rows["queue_entry_time"]).union(
        litho_rows["process_complete_time"]))

    bound = threshold + n_tools_litho
    max_observed = 0
    breaches = 0
    for t in boundaries:
        wip = int(((litho_rows["queue_entry_time"] <= t)
                    & (litho_rows["process_complete_time"] > t)).sum())
        max_observed = max(max_observed, wip)
        if wip > bound:
            breaches += 1

    print(f"  threshold                : {threshold}")
    print(f"  n_tools(LITHO)           : {n_tools_litho}")
    print(f"  asserted bound (thr+n)   : {bound}")
    print(f"  max observed LITHO WIP   : {max_observed}")
    print(f"  boundary breaches        : {breaches}   (must be 0)")

    ok = breaches == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    g1 = gate1_fifo_byte_identity()
    g1b = gate1b_golden_file_regression()
    g2 = gate2_reproducibility()
    g3 = gate3_policy_sanity()
    g4 = gate4_crn_pairing_across_policies()
    g5 = gate5_release_control()
    all_ok = g1 and g1b and g2 and g3 and g4 and g5
    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if all_ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
