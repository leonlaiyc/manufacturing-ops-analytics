"""
M8 Stage A regression + sanity check (run end to end).

Confirms the SEMI E10 tool-state layer (``e10_states.py``) and RAM metrics
(``ram_metrics.py``) behave correctly:

  GATE 1 - partition exactness: for every tool, its E10 state intervals sum
           to exactly the horizon and have zero gaps/overlaps (checked to
           1e-9), on a real injected run (scheduled PM + unscheduled
           breakdown together).
  GATE 2 - reproducibility: rebuilding the timeline twice from the same
           simulate() output is identical (no hidden randomness anywhere in
           this layer - it is a pure function of the log and meta).
  GATE 3 - injection recovery: the injected scheduled and unscheduled windows
           reappear in the E10 log attributed to the correct station, with
           the correct type and matching total duration (allowing for the
           PRODUCTIVE-wins truncation documented in ``e10_states.py`` - total
           attributed downtime is <= the injected window length).
  GATE 4 - null case: with no injections, only PRODUCTIVE and STANDBY appear
           across the whole log, and availability = 1.0 for every tool.
  GATE 5 - hand-check: a tiny constructed 1-tool example (2 production runs +
           1 unscheduled-downtime window) against hand-computed MTBF / MTTR /
           availability / utilization constants.

Run:  py src/equipment/equipment_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
for sub in ("generator", "equipment"):
    sys.path.insert(0, str(SRC / sub))

from factory_generator import (
    default_config, draw_randoms, simulate,
    BreakdownAnomaly, ScheduledDowntimeAnomaly,
)
from e10_states import build_tool_state_timeline, E10_STATES
from ram_metrics import ram_metrics_by_tool, ram_metrics_by_station


def main() -> int:
    cfg = default_config()                       # locked 60-day, seed 42 line
    T1 = cfg.horizon_hours
    draws = draw_randoms(cfg, seed=42)
    ok = True

    print("=" * 64)
    print("GATE 1 - partition exactness (sum to horizon, no gaps/overlaps)")
    print("=" * 64)
    scenario = [
        ScheduledDowntimeAnomaly(station="FURNACE", t_start=10 * 24.0,
                                 t_end=10 * 24.0 + 8.0, tools_removed=1),
        BreakdownAnomaly(station="LITHO", t_start=20 * 24.0,
                         t_end=20 * 24.0 + 12.0, tools_removed=1),
    ]
    log_inj, _, meta_inj = simulate(cfg, draws, anomalies=scenario)
    timeline = build_tool_state_timeline(log_inj, meta_inj)
    # Effective horizon (see e10_states.build_tool_state_timeline docstring):
    # the DES drains in-flight lots past the nominal cfg.horizon_hours, so the
    # partition target is the later of the two.
    eff_horizon = max(T1, float(log_inj["process_complete_time"].max()))

    g1 = True
    for tool_id, tdf in timeline.groupby("tool_id"):
        tdf = tdf.sort_values("t_start").reset_index(drop=True)
        total = float((tdf["t_end"] - tdf["t_start"]).sum())
        span_ok = abs(total - eff_horizon) < 1e-9
        starts_at_zero = abs(tdf["t_start"].iloc[0] - 0.0) < 1e-9
        ends_at_horizon = abs(tdf["t_end"].iloc[-1] - eff_horizon) < 1e-9
        no_gap_overlap = all(
            abs(tdf["t_start"].iloc[i + 1] - tdf["t_end"].iloc[i]) < 1e-9
            for i in range(len(tdf) - 1)
        )
        tool_ok = span_ok and starts_at_zero and ends_at_horizon and no_gap_overlap
        if not tool_ok:
            print(f"  [FAIL] {tool_id}: total={total:.6f} (horizon={eff_horizon:.6f}), "
                  f"span_ok={span_ok} bounds_ok={starts_at_zero and ends_at_horizon} "
                  f"contig={no_gap_overlap}")
        g1 &= tool_ok
    n_tools = timeline["tool_id"].nunique()
    print(f"  {n_tools} tools partition [0, {eff_horizon:.2f}] exactly (<=1e-9) : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 - reproducibility (identical output, fixed seed)")
    print("=" * 64)
    timeline_b = build_tool_state_timeline(log_inj, meta_inj)
    g2 = timeline.equals(timeline_b)
    print(f"  two builds from the same run are identical : {g2}")
    ok &= g2

    print("=" * 64)
    print("GATE 3 - injection recovery (windows reappear, correct type/duration)")
    print("=" * 64)
    sched = timeline[(timeline["station"] == "FURNACE")
                      & (timeline["e10_state"] == "SCHEDULED DOWNTIME")]
    unsched = timeline[(timeline["station"] == "LITHO")
                        & (timeline["e10_state"] == "UNSCHEDULED DOWNTIME")]
    sched_hours = float((sched["t_end"] - sched["t_start"]).sum())
    unsched_hours = float((unsched["t_end"] - unsched["t_start"]).sum())
    g3a = len(sched) > 0 and sched_hours <= 8.0 + 1e-9
    g3b = len(unsched) > 0 and unsched_hours <= 12.0 + 1e-9
    print(f"  SCHEDULED DOWNTIME at FURNACE: {sched_hours:.4f}h "
          f"(injected 8.0000h, PRODUCTIVE-wins truncation allowed) : {g3a}")
    print(f"  UNSCHEDULED DOWNTIME at LITHO: {unsched_hours:.4f}h "
          f"(injected 12.0000h, PRODUCTIVE-wins truncation allowed) : {g3b}")
    # And the convention: attributed to the HIGHEST-index tool of the station.
    g3c = set(sched["tool_id"].unique()) <= {"FURNACE-2"}
    g3d = set(unsched["tool_id"].unique()) <= {"LITHO-2"}
    print(f"  attributed to highest-index tool (FURNACE-2, LITHO-2) : "
          f"{g3c and g3d}")
    ok &= g3a and g3b and g3c and g3d

    print("=" * 64)
    print("GATE 4 - null case (no injections: PRODUCTIVE/STANDBY only, avail=1.0)")
    print("=" * 64)
    log_clean, _, meta_clean = simulate(cfg, draws)
    timeline_clean = build_tool_state_timeline(log_clean, meta_clean)
    observed_states = set(timeline_clean["e10_state"].unique())
    g4a = observed_states <= {"PRODUCTIVE", "STANDBY"}
    print(f"  states observed: {sorted(observed_states)} "
          f"(subset of PRODUCTIVE/STANDBY) : {g4a}")
    ram_clean = ram_metrics_by_tool(timeline_clean)
    g4b = bool((ram_clean["availability"] == 1.0).all())
    print(f"  availability == 1.0 for all {len(ram_clean)} tools : {g4b}")
    ok &= g4a and g4b

    print("=" * 64)
    print("GATE 5 - hand-check (tiny constructed example vs hand-computed values)")
    print("=" * 64)
    # 1 tool "X-1" at station "X", horizon 10h:
    #   PRODUCTIVE [1,3] (2h), UNSCHEDULED DOWNTIME [4,5] (1h),
    #   PRODUCTIVE [6,9] (3h), STANDBY fills [0,1]+[3,4]+[5,6]+[9,10] = 4h.
    # Hand computation:
    #   total=10, up=10-1=9, productive=5
    #   availability = 9/10 = 0.9;  utilization = 5/10 = 0.5
    #   1 failure: up-run before it = 4-0 = 4h -> MTBF = 4.0h
    #   MTTR = duration of the one failure = 5-4 = 1.0h
    HAND_MTBF = 4.0
    HAND_MTTR = 1.0
    HAND_AVAILABILITY = 0.9
    HAND_UTILIZATION = 0.5

    tiny_log = pd.DataFrame([
        {"lot_id": 0, "product_type": "P1", "step_seq": 0, "station": "X",
         "queue_entry_time": 0.5, "process_start_time": 1.0,
         "process_complete_time": 3.0, "tool_id": "X-1"},
        {"lot_id": 1, "product_type": "P1", "step_seq": 0, "station": "X",
         "queue_entry_time": 5.5, "process_start_time": 6.0,
         "process_complete_time": 9.0, "tool_id": "X-1"},
    ])
    tiny_meta = {
        "horizon_hours": 10.0,
        "stations": {"X": {"n_tools": 1, "batch_size": 1,
                            "pt_mean": 2.0, "pt_cv": 0.5}},
        "anomalies": [
            {"type": "breakdown", "station": "X",
             "t_start": 4.0, "t_end": 5.0, "tools_removed": 1},
        ],
    }
    tiny_timeline = build_tool_state_timeline(tiny_log, tiny_meta)
    tiny_ram = ram_metrics_by_tool(tiny_timeline).set_index("tool_id").loc["X-1"]

    g5a = abs(tiny_ram["mtbf_hours"] - HAND_MTBF) < 1e-9
    g5b = abs(tiny_ram["mttr_hours"] - HAND_MTTR) < 1e-9
    g5c = abs(tiny_ram["availability"] - HAND_AVAILABILITY) < 1e-9
    g5d = abs(tiny_ram["utilization"] - HAND_UTILIZATION) < 1e-9
    print(f"  MTBF   {tiny_ram['mtbf_hours']:.6f} == {HAND_MTBF} : {g5a}")
    print(f"  MTTR   {tiny_ram['mttr_hours']:.6f} == {HAND_MTTR} : {g5b}")
    print(f"  avail  {tiny_ram['availability']:.6f} == {HAND_AVAILABILITY} : {g5c}")
    print(f"  util   {tiny_ram['utilization']:.6f} == {HAND_UTILIZATION} : {g5d}")
    g5 = g5a and g5b and g5c and g5d
    ok &= g5

    # Sanity: ENGINEERING is a declared state but never populated by construction.
    all_states_seen = set(timeline["e10_state"]) | set(timeline_clean["e10_state"])
    print(f"  ENGINEERING declared in schema but unused : "
          f"{'ENGINEERING' in E10_STATES and 'ENGINEERING' not in all_states_seen}")

    # Station-level rollup sanity (not a separate gate; smoke-tests the API).
    station_ram = ram_metrics_by_station(timeline)
    print(f"  ram_metrics_by_station produced {len(station_ram)} station rows "
          f"(smoke test)")

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
