"""
M8 Stage C regression + sanity check (run end to end).

Confirms the simulated tool-sensor signatures (``sensor_sim.py``) and the
gradient-boosting health model with SHAP (``pdm_model.py``) behave
correctly. Every gate below is scored against KNOWN synthetic ground truth
this project itself generated (see the framing note at the top of
``sensor_sim.py`` and ``pdm_model.py``): none of these numbers demonstrate
real predictive maintenance capability, only whether the model recovers the
rule this repo wrote.

  GATE 1 - reproducibility: two identical training/scoring runs (fixed seeds)
           produce byte-identical health scores and AUC.
  GATE 2 - causality: recomputing features after corrupting every channel
           value on days at or after a cutoff leaves every row BEFORE that
           cutoff byte-identical, proving no future day leaks into a day's
           rolling features.
  GATE 3 - detection sanity: held-out ROC AUC > 0.7 (comfortably above the
           0.5 chance level on this synthetic ground truth).
  GATE 4 - driver recovery: the two channels ``sensor_sim.DEGRADATION_WIRING``
           wires to the injected ramp (vibration, temperature for LITHO) rank
           in the top 2 of the SHAP global importance table.
  GATE 5 - comparison table integrity: the GB-vs-EWMA table has both
           mean_detection_delay_days and false_alarm_rate populated for both
           methods, scored on the SAME held-out episodes with the SAME
           ``monitoring.evaluation`` vocabulary used for the M5 baseline.

Episode count / runtime: 24 episodes (16 train, 8 held-out test), 90-day
horizon, 5 channels, kept small enough that the whole check (episode
simulation + feature build + GB fit + SHAP + two evaluation passes) runs in
well under a minute on this machine - comfortably inside the ~5 minute budget
even accounting for slower hardware.

Run:  py src/equipment/pdm_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
for sub in ("generator", "monitoring", "equipment"):
    p = str(SRC / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from sensor_sim import make_episodes, DEGRADATION_WIRING
from pdm_model import (
    split_episodes, train_health_model, score_test_episodes, build_features,
    held_out_auc, shap_importance, evaluate_health_scores,
    evaluate_ewma_baseline, comparison_table,
)

STATION = "LITHO"
N_EPISODES = 24
N_TRAIN = 16
HORIZON_DAYS = 90
DURATION_DAYS = 20
BASE_SEED = 9000
AUC_GATE = 0.7


def _run_pipeline():
    """One full episodes -> train -> score pass. Reused by GATE 1 and GATE 3-5."""
    episodes = make_episodes(N_EPISODES, station=STATION, n_tools=2,
                             horizon_days=HORIZON_DAYS, duration_days=DURATION_DAYS,
                             base_seed=BASE_SEED)
    train_eps, test_eps = split_episodes(episodes, n_train=N_TRAIN)
    model, train_table = train_health_model(train_eps)
    scored = score_test_episodes(model, test_eps)
    return episodes, train_eps, test_eps, model, train_table, scored


def main() -> int:
    t0 = time.time()
    ok = True

    print("=" * 64)
    print("GATE 1 - reproducibility (identical health scores, repeated runs)")
    print("=" * 64)
    _, _, _, _, _, scored_a = _run_pipeline()
    _, _, _, _, _, scored_b = _run_pipeline()
    auc_a = held_out_auc(scored_a)
    auc_b = held_out_auc(scored_b)
    g1 = (np.array_equal(scored_a["health_score"].to_numpy(),
                         scored_b["health_score"].to_numpy())
          and auc_a == auc_b)
    print(f"  identical health_score arrays and AUC across two runs "
          f"({auc_a:.6f} == {auc_b:.6f}) : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 - causality (post-label-day masking leaves earlier rows identical)")
    print("=" * 64)
    episodes, train_eps, test_eps, model, train_table, scored = _run_pipeline()
    probe_frame = test_eps[0]["frame"].copy()
    feats_before = build_features(probe_frame)
    cutoff_day = int(probe_frame["day"].median())
    corrupted = probe_frame.copy()
    channels = ["temperature", "vibration", "pressure", "flow", "current"]
    mask = corrupted["day"] >= cutoff_day
    for ch in channels:
        corrupted.loc[mask, ch] = 1.0e9  # garbage future values
    feats_after = build_features(corrupted)
    pre_a = feats_before[feats_before["day"] < cutoff_day].reset_index(drop=True)
    pre_b = feats_after[feats_after["day"] < cutoff_day].reset_index(drop=True)
    g2 = pre_a.equals(pre_b)
    print(f"  feature rows for day < {cutoff_day} unchanged after corrupting "
          f"day >= {cutoff_day} : {g2}")
    ok &= g2

    print("=" * 64)
    print(f"GATE 3 - detection sanity (held-out AUC > {AUC_GATE})")
    print("=" * 64)
    auc = held_out_auc(scored)
    g3 = auc > AUC_GATE
    print(f"  held-out ROC AUC vs known synthetic ground truth = {auc:.4f} "
          f"(> {AUC_GATE}) : {g3}")
    print("  (this measures recovery of a self-generated rule, not real-fab "
          "predictive power - see module framing note)")
    ok &= g3

    print("=" * 64)
    print("GATE 4 - driver recovery (wired channels rank top-2 in SHAP importance)")
    print("=" * 64)
    shap_table = shap_importance(model, train_table)
    wired_channels = set(DEGRADATION_WIRING[STATION].keys())
    top2 = set(shap_table.loc[shap_table["rank"] <= 2, "channel"])
    g4 = wired_channels == top2
    print(shap_table.to_string(index=False))
    print(f"  wired channels {sorted(wired_channels)} == SHAP top-2 "
          f"{sorted(top2)} : {g4}")
    ok &= g4

    print("=" * 64)
    print("GATE 5 - comparison table integrity (GB vs EWMA, same held-out episodes)")
    print("=" * 64)
    strongest_channel = shap_table.iloc[0]["channel"]
    gb_agg = evaluate_health_scores(scored, test_eps)
    ewma_agg = evaluate_ewma_baseline(test_eps, channel=strongest_channel)
    table = comparison_table(gb_agg, ewma_agg, strongest_channel)
    required_cols = {"mean_detection_delay_days", "false_alarm_rate"}
    g5 = (required_cols <= set(table.columns)
          and len(table) == 2
          and table["mean_detection_delay_days"].notna().all()
          and table["false_alarm_rate"].notna().all())
    print(table.to_string(index=False))
    print(f"  table has {required_cols} populated for both methods, "
          f"{len(test_eps)} shared held-out episodes : {g5}")
    print("  EWMA competitive with (or beating) gradient boosting here is an "
          "accepted outcome, not a failure of either method.")
    ok &= g5

    elapsed = time.time() - t0
    print("=" * 64)
    print(f"Runtime: {elapsed:.1f}s for {N_EPISODES} episodes "
          f"({N_TRAIN} train / {N_EPISODES - N_TRAIN} test)")
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
