"""
M11 Stage B regression + sanity check (run end to end).

Confirms the rolling-window drift monitor (``drift.py``) and split conformal
intervals for the M7 virtual-metrology model (``conformal.py``) behave as
documented:

  GATE 1 - drift recovery: on the M5 longer-horizon config with a single
           injected ``DegradationAnomaly`` at LITHO (known onset day 110),
           the monitor's ``daily_mean_process_time`` channel alarms with a
           finite delay; on the CRN-paired clean twin (same draw table, no
           anomaly), the same channel produces zero alarms at the documented
           threshold (test_window_days=5, z_threshold=7.0, k_consecutive=3).
  GATE 2 - arrival-shift recovery: a stylized rate cut-over (two independent
           runs concatenated day after day, the second at a higher
           ``arrival_rate``) is detected on the ``daily_arrival_count``
           channel with a finite delay; a clean concatenation (two runs at
           the SAME rate) stays quiet on that channel at the same threshold.
  GATE 3 - conformal coverage: split conformal intervals for the M7 VM model,
           nominal 90% coverage, measured empirical coverage on a held-out
           test split of size ~286 lots (20% of the ~1430-lot default run)
           falls inside [0.85, 0.96]. The band is documented as a Wilson-type
           tolerance around 0.90 sized for this n: at n=286 the binomial
           standard error of an observed coverage rate is
           sqrt(0.9*0.1/286) ~= 0.0177, so +/-0.05 around 0.90 is roughly a
           2.8-sigma band, wide enough to absorb sampling noise from one
           fixed-seed run without being so wide it would pass a badly
           miscalibrated interval.
  GATE 4 - noise sensitivity: doubling ``metrology_sigma`` in QualityConfig
           (via ``build_lot_quality``'s ``metrology_sigma`` kwarg) strictly
           widens the conformal interval (2*q_alpha), since a noisier target
           has larger calibration residuals by construction.
  GATE 5 - reproducibility: two runs of ``rolling_drift_scan`` on the same
           series/params give identical scores and alarms; two runs of
           ``fit_and_calibrate`` on the same log/QualityConfig seed give an
           identical q_alpha.
  GATE 6 - model-card meta-gate: both
           ``docs/model_cards/virtual_metrology.md`` and
           ``docs/model_cards/pdm_health_model.md`` exist, contain every
           required section header, and contain zero em dash characters.

Run:  py src/dataquality/reliability_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
REPO_ROOT = SRC.parent
for sub in ("generator", "monitoring", "quality", "dataquality"):
    sys.path.insert(0, str(SRC / sub))

from factory_generator import default_config, draw_randoms, simulate, DegradationAnomaly
from m5_config import m5_config
from yield_model import QualityConfig, build_lot_quality
from queue_time import DEFAULT_WINDOW_HOURS

from drift import (
    daily_mean_process_time, build_arrival_shift_series,
    rolling_drift_scan, detection_delay_days, false_alarm_count,
    TEST_WINDOW_DAYS, Z_THRESHOLD, K_CONSECUTIVE,
)
from conformal import fit_and_calibrate, DEFAULT_ALPHA

REF_START, REF_END = 10, 48   # clean baseline window, see monitoring/anomalies.py
EM_DASH = "—"

REQUIRED_SECTIONS = [
    "Purpose", "Data", "Method", "Metrics", "Intended use",
    "Out-of-scope uses", "Failure modes", "Trust boundary",
]
MODEL_CARDS = [
    REPO_ROOT / "docs" / "model_cards" / "virtual_metrology.md",
    REPO_ROOT / "docs" / "model_cards" / "pdm_health_model.md",
]


def main() -> int:
    ok = True

    print("=" * 64)
    print("GATE 1 - drift recovery (injected LITHO degradation, known onset)")
    print("=" * 64)
    cfg = m5_config()  # 120-day horizon, seed 42, clean baseline days 10-48
    draws = draw_randoms(cfg, seed=42)
    onset_day, end_day = 110, 135
    degradation = DegradationAnomaly(station="LITHO", t_onset=onset_day * 24.0,
                                      t_end=end_day * 24.0, alpha=0.00025)

    log_clean, _, _ = simulate(cfg, draws)
    log_anom, _, _ = simulate(cfg, draws, anomalies=[degradation])

    series_clean = daily_mean_process_time(log_clean, "LITHO")
    series_anom = daily_mean_process_time(log_anom, "LITHO")

    report_clean = rolling_drift_scan(series_clean, REF_START, REF_END, channel="litho_pt_clean")
    report_anom = rolling_drift_scan(series_anom, REF_START, REF_END, channel="litho_pt_anomalous")

    delay = detection_delay_days(report_anom, onset_day)
    fa_clean = false_alarm_count(report_clean)
    g1a = delay is not None and delay >= 0
    g1b = fa_clean == 0
    print(f"  monitor params: test_window={TEST_WINDOW_DAYS}d z={Z_THRESHOLD} k={K_CONSECUTIVE}")
    print(f"  injected onset day = {onset_day}, first alarm day = {report_anom.first_alarm_day}")
    print(f"  detection delay (days) = {delay} : finite and non-negative = {g1a}")
    print(f"  clean-run false alarms on this channel = {fa_clean} (target 0) : {g1b}")
    g1 = g1a and g1b
    print(f"  RESULT: {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 - arrival-shift recovery (stylized rate cut-over)")
    print("=" * 64)
    cfg_before = m5_config(horizon_days=60, seed=42)
    draws_before = draw_randoms(cfg_before, seed=42)
    log_before, life_before, _ = simulate(cfg_before, draws_before)

    cfg_after_shift = copy.deepcopy(m5_config(horizon_days=60, seed=99))
    cfg_after_shift.arrival_rate = 1.5   # cut-over to a higher rate
    draws_after_shift = draw_randoms(cfg_after_shift, seed=99)
    _, life_after_shift, _ = simulate(cfg_after_shift, draws_after_shift)

    cfg_after_clean = m5_config(horizon_days=60, seed=99)   # same rate: clean control
    draws_after_clean = draw_randoms(cfg_after_clean, seed=99)
    _, life_after_clean, _ = simulate(cfg_after_clean, draws_after_clean)

    cutover_day = int(cfg_before.horizon_hours / 24)
    series_shift = build_arrival_shift_series(life_before, cfg_before.horizon_hours,
                                              life_after_shift, cfg_after_shift.horizon_hours)
    series_shift_clean = build_arrival_shift_series(life_before, cfg_before.horizon_hours,
                                                     life_after_clean, cfg_after_clean.horizon_hours)

    report_shift = rolling_drift_scan(series_shift, REF_START, REF_END, channel="arrivals_shift")
    report_shift_clean = rolling_drift_scan(series_shift_clean, REF_START, REF_END,
                                            channel="arrivals_no_shift")

    delay2 = detection_delay_days(report_shift, cutover_day)
    fa_clean2 = false_alarm_count(report_shift_clean)
    g2a = delay2 is not None and delay2 >= 0
    g2b = fa_clean2 == 0
    print(f"  stylized cut-over: arrival_rate {cfg_before.arrival_rate} -> "
          f"{cfg_after_shift.arrival_rate} lots/h at day {cutover_day}")
    print(f"  first alarm day = {report_shift.first_alarm_day}, "
          f"detection delay (days) = {delay2} : {g2a}")
    print(f"  same-rate concatenation false alarms = {fa_clean2} (target 0) : {g2b}")
    g2 = g2a and g2b
    print(f"  RESULT: {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    print("=" * 64)
    print("GATE 3 - conformal coverage (nominal 90%, band [0.85, 0.96])")
    print("=" * 64)
    vm_cfg = default_config()   # locked 60-day, seed 42 line
    vm_log, _, _ = simulate(vm_cfg)
    qcfg = QualityConfig()
    lot_q = build_lot_quality(vm_log, vm_cfg.route, qcfg)
    conf_res = fit_and_calibrate(vm_log, vm_cfg.route, lot_q, DEFAULT_WINDOW_HOURS,
                                 alpha=DEFAULT_ALPHA)
    g3 = 0.85 <= conf_res.test_coverage <= 0.96
    print(f"  n_train={conf_res.n_train}  n_cal={conf_res.n_cal}  n_test={conf_res.n_test}")
    print(f"  nominal coverage = {1 - DEFAULT_ALPHA:.2f}, "
          f"empirical test coverage = {conf_res.test_coverage:.4f}")
    print(f"  q_alpha = {conf_res.q_alpha:.5f}, mean interval width = {conf_res.test_mean_width:.5f}")
    print(f"  coverage in [0.85, 0.96] : {g3}")
    print(f"  RESULT: {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    print("=" * 64)
    print("GATE 4 - noise sensitivity (doubled metrology sigma widens interval)")
    print("=" * 64)
    base_sigma = 0.01   # yield_model.DEFAULT_METROLOGY_SIGMA
    doubled_sigma = base_sigma * 2
    lot_q_base = build_lot_quality(vm_log, vm_cfg.route, qcfg, metrology_sigma=base_sigma)
    lot_q_double = build_lot_quality(vm_log, vm_cfg.route, qcfg, metrology_sigma=doubled_sigma)
    res_base = fit_and_calibrate(vm_log, vm_cfg.route, lot_q_base, DEFAULT_WINDOW_HOURS)
    res_double = fit_and_calibrate(vm_log, vm_cfg.route, lot_q_double, DEFAULT_WINDOW_HOURS)
    g4 = res_double.test_mean_width > res_base.test_mean_width
    print(f"  base sigma={base_sigma}      width = {res_base.test_mean_width:.5f}")
    print(f"  doubled sigma={doubled_sigma}  width = {res_double.test_mean_width:.5f}")
    print(f"  doubled-noise width strictly greater : {g4}")
    print(f"  RESULT: {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    print("=" * 64)
    print("GATE 5 - reproducibility (identical reports/quantiles on repeat)")
    print("=" * 64)
    report_anom_b = rolling_drift_scan(series_anom, REF_START, REF_END, channel="litho_pt_anomalous")
    g5a = (report_anom.scores.equals(report_anom_b.scores)
           and report_anom.alarms.equals(report_anom_b.alarms))
    conf_res_b = fit_and_calibrate(vm_log, vm_cfg.route, lot_q, DEFAULT_WINDOW_HOURS,
                                   alpha=DEFAULT_ALPHA)
    g5b = conf_res.q_alpha == conf_res_b.q_alpha
    print(f"  two drift scans identical (scores + alarms) : {g5a}")
    print(f"  two conformal calibrations identical q_alpha "
          f"({conf_res.q_alpha:.6f} == {conf_res_b.q_alpha:.6f}) : {g5b}")
    g5 = g5a and g5b
    print(f"  RESULT: {'PASS' if g5 else 'FAIL'}")
    ok &= g5

    print("=" * 64)
    print("GATE 6 - model-card meta-gate (sections present, zero em dash)")
    print("=" * 64)
    g6 = True
    for path in MODEL_CARDS:
        exists = path.exists()
        if not exists:
            print(f"  {path.name}: MISSING")
            g6 = False
            continue
        text = path.read_text(encoding="utf-8")
        missing_sections = [s for s in REQUIRED_SECTIONS if f"## {s}" not in text]
        has_em_dash = EM_DASH in text
        line_count = len(text.splitlines())
        card_ok = (not missing_sections) and (not has_em_dash)
        print(f"  {path.name}: {line_count} lines, "
              f"missing sections = {missing_sections}, "
              f"em dash present = {has_em_dash} : {'PASS' if card_ok else 'FAIL'}")
        g6 &= card_ok
    print(f"  RESULT: {'PASS' if g6 else 'FAIL'}")
    ok &= g6

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
