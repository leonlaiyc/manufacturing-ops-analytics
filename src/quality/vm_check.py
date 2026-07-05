"""
M7 Stage C regression + sanity check (run end to end).

Confirms the virtual-metrology, chamber-matching, and yield-aware what-if
layers (``virtual_metrology.py``, ``chamber_matching.py``,
``decision/yield_whatif.py``) behave correctly:

  GATE 1 — VM reproducibility: two runs of ``fit_and_evaluate`` on the same
           log with the same QualityConfig seed produce identical metrics
           (the whole pipeline is deterministic given the inputs).
  GATE 2 — VM sanity: test R^2 > 0 against the train-mean baseline (the
           model beats "no model" on a held-out FUTURE window), and both
           queue-time violation-flag coefficients are positive (the OLS
           recovers the ground-truth risk drivers injected by Stage B).
  GATE 3 — chamber matching: on a log generated WITH LITHO tool_offsets
           (1.05, 0.95) the mismatch is detected at alpha 0.01 on both
           processing durations and downstream metrology readings; on the
           offset-free default log the verdict stays "not detected" for
           both (false-positive guard).
  GATE 4 — yield what-if exact pairing: a baseline-vs-baseline paired
           comparison through ``yield_whatif.paired_yield_comparison``
           gives EXACTLY zero delta on violation rate, mean p_latent, and
           mean yield in every replication (the shared-QualityConfig-seed
           scheme really does inherit CRN discipline; any nonzero delta
           would mean randomness is escaping the pairing).

Run:  py src/quality/vm_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
for sub in ("generator", "bottleneck", "monitoring", "decision", "quality"):
    sys.path.insert(0, str(SRC / sub))

from factory_generator import default_config, simulate
from yield_model import QualityConfig, build_lot_quality
from queue_time import DEFAULT_WINDOW_HOURS
from virtual_metrology import fit_and_evaluate
from chamber_matching import compare_litho_tools, mismatch_verdict
from yield_whatif import paired_yield_comparison

ALPHA = 0.01


def main() -> int:
    cfg = default_config()                       # locked 60-day, seed 42 line
    log, life, meta = simulate(cfg)
    ok = True

    print("=" * 64)
    print("GATE 1 — VM reproducibility (same seed, identical metrics)")
    print("=" * 64)
    qcfg = QualityConfig(seed=123)
    lot_q_a = build_lot_quality(log, cfg.route, qcfg)
    lot_q_b = build_lot_quality(log, cfg.route, qcfg)
    res_a = fit_and_evaluate(log, cfg.route, lot_q_a, DEFAULT_WINDOW_HOURS)
    res_b = fit_and_evaluate(log, cfg.route, lot_q_b, DEFAULT_WINDOW_HOURS)
    g1 = (res_a.metrics == res_b.metrics
          and res_a.coef_table["coefficient"].equals(res_b.coef_table["coefficient"]))
    print(f"  metrics run A == run B            : {res_a.metrics == res_b.metrics}")
    print(f"  coefficients run A == run B       : "
          f"{res_a.coef_table['coefficient'].equals(res_b.coef_table['coefficient'])}")
    print(f"  RESULT: {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 — VM sanity (R^2 > 0, violation coefficients positive)")
    print("=" * 64)
    lot_q = build_lot_quality(log, cfg.route, QualityConfig())
    res = fit_and_evaluate(log, cfg.route, lot_q, DEFAULT_WINDOW_HOURS)
    r2 = res.metrics["test_r2"]
    auc = res.metrics["auc"]
    coefs = res.coef_table.set_index("feature")["coefficient"]
    c_v1 = float(coefs["viol_litho1"])
    c_v2 = float(coefs["viol_litho2"])
    g2a = r2 > 0.0
    g2b = c_v1 > 0.0 and c_v2 > 0.0
    print(f"  test R^2 (vs train-mean baseline) : {r2:.4f} > 0 : {g2a}")
    print(f"  risk-ranking AUC (defects > 0)    : {auc:.4f}")
    print(f"  coef viol_litho1 = {c_v1:+.5f} > 0, "
          f"coef viol_litho2 = {c_v2:+.5f} > 0 : {g2b}")
    g2 = g2a and g2b
    print(f"  RESULT: {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    print("=" * 64)
    print(f"GATE 3 — chamber matching (detect offsets, alpha={ALPHA};"
          " silent on offset-free)")
    print("=" * 64)
    cfg_off = copy.deepcopy(default_config())
    cfg_off.stations["LITHO"].tool_offsets = (1.05, 0.95)   # tool 1 slower
    log_off, _, _ = simulate(cfg_off)
    lot_q_off = build_lot_quality(log_off, cfg_off.route,
                                  QualityConfig(off_nominal_tool_label="LITHO-1"))
    res_off = compare_litho_tools(log_off, cfg_off.route, lot_q_off)
    verdict_off = mismatch_verdict(res_off, alpha=ALPHA)
    for name, r in res_off.items():
        print(f"  offset log   {name:<20} t p={r.t_pvalue:.3e}  "
              f"MW p={r.u_pvalue:.3e}  d={r.cohens_d:+.3f}  -> {verdict_off[name]}")
    g3a = all(v == "detected" for v in verdict_off.values())

    lot_q_free = build_lot_quality(log, cfg.route, QualityConfig())
    res_free = compare_litho_tools(log, cfg.route, lot_q_free)
    verdict_free = mismatch_verdict(res_free, alpha=ALPHA)
    for name, r in res_free.items():
        print(f"  offset-free  {name:<20} t p={r.t_pvalue:.3e}  "
              f"MW p={r.u_pvalue:.3e}  d={r.cohens_d:+.3f}  -> {verdict_free[name]}")
    g3b = all(v == "not detected" for v in verdict_free.values())
    print(f"  offsets detected on offset log    : {g3a}")
    print(f"  silent on offset-free log         : {g3b}")
    g3 = g3a and g3b
    print(f"  RESULT: {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    print("=" * 64)
    print("GATE 4 — yield what-if exact pairing (baseline vs baseline = 0)")
    print("=" * 64)
    paired = paired_yield_comparison(cfg, copy.deepcopy(cfg),
                                     "baseline-vs-baseline", n_reps=3,
                                     seed0=5000, qcfg=QualityConfig())
    d_viol = paired["d_violation_rate"].abs().max()
    d_p = paired["d_mean_p_latent"].abs().max()
    d_y = paired["d_mean_lot_yield"].abs().max()
    g4 = (d_viol == 0.0) and (d_p == 0.0) and (d_y == 0.0)
    print(f"  max |d_violation_rate|            : {d_viol:.3e}   (must be 0.0)")
    print(f"  max |d_mean_p_latent|             : {d_p:.3e}   (must be 0.0)")
    print(f"  max |d_mean_lot_yield|            : {d_y:.3e}   (must be 0.0)")
    print(f"  RESULT: {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
