"""
M7 Stage B regression + sanity check (run end to end).

Confirms the lot-level yield ground-truth layer (``queue_time.py`` +
``yield_model.py``) behaves correctly and stays calibrated:

  GATE 1 — reproducibility: two runs of ``build_lot_quality`` with the same
           seed produce identical DataFrames (the dedicated numpy Generator
           is deterministic given the config).
  GATE 2 — null case: all additive effect coefficients set to 0 gives
           p_latent == p_base for every lot exactly, and mean realized yield
           falls within 3 standard errors of the Binomial-implied
           ``1 - p_base``.
  GATE 3 — monotonicity: violating lots (either LITHO visit) have a strictly
           higher mean p_latent than non-violating lots (exact, since the
           additive term is deterministic given the flags), and a higher
           mean realized defect rate (directional, on the seeded default
           run — realized outcomes carry sampling noise, so this is a
           sanity direction check, not an exact equality).
  GATE 4 — calibration band: the post-litho queue-time window's baseline
           violation rate on the default-seed log falls in [0.05, 0.20]
           (the fixed window was calibrated to ~0.10; this gate guards
           against silent drift if the generator or window constant ever
           changes without re-calibration).
  GATE 5 — chamber effect: on a log generated WITH LITHO tool_offsets
           (1.05, 0.95), lots that took the slower (off-nominal) tool at
           either visit have a higher mean p_latent than lots that did not,
           when the chamber effect is enabled via ``off_nominal_tool_label``.

Run:  py src/quality/quality_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / "generator"))
sys.path.insert(0, str(SRC / "quality"))

from factory_generator import default_config, simulate
from queue_time import calibrate_window, DEFAULT_WINDOW_HOURS
from yield_model import QualityConfig, build_lot_quality, WAFERS_PER_LOT


def main() -> int:
    cfg = default_config()                       # locked 60-day, seed 42 line
    log, life, meta = simulate(cfg)
    ok = True

    print("=" * 64)
    print("GATE 1 — reproducibility (same seed, identical output)")
    print("=" * 64)
    qcfg = QualityConfig(seed=123)
    df_a = build_lot_quality(log, cfg.route, qcfg)
    df_b = build_lot_quality(log, cfg.route, qcfg)
    g1 = df_a.equals(df_b)
    print(f"  two runs with seed=123 identical : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 — null case (all effects zeroed)")
    print("=" * 64)
    p_base = 0.02
    null_cfg = QualityConfig(p_base=p_base, a_viol1=0.0, a_viol2=0.0,
                              a_chamber=0.0, a_pt_excess=0.0,
                              off_nominal_tool_label=None, seed=7)
    df_null = build_lot_quality(log, cfg.route, null_cfg)
    g2a = bool((df_null["p_latent"] == p_base).all())
    print(f"  p_latent == p_base for every lot : {g2a}")

    n = len(df_null)
    expected_mean_yield = 1.0 - p_base
    # SE of mean realized yield under Binomial(25, p_base)/25 per lot, n lots.
    var_yield_per_lot = p_base * (1 - p_base) / WAFERS_PER_LOT
    se_mean_yield = np.sqrt(var_yield_per_lot / n)
    observed_mean_yield = float(df_null["lot_yield"].mean())
    z = abs(observed_mean_yield - expected_mean_yield) / se_mean_yield
    g2b = z <= 3.0
    print(f"  mean realized yield {observed_mean_yield:.5f} vs expected "
          f"{expected_mean_yield:.5f} (z={z:.2f}, <=3 SE): {g2b}")
    ok &= g2a and g2b

    print("=" * 64)
    print("GATE 3 — monotonicity (violating lots riskier)")
    print("=" * 64)
    default_cfg = QualityConfig()  # p_base=0.02, defaults per spec
    df_default = build_lot_quality(log, cfg.route, default_cfg)
    violating = df_default["viol_litho1"] | df_default["viol_litho2"]
    p_viol = df_default.loc[violating, "p_latent"].mean()
    p_clean = df_default.loc[~violating, "p_latent"].mean()
    g3a = p_viol > p_clean
    print(f"  mean p_latent violating={p_viol:.4f} > clean={p_clean:.4f} : {g3a}")
    defect_rate_viol = (df_default.loc[violating, "defects"] / WAFERS_PER_LOT).mean()
    defect_rate_clean = (df_default.loc[~violating, "defects"] / WAFERS_PER_LOT).mean()
    g3b = defect_rate_viol > defect_rate_clean
    print(f"  mean defect rate violating={defect_rate_viol:.4f} > "
          f"clean={defect_rate_clean:.4f} : {g3b}")
    ok &= g3a and g3b

    print("=" * 64)
    print("GATE 4 — calibration band (baseline violation rate in [0.05, 0.20])")
    print("=" * 64)
    W, viol_rate = calibrate_window(log, cfg.route, quantile=0.90)
    g4a = abs(W - DEFAULT_WINDOW_HOURS) < 1e-9
    print(f"  recalibrated W={W:.6f} matches fixed DEFAULT_WINDOW_HOURS "
          f"={DEFAULT_WINDOW_HOURS:.6f} : {g4a}")
    g4b = 0.05 <= viol_rate <= 0.20
    print(f"  baseline violation rate {viol_rate:.4f} in [0.05, 0.20] : {g4b}")
    ok &= g4a and g4b

    print("=" * 64)
    print("GATE 5 — chamber effect (off-nominal LITHO tool raises risk)")
    print("=" * 64)
    cfg_offset = copy.deepcopy(default_config())
    cfg_offset.stations["LITHO"].tool_offsets = (1.05, 0.95)  # tool 1 slower
    log_offset, _, _ = simulate(cfg_offset)
    chamber_cfg = QualityConfig(off_nominal_tool_label="LITHO-1")  # slower tool
    df_chamber = build_lot_quality(log_offset, cfg_offset.route, chamber_cfg)
    slow_tool = ((df_chamber["litho1_tool"] == "LITHO-1")
                 | (df_chamber["litho2_tool"] == "LITHO-1"))
    p_slow = df_chamber.loc[slow_tool, "p_latent"].mean()
    p_fast = df_chamber.loc[~slow_tool, "p_latent"].mean()
    g5 = p_slow > p_fast
    print(f"  mean p_latent slow-tool-lots={p_slow:.4f} > "
          f"other-lots={p_fast:.4f} : {g5}")
    ok &= g5

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
