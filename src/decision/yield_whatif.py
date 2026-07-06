"""
Yield-aware what-if (M7 Stage C).

Wraps the EXISTING CRN-paired scenario machinery (``factory_generator``,
``bottleneck/counterfactual.py``, ``decision/whatif.py`` -- imported, never
copy-pasted) so each scenario's simulated event log also flows through the
Stage-B quality layer (``quality/queue_time.py`` + ``quality/yield_model.py``)
with the SAME ``QualityConfig`` seed for every scenario in a comparison.

Why a shared QualityConfig seed preserves CRN discipline
---------------------------------------------------------
``build_lot_quality`` is deterministic given (log, route, QualityConfig):
the latent defect probability ``p_latent`` is a pure function of upstream
process/queue features already present in the log (no additional randomness),
and the ONLY stochastic step is the per-lot Binomial(25, p_latent) defect
draw plus the metrology noise, both drawn from a dedicated ``numpy.random.
Generator`` seeded by ``QualityConfig.seed``. Using the SAME seed for
baseline and every treatment in a comparison means the Binomial/Gaussian
draws are made against the SAME sequence of underlying random numbers; since
``p_latent`` differs between scenarios only through the causal channel of
interest (e.g. more queue-time violations under higher demand), any change
in realized defects/yield reflects that causal channel, not extra sampling
noise from an independently-reseeded yield layer. This is exactly the CRN
logic already used for the DES's own draws (one shared ``RandomDraws`` table
per replication, baseline vs treatment) applied one layer up, to the yield
scoring step. A baseline-vs-baseline comparison under this scheme must
therefore show EXACTLY zero delta on every quality metric (see
``vm_check.py`` GATE 4) -- this is the regression test for the claim above.

Consumers must have src/generator, src/bottleneck, src/monitoring,
src/decision, and src/quality on sys.path (same convention as whatif.py).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from factory_generator import default_config, draw_randoms, simulate
from counterfactual import steady_state_kpis, with_extra_tool
from whatif import with_demand
from queue_time import DEFAULT_WINDOW_HOURS, flag_violations
from yield_model import QualityConfig, build_lot_quality

#: Illustrative cost per defective wafer (scrap), in the same illustrative-
#: rates style as decision/cost_model.py's CostRates: NOT a real figure, used
#: only to rank scenarios against each other on a common scale, re-checked
#: under sensitivity, never presented as a forecast.
DEFAULT_COST_PER_DEFECTIVE_WAFER = 150.0


def _quality_metrics(log: pd.DataFrame, route: list, qcfg: QualityConfig,
                      window_hours: float, cost_per_defective_wafer: float) -> dict:
    """Per-scenario quality summary from one simulated log."""
    viol = flag_violations(log, route, window_hours=window_hours)
    violation_rate = float(viol["violation"].mean()) if len(viol) else float("nan")

    lot_quality = build_lot_quality(log, route, qcfg, window_hours=window_hours)
    mean_p_latent = float(lot_quality["p_latent"].mean())
    mean_lot_yield = float(lot_quality["lot_yield"].mean())
    expected_defective_wafers = float(lot_quality["defects"].sum())
    scrap_cost = expected_defective_wafers * cost_per_defective_wafer

    return {
        "violation_rate": violation_rate,
        "mean_p_latent": mean_p_latent,
        "mean_lot_yield": mean_lot_yield,
        "total_defective_wafers": expected_defective_wafers,
        "scrap_cost": scrap_cost,
    }


def paired_yield_comparison(base_cfg, treat_cfg, scenario_name: str,
                            n_reps: int = 30, seed0: int = 5000,
                            qcfg: QualityConfig | None = None,
                            qcfg_treat: QualityConfig | None = None,
                            window_hours: float | None = None,
                            cost_per_defective_wafer: float =
                            DEFAULT_COST_PER_DEFECTIVE_WAFER) -> pd.DataFrame:
    """CRN-paired baseline-vs-treatment comparison, cycle time + quality.

    ``qcfg_treat`` (default: same object as ``qcfg``) lets the treatment log
    be scored with a different DETERMINISTIC scoring parameterization while
    still sharing the SEED with the baseline (the pairing requirement is on
    the seed, i.e. on the Binomial/Gaussian random stream, not on the
    deterministic risk coefficients). The one intended use is the chamber-
    offset demo: an offset log's slower tool carries the Stage-B chamber
    risk term (``off_nominal_tool_label``), while the offset-free baseline
    has no off-nominal tool and must be scored with the term disabled (see
    ``yield_model.QualityConfig``). ``qcfg_treat`` must carry the same
    ``seed`` as ``qcfg``; a ValueError guards this.

    DES-level CRN: per replication, baseline and treatment each get a
    ``RandomDraws`` table built from THEIR OWN config with the SAME seed
    (``draw_randoms(cfg, seed0 + rep)``). For capacity or tool-offset
    treatments this is bit-identical to the single-shared-table pattern in
    ``counterfactual.py``/``whatif.py``: ``draw_randoms`` depends only on
    the distributional config (arrival_rate, route, pt_mean/pt_cv), never
    on n_tools or tool_offsets, so both tables are the same object
    value-for-value and pairing is EXACT (GATE 4 of ``vm_check.py`` checks
    the degenerate case). For demand treatments the arrival process itself
    is the intervention, so the tables legitimately differ; sharing the
    seed keeps the underlying random stream common (numpy's
    ``exponential(scale)`` is ``scale`` times a standard-exponential draw
    from the same stream, so each interarrival time scales proportionally),
    which is the same convention ``whatif.run_demand_capacity`` uses when
    it draws from the scaled config. Each resulting log is then scored by
    the Stage-B quality layer using the SAME ``qcfg`` (yield-level CRN, see
    module docstring). Returns one row per replication with paired deltas.
    """
    qcfg = qcfg or QualityConfig()
    qcfg_treat = qcfg_treat or qcfg
    if qcfg_treat.seed != qcfg.seed:
        raise ValueError("qcfg_treat must share qcfg's seed (CRN pairing of the "
                         "yield layer's Binomial/Gaussian draws)")
    window_hours = DEFAULT_WINDOW_HOURS if window_hours is None else window_hours
    t0, t1 = base_cfg.warmup_hours, base_cfg.horizon_hours

    rows = []
    for rep in range(n_reps):
        draws_b = draw_randoms(base_cfg, seed0 + rep)
        draws_t = draw_randoms(treat_cfg, seed0 + rep)
        log_b, life_b, _ = simulate(base_cfg, draws_b)
        log_t, life_t, _ = simulate(treat_cfg, draws_t)

        th_b, ct_b = steady_state_kpis(life_b, t0, t1)
        th_t, ct_t = steady_state_kpis(life_t, t0, t1)

        qm_b = _quality_metrics(log_b, base_cfg.route, qcfg, window_hours,
                                cost_per_defective_wafer)
        qm_t = _quality_metrics(log_t, treat_cfg.route, qcfg_treat, window_hours,
                                cost_per_defective_wafer)

        rows.append({
            "rep": rep,
            "scenario": scenario_name,
            "d_cycle_time": ct_t - ct_b,
            "d_throughput": th_t - th_b,
            "d_violation_rate": qm_t["violation_rate"] - qm_b["violation_rate"],
            "d_mean_p_latent": qm_t["mean_p_latent"] - qm_b["mean_p_latent"],
            "d_mean_lot_yield": qm_t["mean_lot_yield"] - qm_b["mean_lot_yield"],
            "d_scrap_cost": qm_t["scrap_cost"] - qm_b["scrap_cost"],
            "base_violation_rate": qm_b["violation_rate"],
            "treat_violation_rate": qm_t["violation_rate"],
            "base_mean_lot_yield": qm_b["mean_lot_yield"],
            "treat_mean_lot_yield": qm_t["mean_lot_yield"],
            "base_scrap_cost": qm_b["scrap_cost"],
            "treat_scrap_cost": qm_t["scrap_cost"],
        })
    return pd.DataFrame(rows)


def summarize_yield_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean delta per scenario across replications (tidy summary row per scenario)."""
    metrics = ["d_cycle_time", "d_throughput", "d_violation_rate",
              "d_mean_p_latent", "d_mean_lot_yield", "d_scrap_cost"]
    return (paired.groupby("scenario", sort=False)[metrics]
            .mean()
            .reset_index())


# --------------------------------------------------------------------------- #
# The three demonstration comparisons Stage D will narrate.
# --------------------------------------------------------------------------- #
def demo_extra_litho_tool(base_cfg=None, n_reps: int = 30, seed0: int = 5000,
                          qcfg: QualityConfig | None = None) -> pd.DataFrame:
    """(i) +1 LITHO tool vs baseline, CRN-paired, cycle time + yield."""
    base_cfg = base_cfg or default_config()
    treat_cfg = with_extra_tool(base_cfg, "LITHO")
    return paired_yield_comparison(base_cfg, treat_cfg, "LITHO+1",
                                   n_reps=n_reps, seed0=seed0, qcfg=qcfg)


def demo_demand_increase(base_cfg=None, factor: float = 1.15, n_reps: int = 30,
                         seed0: int = 5000, qcfg: QualityConfig | None = None
                         ) -> pd.DataFrame:
    """(ii) arrival-rate +15% vs baseline, CRN-paired, cycle time + yield."""
    base_cfg = base_cfg or default_config()
    treat_cfg = with_demand(base_cfg, factor)
    return paired_yield_comparison(base_cfg, treat_cfg,
                                   f"demand x{factor:g}",
                                   n_reps=n_reps, seed0=seed0, qcfg=qcfg)


def demo_chamber_offset(base_cfg=None, offsets: tuple = (1.05, 0.95),
                        n_reps: int = 30, seed0: int = 5000,
                        qcfg: QualityConfig | None = None) -> pd.DataFrame:
    """(iii) LITHO tool_offsets on vs off, CRN-paired, cycle time + yield.

    "Treatment" here is the offset-free baseline's own config with LITHO
    tool_offsets switched on -- the DES-level draw tables are value-
    identical (offsets are applied at simulate time, not in the table), so
    the pairing is exact, and unlike (i)/(ii) this comparison isolates a
    run-to-run/chamber-drift effect rather than a capacity or demand change.

    Scoring follows Stage B's semantics (``yield_model.QualityConfig``): on
    the offset log the SLOWER tool (index 0 of ``offsets`` when its
    multiplier is the larger one, i.e. "LITHO-1" for the default
    (1.05, 0.95)) is the off-nominal chamber and carries the ``a_chamber``
    risk term; the offset-free baseline has no off-nominal tool, so the
    term is disabled there. Both scorings share the same seed (enforced by
    ``paired_yield_comparison``), preserving the yield-layer pairing.
    """
    base_cfg = base_cfg or default_config()
    treat_cfg = copy.deepcopy(base_cfg)
    treat_cfg.stations["LITHO"].tool_offsets = offsets
    qcfg = qcfg or QualityConfig()
    slow_tool_index = int(np.argmax(offsets))          # larger multiplier = slower
    slow_label = f"LITHO-{slow_tool_index + 1}"
    qcfg_treat = copy.deepcopy(qcfg)
    qcfg_treat.off_nominal_tool_label = slow_label
    return paired_yield_comparison(base_cfg, treat_cfg,
                                   f"LITHO offsets {offsets}",
                                   n_reps=n_reps, seed0=seed0, qcfg=qcfg,
                                   qcfg_treat=qcfg_treat)


def run_all_demos(base_cfg=None, n_reps: int = 30, seed0: int = 5000,
                  qcfg: QualityConfig | None = None) -> pd.DataFrame:
    """Run all three demonstration comparisons and return one tidy summary table."""
    base_cfg = base_cfg or default_config()
    paired = pd.concat([
        demo_extra_litho_tool(base_cfg, n_reps=n_reps, seed0=seed0, qcfg=qcfg),
        demo_demand_increase(base_cfg, n_reps=n_reps, seed0=seed0, qcfg=qcfg),
        demo_chamber_offset(base_cfg, n_reps=n_reps, seed0=seed0, qcfg=qcfg),
    ], ignore_index=True)
    return summarize_yield_comparison(paired)
