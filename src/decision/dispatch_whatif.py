"""
CRN-paired dispatching-policy comparison and decision table (M9 Stage B).

Turns the M9 Stage A configurable-dispatch layer (``factory_generator.py``:
``queue_discipline``, due dates, opt-in ``release_control``) into a DECISION:
which policy should the fab run, under which demand regime, for which
business objective? This module compares five configurations against the
locked FIFO baseline, CRN-paired, and assembles a decision table from the
measured deltas (never hand-written).

Configurations compared (all against a FIFO baseline on the SAME draw table):
  - edd             : earliest-due-date dispatch (``queue_discipline="edd"``)
  - critical_ratio  : critical-ratio dispatch (``queue_discipline="critical_ratio"``)
  - queue_time_aware: post-litho slack dispatch (``queue_discipline="queue_time_aware"``)
  - release_control : FIFO dispatch + a bottleneck-WIP release gate
                       (``release_control=ReleaseControlConfig(litho_wip_threshold=14)``)

Reused, not duplicated (composition, matching ``yield_whatif.py`` and
``maintenance_whatif.py``): ``factory_generator`` for the DES and dispatch
config, ``counterfactual.steady_state_kpis`` for throughput/cycle time,
``queue_time.flag_violations`` + ``yield_model.build_lot_quality`` for the
quality layer, ``cost_model.cost_components`` for congestion cost, and
``scipy.stats.t`` for the paired-CI half-width (same recipe as
``counterfactual.summarize``).

CRN pairing discipline
-----------------------
Every replication draws ONE ``RandomDraws`` table (``draw_randoms(cfg,
seed0 + rep)``) from a config that fixes the distributional parameters
(arrival_rate, route, pt_mean/pt_cv) for the regime; FIFO and every treatment
configuration in that regime are simulated against that SAME table, so a
configuration's queue_discipline/release_control is the ONLY thing that
differs (see ``factory_generator``'s module docstring, "Why dispatch-policy
changes cannot desynchronize CRN pairing", for why reordering a station's
pending list cannot desynchronize the (lot, route step) - indexed draws).
The quality layer is scored with the SAME ``QualityConfig`` (same seed) for
every configuration in a comparison, matching ``yield_whatif.py``'s
discipline, so a FIFO-vs-FIFO paired comparison must show an EXACT zero delta
on every metric (checked in ``dispatch_whatif_check.py`` GATE 2).

Release-control threshold pick (documented, not tuned live)
-------------------------------------------------------------
A small sweep of ``litho_wip_threshold`` (10 reps, seed0=9000, baseline mean
LITHO WIP ~4.4 lots) against this line's steady-state throughput and cycle
time showed:

    thr=11  mean_ct=21.01  mean_th=0.995  on_time=0.440  (binds hard, some starvation)
    thr=12  mean_ct=16.79  mean_th=1.001  on_time=0.647
    thr=13  mean_ct=15.42  mean_th=1.001  on_time=0.758
    thr=14  mean_ct=14.90  mean_th=1.001  on_time=0.805  <- chosen
    thr=16  mean_ct=14.71  mean_th=1.001  on_time=0.819  (already close to no-control)
    thr=20  mean_ct=14.71  mean_th=1.000  on_time=0.828  (no longer binding)
    no control (baseline): mean_ct=14.82  mean_th=1.001  on_time=0.812

At the locked arrival rate (rho_LITHO ~ 0.85), thresholds below ~12 visibly
throttle throughput (the pre-release pool cannot drain fast enough) - a
capacity decision the sweep would flag as starving the line, not a dispatch
comparison. Threshold 14 was chosen because it still visibly binds (cycle
time and post-litho violation rate move relative to the low-threshold rows,
see the module's demonstration run) while throughput and on-time rate sit
within noise of the no-control baseline - i.e. it caps WIP without costing
output. This is a fixed constant like ``queue_time.DEFAULT_WINDOW_HOURS``,
not re-swept per call.

Consumers must have src/generator, src/bottleneck, src/quality, and
src/decision on sys.path (same convention as ``yield_whatif.py``).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats

from factory_generator import default_config, draw_randoms, simulate, ReleaseControlConfig
from counterfactual import steady_state_kpis
from cost_model import CostRates, cost_components
from queue_time import DEFAULT_WINDOW_HOURS, flag_violations
from yield_model import QualityConfig, build_lot_quality

#: Illustrative cost per defective wafer (scrap), same figure and framing as
#: ``yield_whatif.py`` / ``maintenance_whatif.py``: not a real price, used
#: only to rank configurations against each other on a common scale.
DEFAULT_COST_PER_DEFECTIVE_WAFER = 150.0

#: Release-control threshold picked by the documented sweep above: binds
#: (visibly reduces cycle time / violation rate relative to lower thresholds)
#: without starving throughput at the locked arrival rate.
DEFAULT_RELEASE_THRESHOLD = 14

#: The five configurations compared, in report order. "fifo" is the baseline
#: every other row is paired against, so it is not itself a treatment row in
#: the output (see ``compare_policies``).
TREATMENT_CONFIGS = ("edd", "critical_ratio", "queue_time_aware", "release_control")

#: Demand regimes: baseline arrival rate and +15% (the M7 demo precedent, see
#: ``yield_whatif.demo_demand_increase``).
DEMAND_REGIMES = {"baseline": 1.0, "demand x1.15": 1.15}


def _make_config(base_cfg, treatment: str):
    """One treatment's FactoryConfig, deep-copied from ``base_cfg`` (FIFO).

    Every branch changes ONLY the field(s) needed for that treatment; the
    draw table (distributional config) is untouched, so pairing against the
    FIFO baseline on the same table stays exact except for the ONE dispatch
    mechanism under test (see module docstring, CRN pairing discipline).
    """
    cfg = copy.deepcopy(base_cfg)
    if treatment == "fifo":
        return cfg
    if treatment == "edd":
        cfg.queue_discipline = "edd"
        return cfg
    if treatment == "critical_ratio":
        cfg.queue_discipline = "critical_ratio"
        return cfg
    if treatment == "queue_time_aware":
        cfg.queue_discipline = "queue_time_aware"
        return cfg
    if treatment == "release_control":
        cfg.release_control = ReleaseControlConfig(
            litho_wip_threshold=DEFAULT_RELEASE_THRESHOLD)
        return cfg
    raise ValueError(f"Unknown treatment: {treatment!r}")


def _on_time_rate(lifecycle: pd.DataFrame, t0: float, t1: float) -> float:
    """Fraction of steady-state-window completions at or before their due date."""
    done = lifecycle.dropna(subset=["completion_time"])
    in_win = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)]
    if len(in_win) == 0:
        return float("nan")
    return float((in_win["completion_time"] <= in_win["due_date"]).mean())


def _scenario_metrics(log: pd.DataFrame, life: pd.DataFrame, route: list,
                       t0: float, t1: float, qcfg: QualityConfig,
                       window_hours: float, rates: CostRates,
                       cost_per_defective_wafer: float) -> dict:
    """One configuration's full metric bundle for one replication.

    Metrics per the spec: mean lot cycle time, output (completed lots),
    on-time delivery rate, post-litho violation rate, mean lot yield,
    congestion cost, and scrap cost (the last two also rolled up into a
    total_cost so "total cost" can be one decision-table objective).
    """
    th, ct = steady_state_kpis(life, t0, t1)
    window = t1 - t0
    output = th * window   # completed lots in [t0, t1] (steady_state_kpis' own numerator)

    on_time_rate = _on_time_rate(life, t0, t1)

    viol = flag_violations(log, route, window_hours=window_hours)
    violation_rate = float(viol["violation"].mean()) if len(viol) else float("nan")

    lot_quality = build_lot_quality(log, route, qcfg, window_hours=window_hours)
    mean_lot_yield = float(lot_quality["lot_yield"].mean())
    total_defective_wafers = float(lot_quality["defects"].sum())
    scrap_cost = total_defective_wafers * cost_per_defective_wafer

    congestion_cost = cost_components(log, t0, t1, rates)["total"]
    total_cost = congestion_cost + scrap_cost

    return {
        "mean_cycle_time": ct,
        "output": output,
        "on_time_rate": on_time_rate,
        "violation_rate": violation_rate,
        "mean_lot_yield": mean_lot_yield,
        "congestion_cost": congestion_cost,
        "scrap_cost": scrap_cost,
        "total_cost": total_cost,
    }


#: Metric columns, in report order, and whether a POSITIVE delta (treatment -
#: baseline) is an IMPROVEMENT. Used both by the raw comparison and by
#: ``decision_table`` to read off "which policy wins" without hand-coding
#: direction per objective a second time.
METRICS_HIGHER_IS_BETTER = {
    "mean_cycle_time": False,
    "output": True,
    "on_time_rate": True,
    "violation_rate": False,
    "mean_lot_yield": True,
    "congestion_cost": False,
    "scrap_cost": False,
    "total_cost": False,
}


def paired_policy_comparison(base_cfg, treatment: str, regime_name: str,
                              arrival_factor: float = 1.0, n_reps: int = 30,
                              seed0: int = 8000,
                              qcfg: QualityConfig | None = None,
                              rates: CostRates | None = None,
                              window_hours: float | None = None,
                              cost_per_defective_wafer: float =
                              DEFAULT_COST_PER_DEFECTIVE_WAFER) -> pd.DataFrame:
    """CRN-paired FIFO-vs-``treatment`` comparison under one demand regime.

    Per replication: draw ONE table from a config with ``arrival_rate``
    scaled by ``arrival_factor`` (nothing else - see ``with_demand`` in
    ``whatif.py`` for the precedent this mirrors), simulate the FIFO baseline
    and the treatment configuration against that table, score both with the
    SAME ``QualityConfig`` (same seed), and record the paired delta
    (treatment - baseline) for every metric.

    Returns one row per replication: rep, regime, policy, d_<metric> for
    every metric in ``METRICS_HIGHER_IS_BETTER``, plus the raw baseline and
    treatment values (base_<metric>, treat_<metric>) for the decision table's
    caveat text.
    """
    qcfg = qcfg or QualityConfig()
    rates = rates or CostRates()
    window_hours = DEFAULT_WINDOW_HOURS if window_hours is None else window_hours

    base_cfg = copy.deepcopy(base_cfg)
    base_cfg.arrival_rate *= arrival_factor
    treat_cfg = _make_config(base_cfg, treatment)
    t0, t1 = base_cfg.warmup_hours, base_cfg.horizon_hours

    rows = []
    for rep in range(n_reps):
        draws = draw_randoms(base_cfg, seed0 + rep)
        log_b, life_b, _ = simulate(base_cfg, draws)
        log_t, life_t, _ = simulate(treat_cfg, draws)

        m_b = _scenario_metrics(log_b, life_b, base_cfg.route, t0, t1, qcfg,
                                 window_hours, rates, cost_per_defective_wafer)
        m_t = _scenario_metrics(log_t, life_t, treat_cfg.route, t0, t1, qcfg,
                                 window_hours, rates, cost_per_defective_wafer)

        row = {"rep": rep, "regime": regime_name, "policy": treatment}
        for metric in METRICS_HIGHER_IS_BETTER:
            row[f"base_{metric}"] = m_b[metric]
            row[f"treat_{metric}"] = m_t[metric]
            row[f"d_{metric}"] = m_t[metric] - m_b[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_policies(base_cfg=None, treatments=TREATMENT_CONFIGS,
                      regimes: dict | None = None, n_reps: int = 30,
                      seed0: int = 8000, qcfg: QualityConfig | None = None,
                      rates: CostRates | None = None,
                      window_hours: float | None = None,
                      cost_per_defective_wafer: float =
                      DEFAULT_COST_PER_DEFECTIVE_WAFER) -> pd.DataFrame:
    """CRN-paired comparison of every configuration in ``treatments``, against
    FIFO, in every regime in ``regimes``. Same seed set (``seed0 + rep``)
    across every configuration within a regime, per the module docstring's
    CRN pairing discipline; each regime gets its OWN draw tables (a demand
    change is itself the intervention that legitimately changes the table's
    arrival process, see ``paired_policy_comparison``).

    Returns a tidy long-format DataFrame, one row per (regime, policy, rep):
    rep, regime, policy, base_<metric>, treat_<metric>, d_<metric> for every
    metric. ``summarize_policy_comparison`` collapses this to one row per
    (regime, policy, metric) with mean delta and 95% CI, matching the
    ``regime, policy, metric, mean delta, ci_lo, ci_hi`` tidy shape the spec
    asks for.
    """
    base_cfg = base_cfg or default_config()
    regimes = regimes if regimes is not None else DEMAND_REGIMES
    frames = []
    for regime_name, factor in regimes.items():
        for treatment in treatments:
            frames.append(paired_policy_comparison(
                base_cfg, treatment, regime_name, arrival_factor=factor,
                n_reps=n_reps, seed0=seed0, qcfg=qcfg, rates=rates,
                window_hours=window_hours,
                cost_per_defective_wafer=cost_per_defective_wafer))
    return pd.concat(frames, ignore_index=True)


def summarize_policy_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``compare_policies`` output to the tidy long-format table the
    spec asks for: one row per (regime, policy, metric), with the paired
    mean delta and a 95% CI (t-based, matching ``counterfactual.summarize``'s
    recipe: half-width = t.ppf(0.975, n-1) * sem).

    Columns: regime, policy, metric, n, mean_delta, ci_lo, ci_hi.
    """
    metrics = list(METRICS_HIGHER_IS_BETTER)
    rows = []
    group_cols = ["regime", "policy"]
    order = list(dict.fromkeys(zip(paired["regime"], paired["policy"])))
    for regime_name, policy_name in order:
        grp = paired[(paired["regime"] == regime_name) & (paired["policy"] == policy_name)]
        for metric in metrics:
            x = grp[f"d_{metric}"].to_numpy()
            n = len(x)
            mean = float(x.mean())
            sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
            rows.append({
                "regime": regime_name,
                "policy": policy_name,
                "metric": metric,
                "n": n,
                "mean_delta": mean,
                "ci_lo": mean - half,
                "ci_hi": mean + half,
            })
    return pd.DataFrame(rows)


#: Business objective -> the metric that measures it. Kept as an explicit
#: mapping (not inferred) so ``decision_table`` and its caveats read off ONE
#: source of truth for "what does this objective mean, numerically".
OBJECTIVE_METRICS = {
    "cycle time": "mean_cycle_time",
    "on-time delivery": "on_time_rate",
    "yield risk": "violation_rate",
    "total cost": "total_cost",
}

#: Caveat metric shown alongside the winner for each objective: the metric
#: the winner is MOST likely to trade away when it wins on its own objective
#: (documented pairing, not exhaustive - the caveat column reports the
#: MEASURED value of this metric for the winning policy, never hand-written
#: text).
OBJECTIVE_CAVEAT_METRIC = {
    "cycle time": "violation_rate",
    "on-time delivery": "mean_cycle_time",
    "yield risk": "mean_cycle_time",
    "total cost": "on_time_rate",
}


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per regime, per business objective, which policy wins, with a caveat.

    For each (regime, objective) the winner is the policy with the BEST mean
    delta on the objective's metric (best = most negative for a
    lower-is-better metric, most positive for a higher-is-better metric - see
    ``METRICS_HIGHER_IS_BETTER``). The caveat column reports the measured
    delta of ``OBJECTIVE_CAVEAT_METRIC[objective]`` for that SAME winning
    policy/regime row (pulled from ``summary``, never hand-written), so
    "what it sacrifices" is always a number that was actually measured.

    "significant" is True when the winning metric's 95% CI excludes zero
    (``ci_lo`` and ``ci_hi`` are both the same sign, or both zero-excluding);
    when the CI includes zero the row is marked not significant instead of
    silently declaring a tie (spec GATE 4).

    Returns one row per (regime, objective):
        regime, objective, winner, metric, mean_delta, ci_lo, ci_hi,
        significant, caveat_metric, caveat_delta, caveat_ci_lo, caveat_ci_hi
    """
    rows = []
    for regime_name in dict.fromkeys(summary["regime"]):
        reg = summary[summary["regime"] == regime_name]
        for objective, metric in OBJECTIVE_METRICS.items():
            higher_better = METRICS_HIGHER_IS_BETTER[metric]
            cand = reg[reg["metric"] == metric]
            if higher_better:
                best_idx = cand["mean_delta"].idxmax()
            else:
                best_idx = cand["mean_delta"].idxmin()
            best = cand.loc[best_idx]

            significant = not (best["ci_lo"] <= 0.0 <= best["ci_hi"])

            caveat_metric = OBJECTIVE_CAVEAT_METRIC[objective]
            caveat_row = reg[(reg["metric"] == caveat_metric)
                              & (reg["policy"] == best["policy"])].iloc[0]

            rows.append({
                "regime": regime_name,
                "objective": objective,
                "winner": best["policy"],
                "metric": metric,
                "mean_delta": best["mean_delta"],
                "ci_lo": best["ci_lo"],
                "ci_hi": best["ci_hi"],
                "significant": significant,
                "caveat_metric": caveat_metric,
                "caveat_delta": caveat_row["mean_delta"],
                "caveat_ci_lo": caveat_row["ci_lo"],
                "caveat_ci_hi": caveat_row["ci_hi"],
            })
    return pd.DataFrame(rows)


def run_all(base_cfg=None, n_reps: int = 30, seed0: int = 8000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: (summary, decision_table) for the notebook.

    ``summary`` is the tidy long-format comparison (regime, policy, metric,
    mean_delta, ci_lo, ci_hi); ``table`` is the decision table built from it.
    """
    base_cfg = base_cfg or default_config()
    paired = compare_policies(base_cfg, n_reps=n_reps, seed0=seed0)
    summary = summarize_policy_comparison(paired)
    table = decision_table(summary)
    return summary, table
