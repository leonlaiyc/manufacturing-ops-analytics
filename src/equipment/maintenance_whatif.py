"""
Maintenance-timing what-if (M8 Stage B): delay PM vs immediate PM.

Turns the M8 Stage A E10 layer into a DECISION: given a bottleneck tool that
is degrading (running slower over time), when should the operator schedule
its preventive maintenance (PM)? Acting early means less accumulated
congestion/yield damage but the scheduled-downtime hit lands sooner; acting
late lets the degradation run longer (more congestion, more yield risk) but
defers the downtime. This module builds that trade-off as a CRN-paired
comparison, reusing existing injection primitives, the M6 cost model, and the
M7 quality layer, with no new randomness introduced anywhere.

Composition idea (verified against ``factory_generator.py`` before writing
this module - see ``_simulate_injected``): ``DegradationAnomaly`` and
``ScheduledDowntimeAnomaly`` are independent anomaly objects that
``_simulate_injected`` composes by SUMMING ``tools_delta`` and MULTIPLYING
``pt_multiplier`` across the whole ``anomalies`` list; nothing about one
anomaly's window depends on another's. So a maintenance-timing scenario is
just:

    DegradationAnomaly(station, t_onset, t_end=T,        alpha)   # runs until PM
    ScheduledDowntimeAnomaly(station, t_start=T, t_end=T+D)       # the PM itself

passed together as ``anomalies=[deg, pm]`` to ``simulate``. This already
existed as a one-anomaly special case in ``decision/whatif.py``
(``run_improvement_raw``'s ``early_fix`` option builds a ``DegradationAnomaly``
whose ``t_end`` is the detection day, with no explicit repair window - just a
flat one-off repair cost). This module generalizes that: it ALSO models the
repair as an explicit ``ScheduledDowntimeAnomaly`` window so the PM's own
downtime cost and its knock-on congestion are visible in the simulated log,
not just charged as a flat fee. No extension to the injection primitives was
needed; both classes already exist from Stage A / M5.

Stylized PM-resets-degradation semantics (documented, not hidden)
------------------------------------------------------------------
"PM resets degradation" is modeled as: the degradation multiplier is exactly
1.0 for every t > T (``DegradationAnomaly.pt_multiplier`` returns 1.0 outside
its own window, and no second degradation anomaly is added after the PM), and
the tool is unavailable (capacity reduced) for ``[T, T+D]``. This is a
STYLIZED idealization - a real PM's effectiveness at reversing accumulated
wear is itself uncertain and often partial - not a physical maintenance
model. The point of this module is the TIMING trade-off (when to act) under
that idealization, not a claim about real PM efficacy.

CRN pairing discipline (reused from ``decision/yield_whatif.py``)
--------------------------------------------------------------------
Every scenario in one ``compare_pm_timings`` call, INCLUDING the no-
degradation baseline, is simulated from ``draw_randoms(cfg, seed0 + rep)`` -
the SAME table for the same ``rep`` across scenarios - so scenarios differ
only through their anomaly list, never through re-sampled arrivals or
processing times. The quality layer is scored with the SAME ``QualityConfig``
(same seed) for every scenario in a comparison, exactly as in
``yield_whatif.paired_yield_comparison``, so a baseline-vs-baseline pairing
(same anomaly list twice) must show an EXACT zero delta on cycle time,
violation rate, and yield (checked in ``maintenance_check.py`` GATE 2).

Consumers must have src/generator, src/quality, and src/decision on
sys.path (same convention as the rest of src/equipment and yield_whatif.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factory_generator import (
    default_config, draw_randoms, simulate,
    ScheduledDowntimeAnomaly, DegradationAnomaly,
)
from cost_model import CostRates, cost_components
from queue_time import DEFAULT_WINDOW_HOURS, flag_violations
from yield_model import QualityConfig, build_lot_quality

#: Illustrative cost per defective wafer (scrap), same figure and same
#: illustrative-rates framing as ``decision/yield_whatif.py``: not a real
#: price, used only to rank scenarios against each other on a common scale.
DEFAULT_COST_PER_DEFECTIVE_WAFER = 150.0

#: Illustrative PM downtime cost rate, $ per hour of scheduled downtime on the
#: bottleneck tool. Distinct from ``CostRates.repair_cost`` (a flat one-off
#: fee used elsewhere in this repo): here the PM's duration D is itself a
#: decision input (the demonstration varies it), so the cost scales with
#: hours, in the same "one illustrative rate, defensible in review" style as
#: every other rate in ``cost_model.py``.
DEFAULT_PM_DOWNTIME_RATE = 5000.0


def build_pm_scenario(station: str, t_onset: float, severity_alpha: float,
                       pm_time: float, pm_duration: float) -> list:
    """Build the anomaly list for one (degrade-until-T, repair-at-T) scenario.

    Parameters
    ----------
    station : str
        Station undergoing degradation and PM (the demonstration uses LITHO,
        the bottleneck; ``alert_priority.py`` also exercises METRO for
        contrast).
    t_onset : float
        Hour the degradation begins (``DegradationAnomaly.t_onset``).
    severity_alpha : float
        Fractional processing-time increase per hour of degradation
        (``DegradationAnomaly.alpha``; same meaning as elsewhere in this
        repo - see ``factory_generator.DegradationAnomaly``).
    pm_time : float
        Hour T at which PM starts. Must be >= ``t_onset``. The degradation
        window is ``[t_onset, pm_time]`` (ends exactly at T - the "PM resets
        degradation" idealization documented in the module docstring).
    pm_duration : float
        Hours D the PM takes; the tool is down (capacity reduced by one)
        for ``[T, T+D]``.

    Returns
    -------
    list of [DegradationAnomaly, ScheduledDowntimeAnomaly], ready to pass as
    ``simulate(cfg, draws, anomalies=...)``.
    """
    if pm_time < t_onset:
        raise ValueError("pm_time must be >= t_onset (PM cannot precede degradation onset)")
    deg = DegradationAnomaly(station=station, t_onset=t_onset, t_end=pm_time,
                              alpha=severity_alpha)
    pm = ScheduledDowntimeAnomaly(station=station, t_start=pm_time,
                                  t_end=pm_time + pm_duration, tools_removed=1)
    return [deg, pm]


def _scenario_metrics(log: pd.DataFrame, life: pd.DataFrame, route: list,
                      t0: float, t1: float, qcfg: QualityConfig,
                      window_hours: float, rates: CostRates,
                      cost_per_defective_wafer: float,
                      pm_duration: float, pm_downtime_rate: float) -> dict:
    """One scenario's full metric bundle: cycle time, quality, and all costs."""
    done = life.dropna(subset=["completion_time"])
    in_win = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)]
    ct = (in_win["completion_time"] - in_win["arrival_time"])
    mean_cycle_time = float(ct.mean()) if len(in_win) else float("nan")

    viol = flag_violations(log, route, window_hours=window_hours)
    violation_rate = float(viol["violation"].mean()) if len(viol) else float("nan")

    lot_quality = build_lot_quality(log, route, qcfg, window_hours=window_hours)
    mean_lot_yield = float(lot_quality["lot_yield"].mean())
    total_defective_wafers = float(lot_quality["defects"].sum())
    scrap_cost = total_defective_wafers * cost_per_defective_wafer

    congestion = cost_components(log, t0, t1, rates)
    pm_downtime_cost = pm_duration * pm_downtime_rate
    total_cost = congestion["total"] + scrap_cost + pm_downtime_cost

    return {
        "mean_cycle_time": mean_cycle_time,
        "violation_rate": violation_rate,
        "mean_lot_yield": mean_lot_yield,
        "total_defective_wafers": total_defective_wafers,
        "congestion_cost": congestion["total"],
        "scrap_cost": scrap_cost,
        "pm_downtime_cost": pm_downtime_cost,
        "total_cost": total_cost,
    }


def compare_pm_timings(station: str = "LITHO", t_onset: float = 20 * 24.0,
                       severity_alpha: float = 0.01,
                       pm_times: dict | None = None,
                       pm_duration: float = 24.0,
                       cfg=None, n_reps: int = 15, seed0: int = 6000,
                       qcfg: QualityConfig | None = None,
                       rates: CostRates | None = None,
                       window_hours: float | None = None,
                       cost_per_defective_wafer: float = DEFAULT_COST_PER_DEFECTIVE_WAFER,
                       pm_downtime_rate: float = DEFAULT_PM_DOWNTIME_RATE) -> pd.DataFrame:
    """CRN-paired comparison of PM-timing alternatives against a clean baseline.

    Every replication draws ONE table (``draw_randoms(cfg, seed0 + rep)``) and
    runs FOUR scenarios against it: ``baseline`` (no degradation, no PM at
    all - the "nothing ever breaks" reference), then one scenario per entry
    in ``pm_times`` (default: immediate/mid/late - see below). All four
    scenarios share the table, so any difference between them is caused by
    their anomaly list, never by re-sampled randomness (see module
    docstring). The quality layer also shares one ``QualityConfig`` (same
    seed) across all four scenarios per replication, matching
    ``yield_whatif.paired_yield_comparison``'s discipline.

    Parameters
    ----------
    station : str
        Station undergoing degradation and PM.
    t_onset : float
        Hour degradation begins, shared by every PM-timing scenario.
    severity_alpha : float
        Degradation severity (fractional processing-time inflation per hour).
    pm_times : dict | None
        ``{label: pm_time_hours}``. Default (None) is the reference
        three-timing demonstration: immediate PM shortly after onset, a mid
        delay, and a long delay (see ``DEFAULT_PM_TIMES`` below).
    pm_duration : float
        Hours the PM takes (``D``), same for every timing (only WHEN differs).
    cfg :
        Base ``FactoryConfig``; defaults to ``default_config()``.
    n_reps : int
        Replications. Kept small (default 15) so the reference demonstration
        runs in well under a minute per scenario set on this machine while
        still giving a stable directional read for GATE 3 (see
        ``maintenance_check.py``): the injected effect (LITHO degradation
        exposure) is large relative to run-to-run queueing noise at rho ~
        0.85, so 15 CRN-paired reps are enough for the delay-cost direction
        to be robust, unlike a fine-grained confidence-interval claim which
        would need many more.
    seed0 : int
        Base seed for ``draw_randoms``; replication ``rep`` uses ``seed0 + rep``.
    qcfg : QualityConfig | None
        Shared quality-scoring config; defaults to ``QualityConfig()``.
    rates : CostRates | None
        Congestion (processing + holding) cost rates; defaults to ``CostRates()``.
    window_hours : float | None
        Post-litho queue-time window; defaults to the fixed, calibrated
        ``queue_time.DEFAULT_WINDOW_HOURS``.
    cost_per_defective_wafer, pm_downtime_rate : float
        Illustrative rates (see module-level constants).

    Returns
    -------
    Tidy pd.DataFrame, one row per (rep, scenario), columns:
        rep, scenario, pm_time (nan for baseline), mean_cycle_time,
        violation_rate, mean_lot_yield, total_defective_wafers,
        congestion_cost, scrap_cost, pm_downtime_cost, total_cost
    """
    cfg = cfg or default_config()
    qcfg = qcfg or QualityConfig()
    rates = rates or CostRates()
    window_hours = DEFAULT_WINDOW_HOURS if window_hours is None else window_hours
    pm_times = pm_times if pm_times is not None else dict(DEFAULT_PM_TIMES)
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours

    rows = []
    for rep in range(n_reps):
        draws = draw_randoms(cfg, seed0 + rep)

        log_b, life_b, _ = simulate(cfg, draws)
        m_b = _scenario_metrics(log_b, life_b, cfg.route, t0, t1, qcfg,
                                window_hours, rates, cost_per_defective_wafer,
                                pm_duration=0.0, pm_downtime_rate=pm_downtime_rate)
        rows.append({"rep": rep, "scenario": "baseline", "pm_time": float("nan"),
                    **m_b})

        for label, pm_time in pm_times.items():
            anomalies = build_pm_scenario(station, t_onset, severity_alpha,
                                          pm_time, pm_duration)
            log_s, life_s, _ = simulate(cfg, draws, anomalies=anomalies)
            m_s = _scenario_metrics(log_s, life_s, cfg.route, t0, t1, qcfg,
                                    window_hours, rates, cost_per_defective_wafer,
                                    pm_duration=pm_duration,
                                    pm_downtime_rate=pm_downtime_rate)
            rows.append({"rep": rep, "scenario": label, "pm_time": float(pm_time),
                        **m_s})

    return pd.DataFrame(rows)


#: Reference three-timing demonstration (hours from horizon start). Onset is
#: fixed at 20 days (``t_onset`` default above); these are the candidate PM
#: hours T relative to that onset: immediate (1 day after onset - just enough
#: to notice and react), mid (10 days of accumulated exposure), late (25 days
#: - close to the full remaining horizon), each followed by a fixed
#: ``pm_duration`` repair window.
DEFAULT_PM_TIMES = {
    "immediate_pm": 21 * 24.0,
    "mid_pm": 30 * 24.0,
    "late_pm": 45 * 24.0,
}


def summarize_pm_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per scenario across replications (tidy summary row per scenario)."""
    metrics = ["mean_cycle_time", "violation_rate", "mean_lot_yield",
              "total_defective_wafers", "congestion_cost", "scrap_cost",
              "pm_downtime_cost", "total_cost"]
    scenario_order = list(dict.fromkeys(paired["scenario"]))
    out = (paired.groupby("scenario", sort=False)[metrics]
           .mean()
           .reindex(scenario_order)
           .reset_index())
    return out


def demo_litho_pm_timing(cfg=None, n_reps: int = 15, seed0: int = 6000) -> pd.DataFrame:
    """Reference demonstration: LITHO degradation, three PM timings vs baseline.

    Fixed seeds (``seed0=6000``), ``n_reps=15`` (see ``compare_pm_timings``
    docstring for why this is enough for a directional read). Shows the
    trade-off: acting early means less accumulated congestion/yield damage
    but the downtime cost lands sooner; acting late defers the downtime but
    lets degradation and its congestion/yield cost accumulate for longer.
    """
    cfg = cfg or default_config()
    paired = compare_pm_timings(station="LITHO", cfg=cfg, n_reps=n_reps, seed0=seed0)
    return summarize_pm_comparison(paired)
