"""
Transparent alert-priority scoring (M8 Stage B).

Turns an E10 degradation signal into a single priority NUMBER an operator can
rank alerts by, using ONE documented formula with no tuned weights:

    priority = severity * bottleneck_criticality * cost_exposure_per_hour

Each factor is a plain, explainable quantity already computed elsewhere in
this project; this module does not introduce a new model, only multiplies
three existing readings together. That is a deliberate simplicity choice
(CLAUDE.md interpretability-first rule): the score's ranking claim is checked
against simulated reality in ``maintenance_check.py`` GATE 4 rather than
tuned to produce a particular ranking.

Factor definitions
---------------------
  severity (dimensionless, >= 0)
      Relative process-time inflation of the degradation: for
      ``factory_generator.DegradationAnomaly`` the multiplier at time t is
      ``1 + alpha * (t - t_onset)``, so a natural single-number severity read
      is the multiplier at a chosen HORIZON hours after onset (e.g. 24h):
      ``severity = alpha * horizon_hours``. This is exactly "how much slower,
      as a fraction, has the tool gotten after running degraded for
      ``horizon_hours``" - the same ``alpha`` already used everywhere else in
      this repo (M5 detectors, ``whatif.run_degradation_impact``), just read
      off at a fixed common horizon so two different stations' severities are
      comparable on the same scale.

  bottleneck_criticality (dimensionless, 0 to 1)
      The station's empirical slot utilization from the KPI layer
      (``kpi.kpi_metrics.station_utilization``) on a CLEAN baseline run - how
      saturated the station already is. A near-saturated station (LITHO,
      ~0.85) has almost no spare capacity to absorb a slowdown, so the same
      relative severity does far more damage there than at a station with
      slack (METRO, ~0.45). This is the Theory-of-Constraints logic already
      locked into this project (CLAUDE.md: utilization / ToC for
      bottlenecks) applied to alert triage rather than capacity planning.

  cost_exposure_per_hour ($/hour, >= 0)
      What one hour of this station's capacity is worth, from the M6 cost
      model's PROCESSING rate: ``CostRates.proc_rate`` scaled by the
      station's tool count (``n_tools``), i.e. the $/hour value of the
      station's full serving capacity being unavailable. This reuses the
      cost model's own rate rather than inventing a new one, keeping the
      score's dollar-flavored factor defensible on the same terms as every
      other cost figure in this repo (illustrative, ranking-not-forecast).

Why multiplicative, not a weighted sum
-----------------------------------------
Each factor answers a different "how much": severity answers "how much
slower", bottleneck_criticality answers "how much of the plant's capacity is
already at risk", cost_exposure_per_hour answers "how much is an hour of that
capacity worth". Multiplying is the natural combination when any one factor
being near zero should drive the priority near zero (a severe degradation at
a station that is barely used, or that costs nothing per hour, should not
alarm the operator) - exactly the behavior a weighted SUM would not have
(a sum keeps ranking alerts up even when one factor is negligible, since the
other terms still contribute additively). No factor is weighted or tuned;
the formula is the plain product, and its ranking claim is validated against
simulated cost impact in ``maintenance_check.py`` GATE 4, not fit to match it.

Consumers must have src/generator, src/kpi, and src/decision on sys.path.
"""

from __future__ import annotations

from dataclasses import dataclass

from factory_generator import default_config, draw_randoms, simulate
from kpi_metrics import station_utilization
from cost_model import CostRates

#: Common horizon (hours) at which severity is read off, so degradations at
#: different stations/alphas are compared on the same "how much slower after
#: this many hours" basis. 24h = one calendar day of exposure, a natural
#: operator-facing unit ("if this ran degraded for a day, how much worse
#: would it be").
DEFAULT_SEVERITY_HORIZON_HOURS = 24.0


@dataclass
class PriorityInputs:
    """One alert's three raw readings, for inspection/debugging before scoring."""
    station: str
    severity: float
    bottleneck_criticality: float
    cost_exposure_per_hour: float

    @property
    def priority(self) -> float:
        """The one documented formula: plain product, no weights."""
        return self.severity * self.bottleneck_criticality * self.cost_exposure_per_hour


def severity_from_alpha(alpha: float,
                        horizon_hours: float = DEFAULT_SEVERITY_HORIZON_HOURS) -> float:
    """Relative process-time inflation after ``horizon_hours`` of degradation.

    ``DegradationAnomaly.pt_multiplier`` = ``1 + alpha * (t - t_onset)``, so
    the fractional inflation at ``t_onset + horizon_hours`` is
    ``alpha * horizon_hours`` (the "+1" is the no-degradation baseline,
    subtracted off so severity is 0 when alpha is 0).
    """
    return alpha * horizon_hours


def bottleneck_criticality_from_baseline(station: str, cfg=None) -> float:
    """Empirical slot utilization of ``station`` on a clean baseline run.

    Uses ``kpi.kpi_metrics.station_utilization`` (the KPI layer's own
    utilization function) over the steady-state window
    ``[warmup_hours, horizon_hours]`` of one clean (no-anomaly) run of
    ``cfg`` (default: ``default_config()``, seed 42 - the same baseline
    used throughout this project). Returns a value in [0, 1].
    """
    cfg = cfg or default_config()
    draws = draw_randoms(cfg, seed=cfg.seed)
    log, _, meta = simulate(cfg, draws)
    util = station_utilization(log, cfg.warmup_hours, cfg.horizon_hours,
                               meta["stations"])
    row = util.loc[util["station"] == station, "utilization"]
    if row.empty:
        raise KeyError(f"station {station!r} not found in station_utilization output")
    return float(row.iloc[0])


def cost_exposure_per_hour(station: str, cfg=None,
                           rates: CostRates | None = None) -> float:
    """$/hour value of ``station``'s full serving capacity, from the M6 cost model.

    ``CostRates.proc_rate`` ($ per tool-hour of processing) times the
    station's ``n_tools`` - the dollar value of one hour of that station
    being fully unavailable, reusing the cost model's own processing rate
    rather than inventing a new figure.
    """
    cfg = cfg or default_config()
    rates = rates or CostRates()
    n_tools = cfg.stations[station].n_tools
    return rates.proc_rate * n_tools


def compute_priority(station: str, alpha: float, cfg=None,
                     rates: CostRates | None = None,
                     horizon_hours: float = DEFAULT_SEVERITY_HORIZON_HOURS
                     ) -> PriorityInputs:
    """Build the ``PriorityInputs`` (and hence the priority score) for one alert.

    ``station`` and ``alpha`` describe the degradation (same meaning as
    ``factory_generator.DegradationAnomaly``); ``cfg``/``rates`` default to
    this project's locked baseline config and illustrative cost rates.
    """
    cfg = cfg or default_config()
    rates = rates or CostRates()
    return PriorityInputs(
        station=station,
        severity=severity_from_alpha(alpha, horizon_hours),
        bottleneck_criticality=bottleneck_criticality_from_baseline(station, cfg),
        cost_exposure_per_hour=cost_exposure_per_hour(station, cfg, rates),
    )


def compare_bottleneck_vs_nonbottleneck(alpha: float = 0.01,
                                        bottleneck_station: str = "LITHO",
                                        nonbottleneck_station: str = "METRO",
                                        cfg=None, rates: CostRates | None = None
                                        ) -> tuple[PriorityInputs, PriorityInputs]:
    """Same-severity degradation on the bottleneck vs a non-bottleneck station.

    Demonstration helper for the priority formula's central claim: identical
    ``alpha`` (severity) on LITHO (the engineered bottleneck, slot
    utilization ~0.85) must rank ABOVE the same severity on METRO (slack
    station, ~0.45), because bottleneck_criticality is the only factor that
    differs between the two calls (severity and the processing cost rate are
    shared - ``cost_exposure_per_hour`` also differs only through each
    station's own ``n_tools``, which both stations share at 2 in the locked
    config, so the demonstrated ranking is driven by criticality).

    Returns (priority_bottleneck, priority_nonbottleneck) as ``PriorityInputs``.
    """
    cfg = cfg or default_config()
    rates = rates or CostRates()
    p_bn = compute_priority(bottleneck_station, alpha, cfg, rates)
    p_nb = compute_priority(nonbottleneck_station, alpha, cfg, rates)
    return p_bn, p_nb
