"""
Tool registry for the M10 agentic decision-support layer (Stage A).

Exposes the existing what-if engines (``src/decision/whatif.py``,
``src/decision/yield_whatif.py``, ``src/decision/dispatch_whatif.py``,
``src/equipment/maintenance_whatif.py``) and the read-only KPI baseline
(``src/kpi/kpi_metrics.py``) as plain, provider-agnostic callable tools:
a JSON-schema dict (name, description, input schema) plus a Python callable,
compatible with both Anthropic and OpenAI tool-calling conventions. The LLM
client itself is OUT of scope for this stage (see V2-PLAN.md M10); this
module only guarantees that IF a tool is called, its result is bounded,
deterministic, and logged.

Nothing here modifies the wrapped engines - they are imported and called
exactly as their own modules use them.

Each registered callable:
  1. validates its inputs against documented bounds (raises ValueError with a
     clear message on an out-of-bounds argument - see ``_check_bounds``);
  2. runs the underlying engine with a fixed seed (``seed0``, itself a bounded
     input, defaulting to the engine module's own default);
  3. returns a JSON-serializable dict that ALWAYS includes ``run_id``,
     produced by ``run_log.compute_run_id`` and logged via a ``RunLogger``
     the caller supplies (``registry.call(name, args, logger)``).

Consumers must have src/generator, src/bottleneck, src/quality, src/decision,
src/equipment, and src/kpi on sys.path (same convention as the modules being
wrapped); ``_ensure_sys_path`` below does this once at import time so callers
of this module do not need to repeat the boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]


def _ensure_sys_path() -> None:
    for sub in ("generator", "bottleneck", "quality", "decision", "equipment", "kpi"):
        p = str(SRC / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_sys_path()

from factory_generator import default_config                      # noqa: E402
from counterfactual import steady_state_kpis                      # noqa: E402
from whatif import with_demand, run_demand_capacity                # noqa: E402
from yield_whatif import (                                          # noqa: E402
    demo_extra_litho_tool, demo_demand_increase, summarize_yield_comparison,
)
from dispatch_whatif import compare_policies, summarize_policy_comparison, decision_table  # noqa: E402
from maintenance_whatif import compare_pm_timings, summarize_pm_comparison  # noqa: E402
from kpi_metrics import station_utilization                         # noqa: E402

from run_log import compute_run_id, RunLogger                       # noqa: E402

#: Documented input bounds. Kept small and explicit (not inferred from the
#: engines) so every registered tool's runtime is predictable: n_reps caps
#: keep a single tool call bounded to at most a few tens of DES replications
#: (seconds, not minutes) on this machine, matching the reference n_reps
#: (30 for whatif/yield/dispatch, 15 for maintenance-timing) used throughout
#: M6-M9's own check scripts and notebooks.
MAX_N_REPS = 30
MIN_N_REPS = 1
MAX_DEMAND_FACTOR = 2.0
MIN_DEMAND_FACTOR = 0.5
VALID_STATIONS = ("CLEAN", "FURNACE", "DEPO", "LITHO", "ETCH", "IMPLANT", "METRO")
VALID_PM_STATIONS = ("LITHO", "METRO")  # bottleneck + the alert_priority.py contrast station


def _check_bounds(name: str, value, lo, hi, label: str) -> None:
    if value is None or not (lo <= value <= hi):
        raise ValueError(f"{name}: {label}={value!r} out of bounds [{lo}, {hi}]")


def _check_choice(name: str, value, choices, label: str) -> None:
    if value not in choices:
        raise ValueError(f"{name}: {label}={value!r} not one of {choices}")


def _df_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of plain-Python-typed dict rows (JSON-serializable)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Tool 1 - capacity what-if (extra tool at a station, paired throughput delta)
# --------------------------------------------------------------------------- #
CAPACITY_SCHEMA = {
    "name": "run_capacity_whatif",
    "description": (
        "Compare baseline capacity against adding one tool at a given station, "
        "CRN-paired, across demand levels. Returns per-(factor) mean paired "
        "delta in throughput (lots/hour) versus a baseline run at the same "
        "demand factor. Use to answer 'is an extra tool at station X worth it'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "station": {"type": "string", "enum": list(VALID_STATIONS),
                        "description": "Station to add one tool to."},
            "demand_factors": {
                "type": "array", "items": {"type": "number"},
                "description": f"Arrival-rate multipliers, each in "
                                f"[{MIN_DEMAND_FACTOR}, {MAX_DEMAND_FACTOR}].",
                "default": [1.0],
            },
            "n_reps": {"type": "integer", "minimum": MIN_N_REPS, "maximum": MAX_N_REPS,
                       "default": 30, "description": "CRN-paired replications."},
            "seed0": {"type": "integer", "default": 1000,
                      "description": "Base seed; replication rep uses seed0+rep."},
        },
        "required": ["station"],
    },
}


def run_capacity_whatif(station: str, demand_factors: list[float] | None = None,
                         n_reps: int = 30, seed0: int = 1000) -> dict:
    demand_factors = demand_factors if demand_factors is not None else [1.0]
    _check_choice("run_capacity_whatif", station, VALID_STATIONS, "station")
    _check_bounds("run_capacity_whatif", n_reps, MIN_N_REPS, MAX_N_REPS, "n_reps")
    if not demand_factors:
        raise ValueError("run_capacity_whatif: demand_factors must be non-empty")
    for f in demand_factors:
        _check_bounds("run_capacity_whatif", f, MIN_DEMAND_FACTOR, MAX_DEMAND_FACTOR,
                      "demand_factors[i]")

    cfg = default_config()
    df = run_demand_capacity(cfg, factors=tuple(demand_factors), n_reps=n_reps,
                             seed0=seed0, station=station)
    per_factor = (df.groupby("factor", sort=False)["d_throughput"]
                  .agg(["mean", "std", "count"]).reset_index())
    per_factor.columns = ["factor", "mean_d_throughput", "std_d_throughput", "n"]

    return {
        "station": station,
        "demand_factors": list(demand_factors),
        "n_reps": n_reps,
        "seed0": seed0,
        "per_factor": _df_records(per_factor),
    }


# --------------------------------------------------------------------------- #
# Tool 2 - demand what-if (yield-aware, CRN-paired vs baseline)
# --------------------------------------------------------------------------- #
DEMAND_SCHEMA = {
    "name": "run_demand_whatif",
    "description": (
        "Compare a demand increase (arrival-rate multiplier) against baseline, "
        "CRN-paired, reporting cycle time AND yield-risk deltas (M7 yield-aware "
        "what-if). Use to answer 'what happens to cycle time and quality if "
        "demand grows by X percent'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "demand_factor": {
                "type": "number",
                "description": f"Arrival-rate multiplier, in "
                                f"[{MIN_DEMAND_FACTOR}, {MAX_DEMAND_FACTOR}] "
                                f"(1.15 = +15% demand).",
            },
            "n_reps": {"type": "integer", "minimum": MIN_N_REPS, "maximum": MAX_N_REPS,
                       "default": 30},
            "seed0": {"type": "integer", "default": 5000},
        },
        "required": ["demand_factor"],
    },
}


def run_demand_whatif(demand_factor: float, n_reps: int = 30, seed0: int = 5000) -> dict:
    _check_bounds("run_demand_whatif", demand_factor, MIN_DEMAND_FACTOR,
                  MAX_DEMAND_FACTOR, "demand_factor")
    _check_bounds("run_demand_whatif", n_reps, MIN_N_REPS, MAX_N_REPS, "n_reps")

    cfg = default_config()
    paired = demo_demand_increase(cfg, factor=demand_factor, n_reps=n_reps, seed0=seed0)
    summary = summarize_yield_comparison(paired)

    return {
        "demand_factor": demand_factor,
        "n_reps": n_reps,
        "seed0": seed0,
        "summary": _df_records(summary),
    }


# --------------------------------------------------------------------------- #
# Tool 3 - dispatching policy comparison + decision table
# --------------------------------------------------------------------------- #
DISPATCH_SCHEMA = {
    "name": "run_dispatch_comparison",
    "description": (
        "Compare dispatching policies (edd, critical_ratio, queue_time_aware, "
        "release_control) against the FIFO baseline, CRN-paired, across demand "
        "regimes, and return the decision table (which policy wins which "
        "business objective, with a measured caveat and significance flag). "
        "Use to answer 'which dispatch policy should we run'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "treatments": {
                "type": "array",
                "items": {"type": "string", "enum": ["edd", "critical_ratio",
                                                      "queue_time_aware", "release_control"]},
                "default": ["edd", "critical_ratio", "queue_time_aware", "release_control"],
            },
            "n_reps": {"type": "integer", "minimum": MIN_N_REPS, "maximum": MAX_N_REPS,
                       "default": 30},
            "seed0": {"type": "integer", "default": 8000},
        },
        "required": [],
    },
}

_ALL_TREATMENTS = ("edd", "critical_ratio", "queue_time_aware", "release_control")


def run_dispatch_comparison(treatments: list[str] | None = None,
                            n_reps: int = 30, seed0: int = 8000) -> dict:
    treatments = tuple(treatments) if treatments is not None else _ALL_TREATMENTS
    for t in treatments:
        _check_choice("run_dispatch_comparison", t, _ALL_TREATMENTS, "treatments[i]")
    _check_bounds("run_dispatch_comparison", n_reps, MIN_N_REPS, MAX_N_REPS, "n_reps")

    cfg = default_config()
    paired = compare_policies(cfg, treatments=treatments, n_reps=n_reps, seed0=seed0)
    summary = summarize_policy_comparison(paired)
    table = decision_table(summary)

    return {
        "treatments": list(treatments),
        "n_reps": n_reps,
        "seed0": seed0,
        "decision_table": _df_records(table),
    }


# --------------------------------------------------------------------------- #
# Tool 4 - PM timing comparison
# --------------------------------------------------------------------------- #
PM_SCHEMA = {
    "name": "run_pm_timing_comparison",
    "description": (
        "Compare preventive-maintenance timing alternatives (immediate/mid/late) "
        "against a no-degradation baseline for a degrading tool, CRN-paired, "
        "reporting cycle time, yield risk, and total cost per scenario. Use to "
        "answer 'when should we schedule PM on a degrading tool'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "station": {"type": "string", "enum": list(VALID_PM_STATIONS), "default": "LITHO"},
            "n_reps": {"type": "integer", "minimum": MIN_N_REPS, "maximum": 15,
                       "default": 15,
                       "description": "Capped at 15, matching maintenance_whatif's "
                                      "own reference n_reps."},
            "seed0": {"type": "integer", "default": 6000},
        },
        "required": [],
    },
}


def run_pm_timing_comparison(station: str = "LITHO", n_reps: int = 15,
                             seed0: int = 6000) -> dict:
    _check_choice("run_pm_timing_comparison", station, VALID_PM_STATIONS, "station")
    _check_bounds("run_pm_timing_comparison", n_reps, MIN_N_REPS, 15, "n_reps")

    cfg = default_config()
    paired = compare_pm_timings(station=station, cfg=cfg, n_reps=n_reps, seed0=seed0)
    summary = summarize_pm_comparison(paired)

    return {
        "station": station,
        "n_reps": n_reps,
        "seed0": seed0,
        "summary": _df_records(summary),
    }


# --------------------------------------------------------------------------- #
# Tool 5 - read-only KPI baseline
# --------------------------------------------------------------------------- #
KPI_SCHEMA = {
    "name": "get_kpi_baseline",
    "description": (
        "Read-only baseline KPI snapshot of the locked default configuration: "
        "steady-state throughput, mean cycle time, and per-station utilization. "
        "No what-if, no randomness beyond the locked default seed. Use to "
        "anchor a memo's 'current state' section."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "seed": {"type": "integer", "default": 42,
                     "description": "Locked default-config seed; bounded to the "
                                     "documented default to keep the baseline "
                                     "reproducible across memos."},
        },
        "required": [],
    },
}


def get_kpi_baseline(seed: int = 42) -> dict:
    _check_bounds("get_kpi_baseline", seed, 42, 42, "seed")  # only the locked default

    from factory_generator import draw_randoms, simulate  # local import, avoids top-level cost
    cfg = default_config(seed=seed)
    draws = draw_randoms(cfg, seed)
    log, life, _ = simulate(cfg, draws)
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours
    throughput, cycle_time = steady_state_kpis(life, t0, t1)
    util = station_utilization(log, t0, t1, {
        s: {"n_tools": st.n_tools, "batch_size": st.batch_size}
        for s, st in cfg.stations.items()
    }, order=list(dict.fromkeys(cfg.route)))

    return {
        "seed": seed,
        "throughput_per_hour": float(throughput),
        "mean_cycle_time_hours": float(cycle_time),
        "utilization": _df_records(util),
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """Maps tool name -> (JSON schema dict, python callable).

    ``call`` validates via the callable itself (which raises ValueError on
    out-of-bounds args - GATE 4 in ``agent_check.py``), computes the
    deterministic run_id from the tool name / canonicalized args / seed,
    executes, attaches ``run_id`` to the result, and logs the invocation via
    the supplied ``RunLogger`` if given.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict, Callable[..., dict]]] = {}
        for schema, fn in (
            (CAPACITY_SCHEMA, run_capacity_whatif),
            (DEMAND_SCHEMA, run_demand_whatif),
            (DISPATCH_SCHEMA, run_dispatch_comparison),
            (PM_SCHEMA, run_pm_timing_comparison),
            (KPI_SCHEMA, get_kpi_baseline),
        ):
            self._tools[schema["name"]] = (schema, fn)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schema(self, name: str) -> dict:
        return self._tools[name][0]

    def schemas(self) -> list[dict]:
        return [s for s, _ in self._tools.values()]

    def callable(self, name: str) -> Callable[..., dict]:
        return self._tools[name][1]

    def call(self, name: str, args: dict, logger: RunLogger | None = None,
             timestamp: str = "1970-01-01T00:00:00+00:00") -> dict:
        """Validate + execute a registered tool, returning its result with run_id.

        ``args`` must NOT include ``seed0``/``seed`` twice; the seed used for
        the run_id is whichever of ``seed0``/``seed`` the tool accepts
        (defaulting to the tool's own default when not supplied), so the same
        tool+args always yields the same run_id (GATE 5).
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}. Registered: {self.names()}")
        _schema, fn = self._tools[name]
        seed = args.get("seed0", args.get("seed"))
        if seed is None:
            # fall back to the callable's own default seed argument
            defaults = fn.__defaults__ or ()
            names_ = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            defaults_map = dict(zip(names_[-len(defaults):], defaults)) if defaults else {}
            seed = defaults_map.get("seed0", defaults_map.get("seed", 0))

        run_id = compute_run_id(name, args, seed)
        result = fn(**args)
        result = dict(result)
        result["run_id"] = run_id

        if logger is not None:
            logger.log(run_id=run_id, tool=name, args=args, seed=seed,
                       engine_config={"default_config_seed": 42}, result=result,
                       timestamp=timestamp)
        return result
