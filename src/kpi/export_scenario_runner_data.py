"""
Export the scenario-runner demo data from engine outputs.

This is intentionally a hard-gated exporter: if any published value cannot be
reproduced at the displayed rounding, the script exits before writing JSON.
The bilingual memo copy stays in docs/scenario-runner.html; this file exports
only data fields and structured analysis-record values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for rel in (
    "generator",
    "bottleneck",
    "decision",
    "equipment",
    "quality",
    "monitoring",
    "kpi",
):
    path = str(SRC / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from cost_model import CostRates  # noqa: E402
from counterfactual import summarize  # noqa: E402
from dispatch_whatif import run_all as run_dispatch_all  # noqa: E402
from maintenance_whatif import demo_litho_pm_timing  # noqa: E402
from precompute_findings import STRESS_TOOL_COSTS  # noqa: E402
from whatif import run_capacity_cost  # noqa: E402
from yield_whatif import demo_extra_litho_tool, summarize_yield_comparison  # noqa: E402

OUT = ROOT / "docs" / "assets" / "scenario_runner_data.json"
CACHE = ROOT / "data" / "synthetic" / "findings_cache.json"

STATION_ORDER = ("LITHO", "FURNACE", "DEPO", "METRO")
INTERVENTION = {station: f"{station}+1" for station in STATION_ORDER}


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: got {actual!r}, expected {expected!r}")


def _fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _fmt_money_k(value: float) -> str:
    return f"{value:+.1f}k"


def _policy_label(policy: str) -> str:
    return "EDD" if policy == "edd" else policy.replace("_", " ").capitalize()


def _capacity() -> dict:
    rates = CostRates()
    base = run_capacity_cost(n_reps=30, seed0=1000, rates=rates)
    stress = run_capacity_cost(
        n_reps=30,
        seed0=1000,
        rates=rates,
        added_tool_costs=STRESS_TOOL_COSTS,
    )
    ct_summary = summarize(base, "d_cycle_time")
    stress_cost = (
        stress.groupby("intervention")["d_cost"]
        .mean()
        .reindex([INTERVENTION[s] for s in STATION_ORDER])
    )

    expected_ct = {"LITHO": 2.46, "FURNACE": 0.70, "DEPO": 0.26, "METRO": 0.05}
    expected_cost = {"LITHO": 8.0, "FURNACE": -1.7, "DEPO": 1.7, "METRO": 1.4}
    expected_ci = {"LITHO": (2.13, 2.79)}
    slot_labels = {
        "LITHO": "+1 lot",
        "FURNACE": "+4 lots",
        "DEPO": "+1 lot",
        "METRO": "+1 lot",
    }

    rows = {}
    scenarios = {}
    for station in STATION_ORDER:
        intervention = INTERVENTION[station]
        ct = abs(float(ct_summary.loc[intervention, "mean"]))
        ci_low = abs(float(ct_summary.loc[intervention, "ci95_high"]))
        ci_high = abs(float(ct_summary.loc[intervention, "ci95_low"]))
        cost_k = float(stress_cost.loc[intervention]) / 1000

        _assert_equal(f"{station} cycle-time reduction", round(ct, 2), expected_ct[station])
        _assert_equal(f"{station} stress net cost", round(cost_k, 1), expected_cost[station])
        if station in expected_ci:
            _assert_equal(f"{station} cycle-time CI", (round(ci_low, 2), round(ci_high, 2)),
                          expected_ci[station])

        cycle_label = f"-{ct:.2f} h"
        cost_label = f"-${abs(cost_k):.1f}k" if cost_k < 0 else f"+${cost_k:.1f}k"
        kpi = f"delta_cycle_time = {cycle_label}"
        if station == "LITHO":
            kpi += f" (95% CI {ci_low:.2f}-{ci_high:.2f})"
        kpi += f" &middot; net_cost = {cost_label}"

        rows[station] = {
            "slot_label": slot_labels[station],
            "cycle_label": cycle_label,
            "cost_label": cost_label,
        }
        scenarios[station] = {
            "kpi": kpi,
            "delta_cycle_time_h": round(-ct, 2),
            "net_cost_k": round(cost_k, 1),
        }

    return {
        "records": {
            "engine": "src/decision/whatif.py + cost_model.py",
            "run": "n_reps=30, CRN-paired",
            "seeds": "1000-1029",
            "assumptions": (
                "FIFO dispatch; Poisson arrivals; lognormal PT; "
                "illustrative CostRates (ranking only)"
            ),
        },
        "rows": rows,
        "scenarios": scenarios,
    }


def _maintenance() -> dict:
    summary = demo_litho_pm_timing(n_reps=15, seed0=6000).set_index("scenario")
    mapping = {
        "immediate": ("immediate_pm", 1),
        "mid": ("mid_pm", 2),
        "late": ("late_pm", 3),
    }
    expected = {"immediate": 1.75, "mid": 2.45, "late": 4.59}

    scenarios = {}
    for key, (scenario, rank) in mapping.items():
        total_m = float(summary.loc[scenario, "total_cost"]) / 1_000_000
        _assert_equal(f"{key} PM total cost", round(total_m, 2), expected[key])
        scenarios[key] = {
            "value": f"${total_m:.2f}M",
            "kpi": f"total_cost = ${total_m:.2f}M (rank {rank} of 3)",
            "total_cost_m": round(total_m, 2),
        }

    return {
        "records": {
            "engine": "src/equipment/maintenance_whatif.py",
            "run": "n_reps=15, CRN-paired",
            "seeds": "6000-6014",
            "assumptions": (
                "station=LITHO (engineered bottleneck); E10 state layer; "
                "illustrative downtime + delay cost rates"
            ),
        },
        "scenarios": scenarios,
    }


def _dispatch() -> dict:
    _summary, decision = run_dispatch_all(n_reps=30, seed0=8000)
    base = decision[decision["regime"] == "baseline"].set_index("objective")
    objectives = {
        "ct": ("cycle time", "avg cycle time"),
        "ontime": ("on-time delivery", "on-time delivery"),
        "yieldrisk": ("yield risk", "yield risk"),
        "cost": ("total cost", "total cost"),
    }
    expected = {
        "ct": "EDD",
        "ontime": "EDD",
        "yieldrisk": "Release control",
        "cost": "Release control",
    }

    scenarios = {}
    winners = {}
    for key, (objective, label) in objectives.items():
        winner = _policy_label(str(base.loc[objective, "winner"]))
        _assert_equal(f"{objective} dispatch winner", winner, expected[key])
        display = "release_control" if winner == "Release control" else winner
        scenarios[key] = {
            "winner": winner,
            "kpi": f"best_policy = {display} (objective: {label})",
        }
        winners[key] = winner

    return {
        "records": {
            "engine": "src/decision/dispatch_whatif.py",
            "run": "n_reps=30, CRN-paired",
            "seeds": "8000-8029",
            "assumptions": (
                "policies = fifo, edd, critical_ratio, release_control; "
                "illustrative yield + congestion cost rates"
            ),
        },
        "winners": winners,
        "scenarios": scenarios,
    }


def _yield() -> dict:
    summary = summarize_yield_comparison(
        demo_extra_litho_tool(n_reps=30, seed0=5000)
    ).set_index("scenario")
    row = summary.loc["LITHO+1"]
    d_ct = float(row["d_cycle_time"])
    d_vio = float(row["d_violation_rate"])
    _assert_equal("yield cycle-time delta", round(d_ct, 2), -2.87)
    _assert_equal("yield violation-rate delta", round(d_vio, 3), 0.038)

    return {
        "records": {
            "engine": "src/decision/yield_whatif.py",
            "run": "n_reps=30, CRN-paired",
            "seeds": "5000-5029 (shared QualityConfig seed)",
            "assumptions": (
                "station=LITHO (engineered bottleneck); queue-time-driven "
                "violations; linear-additive ground truth"
            ),
        },
        "scenarios": {
            "baseline": {
                "ct": "0 h",
                "vio": "0",
                "kpi": "delta_cycle_time = 0 &middot; delta_violation_rate = 0 (reference run)",
            },
            "litho1": {
                "ct": f"{d_ct:.2f} h",
                "vio": _fmt_signed(d_vio, 3),
                "kpi": (
                    f"delta_cycle_time = {d_ct:.2f} h &middot; "
                    f"delta_violation_rate = {_fmt_signed(d_vio, 3)}"
                ),
            },
        },
    }


def _monitoring() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(
            "data/synthetic/findings_cache.json not found. Run "
            "py src/kpi/precompute_findings.py first."
        )
    finding = json.loads(CACHE.read_text(encoding="utf-8"))["finding_03"]
    alert_day = int(finding["alert_day"])
    horizon = int(finding["horizon_days"])
    onset = int(finding["onset_day"])
    remaining = horizon - alert_day
    output_alert = finding["output_only_alert_day"]

    _assert_equal("monitoring alert day", alert_day, 84)
    _assert_equal("monitoring horizon", horizon, 160)
    _assert_equal("monitoring onset day", onset, 30)
    _assert_equal("monitoring output-only alert", output_alert, None)
    _assert_equal("monitoring remaining horizon", remaining, 76)

    return {
        "records": {
            "engine": "src/monitoring + src/kpi/precompute_findings.py",
            "run": f"deterministic backtest, horizon={horizon} days",
            "seeds": f"config seed 42; labeled onset = day {onset}",
            "assumptions": (
                "same locked line observed for 160 days; synthetic labeled "
                "anomaly, so detection delay is measured against known ground truth"
            ),
        },
        "rows": {
            "output": {
                "result_en": f"No alert within {horizon} days",
                "result_zh": f"{horizon} 天內沒有 alert",
            },
            "ewma": {
                "result_en": f"Alert on day {alert_day}",
                "result_zh": f"day {alert_day} alert",
            },
        },
        "scenarios": {
            "output": {
                "kpi": f"alert_day = none (silent over {horizon}-day horizon)",
            },
            "ewma": {
                "kpi": (
                    f"alert_day = {alert_day} &middot; "
                    f"remaining_horizon = {remaining} days"
                ),
            },
        },
    }


def build_payload() -> dict:
    return {
        "schema_version": 1,
        "source": "src/kpi/export_scenario_runner_data.py",
        "capacity": _capacity(),
        "maintenance": _maintenance(),
        "dispatch": _dispatch(),
        "yield": _yield(),
        "monitoring": _monitoring(),
    }


def main() -> int:
    payload = build_payload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
