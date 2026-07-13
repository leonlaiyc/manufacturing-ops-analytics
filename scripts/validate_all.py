"""Run the repository acceptance checks in one command.

This wrapper does not replace the individual validation scripts. It gives
reviewers one stable entry point while preserving each module's focused checks.

Run:
    py scripts/validate_all.py
    py scripts/validate_all.py --with-notebooks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The interpreter that launched this script; portable across Windows (py
# launcher) and CI runners where no `py` shim exists.
PY = sys.executable


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]


CORE_CHECKS: tuple[Check, ...] = (
    Check("compileall", (PY,"-m", "compileall", "-q", "src")),
    Check("M2 generator validation", (PY,"src/generator/validate_m2.py")),
    Check("CRN determinism", (PY,"src/generator/crn_check.py")),
    Check("M5 monitoring", (PY,"src/monitoring/monitoring_check.py")),
    Check("M7 quality", (PY,"src/quality/quality_check.py")),
    Check("M7 virtual metrology", (PY,"src/quality/vm_check.py")),
    Check("M8 equipment E10", (PY,"src/equipment/equipment_check.py")),
    Check("M8 maintenance what-if", (PY,"src/equipment/maintenance_check.py")),
    Check("M8 PdM", (PY,"src/equipment/pdm_check.py")),
    Check("M9 dispatch generator", (PY,"src/generator/dispatch_check.py")),
    Check("M9 dispatch what-if", (PY,"src/decision/dispatch_whatif_check.py")),
    Check("M10 agent tools", (PY,"src/agent/agent_check.py")),
    Check("M10 agent loop", (PY,"src/agent/loop_check.py")),
    Check("M10 MCP wrapper", (PY,"src/agent/mcp_check.py")),
    Check("M11 data quality", (PY,"src/dataquality/dq_check.py")),
    Check("M11 reliability", (PY,"src/dataquality/reliability_check.py")),
    Check("scenario runner data", (PY,"src/kpi/export_scenario_runner_data.py")),
    Check("index static assets", (PY,"src/kpi/export_index_assets.py")),
    Check("baseline HTML dashboard", (PY,"src/kpi/export_html_dashboard.py")),
)

NOTEBOOKS: tuple[str, ...] = (
    "notebooks/01_process_mining_real_log.ipynb",
    "notebooks/02_synthetic_generator_demo.ipynb",
    "notebooks/03_kpi_dashboard.ipynb",
    "notebooks/04_bottleneck_identification.ipynb",
    "notebooks/05_kpi_anomaly_monitoring.ipynb",
    "notebooks/06_capacity_demand_cost_whatif.ipynb",
    "notebooks/07_quality_yield_virtual_metrology.ipynb",
    "notebooks/08_equipment_health_e10_monitoring.ipynb",
    "notebooks/09_dispatching_policy_comparison.ipynb",
    "notebooks/10_agentic_decision_support.ipynb",
    "notebooks/11_data_quality_model_reliability.ipynb",
)


def notebook_checks(output_dir: Path) -> tuple[Check, ...]:
    return tuple(
        Check(
            f"notebook {Path(path).name}",
            (
                PY,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                path,
                "--output-dir",
                str(output_dir),
                "--ExecutePreprocessor.kernel_name=py310",
                "--ExecutePreprocessor.timeout=900",
            ),
        )
        for path in NOTEBOOKS
    )


def run_check(check: Check) -> bool:
    start = time.perf_counter()
    print(f"[RUN ] {check.name}")
    result = subprocess.run(check.command, cwd=ROOT, text=True)
    elapsed = time.perf_counter() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"[{status}] {check.name} ({elapsed:.1f}s)")
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-notebooks",
        action="store_true",
        help="also execute all notebooks with the py310 Jupyter kernel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = list(CORE_CHECKS)

    with tempfile.TemporaryDirectory(prefix="moa-nb-check-") as tmp:
        if args.with_notebooks:
            checks.extend(notebook_checks(Path(tmp)))

        failed: list[str] = []
        for check in checks:
            if not run_check(check):
                failed.append(check.name)

    passed = len(checks) - len(failed)
    print(f"\nSummary: {passed} passed, {len(failed)} failed")
    if failed:
        print("Failed checks:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
