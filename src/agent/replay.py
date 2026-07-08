"""
Offline scripted-session replay harness for the M10 agentic decision-support
layer (Stage A).

A canned operational question, a fixed sequence of tool calls answering it,
and the rendered memo, all deterministic (fixed seeds, fixed timestamp, no
LLM anywhere). This is the fixture ``agent_check.py`` uses for GATES 1-3
(reproducibility, traceability, tamper detection): running ``run_session()``
twice must produce byte-identical logs and memos.

Canned question: "Is one more litho tool worth it under 15 percent demand
growth?" Answered by:
  1. get_kpi_baseline           - anchor: current throughput/cycle time.
  2. run_demand_whatif          - what a 15% demand increase alone does to
                                   cycle time and yield risk.
  3. run_capacity_whatif        - what adding one LITHO tool does to
                                   throughput, at baseline and +15% demand.

Consumers must have src/agent on sys.path (or run from this file's directory)
in addition to the usual generator/bottleneck/quality/decision/equipment/kpi
paths that ``tools.py`` adds itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import ToolRegistry           # noqa: E402
from run_log import RunLogger            # noqa: E402
from memo import render_memo             # noqa: E402

#: Fixed so GATE 1 (reproducibility) can compare two runs byte-for-byte.
FIXED_TIMESTAMP = "2026-07-07T00:00:00+00:00"

QUESTION = "Is one more litho tool worth it under 15 percent demand growth?"

#: The fixed sequence of (tool_name, args) this scripted session answers the
#: question with. n_reps kept modest so the replay harness runs in seconds.
SCRIPTED_CALLS = [
    ("get_kpi_baseline", {}),
    ("run_demand_whatif", {"demand_factor": 1.15, "n_reps": 20, "seed0": 5000}),
    ("run_capacity_whatif", {"station": "LITHO", "demand_factors": [1.0, 1.15],
                              "n_reps": 20, "seed0": 1000}),
]


def run_session() -> tuple[RunLogger, str]:
    """Execute the scripted sequence of tool calls and render the memo.

    Returns (logger, memo_text). Deterministic: same registry, same fixed
    seeds/args, same fixed timestamp every time this is called.
    """
    registry = ToolRegistry()
    logger = RunLogger()
    tool_results = []
    for name, args in SCRIPTED_CALLS:
        result = registry.call(name, args, logger=logger, timestamp=FIXED_TIMESTAMP)
        result = dict(result)
        result["_tool_name"] = name
        tool_results.append(result)

    demand_result = tool_results[1]
    capacity_result = tool_results[2]
    demand_summary = demand_result["summary"][0]
    capacity_rows = {row["factor"]: row for row in capacity_result["per_factor"]}

    recommendation_stub = (
        "Framing for the human decision-maker: at the locked demand level "
        "(factor 1.0) the extra LITHO tool's throughput gain is measured "
        f"as {capacity_rows[1.0]['mean_d_throughput']:.4f} [run:{capacity_result['run_id']}] "
        "lots/hour, while at +15% demand it is measured as "
        f"{capacity_rows[1.15]['mean_d_throughput']:.4f} [run:{capacity_result['run_id']}] "
        "lots/hour; demand growth alone (no extra tool) is measured to move "
        f"cycle time by {demand_summary['d_cycle_time']:.4f} [run:{demand_result['run_id']}] "
        "hours. The recommendation above states the largest single measured "
        "effect; a human reviewer should weigh it against the capacity cost "
        "(decision/cost_model.py) before acting - this stage does not."
    )

    memo_text = render_memo(QUESTION, tool_results, recommendation_stub=recommendation_stub)
    return logger, memo_text


if __name__ == "__main__":
    logger, memo_text = run_session()
    print(memo_text)
    print()
    print(f"[{len(logger.entries)} run(s) logged]")
