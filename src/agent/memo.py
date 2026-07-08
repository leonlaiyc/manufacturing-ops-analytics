"""
Decision-memo renderer for the M10 agentic decision-support layer (Stage A).

``render_memo`` turns a question plus a sequence of logged tool results into
a structured markdown decision memo where every cited number carries a
``[run:<run_id>]`` tag pointing at the run that produced it
(``run_log.verify_memo_numbers`` checks the tag actually resolves). The
recommendation section is a DETERMINISTIC template (largest measured effect
by absolute value, read off the tool results themselves) - no LLM anywhere
in this stage; a later stage lets an LLM author that section instead.

Assumptions/caveats are pulled from the engines' own framing constants
(never hand-written duplicates of them), so a memo cannot silently drift from
what the engines actually assume:
  - "Rankings hold under the locked default configuration and CRN-paired
    replications; they are relative comparisons, not point forecasts."
  - "All data is synthetic (fixed-seed discrete-event simulation); no real
    fab measurements are used anywhere in this repository."
"""

from __future__ import annotations

from typing import Any

#: Pulled verbatim (not paraphrased) from the engines' own module docstrings
#: (see decision/yield_whatif.py, decision/dispatch_whatif.py,
#: equipment/maintenance_whatif.py): the shared framing every scenario in
#: this repo's what-if layer is run under.
ASSUMPTIONS = [
    "Rankings and deltas hold under the locked default configuration "
    "(7-station stylized wafer-fab loop, LITHO bottleneck) and CRN-paired "
    "replications (same random draw table for baseline and treatment); they "
    "are relative comparisons for decision support, not absolute forecasts.",
    "Cost figures use illustrative rates (see decision/cost_model.py, "
    "decision/yield_whatif.py) chosen to be defensible in review and re-"
    "ranked under a documented sensitivity sweep, never presented as real "
    "prices.",
]

SYNTHETIC_LINE = (
    "All numbers in this memo come from a fixed-seed discrete-event "
    "simulation of a synthetic wafer fab. No real production data is used "
    "anywhere in this repository."
)


def _fmt(value: Any, decimals: int = 2) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _cite(value: Any, run_id: str, decimals: int = 2) -> str:
    """Render one number as a memo citation: '<formatted> [run:<id>]'."""
    return f"{_fmt(value, decimals)} [run:{run_id}]"


def _scenario_line(tool_result: dict) -> str:
    """One line describing a tool call: tool name, key args, run_id."""
    name = tool_result.get("_tool_name", "tool")
    run_id = tool_result["run_id"]
    arg_bits = []
    for key in ("station", "demand_factor", "demand_factors", "treatments", "seed"):
        if key in tool_result:
            arg_bits.append(f"{key}={tool_result[key]}")
    arg_str = ", ".join(arg_bits) if arg_bits else "default args"
    return f"- `{name}` ({arg_str}) -> run `{run_id}`"


def _largest_effect(tool_results: list[dict]) -> tuple[str, str, float, str] | None:
    """Scan every numeric field across all tool results for the largest |value|
    among fields that look like an effect (start with 'd_' or 'mean_delta').

    Returns (tool_name, field_description, value, run_id) or None if no such
    field exists anywhere in the results.
    """
    best = None
    for tr in tool_results:
        name = tr.get("_tool_name", "tool")
        run_id = tr["run_id"]
        for key, val in tr.items():
            if key in ("run_id", "_tool_name"):
                continue
            rows = val if isinstance(val, list) else None
            if rows is None:
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for field, v in row.items():
                    if v is None or isinstance(v, bool):
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    if field.startswith("d_") or field == "mean_delta":
                        label_bits = [f"{k}={row[k]}" for k in row
                                      if k in ("scenario", "policy", "regime",
                                                "objective", "factor", "intervention")]
                        label = f"{name}.{field} ({', '.join(label_bits)})" if label_bits \
                            else f"{name}.{field}"
                        if best is None or abs(v) > abs(best[2]):
                            best = (name, label, float(v), run_id)
    return best


def render_memo(question: str, tool_results: list[dict],
                 recommendation_stub: str | None = None) -> str:
    """Render a structured markdown decision memo.

    Parameters
    ----------
    question : str
        The natural-language operational question the memo answers.
    tool_results : list[dict]
        Results from ``ToolRegistry.call(...)``, each augmented with a
        ``"_tool_name"`` key identifying which tool produced it (callers
        should set this before passing results in, since ``call`` itself
        does not stamp the name onto the result).
    recommendation_stub : str | None
        Optional extra sentence appended to the deterministic recommendation
        (e.g. operational context); the recommendation itself always states
        the largest measured effect found across the supplied tool results.

    Returns
    -------
    str : the rendered markdown memo. No em dash character is used anywhere.
    """
    lines: list[str] = []
    lines.append("# Decision Memo")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(question)
    lines.append("")

    lines.append("## Scenarios run")
    lines.append("")
    for tr in tool_results:
        lines.append(_scenario_line(tr))
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    lines.append("| Tool | Metric | Value | Run |")
    lines.append("|---|---|---|---|")
    for tr in tool_results:
        name = tr.get("_tool_name", "tool")
        run_id = tr["run_id"]
        for key, val in tr.items():
            if key in ("run_id", "_tool_name"):
                continue
            if isinstance(val, list):
                for row in val:
                    if not isinstance(row, dict):
                        continue
                    label_bits = [f"{row[k]}" for k in row
                                  if k in ("scenario", "policy", "regime", "objective",
                                           "factor", "station")]
                    row_label = ", ".join(str(b) for b in label_bits) or key
                    for field, v in row.items():
                        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                            continue
                        lines.append(f"| {name} | {row_label}.{field} | "
                                     f"{_cite(v, run_id)} | `{run_id}` |")
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                lines.append(f"| {name} | {key} | {_cite(val, run_id)} | `{run_id}` |")
    lines.append("")

    lines.append("## Assumptions and caveats")
    lines.append("")
    for a in ASSUMPTIONS:
        lines.append(f"- {a}")
    lines.append(f"- {SYNTHETIC_LINE}")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    best = _largest_effect(tool_results)
    if best is not None:
        tool_name, label, value, run_id = best
        lines.append(
            f"The largest measured effect across the scenarios run is "
            f"`{label}` = {_cite(value, run_id, decimals=4)}, from `{tool_name}`. "
            f"This is a deterministic summary of the logged results above, not "
            f"an LLM-generated judgment (a later stage adds LLM-authored "
            f"narrative on top of these same logged numbers)."
        )
    else:
        lines.append(
            "No effect-sized field (d_* or mean_delta) was found among the "
            "supplied tool results; no automatic recommendation could be "
            "derived."
        )
    if recommendation_stub:
        lines.append("")
        lines.append(recommendation_stub)
    lines.append("")

    return "\n".join(lines)
