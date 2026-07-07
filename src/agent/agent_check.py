"""
M10 Stage A regression + sanity check (run end to end).

Confirms the agentic decision-support foundation (``tools.py``, ``run_log.py``,
``memo.py``, ``replay.py``) behaves correctly. No LLM anywhere in this stage;
every gate is a mechanical check against logged, reproducible simulation
output.

  GATE 1 - reproducibility: two runs of ``replay.run_session()`` produce an
           identical run log (JSONL) and an identical memo.
  GATE 2 - traceability: ``verify_memo_numbers`` on the replay memo reports
           every cited number found in its run log (100 percent), and at
           least 5 citations exist.
  GATE 3 - tamper detection: corrupting one number in the memo makes
           ``verify_memo_numbers`` flag exactly that citation (and no other).
  GATE 4 - schema validity: every registered tool's JSON schema validates
           (draft-07 via jsonschema if installed, else a documented
           structural check), and every callable rejects an out-of-bounds
           argument with a clear error.
  GATE 5 - run_id integrity: same tool+args+seed gives the same run_id;
           changing any argument changes it.

Run:  py src/agent/agent_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import ToolRegistry                        # noqa: E402
from run_log import RunLogger, compute_run_id, verify_memo_numbers  # noqa: E402
import replay                                          # noqa: E402

try:
    import jsonschema
    _HAVE_JSONSCHEMA = True
except ImportError:
    _HAVE_JSONSCHEMA = False


#: Minimal draft-07 meta-schema subset check used only if jsonschema is not
#: installed. Confirms the structural shape every registered schema must
#: have: a JSON Schema object with "type": "object" and a "properties" dict.
def _structural_schema_check(schema: dict) -> list[str]:
    problems = []
    input_schema = schema.get("input_schema")
    if not isinstance(input_schema, dict):
        problems.append("input_schema missing or not a dict")
        return problems
    if input_schema.get("type") != "object":
        problems.append("input_schema.type must be 'object'")
    if not isinstance(input_schema.get("properties"), dict):
        problems.append("input_schema.properties must be a dict")
    if "required" in input_schema and not isinstance(input_schema["required"], list):
        problems.append("input_schema.required must be a list")
    if not isinstance(schema.get("name"), str) or not schema["name"]:
        problems.append("schema.name must be a non-empty string")
    if not isinstance(schema.get("description"), str) or not schema["description"]:
        problems.append("schema.description must be a non-empty string")
    return problems


def gate1_reproducibility() -> bool:
    print("=" * 64)
    print("GATE 1 - reproducibility (replay session run twice, identical)")
    print("=" * 64)

    logger_a, memo_a = replay.run_session()
    logger_b, memo_b = replay.run_session()

    log_ok = logger_a.to_jsonl() == logger_b.to_jsonl()
    memo_ok = memo_a == memo_b
    print(f"  run log identical across two runs  : {log_ok}")
    print(f"  memo text identical across two runs : {memo_ok}")

    ok = log_ok and memo_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def gate2_traceability() -> tuple[bool, "RunLogger", str]:
    print("=" * 64)
    print("GATE 2 - traceability (every cited number found in its run log)")
    print("=" * 64)

    logger, memo_text = replay.run_session()
    report = verify_memo_numbers(memo_text, logger)

    enough_citations = report["total_citations"] >= 5
    all_found = report["all_found"]
    print(f"  total citations found in memo       : {report['total_citations']}"
          f"  (>= 5 required: {enough_citations})")
    print(f"  every citation resolves to its run   : {all_found}")
    if not all_found:
        for r in report["report"]:
            if not (r["run_found"] and r["value_found"]):
                print(f"    UNRESOLVED: {r['citation_text']}")

    ok = enough_citations and all_found
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok, logger, memo_text


def gate3_tamper_detection(logger: "RunLogger", memo_text: str) -> bool:
    print("=" * 64)
    print("GATE 3 - tamper detection (corrupt one citation, exactly it is flagged)")
    print("=" * 64)

    baseline_report = verify_memo_numbers(memo_text, logger)
    assert baseline_report["all_found"], "precondition: clean memo must verify first"

    # Find the first citation and corrupt its NUMBER (not its run tag), by a
    # margin far outside any formatting/rounding tolerance.
    citation_re = re.compile(r"(-?\d[\d,]*\.?\d*)(\s*[A-Za-z%$]*\s*\[run:[0-9a-fA-F]+\])")
    m = citation_re.search(memo_text)
    assert m is not None, "precondition: memo must contain at least one citation"

    original_number = m.group(1)
    corrupted_number = "999999.999999"
    corrupted_memo = memo_text[:m.start(1)] + corrupted_number + memo_text[m.end(1):]

    tampered_report = verify_memo_numbers(corrupted_memo, logger)

    flagged = [r for r in tampered_report["report"]
               if not (r["run_found"] and r["value_found"])]
    exactly_one_flagged = len(flagged) == 1
    is_the_right_one = (exactly_one_flagged
                        and flagged[0]["cited_value"] == float(corrupted_number))
    others_still_pass = (tampered_report["total_citations"] - len(flagged)
                         == baseline_report["total_citations"] - 1)

    print(f"  original citation number tampered   : {original_number!r} -> {corrupted_number!r}")
    print(f"  exactly one citation now flagged     : {exactly_one_flagged}"
          f"  (flagged count={len(flagged)})")
    print(f"  the flagged citation is the tampered one : {is_the_right_one}")
    print(f"  every other citation still verifies  : {others_still_pass}")

    ok = exactly_one_flagged and is_the_right_one and others_still_pass
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def gate4_schema_validity() -> bool:
    print("=" * 64)
    print("GATE 4 - schema validity + bounds rejection")
    print("=" * 64)

    registry = ToolRegistry()
    schema_ok = True
    for schema in registry.schemas():
        if _HAVE_JSONSCHEMA:
            try:
                jsonschema.Draft7Validator.check_schema(schema["input_schema"])
                problems = []
            except jsonschema.exceptions.SchemaError as e:
                problems = [str(e)]
        else:
            problems = _structural_schema_check(schema)
        this_ok = not problems
        schema_ok = schema_ok and this_ok
        print(f"  {schema['name']:<28} schema valid: {this_ok}"
              + (f"  ({'; '.join(problems)})" if problems else ""))

    # Out-of-bounds rejection: one deliberately invalid call per tool.
    bad_calls = [
        ("run_capacity_whatif", {"station": "LITHO", "n_reps": 999}),
        ("run_capacity_whatif", {"station": "NOT_A_STATION"}),
        ("run_demand_whatif", {"demand_factor": 99.0}),
        ("run_dispatch_comparison", {"treatments": ["not_a_policy"]}),
        ("run_pm_timing_comparison", {"station": "CLEAN"}),
        ("run_pm_timing_comparison", {"n_reps": 1000}),
        ("get_kpi_baseline", {"seed": 7}),
    ]
    bounds_ok = True
    for name, args in bad_calls:
        try:
            registry.call(name, args)
            print(f"  REJECTED CALL DID NOT RAISE: {name}({args})")
            bounds_ok = False
        except ValueError as e:
            print(f"  {name}({args}) correctly rejected: {e}")

    ok = schema_ok and bounds_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def gate5_run_id_integrity() -> bool:
    print("=" * 64)
    print("GATE 5 - run_id integrity (same args -> same id, changed arg -> different id)")
    print("=" * 64)

    args = {"station": "LITHO", "demand_factors": [1.0, 1.15], "n_reps": 5}
    seed = 1000
    id_a = compute_run_id("run_capacity_whatif", args, seed)
    id_b = compute_run_id("run_capacity_whatif", dict(args), seed)
    same_ok = id_a == id_b
    print(f"  identical tool+args+seed -> identical run_id : {same_ok}"
          f"  ({id_a} == {id_b})")

    variants = [
        ("station", "DEPO"),
        ("demand_factors", [1.0, 1.25]),
        ("n_reps", 6),
    ]
    changed_ok = True
    for key, new_val in variants:
        changed_args = dict(args)
        changed_args[key] = new_val
        id_c = compute_run_id("run_capacity_whatif", changed_args, seed)
        differs = id_c != id_a
        changed_ok = changed_ok and differs
        print(f"  changing {key:<16} -> run_id differs: {differs}  ({id_c})")

    seed_id = compute_run_id("run_capacity_whatif", args, seed + 1)
    seed_differs = seed_id != id_a
    changed_ok = changed_ok and seed_differs
    print(f"  changing seed              -> run_id differs: {seed_differs}  ({seed_id})")

    # Argument ORDER must not matter (canonicalization), only VALUES.
    reordered = {"n_reps": 5, "station": "LITHO", "demand_factors": [1.0, 1.15]}
    id_reordered = compute_run_id("run_capacity_whatif", reordered, seed)
    order_invariant = id_reordered == id_a
    print(f"  key order does not affect run_id    : {order_invariant}")

    ok = same_ok and changed_ok and order_invariant
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    g1 = gate1_reproducibility()
    g2, logger, memo_text = gate2_traceability()
    g3 = gate3_tamper_detection(logger, memo_text)
    g4 = gate4_schema_validity()
    g5 = gate5_run_id_integrity()

    ok = g1 and g2 and g3 and g4 and g5
    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
