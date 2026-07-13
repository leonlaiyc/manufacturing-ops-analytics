"""
Offline gates for the MCP wrapper (``src/agent/mcp_server.py``).

Runs the server and a client in-process over the MCP SDK's in-memory
transport: no subprocess, no network, no API key. What is being proven is
that the MCP layer is a pure transport: schemas, bounds validation,
deterministic run_ids, and run logging behave exactly as they do when the
in-repo agent loop calls the registry directly.

GATE 1  tool list parity: list_tools over MCP returns exactly the registry's
        tool names, and every inputSchema equals the registry schema
        (deep equality, no normalization drift).
GATE 2  traceability end to end: a get_kpi_baseline call over MCP returns a
        result carrying a run_id, and that run_id is present in the JSONL
        run log the server wrote to disk.
GATE 3  bounds still bite: an out-of-bounds argument (invalid station)
        surfaces as an MCP tool error and appends nothing to the run log.
GATE 4  determinism through the transport: two MCP calls with identical
        args return the identical run_id (fixed-seed discipline survives
        serialization).

Usage: ``py src/agent/mcp_check.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from mcp_server import build_server     # noqa: E402
from run_log import RunLogger           # noqa: E402
from tools import ToolRegistry          # noqa: E402


def _log_lines(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in
            log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_gates(log_path: Path) -> list[tuple[str, bool, str]]:
    registry = ToolRegistry()
    logger = RunLogger(log_path=log_path)
    server = build_server(logger, registry=registry)
    results: list[tuple[str, bool, str]] = []

    async with create_connected_server_and_client_session(server) as session:
        # GATE 1 - tool list parity
        listed = (await session.list_tools()).tools
        names_mcp = sorted(t.name for t in listed)
        names_reg = sorted(registry.names())
        schema_ok = all(
            t.inputSchema == registry.schema(t.name)["input_schema"]
            and t.description == registry.schema(t.name)["description"]
            for t in listed
        )
        ok1 = names_mcp == names_reg and schema_ok
        results.append((
            "GATE 1 tool list parity (names + verbatim schemas)", ok1,
            f"mcp={names_mcp} registry={names_reg} schemas_equal={schema_ok}"))

        # GATE 2 - traceability end to end
        res = await session.call_tool("get_kpi_baseline", {})
        ok_call = not res.isError and len(res.content) == 1
        payload = json.loads(res.content[0].text) if ok_call else {}
        run_id = payload.get("run_id")
        logged_ids = [e["run_id"] for e in _log_lines(log_path)]
        ok2 = ok_call and run_id is not None and run_id in logged_ids
        results.append((
            "GATE 2 result carries run_id present in on-disk run log", ok2,
            f"run_id={run_id!r} logged={logged_ids}"))

        # GATE 3 - out-of-bounds arg -> MCP error, nothing logged
        n_before = len(_log_lines(log_path))
        bad = await session.call_tool("run_capacity_whatif",
                                      {"station": "NOT_A_STATION", "n_reps": 1})
        n_after = len(_log_lines(log_path))
        ok3 = bad.isError and n_after == n_before
        results.append((
            "GATE 3 out-of-bounds arg is an MCP error and is not logged", ok3,
            f"isError={bad.isError} log_entries {n_before}->{n_after}"))

        # GATE 4 - determinism through the transport
        res2 = await session.call_tool("get_kpi_baseline", {})
        payload2 = json.loads(res2.content[0].text) if not res2.isError else {}
        ok4 = payload2.get("run_id") == run_id and run_id is not None
        results.append((
            "GATE 4 identical args -> identical run_id over MCP", ok4,
            f"first={run_id!r} second={payload2.get('run_id')!r}"))

    return results


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "mcp_run_log.jsonl"
        results = anyio.run(run_gates, log_path)

    ok = all(passed for _, passed, _ in results)
    for label, passed, detail in results:
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
        print(f"  {detail}")
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
