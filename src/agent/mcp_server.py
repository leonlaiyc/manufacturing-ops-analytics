"""
MCP stdio server exposing the M10 agent tool layer (``src/agent/tools.py``).

Any MCP client (Claude Code, Claude Desktop) can call the five bounded what-if
tools directly; the server is a thin transport wrapper and adds no behavior:

- tool schemas are the registry's own hand-written JSON schemas, passed
  through verbatim (GATE 1 in ``mcp_check.py`` asserts deep equality);
- every call goes through ``ToolRegistry.call``, so it keeps the exact same
  bounds validation, deterministic ``run_id``, and JSONL run logging as the
  in-repo agent loop (default log: ``reports/mcp_sessions/mcp_run_log.jsonl``);
- out-of-bounds or malformed arguments raise before anything is logged, and
  surface to the client as an MCP tool error.

Honest scope: these tools run the fixed-seed synthetic DES line. Results are
method demonstrations with known ground truth, not predictions about any real
factory. Every result carries a ``run_id`` traceable to the run log.

No LLM anywhere in this module; the client on the other side of stdio is the
one doing the reasoning.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import ToolRegistry          # noqa: E402
from run_log import RunLogger           # noqa: E402

SERVER_NAME = "fab-whatif"
DEFAULT_LOG_PATH = REPO_ROOT / "reports" / "mcp_sessions" / "mcp_run_log.jsonl"


def build_server(logger: RunLogger, registry: ToolRegistry | None = None) -> Server:
    """Wire a ToolRegistry to an MCP low-level Server.

    The registry's schemas become the MCP tool list unchanged; calls are
    dispatched through ``registry.call`` so run_id computation and logging
    stay identical to the in-repo agent loop. Exceptions raised by the
    registry (unknown tool, out-of-bounds args) propagate to the MCP layer,
    which reports them to the client as a tool error (``isError``).
    """
    registry = registry if registry is not None else ToolRegistry()
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["input_schema"],
            )
            for schema in registry.schemas()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        timestamp = datetime.now(timezone.utc).isoformat()
        result = registry.call(name, dict(arguments), logger=logger,
                               timestamp=timestamp)
        return [types.TextContent(type="text",
                                  text=json.dumps(result, sort_keys=True))]

    return server


async def _run_stdio(log_path: Path) -> None:
    logger = RunLogger(log_path=log_path)
    server = build_server(logger)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def main() -> None:
    log_path = DEFAULT_LOG_PATH
    if len(sys.argv) > 2 and sys.argv[1] == "--log-path":
        log_path = Path(sys.argv[2])
    anyio.run(_run_stdio, log_path)


if __name__ == "__main__":
    main()
