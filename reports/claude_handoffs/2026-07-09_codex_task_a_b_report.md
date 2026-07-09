# Codex Progress Report for Claude, 2026-07-09

## Summary

Codex completed both commissioned items that were active after the scenario
runner handover.

1. Task A, live agent transcript evidence, is resolved and pushed.
2. Task B, scenario-runner JSON extraction, is resolved and pushed as a
   behavior-preserving refactor.

The remaining publishing decision is still owner-controlled: merge `v2` into
`main` when the owner is ready.

## Task A, Live Agent Evidence

Commit:

- `fc55c4e feat: record first verified live agent session (M10)`

Committed evidence:

- `reports/agent_sessions/session_20260708T161010Z/memo.md`
- `reports/agent_sessions/session_20260708T161010Z/transcript.json`

Result:

- Provider: OpenAI
- Model: `gpt-4.1-mini`
- Question: `Is one more litho tool worth it under 15 percent demand growth?`
- Status: `VERIFIED`
- Tool runs logged: `1`
- Citations: `7`
- `verification.all_found`: `true`
- `uncited_numbers`: `[]`
- Estimated API cost recorded in transcript: `$0.001150`
- Local budget guard: `$1.00`

What happened:

- The live LLM requested `run_capacity_whatif`.
- The local engine ran `station=LITHO`, `demand_factors=[1.15]`,
  `n_reps=30`, `seed0=1000`.
- The engine returned `mean_d_throughput = 0.007947530864197534` with run id
  `90894f5209e03906`.
- The live memo cited `0.00795 [run:90894f5209e03906]`, and verification
  traced it back to the run log.

Notes:

- Two earlier live attempts failed verification because the model emitted
  uncited numeric values. They remain local only and were not committed as
  evidence.
- The OpenAI runner now supports `.env.local`, `LIVE_AGENT_MAX_USD`, usage
  recording, and stricter live citation prompting.

## Task B, Scenario Runner JSON Extraction

Commit:

- `ff4ddcf refactor: generate scenario runner data from engines (V2)`

Files changed:

- `src/kpi/export_scenario_runner_data.py`
- `docs/assets/scenario_runner_data.json`
- `docs/scenario-runner.html`

Result:

- New exporter derives data from engine entry points and writes
  `docs/assets/scenario_runner_data.json`.
- `docs/scenario-runner.html` now fetches that JSON and fills numeric/data
  fields from it.
- Owner-reviewed bilingual memo prose remains in HTML.
- No visible wording or layout was intentionally changed.

Exporter sources and seed conventions:

- Capacity: `src/decision/whatif.py`, `n_reps=30`, `seed0=1000`
- Maintenance: `src/equipment/maintenance_whatif.py`, `n_reps=15`,
  `seed0=6000`
- Dispatch: `src/decision/dispatch_whatif.py`, `n_reps=30`, `seed0=8000`
- Yield: `src/decision/yield_whatif.py`, `n_reps=30`, `seed0=5000`
- Monitoring: `data/synthetic/findings_cache.json`, config seed `42`,
  onset day `30`, alert day `84`

Hard-gated published values:

| Area | Published value checked |
|---|---|
| Capacity | `2.46`, `0.70`, `0.26`, `0.05` hours |
| Stress cost | `+$8.0k`, `-$1.7k`, `+$1.7k`, `+$1.4k` |
| Maintenance | `$1.75M`, `$2.45M`, `$4.59M` |
| Dispatch | `EDD`, `EDD`, `Release control`, `Release control` |
| Yield | `-2.87 h`, `+0.038` |
| Monitoring | day `84` of `160`, remaining horizon `76` |

Verification performed:

- `py src/kpi/export_scenario_runner_data.py` exits `0`.
- Exporter runtime measured by `Measure-Command`: `70.8` seconds.
- Re-running the exporter produced an identical JSON hash:
  `6D6D53B77D95C54B2484F6689BCBA7F3000DE9F99522ED9CE5AA1A887190AF73`.
- Inline JS extracted from `docs/scenario-runner.html` passes `node --check`.
- `git diff --check` is clean.
- No em dash found in the changed Task B files.
- Local static preview at `http://127.0.0.1:8789/scenario-runner.html`
  rendered the JSON-backed values correctly.

Browser spot-checks:

- Initial capacity row: `+1 lot`, `-2.46 h`, `+$8.0k`.
- Maintenance bars: `$1.75M`, `$2.45M`, `$4.59M`.
- Dispatch winners: `EDD`, `EDD`, `Release control`, `Release control`.
- Yield tab: `-2.87 h`, `+0.038`.
- Monitoring rows: `No alert within 160 days`, `Alert on day 84`.
- Interactive checks also passed for FURNACE, late PM, yield-risk dispatch,
  baseline yield, and output-only monitoring.

## Suggested Claude Review Focus

- Decide whether the live agent evidence should get a small visible entry point
  on `docs/index.html` or stay as a README/report artifact.
- Review the scenario-runner JSON extraction for maintainability. The exporter
  is intentionally explicit rather than abstract, because exact published-value
  gates are easier to audit that way.
- If moving toward publication, the main remaining step is still the owner
  decision to merge `v2` into `main`.
