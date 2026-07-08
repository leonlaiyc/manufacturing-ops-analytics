# V2 Plan: Quality-Aware Fab Operations Analytics

Execution contract for the V2 iteration. Any agent (Claude Code or Codex)
picking up a module starts here, then reads `CLAUDE.md` / `AGENTS.md` for hard
rules. Owner decides scope and methods; agents execute.

- Branch: all V2 work lands on `v2`. `main` stays at the reviewed V1
  (tagged `v1.0`). Merge to `main` only on owner instruction.
- Reader priority for docs: `docs/index.html`, then dashboard, then `README.md`.
  Keep the three consistent whenever positioning text changes.
- Style: no em dash character in reader-facing text. English only in the repo.

## Positioning (decided 2026-07-04)

- Statement: quality-aware fab operations analytics with simulation-in-the-loop
  decision support.
- The project is NOT called a digital twin. It is the simulation and what-if
  decision core that agentic fab architectures (Samsung GTC 2026 session
  S81834, with NVIDIA and Synopsys) call before acting. The missing live
  MES/physical link is stated openly as the boundary. The agent layer is
  positioned as task-level (L2 to L3 on the Synopsys autonomy ladder), never
  autonomous.
- Why-now anchors: (1) 2026's binding constraint on AI hardware is advanced
  packaging capacity (batch tools, queue-time limits, re-entrant flows), the
  structures this project models; (2) fabs are wiring simulators into agentic
  decision loops, the pattern M10 demonstrates at small scale.

## Modules (build order; README roadmap table is the public contract)

### M7 Quality / yield layer (SHIPPED 2026-07-05, commits 2346a5f..d78ce6e)
- Add lot-level yield risk to the DES output: LITHO/ETCH/METRO steps produce a
  quality risk score; a queue-time window before selected steps (photoresist
  analogy) raises rework/scrap probability when violated.
- Tabular virtual metrology: predict metrology outcome from upstream process
  time, queue time, tool state, utilization. Interpretable baseline first.
- Run-to-run flavor: give the two LITHO tools slightly different means and add
  a chamber-matching comparison.
- Yield-aware what-if: extend M6 so scenarios report cycle time AND yield risk.
- Guardrails: yield model parameters are synthetic and labeled as such; no
  claim of real-fab yield prediction. Fixed seeds. Schema changes must keep
  `validate_m2.py`, `crn_check.py`, `monitoring_check.py` passing.
- Done when: notebook 07 runs end to end; what-if output shows the
  cycle-time/yield trade-off with CRN pairing; validation commands pass.

### M8 Equipment health (SEMI E10) (SHIPPED 2026-07-06, commits 032efe0..d19a018)
- Tool-state event log (tool_id, station, start, end, e10_state, reason,
  lot_id) derived from or alongside the DES; states: Production, Standby,
  Scheduled Down, Unscheduled Down, Engineering.
- Metrics: MTBF, MTTR, availability, state decomposition per tool.
- Maintenance-timing trade-off: delay PM vs immediate PM, costed via the M6
  cost model; alert priority = severity times bottleneck criticality.
- Owner-approved exception: gradient boosting + SHAP allowed here, framed as
  measuring detection quality against known synthetic ground truth. Never
  claim predictive power; the sensor signatures are self-generated, so wording
  must avoid the circularity trap.
- Done when: notebook 08 reproduces E10 decomposition and the maintenance
  decision comparison; detection quality scored on labeled ground truth.

### M9 Dispatching policies (SHIPPED 2026-07-07, commits 56fe2c5..c94f296)

(A mid-stage handover note lived here on 2026-07-07 while a usage-limit
interruption was open; resolved the same day by commit 2544688. History in
git if ever needed.)
- Policies: FIFO (baseline), EDD, critical ratio, queue-time-aware,
  bottleneck-WIP control. CRN-paired runs on identical arrival streams.
- Output: decision table stating which policy wins under which demand and
  yield-risk conditions, with confidence intervals.
- Done when: notebook 09 produces the paired comparison; FIFO reproduces the
  V1 baseline exactly (regression gate).

### M10 Agentic decision support (SHIPPED 2026-07-08, commits e8feff3..b238f31; live transcript pending owner API credentials)
- An LLM agent exposes the what-if runner as a callable tool. Input: a natural
  language operational question. Output: a decision memo where every number
  cites a logged simulation run id. No number may originate from the LLM.
- Trigger chain demo: EWMA alert (M5) leads to proposed scenarios, CRN-paired
  results, memo.
- Guardrails: log every tool call and parameters; memos are reproducible;
  scope is decision support, no autonomous action. Do not claim NVIDIA
  tooling; the README frames this as a small-scale illustration of the
  simulation-in-the-loop pattern.

### M11 Data quality and model reliability (SHIPPED 2026-07-08, commits b6d3bbc..667e3da)
- Event schema contract and validators (missing timestamps, negative
  durations, duplicates, impossible routes); leakage-safe as-of joins for
  M7 labels; drift monitoring; conformal prediction intervals for M7/M8
  models; a short model card each (assumptions, limits, failure modes).

### HANDOVER: dashboard narrative refresh (superseded 2026-07-08)

This handover was completed by retiring the standalone dashboard page instead
of rewriting it. The owner decided the dashboard duplicated `docs/index.html`.
The evidence visuals were folded into the index page, and
`docs/dashboard.html` was removed in commit `7e06914`.

Current rule:
1. `docs/dashboard.html` is retired. Do not recreate it unless the owner
   explicitly asks.
2. `src/kpi/export_html_dashboard.py` now exports the baseline page only and
   removes the retired dashboard if present.
3. `docs/index.html` is now the primary public narrative page. Treat it as the
   owner-reviewed copy surface.

### HANDOVER: index page continuation for Claude Code (2026-07-08)

Goal: continue owner-directed refinement of `docs/index.html` on branch `v2`.
The owner is reviewing the public page section by section and wants concise,
reader-clear wording that avoids overclaiming.

Latest relevant commits on `v2`:
1. `7e06914` folded dashboard evidence into `docs/index.html`, retired
   `docs/dashboard.html`, added `src/kpi/export_index_assets.py`, and exported
   `docs/assets/finding03_output_cycle_time.png`.
2. `6048b8b` simplified the index narrative, removed long explanatory blocks,
   and replaced the three intro cards with Diagnose, Model, Decide.
3. `faf80a2` refined decision copy, removed the cost-method explanation from
   Finding 01, removed the Finding 02 consequence column, and simplified the
   agent section.
4. `50dac34` clarified that the real log is a public 4TU.ResearchData
   job-shop manufacturing production log, not private fab data.

Current index positioning:
1. The first intro card says the project diagnoses a public 4TU job-shop
   manufacturing production log.
2. The second intro card says the project builds a fixed-seed synthetic fab
   with batch FURNACE and re-entrant LITHO, then checks methods against known
   bottlenecks and injected anomalies.
3. The third intro card says the project turns capacity, maintenance,
   dispatching, and yield trade-off questions into CRN-paired what-if analysis
   flows. The owner is sensitive to wording that sounds like a full
   user-facing product. Use "prototype", "bounded options", or "analysis flow"
   if this topic returns.
4. Finding 01 title in Chinese is now "全方位分析，避免直覺誤判".
5. Finding 02 uses the original dashboard Finding 03 chart inside
   `docs/index.html`; its visual label uses "圖表觀察：" rather than a raw
   arrow.
6. Finding 03 title in Chinese is now "一個agent，回答四類 fab 營運問題".

Important owner preferences learned today:
1. Dashboard was removed because it repeated the index. Do not add a second
   roadmap-style page.
2. Concrete visuals must support the three core index highlights.
3. Every chart or visual block needs an immediate takeaway or observation.
4. Avoid phrases that make the project sound like private real-fab data,
   production optimizer, completed digital twin, or completed packaging model.
5. Public data source must be explicit when mentioning the real production log.
6. Costs are illustrative assumptions for ranking, never real-fab price or
   savings forecasts.
7. The owner may next ask for a lightweight scenario runner UI. If so, frame it
   as a bounded scenario runner demo, not a production decision platform.

Validation already performed during this Codex pass:
1. `py src/kpi/export_index_assets.py` regenerated the Finding 02 chart asset.
2. `py src/kpi/export_html_dashboard.py` removed the retired dashboard and
   regenerated the baseline page.
3. The fourteen validation scripts in `AGENTS.md` passed after installing the
   owner-approved M8 dependencies `scikit-learn` and `shap` into the Python 3.10
   environment used by `py`.
4. Later copy-only edits checked `docs/index.html` with `rg`, `git diff --check`,
   and the local preview URL.

Local preview:
`http://127.0.0.1:8787/` is the current local server for `docs/` if the process
is still running. GitHub Pages publishes from `main:/docs`, so the public URL
does not show `v2` changes until the owner merges.

Suggested Claude Code prompt:
Read `CLAUDE.md` in full, then read `V2-PLAN.md` section "HANDOVER: index page
continuation for Claude Code". Continue only with the newest owner request.
Before editing, explain the exact small changes you will make. Keep edits to
`docs/index.html` unless the owner explicitly asks for code or generated asset
changes.

### HANDOVER: scenario runner demo for Codex (2026-07-08)

Goal: continue owner-directed work on the public pages on branch `v2`. Claude
Code built the scenario runner demo the previous handover anticipated. This
section is the current state; the index-page handover above is still valid for
`docs/index.html` copy work.

What shipped (two commits on `v2`):
1. `663c0ff` added `docs/scenario-runner.html`, a bounded interactive demo over
   the precomputed what-if library, plus a primary entry button in the
   `docs/index.html` `.links` block (where the retired dashboard link used to
   be). Button copy: "Try the scenario runner demo" / "操作情境模擬 demo".
2. `c005ece` applied the owner's first review round (see locked rules below).

What the page is (locked framing, do not loosen without owner instruction):
1. It is a "Scenario Runner Demo" / "情境模擬 demo". NEVER call it a simulator,
   decision platform, or optimization tool. It runs nothing live: every number
   is read from a precomputed, fixed-seed, CRN-paired run and matches the
   index page exactly. No new Python code, no free-form parameter input.
2. Five tabs. The first four (capacity, maintenance, dispatch, yield) map to
   the agent's four what-if decision modules in `src/agent/tools.py`. The fifth
   (monitoring) is the detection layer that feeds those decisions, NOT a fifth
   agent tool. The hero states this 4+1 relationship in both languages.
3. Per-tab pattern: pick a scenario, see the KPI visual, then a decision-memo
   preview, then a structured analysis record. There are no separate takeaway
   lines (owner cut them as duplicating the memo). Memo sits ABOVE the record.
4. The decision memo is a fixed template written index-style: it compares the
   options and ends with objective-conditional guidance ("if the priority is
   X ... if the priority is Y ..."). It is explicitly labeled "no live LLM, no
   numbers outside the record". A real LLM memo wrapper stays in the local M10
   agent flow, never on GitHub Pages.
5. The analysis record is code-style structured fields: scenario id, engine
   (module path), run (n_reps, CRN), seeds, assumptions, KPI result. Seed
   ranges are real: capacity 1000-1029, maintenance 6000-6014, dispatch
   8000-8029, yield 5000-5029 (shared QualityConfig seed), monitoring config
   seed 42 with labeled onset day 30.
6. Maintenance and yield tabs are LITHO-only by design: only LITHO runs are
   published, and LITHO is the engineered bottleneck where each trade-off is
   sharpest. Each tab carries a scope note saying the engine accepts a station
   parameter but only the LITHO numbers are published. Do NOT fabricate other
   stations to fill a selector.
7. Bottom notes are titled "Scope" / "聲明" (owner changed "Honest scope" /
   "誠實聲明" to a plain statement).

Open owner question (do not act without instruction):
- Whether `docs/index.html` Finding 03 should gain a half-sentence tying the
  four agent decision modules to the Finding 02 monitoring detection layer
  (the 4+1 framing). Claude Code did not touch the index narrative for this;
  index copy is the owner's hand-tuned surface, surgical bilingual edits only.

Validation performed:
- Local preview at `http://127.0.0.1:8123/scenario-runner.html`
  (`.claude/launch.json` config `docs-static`, `py -m http.server 8123`).
- Programmatic checks across all five tabs: tab switching, single-select
  highlight, memo/record swap per selection, structured record fields, cost
  column always visible, memo-before-record order, scope notes present, chart
  asset loads, EN/ZH toggle. No console errors.
- No em dash in the file; `git diff --check` clean. No `src/` change, so the
  fourteen validation scripts were not required.

Suggested Codex prompt:
Read `CLAUDE.md` in full, then read `V2-PLAN.md` sections "HANDOVER: scenario
runner demo for Codex" and "HANDOVER: index page continuation". Continue only
with the newest owner request. Keep edits to `docs/` unless the owner asks for
code or generated-asset changes. Match the index visual system and the locked
demo framing; make surgical, bilingual (.en/.zh paired) edits.

### HANDOVER: two commissioned tasks for Codex (2026-07-08, owner-approved)

The owner commissioned both tasks below on 2026-07-08. Do Task A first: it
closes the last README claim without evidence. Task B is an invisible
refactor; the rendered page must not change. Read `CLAUDE.md` in full first
and obey its hard rules (no em dash, English only, fixed seeds, `py` not
`python` on this machine).

#### Task A: record the live agent session (M10 evidence)

GOAL: produce and commit the first VERIFIED live agent transcript under
`reports/agent_sessions/`, so the M10 claim ("a live-session runner records
transcripts as public evidence") is backed by an actual artifact. Everything
else about M10 is already shipped and gated offline with MockLLM.

SPEC:
- Precondition: the owner sets `OPENAI_API_KEY` in the environment for the
  session. Never echo, log, or commit the key; abort if it is absent (the
  runner exits 1 with a clear message). The runner defaults to OpenAI and
  supports `OPENAI_MODEL` if a different model is needed.
- Run exactly: `py scripts/run_live_agent_session.py` (default provider,
  default question, one session per invocation). Cost is a single short
  session.
- Exit 0 (VERIFIED): commit the produced
  `reports/agent_sessions/session_<UTC stamp>/` (transcript.json + memo.md).
  First confirm `.gitignore` does not swallow the path (only `data/**` and
  generated synthetic artifacts are ignored; `reports/figures` is tracked).
- Exit 2 (ran but FAILED verification): keep the artifacts locally, do NOT
  commit them as evidence, and report the `verification` block to the owner.
  One retry is allowed only for a clearly transient API failure. Never touch
  the verification logic in `src/agent/` to make a session pass.
- After a successful commit, update `README.md`: the Roadmap bullet about the
  recorded live agent session becomes a link to the committed transcript
  directory. Do not change the M10 table row (it becomes plainly true).
- Do NOT touch: `src/agent/**` (verification logic), locked design decisions,
  notebook 10.

ACCEPTANCE (all must hold):
- `transcript.json` has `"status": "VERIFIED"`, `verification.all_found`
  true, and an empty `uncited_numbers` list.
- `py src/agent/agent_check.py` and `py src/agent/loop_check.py` still exit 0.
- No credential material anywhere in the committed diff (search the staged
  diff for raw provider-token prefixes before committing; environment
  variable names are allowed).
- Committed as `feat: record first verified live agent session (M10)` and
  pushed to `v2`; V2-PLAN.md Log gains one entry marking open item (1)
  resolved.

REPORT: session status line, citation counts from the runner output, the
committed path, and one line "verified:" / one line "assumed:".

#### Task B: scenario runner data extraction (single source of truth)

GOAL: move the numbers hardcoded in `docs/scenario-runner.html` into a
generated `docs/assets/scenario_runner_data.json`, produced by a new exporter
`src/kpi/export_scenario_runner_data.py`, so page numbers can never silently
drift from engine outputs. This is a BEHAVIOR-PRESERVING change: the rendered
page must show byte-identical numbers and text before vs after.

SPEC:
- Exporter: derive every numeric field from the same module entry points the
  published numbers came from, with the published seeds: capacity
  `src/decision/whatif.py` (n_reps=30, seed0=1000; stress costs via
  `cost_model.py`, reuse `data/synthetic/findings_cache.json` /
  `src/kpi/precompute_findings.py` where the value is already cached),
  maintenance `src/equipment/maintenance_whatif.py` demo (n_reps=15,
  seed0=6000), dispatch `src/decision/dispatch_whatif.py` run_all (n_reps=30,
  seed0=8000), yield `src/decision/yield_whatif.py` demo_extra_litho_tool
  (n_reps=30, seed0=5000), monitoring from the findings cache (seed 42,
  onset day 30, alert day 84).
- Hard gate inside the exporter: assert each derived value matches the
  published value (2.46 / 0.70 / 0.26 / 0.05 h; +8.0k / -1.7k / +1.7k /
  +1.4k; 1.75M / 2.45M / 4.59M; EDD and release_control winners; -2.87 h /
  +0.038; day 84 of 160). If any value cannot be reproduced exactly at the
  published rounding, STOP and report; do not widen tolerances and do not
  ship the JSON.
- Page: load the JSON (same-origin fetch works on GitHub Pages and the local
  `docs-static` server; note in a code comment that file:// will not work)
  and fill the numeric/data fields from it. The bilingual memo and note prose
  STAYS in the HTML: it is the owner-reviewed copy surface. Structured record
  fields (engine, run, seeds, assumptions, KPI strings) may move to the JSON.
- Do NOT change any visible wording, layout, CSS, or the locked demo framing
  (see the scenario runner handover above). No new JS libraries.

ACCEPTANCE (all must hold):
- `py src/kpi/export_scenario_runner_data.py` exits 0 and regenerating twice
  produces an identical file (fixed seeds).
- With the local `docs-static` server, every number and label shown on all
  five tabs is identical to the pre-change page (record the pre-change values
  first, then compare).
- `git diff --check` clean; no em dash; no change outside
  `src/kpi/export_scenario_runner_data.py`, `docs/scenario-runner.html`,
  `docs/assets/scenario_runner_data.json`.
- Committed as `refactor: generate scenario runner data from engines (V2)`.

REPORT: before/after value comparison table, exporter runtime, file:line map
of the page changes, anything that smelled wrong but was left alone.

### Deferred (owner decision required)
- Stylized advanced-packaging (HBM-class) back-end line scenario: reuses the
  DES engine but changes the locked single-product line design. Do not start
  without explicit owner unlock.
- Scenario runner UI: SHIPPED 2026-07-08 as `docs/scenario-runner.html` (see
  the handover above). The JSON extraction follow-up is now commissioned as
  Task B in "two commissioned tasks for Codex".

## Numbering note

V1 historically used "M7" for the fab-ization iteration (see old commits).
From V2 onward M7 means the quality/yield module. AGENTS.md carries the same
note.

## Validation commands (after touching generator/monitoring code)

Use `py`, not `python`, on this machine:
`py src/generator/validate_m2.py`, `py src/generator/crn_check.py`,
`py src/monitoring/monitoring_check.py`.

## Log

- 2026-07-04: v1.0 tagged; `v2` branch opened; CLAUDE.md/AGENTS.md style rule
  and M8 exception added; README repositioned with roadmap contract;
  index.html hero and V2 teaser updated (EN/ZH). Next: M7.
- 2026-07-05: index.html rebuilt as one integrated pitch (no version labels,
  methodology section with verified references). PR #4 opened (v2 to main).
- 2026-07-05: M7 shipped. Stage A tool_id + opt-in chamber offsets
  (bit-identical defaults, CRN-safe post-draw scaling); Stage B src/quality
  yield layer (W = 0.4102 h at ~10% baseline violations, linear-additive
  ground truth, 5 gates); Stage C virtual metrology (test R^2 0.93, AUC 0.73,
  coefficients recovered), chamber matching (detects (1.05, 0.95), silent on
  clean logs), yield-aware CRN what-if with exact-zero pairing gate; Stage D
  notebook 07 (30 cells, clean execute) + five m7 figures. Headline: adding a
  LITHO tool cuts cycle time 2.87 h but raises post-litho violation rate by
  0.038, so capacity decisions need the yield axis. Next: M8 equipment health.
- 2026-07-06: PR #4 merged to main (2c73a5c); new index + M7 + M8 A/B live on
  GitHub Pages.
- 2026-07-06: M8 shipped. Stage A E10 state layer (exact per-tool timeline
  partition, MTBF/MTTR/availability, scheduled-PM injection reusing the
  breakdown capacity path); Stage B maintenance-timing what-if (immediate
  1.75M vs late 4.59M total cost, exact-zero pairing) and alert priority
  (score ordering validated against simulated cost impact); Stage C sensor
  simulation + GradientBoosting health model (held-out AUC 0.79; honest
  comparison: EWMA detects slightly faster, GB cuts false alarms 3.7x; SHAP
  recovers the two wired channels); Stage D notebook 08 + five m8 figures.
  Checks now number eight. Notebooks execute on the py310 kernel (Anaconda
  python3 kernel is broken and out of scope; owner defers repair).
  Next: M9 dispatching policies.
- 2026-07-07: M9 shipped. Stage A dispatch policies (fifo/edd/critical_ratio/
  queue_time_aware + CONWIP-style release control; FIFO byte-identity gate;
  bit-exact CRN-pairing proof across policies). Stage B paired comparison and
  decision table (n_reps=30, both regimes; winners: EDD for cycle time and
  on-time, release control for yield risk and total cost; all CIs exclude
  zero). Stage C notebook 09 + three m9 figures, centerpiece finding: the
  queue-time-aware rule is STRUCTURALLY identical to FIFO on this line (same
  window W per station, zero transport time, so least-slack order equals
  arrival order at any queue depth; proven under rho 0.90 stress, 0 of 1464
  dispatch rows differ). Checks now number ten. Next: M10 agentic decision
  support.
- 2026-07-08: M10 shipped. Stage A tool layer (five bounded, logged tools with
  deterministic run_ids; memo citation format [run:<id>]; tamper-detection
  gate). Stage B LLM agent loop (claude-opus-4-8 adapter, MockLLM driving the
  identical loop for five offline gates; two-layer verification: citations
  must resolve in the run log AND no substantive uncited number may appear;
  fabrication catch proven). Stage C notebook 10 + verification figure.
  Checks now number twelve. Live session runner
  (scripts/run_live_agent_session.py) is key-gated; recorded transcript will
  be committed under reports/agent_sessions/ when the owner provides
  OPENAI_API_KEY. Next: M11 data quality and model reliability.
- 2026-07-08: M11 shipped; THE ROADMAP MODULE SERIES (M7 to M11) IS COMPLETE.
  Stage A schema contract (C1-C6 clauses, seven corruption injectors mapped
  exactly to clauses, leakage-safe as-of join with audit). Stage B drift
  monitoring (degradation delay 4 days, arrival-shift delay 3 days, zero
  clean-run false alarms), split conformal for the VM model (coverage 0.9545
  at nominal 0.90; noise doubling doubles interval width), model cards with
  trust boundaries. Stage C notebook 11 + three m11 figures. Checks now number
  fourteen. Open items requiring owner decisions: (1) live agent transcript
  (needs OPENAI_API_KEY), (2) dashboard narrative refresh (owner copy
  review), (3) advanced-packaging back-end scenario (locked-design unlock).
- 2026-07-08: scenario runner demo shipped (`docs/scenario-runner.html`,
  commits 663c0ff and c005ece on `v2`). Bounded interactive view over the
  precomputed what-if library: five tabs (capacity, maintenance, dispatch,
  yield decision modules + monitoring detection layer), numbers matching the
  index page, per-tab decision-memo preview (fixed template, no live LLM) above
  a structured analysis record with real seed ranges, plus a primary entry
  button in the index links block. Owner review round one applied: cost column
  always shown, takeaway lines removed, memo above record, comparative
  index-style memos, LITHO-only scope notes on maintenance/yield, notes retitled
  to plain "Scope"/"聲明". No `src/` change. Open owner question: whether index
  Finding 03 should gain a half-sentence on the 4+1 (four decision modules plus
  monitoring detection layer) framing; index narrative left untouched pending
  owner instruction. Full handover in the "scenario runner demo for Codex"
  section above.
- 2026-07-08: pre-interview readiness pass. All fourteen validation scripts
  re-run and pass. README repository-structure section updated to the real
  tree (nine src modules, notebooks 01-11, scripts/, docs pages, agent-session
  landing path) and the stale dashboard export comment fixed. Owner
  commissioned two Codex tasks, spec'd in "two commissioned tasks for Codex":
  (A) record and commit the first verified live agent session (needs
  OPENAI_API_KEY from the owner), (B) extract scenario runner page data
  into a generated JSON with an exact-match gate against published values.
  Remaining owner decision for publishing: merge `v2` into `main` (47 commits
  ahead; GitHub Pages serves main, which still shows the V1 state).
