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

### M9 Dispatching policies (IN FLIGHT: Stage A shipped 56fe2c5; Stage B handover below)

**HANDOVER NOTE (2026-07-07, written when the Claude usage limit hit mid-Stage-B).**
State: `src/decision/dispatch_whatif.py` (341 lines) exists as a `wip:` commit
and is UNVERIFIED (its author agent died before writing the check script or
running any gate). Do not trust it until the gates below pass; review it
against this spec first, fix or rewrite as needed.

Remaining work for Stage B (spec the module was written against):
1. `src/decision/dispatch_whatif.py`: compare_policies(...) = CRN-paired
   comparison of EDD, critical_ratio, queue_time_aware, and FIFO+release_control
   against the FIFO baseline; same seed set per rep across configurations;
   metrics per rep: mean lot cycle time, output, on-time delivery rate,
   post-litho violation rate, mean lot yield, congestion + scrap cost; paired
   deltas vs FIFO with mean and t-based 95% CI; two demand regimes (baseline
   and arrival rate x1.15); tidy long-format output. decision_table(...):
   per regime x objective (cycle time / on-time / yield risk / total cost),
   best policy plus a caveat column populated from measured deltas, never
   hand-written. Reuse the pairing pattern from src/decision/yield_whatif.py
   and src/equipment/maintenance_whatif.py.
2. `src/decision/dispatch_whatif_check.py`, exit 0, house style:
   GATE 1 reproducibility; GATE 2 FIFO-vs-FIFO paired deltas exactly zero on
   every metric; GATE 3 directional sanity (EDD improves on-time rate,
   queue_time_aware reduces violation rate, baseline regime, seeded); GATE 4
   decision-table integrity (every cell populated; winner CI excludes zero or
   the cell is marked not significant). If a directional gate fails because
   the simulated system genuinely disagrees, report the measured direction
   with evidence instead of forcing the gate; that is an acceptable outcome
   requiring an owner decision.
3. Run ALL nine prior checks (validate_m2, crn_check, monitoring_check,
   quality_check, vm_check, equipment_check, maintenance_check, pdm_check,
   dispatch_check) plus the new one; use `py`, never bare `python`.
4. Commit as `feat: CRN-paired dispatching policy comparison and decision
   table (M9)`; push to origin/v2 (PR #6 tracks this branch).

Then Stage C: notebook 09 mirroring the notebook 07/08 pattern (build with
nbformat, execute with `--ExecutePreprocessor.kernel_name=py310`, figures
prefixed m9_ in reports/figures/, zero em dash, honest-scope framing), then
docs closure (README shipped row + roadmap removal + check command lists in
README/CLAUDE.md/AGENTS.md), fresh-eyes review, merge PR #6.
- Policies: FIFO (baseline), EDD, critical ratio, queue-time-aware,
  bottleneck-WIP control. CRN-paired runs on identical arrival streams.
- Output: decision table stating which policy wins under which demand and
  yield-risk conditions, with confidence intervals.
- Done when: notebook 09 produces the paired comparison; FIFO reproduces the
  V1 baseline exactly (regression gate).

### M10 Agentic decision support
- An LLM agent exposes the what-if runner as a callable tool. Input: a natural
  language operational question. Output: a decision memo where every number
  cites a logged simulation run id. No number may originate from the LLM.
- Trigger chain demo: EWMA alert (M5) leads to proposed scenarios, CRN-paired
  results, memo.
- Guardrails: log every tool call and parameters; memos are reproducible;
  scope is decision support, no autonomous action. Do not claim NVIDIA
  tooling; the README frames this as a small-scale illustration of the
  simulation-in-the-loop pattern.

### M11 Data quality and model reliability
- Event schema contract and validators (missing timestamps, negative
  durations, duplicates, impossible routes); leakage-safe as-of joins for
  M7 labels; drift monitoring; conformal prediction intervals for M7/M8
  models; a short model card each (assumptions, limits, failure modes).

### Deferred (owner decision required)
- Stylized advanced-packaging (HBM-class) back-end line scenario: reuses the
  DES engine but changes the locked single-product line design. Do not start
  without explicit owner unlock.
- Dashboard narrative refresh to match V2 positioning (index.html hero done;
  dashboard framing pending owner review of V2 copy).

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
