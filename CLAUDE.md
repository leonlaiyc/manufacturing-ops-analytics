# CLAUDE.md — Manufacturing Operations Analytics

Persistent context for this repository. Full project description: `README.md`.
Project status: see the README status table and `git log` — status is NOT
tracked in this file.

## Operating standards (cross-project) — routing

Shared operating rules live in `F:\ai-dev-standards\`. Read the file whose
trigger applies; do not paraphrase from memory:

| Trigger | Read |
|---|---|
| Task needs >3 files opened, repo-wide search, web research, or batch edits | `F:\ai-dev-standards\10-model-dispatch.md` |
| About to write any subagent prompt | `F:\ai-dev-standards\30-delegation-templates.md` |
| About to declare a task done, or verifying work | `F:\ai-dev-standards\20-judgment-rubrics.md` |
| Same command failed twice; context filling fast; unsure whether to ask user | `F:\ai-dev-standards\00-diagnosis.md` and `20-judgment-rubrics.md` |
| Editing CLAUDE.md, AGENTS.md, or anything in ai-dev-standards | `F:\ai-dev-standards\40-maintenance-protocol.md` |

## Hard rules for this repo

- **NEVER `Read` these directly** (up to 1.3 MB — one Read can consume the
  whole context): `notebooks/*.ipynb`, `docs/*.html`, `data/**`. To see
  notebook code: `py -m jupyter nbconvert --to script <nb> --stdout`. To answer a
  question about them: Grep with `head_limit`, or delegate to an Explore
  subagent. Details: `F:\ai-dev-standards\00-diagnosis.md` Leak #1.
- **Language:** all code, comments, notebooks, README, and commit messages in
  English — even when the user writes to you in Chinese.
- **Honest scope:** synthetic data is always labeled synthetic, never presented
  as real. Note where AI assistance was used. Do not overstate results.
- **Interpretability first:** simple, explainable methods only (utilization /
  Theory of Constraints for bottlenecks; control-chart / EWMA for monitoring).
  If a technique cannot be explained from first principles, choose a simpler one.
  Owner-approved exception (2026-07-04): gradient boosting + SHAP is allowed for
  the M8 equipment-health module, framed as measuring detection quality against
  known synthetic ground truth, never as claimed predictive power.
- **Style:** never use the em dash character (—) in reader-facing text: README,
  docs/, notebooks, dashboard copy, reports, commit messages. Use a comma,
  colon, period, or parentheses instead. (Owner instruction, 2026-07-04.)
- **Reproducibility:** all synthetic generation uses a fixed seed.
- **Charts on GitHub = matplotlib static images** (GitHub does not run Plotly's
  JavaScript). Interactive versions are exported separately as standalone HTML.
- `AGENTS.md` (for Codex) mirrors this file's principles/decisions sections.
  If you edit shared content here, apply the same edit there (see maintenance
  protocol).

## Locked design decisions (do not change without explicit instruction)

- Generator: 7 fab-role stations, single product, stylized wafer-fab loop
  `CLEAN FURNACE DEPO LITHO ETCH LITHO IMPLANT METRO` (LITHO visited twice,
  re-entrant). 1 lot = one 25-wafer carrier (FOUP).
  - LITHO is engineered as the bottleneck (highest slot utilization ≈ 0.85).
  - FURNACE is a batch tool: 2 tools × 4-lot greedy batches, low cv (recipe-like);
    run time = first-loaded lot's draw; slot ρ ≈ 0.375 (deliberately NOT the
    constraint). Batch capacity is always measured as **slot utilization**
    (busy / (n_tools × batch_size × window)), never raw busy time.
  - Lognormal processing times, FIFO dispatch, Poisson arrivals.
  - Validated: Little's Law gap < 1%; empirical bottleneck = LITHO; CRN
    baseline-vs-baseline exact zero (`crn_check.py`); injection gates
    (`monitoring_check.py`).
- Event-log schema (one row per operation; batch members one row each):
  `lot_id, product_type, step_seq, station, queue_entry_time, process_start_time, process_complete_time`.
- Fab KPI conventions: X-factor = cycle time / raw process time; M5 anomalies map
  to OEE loss categories (breakdown = Availability loss, degradation = Performance
  loss).

## Stack

Python (Jupyter), pandas / numpy / scipy, plotly + matplotlib. Process-mining
and DES logic are hand-built (transparency by design — no PM4Py, no DES
library, no Streamlit). The public narrative is `docs/index.html`;
`src/kpi/export_html_dashboard.py` now exports the baseline Plotly HTML page
only and removes the retired `docs/dashboard.html` page if present.

Validation commands (run after touching generator/monitoring/quality/equipment
code; use `py`, not `python` — bare `python` is the broken Windows Store stub
on this machine): `py src/generator/validate_m2.py` ·
`py src/generator/crn_check.py` · `py src/monitoring/monitoring_check.py` ·
`py src/quality/quality_check.py` · `py src/quality/vm_check.py` ·
`py src/equipment/equipment_check.py` · `py src/equipment/maintenance_check.py` ·
`py src/equipment/pdm_check.py` · `py src/generator/dispatch_check.py` ·
`py src/decision/dispatch_whatif_check.py` · `py src/agent/agent_check.py` ·
`py src/agent/loop_check.py` · `py src/dataquality/dq_check.py` ·
`py src/dataquality/reliability_check.py`

M8 stack exception (owner-approved 2026-07-04): scikit-learn (gradient
boosting) + shap are allowed in `src/equipment/` only, always framed as
detection-quality measurement on known synthetic ground truth. Notebooks
execute on the registered `py310` Jupyter kernel
(`--ExecutePreprocessor.kernel_name=py310`); never use the Anaconda-backed
`python3` kernel and never install into `C:\Users\User\anaconda3`.

## Working agreement

- Claude Code's role is **execution only**. Planning, scope, method selection,
  and interpretation are decided by the project owner and handed over as
  concrete instructions. Do not independently choose methods, redefine scope,
  or add analysis beyond what is asked.
- If something looks missing, wrong, or ambiguous, surface it as a question —
  do not self-direct or guess.
- Use plan mode to show the concrete steps of a given task before running, for
  approval.
- Keep modules small and documented so each method can be explained in review.

## Version control

- Commit at **every small completed step**; push after each commit once the
  remote is set. Conventional Commits, milestone-tagged where applicable,
  e.g. `feat: process mining on real production log (M1)`.
- Generated artifacts (`data/synthetic/*`, `data/raw/*`) stay gitignored.
