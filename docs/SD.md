# System Design

## Purpose

This document describes how the Manufacturing Operations Analytics project is
structured. It is the system design counterpart to `docs/SA.md`.

The design principle is transparency: each module has a small responsibility,
each method can be explained from first principles, and every major claim has a
validation script or generated artifact.

## Architecture Overview

```text
real production log
        |
        v
notebook 01 process mining

synthetic fab generator
        |
        +--> event_log.csv / lot_lifecycle.csv / metadata.json
        |
        +--> KPI baseline and public figures
        |
        +--> bottleneck, monitoring, quality, equipment, dispatching,
             decision support, agent tools, and data-quality checks
```

The project has two analysis tracks:

1. Real-log diagnosis, limited to what the public log can support.
2. Synthetic fab decision support, where known ground truth enables validation.

## Module Map

| Module | Responsibility | Main checks |
|---|---|---|
| `src/generator` | Hand-built discrete-event generator, CRN draw tables, dispatch policies | `validate_m2.py`, `crn_check.py`, `dispatch_check.py` |
| `src/kpi` | KPI metrics and public artifact exporters | `export_*.py` |
| `src/bottleneck` | Multi-signal bottleneck evidence and counterfactual plots | notebook 04 |
| `src/monitoring` | Anomaly injection, KPI series, control chart, EWMA, detection scoring | `monitoring_check.py` |
| `src/decision` | Cost model, capacity and demand what-if, dispatch comparison, yield what-if | `dispatch_whatif_check.py` |
| `src/quality` | Queue-time windows, synthetic yield model, virtual metrology, chamber matching | `quality_check.py`, `vm_check.py` |
| `src/equipment` | SEMI E10-style states, RAM metrics, maintenance timing, alert priority, PdM model | `equipment_check.py`, `maintenance_check.py`, `pdm_check.py` |
| `src/agent` | Bounded logged tools, memo citation verification, offline and live agent loop | `agent_check.py`, `loop_check.py` |
| `src/dataquality` | Event-log schema contract, corruptions, leakage-safe joins, drift, conformal intervals | `dq_check.py`, `reliability_check.py` |
| `scripts` | Reviewer and operator entry points | `validate_all.py`, `run_live_agent_session.py` |
| `docs` | Public pages, design docs, glossary, model cards | manual review plus exporters |

## Data Products

| Artifact | Producer | Consumer | Git status |
|---|---|---|---|
| `data/raw/Production_Data.csv` | Reviewer downloads locally | Notebook 01 | gitignored |
| `data/synthetic/event_log.csv` | `src/generator/validate_m2.py` | KPI, bottleneck, monitoring, quality, equipment, decision modules | gitignored |
| `data/synthetic/lot_lifecycle.csv` | `src/generator/validate_m2.py` | KPI and cycle-time modules | gitignored |
| `data/synthetic/metadata.json` | `src/generator/validate_m2.py` | KPI and validation modules | gitignored |
| `data/synthetic/findings_cache.json` | `src/kpi/precompute_findings.py` | index asset exporter and scenario runner exporter | gitignored |
| `reports/figures/*.png` | notebooks and exporters | README and GitHub Pages | tracked |
| `docs/assets/*.json` | `src/kpi/export_scenario_runner_data.py` | scenario runner page | tracked |
| `docs/assets/*.png` | `src/kpi/export_index_assets.py` | index page | tracked |
| `docs/baseline.html` | `src/kpi/export_html_dashboard.py` | GitHub Pages | tracked |
| `reports/html/03_kpi_dashboard.html` | `src/kpi/export_html_dashboard.py` | report copy of M3 artifact | tracked |

## Event-Log Schema

The synthetic event log has one row per operation. Batch members each receive
one row, which preserves lot-level accounting while allowing batch-tool slot
utilization.

Required columns:

| Column | Meaning |
|---|---|
| `lot_id` | Lot identifier |
| `product_type` | Product family, locked to a single product in the current design |
| `step_seq` | Route step sequence |
| `station` | Station name |
| `queue_entry_time` | Time when the lot joins the station queue |
| `process_start_time` | Time when service starts |
| `process_complete_time` | Time when service completes |

The data-quality module validates timestamp types, nonnegative durations,
route consistency, duplicates, and tool attribution where applicable.

## Synthetic Generator Design

The generator is a hand-built discrete-event simulation. It does not use a DES
library so the queueing and dispatch logic remain auditable.

Locked line design:

```text
CLEAN -> FURNACE -> DEPO -> LITHO -> ETCH -> LITHO -> IMPLANT -> METRO
```

Key choices:

- One product type.
- One lot equals one 25-wafer FOUP.
- Poisson lot arrivals.
- Lognormal processing times.
- FIFO dispatch by default.
- FURNACE is a batch tool with 2 tools and 4 lot-slots per run.
- LITHO is visited twice and is engineered as the bottleneck.
- FURNACE capacity is measured by slot utilization, not raw busy time.
- CRN draw tables pair baseline and scenario runs.

## KPI and Bottleneck Design

KPI metrics are computed from the event log and lot lifecycle:

- Throughput.
- WIP.
- Slot utilization.
- Cycle time.
- X-factor.

Bottleneck evidence uses several independent signals rather than a weighted
score:

- Utilization.
- Average queue length.
- Average wait before station.
- Share of plant-wide waiting.
- Idle fraction as a secondary starvation proxy.

The system claim is not based only on descriptive signals. It is checked with
CRN-paired capacity counterfactuals, where adding one tool at LITHO should
produce the strongest system-level cycle-time relief.

## Monitoring Design

M5 injects labeled OEE-style anomalies into the synthetic line:

- Breakdown, mapped to Availability loss.
- Degradation, mapped to Performance loss.
- Demand surge, used to test load-shift behavior.

Detection uses interpretable methods:

- Control chart.
- EWMA.

The monitoring check enforces identity for clean and inactive cases, causality
before anomaly start, and expected effect direction after injection.

## Quality and Equipment Design

Quality modules add a known-ground-truth yield layer:

- Queue-time window risk after LITHO.
- Transparent lot-level yield model.
- Virtual metrology using hand-built OLS.
- Chamber matching to detect tool offsets.
- Yield-aware CRN what-if comparison.

Equipment modules add a tool-state layer:

- SEMI E10-style state partitioning.
- MTBF, MTTR, availability, and utilization.
- Preventive-maintenance timing comparison.
- Alert priority checked against simulated cost impact.
- Gradient boosting plus SHAP only for the owner-approved M8 exception:
  detection-quality measurement on synthetic ground truth.

## Decision and Agent Design

Decision modules compare bounded scenarios with paired CRN replications. The
core outputs are effect sizes, confidence intervals, caveats, and cost rankings
under explicit assumptions.

The agent layer does not run free-form code. It exposes bounded tools that:

- Validate parameters.
- Run existing what-if engines.
- Record tool inputs, outputs, seeds, and run IDs.
- Generate memos where every cited number can be checked against the run log.
- Reject fabricated or tampered numbers during verification.

The public scenario runner is separate from the agent loop. It reads
precomputed JSON and runs no live simulation or LLM call.

## Validation Design

Validation is organized as executable gates rather than only narrative claims.

Core entry point:

```bash
py scripts/validate_all.py
```

Flow:

1. Compile `src` Python files.
2. Run generator and CRN gates.
3. Run monitoring gates.
4. Run quality and virtual-metrology gates.
5. Run equipment, maintenance, and PdM gates.
6. Run dispatching and decision gates.
7. Run agent traceability gates.
8. Run data-quality and reliability gates.
9. Regenerate public scenario, index, and baseline artifacts.

Optional full notebook execution:

```bash
py scripts/validate_all.py --with-notebooks
```

The notebook option is slower and is meant for release or handoff review.

## Public Artifact Design

GitHub-visible charts are exported as static PNG files because GitHub does not
execute notebook JavaScript. Interactive artifacts are standalone HTML pages
under `docs`.

Published pages:

- `docs/index.html`: primary narrative page.
- `docs/baseline.html`: interactive M3 KPI baseline.
- `docs/scenario-runner.html`: bounded precomputed scenario runner demo.

Exporter design:

- `src/kpi/export_index_assets.py` renders static assets for the index page.
- `src/kpi/export_html_dashboard.py` renders the baseline page only and removes
  the retired `docs/dashboard.html` if present.
- `src/kpi/export_scenario_runner_data.py` produces the JSON consumed by the
  scenario runner.

## Failure Handling

The project favors explicit failure over silent acceptance:

- Validation scripts exit nonzero on gate failure.
- Agent tools reject out-of-bounds parameters.
- Baseline-vs-baseline CRN deltas must be exact zero.
- Data-quality corruptions must trigger the expected contract clause.
- Live agent sessions that fail citation verification must not be committed as
  evidence.

## Extension Points

Potential future changes should preserve the locked design unless the owner
approves a scope change.

Safe extensions:

- Add new bounded scenarios to the scenario runner data exporter.
- Add new model-card sections when new reliability gates are introduced.
- Add public copy that clarifies scope without adding claims.

Owner-approval extensions:

- Change the line design or route.
- Add an advanced-packaging scenario.
- Add real MES integration.
- Replace transparent methods with black-box models outside the M8 exception.
- Expose free-form simulation controls in public pages.

## Traceability

| Design claim | Verification path |
|---|---|
| Generator is stable and LITHO is the empirical bottleneck | `py src/generator/validate_m2.py` |
| CRN comparisons are paired correctly | `py src/generator/crn_check.py` |
| Monitoring injection is causal | `py src/monitoring/monitoring_check.py` |
| Quality and VM methods recover known synthetic effects | `py src/quality/quality_check.py`, `py src/quality/vm_check.py` |
| Equipment state and PM comparisons are reproducible | `py src/equipment/equipment_check.py`, `py src/equipment/maintenance_check.py` |
| M8 PdM is scored only against synthetic ground truth | `py src/equipment/pdm_check.py`, `docs/model_cards/pdm_health_model.md` |
| Dispatch policies are CRN-paired and objective-dependent | `py src/generator/dispatch_check.py`, `py src/decision/dispatch_whatif_check.py` |
| Agent memos are traceable | `py src/agent/agent_check.py`, `py src/agent/loop_check.py` |
| Data quality and reliability gates are executable | `py src/dataquality/dq_check.py`, `py src/dataquality/reliability_check.py` |
| Public artifacts rebuild | `py scripts/validate_all.py` |
