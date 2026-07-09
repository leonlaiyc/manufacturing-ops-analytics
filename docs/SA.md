# System Analysis

## Purpose

This document states what the Manufacturing Operations Analytics project is
meant to solve, what it does not claim, and how the shipped modules are
accepted. It is the system analysis counterpart to `docs/SD.md`.

The project is a manufacturing operations analytics and decision-support
prototype. It combines process mining on a public real production log with a
fixed-seed synthetic fab-style line, then uses transparent methods to diagnose
constraints, monitor KPI drift, and compare bounded what-if decisions.

## Business Problem

Manufacturing operations teams need to decide where to focus improvement
effort. Local signals can mislead: a slow batch tool can look like the
bottleneck even when the re-entrant lithography step constrains system flow.
Output can also hide early equipment degradation while cycle time and WIP
already worsen.

The project addresses three decision questions:

1. Which station is the effective system constraint?
2. Is a KPI or equipment condition drifting in a way that needs action?
3. What happens to cycle time, delivery, yield risk, and cost if a bounded
   capacity, maintenance, dispatching, or quality scenario is selected?

## Stakeholders

| Stakeholder | Need | Project response |
|---|---|---|
| Operations analyst | Explain why a station is or is not the constraint | Multi-signal bottleneck evidence and CRN-paired counterfactuals |
| Fab or manufacturing manager | Compare options before spending money | Capacity, maintenance, dispatching, and yield what-if tables with 95% CIs |
| Process or industrial engineer | See whether methods respect factory physics | Re-entrant route, batch furnace, FIFO dispatch, slot utilization, Little's Law validation |
| Quality or equipment engineer | Connect KPI drift to yield and equipment state | OEE-style anomalies, SEMI E10-style states, VM, chamber matching, PdM scoring |
| Interview reviewer | Audit rigor and reproducibility | Fixed seeds, explicit synthetic labels, check scripts, model cards, and one-command validation |

## Scope

In scope:

- Process mining on a public real job-shop manufacturing event log.
- A synthetic, fixed-seed, fab-style discrete-event model with one product,
  seven fab-role stations, re-entrant LITHO, and batch FURNACE.
- KPI monitoring using interpretable control chart and EWMA methods.
- Bottleneck analysis using utilization, queue evidence, and CRN-paired
  capacity counterfactuals.
- Yield, equipment health, dispatching, data quality, reliability, and bounded
  agentic decision-support modules.
- GitHub-visible static figures and standalone HTML pages for public review.

Out of scope:

- A production digital twin with live MES synchronization.
- Real fab optimization or autonomous control.
- Real-fab cost, savings, or predictive-power claims.
- Free-form scenario simulation from the public page.
- Private semiconductor production data.
- A stylized advanced-packaging line until the owner explicitly changes the
  locked line design.

## Data Boundary

| Layer | Source | Use | Boundary |
|---|---|---|---|
| Real log | Public 4TU production event log | Process discovery, cycle time, waiting-time decomposition, variant and repeated-activity analysis | No tool counts, no arrival model, no defensible utilization or counterfactuals |
| Synthetic fab line | Hand-built generator in `src/generator` | Known-ground-truth validation, bottleneck proof, anomaly scoring, what-if decision support | Synthetic only, not a model of any named factory |
| Scenario runner data | Generated JSON under `docs/assets` | Bounded public demo | Reads precomputed fixed-seed scenarios, runs no live simulation |
| Agent sessions | Logged tool calls under `reports/agent_sessions` | Traceable decision memo evidence | Only verified sessions should be committed as evidence |

## Core Use Cases

### UC1: Diagnose the Actual Flow of a Real Production Log

Input: `data/raw/Production_Data.csv`, downloaded locally and not committed.

Output: process profile, directly-follows graph, cycle-time distribution,
waiting-time decomposition, and repeated-activity analysis.

Acceptance: notebook 01 executes, real-data caveats are explicit, and no
counterfactual or utilization claim is made for the real log.

### UC2: Validate the Synthetic Fab-Style Testbed

Input: generator configuration and fixed random seed.

Output: synthetic event log, lot lifecycle table, and metadata.

Acceptance:

- Little's Law gap is below the project threshold.
- Empirical bottleneck is LITHO.
- FURNACE slot utilization is measured as used lot-slots over available
  lot-slots, not raw busy time.
- CRN baseline-vs-baseline comparison is exactly zero.

### UC3: Prove the Bottleneck

Input: synthetic event log, station capacities, and CRN draw tables.

Output: evidence table, naive-baseline refutation, and paired capacity
counterfactuals.

Acceptance: independent evidence signals converge on LITHO and the LITHO
capacity counterfactual produces the strongest cycle-time relief.

### UC4: Monitor KPI Anomalies

Input: clean baseline and injected anomaly scenarios.

Output: control chart and EWMA detections with delay, false-alarm rate,
precision, and recall.

Acceptance: anomaly injection is causal, inactive anomalies have no effect, and
detected effects move in the expected direction.

### UC5: Compare Decision Scenarios

Input: bounded capacity, demand, maintenance, dispatching, or yield scenarios.

Output: scenario comparison tables, 95% CIs, cost rankings, and decision memos.

Acceptance: paired CRN comparisons are reproducible, baseline-vs-baseline
deltas are zero, and cost conclusions are framed as rankings under explicit
assumptions.

### UC6: Validate Data Quality and Reliability

Input: event logs, synthetic corruptions, model outputs, and metrology labels.

Output: schema-contract reports, as-of join audits, drift alarms, conformal
interval coverage, and model cards.

Acceptance: every schema clause has a corruption test, leakage is detected in a
deliberate bad join, drift and conformal gates pass, and model cards contain
trust boundaries.

### UC7: Produce Traceable Agentic Decision Memos

Input: bounded agent tools and a user question.

Output: a decision memo with citations to logged tool runs.

Acceptance: every cited number resolves to the run log, tampered citations are
detected, credentials are not required for offline gates, and live sessions are
committed only when verification status is `VERIFIED`.

## Functional Requirements

| ID | Requirement | Evidence |
|---|---|---|
| FR-01 | Ingest and analyze the public real production log without overstating its fields | Notebook 01 |
| FR-02 | Generate a fixed-seed synthetic fab-style event log | `src/generator/validate_m2.py` |
| FR-03 | Preserve the locked line design: CLEAN, FURNACE, DEPO, LITHO, ETCH, LITHO, IMPLANT, METRO | `src/generator/factory_generator.py` |
| FR-04 | Treat FURNACE as a batch tool and report slot utilization | `src/generator/validate_m2.py` |
| FR-05 | Identify LITHO as the engineered and empirical bottleneck | `src/generator/validate_m2.py`, notebook 04 |
| FR-06 | Support CRN-paired what-if comparisons | `src/generator/crn_check.py`, `src/decision/*_check.py` |
| FR-07 | Monitor labeled anomalies with interpretable detectors | `src/monitoring/monitoring_check.py` |
| FR-08 | Compare capacity, demand, maintenance, dispatching, and yield scenarios | notebooks 06 to 09, `src/decision`, `src/equipment`, `src/quality` |
| FR-09 | Log agent tool runs and verify memo citations | `src/agent/agent_check.py`, `src/agent/loop_check.py` |
| FR-10 | Validate data contracts, drift, intervals, and model cards | `src/dataquality/dq_check.py`, `src/dataquality/reliability_check.py` |
| FR-11 | Export public static and interactive artifacts | `src/kpi/export_*.py`, `docs/` |

## Nonfunctional Requirements

| ID | Requirement | Rationale | Evidence |
|---|---|---|---|
| NFR-01 | Reproducible fixed-seed generation | Reviewers must be able to re-run results | `py scripts/validate_all.py` |
| NFR-02 | Interpretability first | Operations decisions must be explainable | README, notebooks, model cards |
| NFR-03 | Honest scope | Synthetic results must not be presented as real fab results | README, `data/README.md`, public docs |
| NFR-04 | English documentation and code | Repository standard | AGENTS.md, CLAUDE.md |
| NFR-05 | No em dash in reader-facing text | Owner style rule | `rg -n "\x{2014}" README.md docs notebooks reports` |
| NFR-06 | GitHub-visible charts use static images | GitHub does not execute Plotly in notebooks | `reports/figures`, `docs/assets` |
| NFR-07 | Bounded public interaction | The public page must not imply live optimization | `docs/scenario-runner.html` |

## Acceptance Map

The one-command core validation entry point is:

```bash
py scripts/validate_all.py
```

This runs:

- Python compile check for `src`.
- M2 generator and CRN gates.
- M5 monitoring gates.
- M7 quality and virtual metrology gates.
- M8 equipment, maintenance, and PdM gates.
- M9 dispatching gates.
- M10 agent tool and loop gates.
- M11 data quality and reliability gates.
- Scenario runner, index asset, and baseline HTML exporters.

Optional full notebook execution:

```bash
py scripts/validate_all.py --with-notebooks
```

## Open Boundaries

The project is ready for review as a transparent analytics prototype. It should
not be described as a production system unless these boundaries change:

- Live MES or equipment integration.
- Real semiconductor-fab event logs.
- Real cost accounting.
- Deployment packaging, user authentication, and operational monitoring.
- Formal advanced-packaging line design.

## AI Assistance Note

AI assistance was used during implementation and documentation work. The
project keeps modeling assumptions, validation gates, and generated synthetic
scope explicit so reviewers can audit the logic rather than trust the assistant.
