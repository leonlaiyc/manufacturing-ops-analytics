# Manufacturing Operations Analytics & Decision Support

Production data tells you what happened. This project turns it into **what to
do**: which station is the real constraint (with proof), when a KPI is drifting
out of control (with measured detection quality), and what a capacity or demand
change would do to cycle time, throughput, and cost — before spending the money.

**[▶ Interactive KPI baseline](https://leonlaiyc.github.io/manufacturing-ops-analytics/)** ·
built on a validated discrete-event model of a fab-style line
(batch furnace, re-entrant litho bottleneck), plus process mining on a real
production log.

## Three findings that matter

**1. Local bottleneck signals need system proof.**
The slowest per-run station and raw local signals can point to the wrong
candidate. Slot utilization and paired what-if tests converge on LITHO as the
effective constraint. Adding one litho tool cuts mean cycle time by **2.46 h
(95% CI 2.13–2.79)**; adding one furnace tool adds four lot-processing slots but
buys only **0.70 h**, all of it batching-delay relief, and recovers only about
half its cost.

![Capacity what-if with cost](reports/figures/capacity_cost_tradeoff.png)

**2. A KPI improvement is not a business case.** The cost model is illustrative:
it ranks options under explicit assumptions, not real factory quotes or a
financial forecast. In the equal-tool-cost base case, LITHO wins both filters:
largest average cycle-time reduction and the only negative net cost (≈ −$12k).
In an investment-stress scenario with station-specific tool costs (LITHO $40k,
FURNACE $8k, DEPO $5k, METRO $2k), LITHO still has the largest operational
impact but becomes ≈ +$8.0k, while FURNACE becomes the best financial result
(≈ −$1.7k) with a smaller cycle-time gain. The right recommendation therefore
depends on the business objective: lead-time / WIP reduction, delivery
commitment, financial return, or a minimum improvement threshold
([notebook 06](notebooks/06_capacity_demand_cost_whatif.ipynb)).

**3. Slow equipment drift is expensive and catchable.** A gentle bottleneck
degradation (an OEE Performance loss) barely touches output at first, but it
raises cycle time and congestion cost. [Notebook 05](notebooks/05_kpi_anomaly_monitoring.ipynb)
validates the EWMA monitoring method with labeled anomalies and clean
pre-anomaly baselines. [Notebook 06](notebooks/06_capacity_demand_cost_whatif.ipynb)
uses a 160-day synthetic backtest: LITHO starts degrading on day 30, an
output-only throughput monitor gives no alert within the horizon, and EWMA on
daily median lot cycle time alerts on day 84. The same scenario translates alert
timing into avoidable cost: about $249k total extra cost in the reference run,
with about 95% still avoidable after the EWMA alert. This is a backtested
synthetic scenario, not a real-world guarantee that every drift will be caught
on the same day.

## What's inside

| Stage | What it does | Where |
|---|---|---|
| M1 Process mining | Reconstructs the actual flow of a **real** production log: DFG, cycle time, waiting-time decomposition, rework/variant analysis | [notebook 01](notebooks/01_process_mining_real_log.ipynb) |
| M2 Fab-style simulator | Transparent hand-built DES: batch furnace, re-entrant litho, engineered ground-truth bottleneck; validated via Little's Law (gap 0.2%) | [notebook 02](notebooks/02_synthetic_generator_demo.ipynb), `src/generator` |
| M3 KPI baseline | Shared baseline signals for the three findings: output, WIP, slot utilization, cycle time, **X-factor**; static + [interactive](https://leonlaiyc.github.io/manufacturing-ops-analytics/) | [notebook 03](notebooks/03_kpi_dashboard.ipynb) |
| M4 Bottleneck proof | Multi-evidence convergence + naive-baseline refutation + **CRN paired counterfactual**; then applied to the real log with stated limits | [notebook 04](notebooks/04_bottleneck_identification.ipynb) |
| M5 Anomaly monitoring | Injected, labeled OEE-style anomalies (Availability / Performance losses); control chart + EWMA scored on delay, FAR, precision/recall | [notebook 05](notebooks/05_kpi_anomaly_monitoring.ipynb) |
| M6 Decision support | Capacity / demand / degradation what-ifs with a transparent cost model; improvement ranking under ±50% sensitivity | [notebook 06](notebooks/06_capacity_demand_cost_whatif.ipynb) |

## The line

A stylized single-layer wafer-fab loop — the unit of flow is a lot
(25-wafer FOUP):

```
CLEAN → FURNACE → DEPO → LITHO → ETCH → LITHO → IMPLANT → METRO
         (batch:            ↑ engineered bottleneck,
       2 tools × 4 lots)      re-entrant, slot ρ ≈ 0.85
```

The two defining fab structures are both present: **re-entrance** (litho visited
per layer) and **batch processing** (furnace runs carry 4 lots — which is why
"slowest per operation" misidentifies it as the constraint, and why capacity is
measured in **slot utilization**, not busy time).

## Methods are deliberately interpretable

Bottleneck logic is queueing / Theory-of-Constraints evidence — no weighted
composite scores. Monitoring is control-chart / EWMA with baselines fit only on
clean pre-anomaly data. Cost is three transparent components used to **rank**
options under explicit assumptions, never to predict absolute dollars. The M6
stress scenario also tests station-specific added-tool costs, so operational and
financial rankings can point in different directions. When they do, the business
objective decides; sensitivity checks make the assumption boundary visible.
These choices are on purpose: operations decisions have to be explainable to —
and challengeable by — the people who act on them.

The same discipline applies to validation: methods are first proven on the
synthetic line, where the answer is known by construction (engineered
bottleneck, labeled anomaly windows, CRN-paired clean twins), and only then
applied to real data with the unprovable parts stated as limits.

## Stack

Python (Jupyter) · pandas / numpy / scipy · matplotlib + plotly ·
hand-built discrete-event simulation and process-mining logic (no black-box
dependencies — every number is traceable to a raw log row or a documented
model assumption).

## Repository structure

```
manufacturing-ops-analytics/
├── data/
│   ├── raw/          # real production log (downloaded locally, gitignored)
│   ├── synthetic/    # generated synthetic event log + ground-truth metadata
│   └── README.md     # data provenance + honest-scope note
├── src/
│   ├── generator/    # fab-style DES (batch tool, re-entrant route, CRN)
│   ├── kpi/          # KPI computation + interactive dashboard export
│   ├── bottleneck/   # evidence signals, naive baselines, CRN counterfactual
│   ├── monitoring/   # anomaly injection, detectors, detection-quality scoring
│   └── decision/     # cost model + capacity/demand/degradation what-ifs
├── notebooks/        # one notebook per analysis stage (01–06)
├── docs/             # glossary + GitHub Pages (interactive dashboard)
└── reports/          # exported figures + standalone HTML dashboard
```

## Reproducibility

All synthetic data is generated from fixed seeds. Regenerate and re-validate:

```bash
python src/generator/validate_m2.py      # Little's Law + bottleneck recovery
python src/generator/crn_check.py        # CRN determinism gates (exact zero delta)
python src/monitoring/monitoring_check.py # anomaly-injection regression gates
python src/kpi/export_html_dashboard.py  # rebuild the interactive dashboard
```

## Scope & honest notes

- The real production log (a job-shop machining process from the 4TU research
  repository) is used for **diagnosis only**; it has no tool counts or arrival
  model, so utilization, counterfactuals, and cost are computed only where they
  are defensible — and the notebooks say so explicitly where they are not.
- The synthetic line is **clearly labeled synthetic** throughout. It exists to
  give the methods a known ground truth (engineered bottleneck, labeled
  anomalies) so they can be **measured**, not just demonstrated. It is not a
  model of any specific factory.
- What is borrowed from the public **SMT2020** semiconductor testbed is the
  structural idea — re-entrant flow and batch tools — not its tool sets,
  routes, or process-time distributions.
- Cost rates and station-specific tool costs are illustrative; every cost
  conclusion is a **ranking** under explicit assumptions, tested for sensitivity,
  never a dollar forecast.
- Where AI assistance was used for implementation, the modeling assumptions and
  method choices are documented so they can be explained and challenged.
