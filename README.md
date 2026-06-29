# Manufacturing Operations Analytics & Decision Support

Production data tells you what happened. This project turns it into what to do:
where the line is slow, which station is the constraint, when a KPI is drifting
out of control, and what a change in capacity or demand would do to output,
cycle time, and cost.

It combines the two things an operations team actually relies on.

**Diagnosis from real data.** Process mining on a real manufacturing production
log reconstructs the actual flow, measures where time is spent, and surfaces
rework loops and the steps that dominate cycle time.

**A decision sandbox.** A transparent discrete-event model of a re-entrant
production line (structure informed by the public SMT2020 semiconductor
manufacturing testbed) is used to monitor KPIs, detect anomalies early, and run
capacity / scenario what-ifs against a known ground truth so the methods can be
validated, not just demonstrated.

## What it does

- Visualizes line state over time: WIP, output, station utilization, cycle time.
- Identifies the bottleneck station and quantifies its cost in cycle time.
- Flags abnormal KPI shifts early (process drift, equipment downtime, demand surge).
- Answers "what if": add capacity at a station, or demand rises by X% — projected
  output, cycle time, and cost per unit.

## Approach & honest scope

The real production log is used for diagnosis; the discrete-event line is
synthetic and is **clearly labeled as synthetic** throughout. The synthetic line
exists to provide volume, a known ground-truth bottleneck, and labeled injected
anomalies, so that detection and monitoring methods can be measured objectively.
It is not intended to predict any specific real fab. Where AI assistance was used
for implementation, the modeling assumptions and method choices are documented so
they can be explained and challenged.

## Methods are deliberately interpretable

Bottleneck logic is utilization / Theory-of-Constraints based; KPI monitoring is
control-chart style. These are chosen over black-box models on purpose, because
operations decisions have to be explainable to and challengeable by the people
who act on them.

## Stack

Python (Jupyter), PM4Py (process mining), pandas / numpy, plotly + matplotlib,
Streamlit (dashboard).

## Repository structure

```
manufacturing-ops-analytics/
├── data/
│   ├── raw/          # real production log (downloaded locally, gitignored)
│   ├── synthetic/    # generated synthetic event log + ground-truth metadata
│   └── README.md     # data provenance + honest-scope note
├── src/
│   ├── generator/    # synthetic re-entrant line (discrete-event simulation)
│   ├── kpi/          # KPI computation
│   ├── bottleneck/   # bottleneck detection
│   ├── monitoring/   # anomaly monitoring
│   └── decision/     # scenario / what-if + cost
├── notebooks/        # one notebook per analysis stage
├── docs/             # glossary
└── app/              # Streamlit dashboard
```

## Reproducibility

All synthetic data is generated from a fixed random seed. Regenerate and
re-validate with:

```bash
python src/generator/validate_m2.py
```

This checks Little's Law self-consistency (WIP = throughput x cycle time) and
confirms the engineered bottleneck is recovered empirically.
