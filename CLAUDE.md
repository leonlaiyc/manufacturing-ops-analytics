# CLAUDE.md — Manufacturing Operations Analytics

Persistent context for this repository. See `README.md` for the full description.

## What this project is

A manufacturing operations analytics and decision-support project: real
production-flow diagnosis via process mining, plus a transparent discrete-event
model of a re-entrant production line for KPI monitoring, bottleneck analysis,
anomaly detection, and capacity / scenario planning.

## Engineering principles (do not violate)

- **Language:** all code, comments, notebooks, README, and commit messages in English.
- **Honest scope:** synthetic data must be clearly labeled as synthetic and never
  presented as real. Note where AI assistance was used. Do not overstate results.
- **Interpretability first:** prefer simple, explainable methods (utilization /
  Theory of Constraints for bottlenecks; control-chart / EWMA for monitoring) over
  black-box models. Every method must be explainable from first principles. If a
  technique cannot be explained simply, choose a simpler one.
- **Reproducibility:** all synthetic generation uses a fixed seed.

## Locked design decisions (do not change without explicit instruction)

- Generator: 7 stations (S1–S7), single product, re-entrant route
  `S1 S2 S3 S4 S5 S4 S6 S7` (S4 visited twice). S4 is engineered as the bottleneck
  (highest planned utilization ≈ 0.85). Lognormal processing times, FIFO dispatch,
  Poisson arrivals. Validated: Little's Law gap < 1%; empirical bottleneck = S4.
- Event-log schema (one row per operation):
  `lot_id, product_type, step_seq, station, queue_entry_time, process_start_time, process_complete_time`.

## Stack

Python (Jupyter), PM4Py (process mining), pandas / numpy, plotly + matplotlib, Streamlit.

## Status

- M0 scaffold: done.
- M2 synthetic generator: done. Code in `src/generator/`; data in `data/synthetic/`.
  Regenerate and re-validate with `python src/generator/validate_m2.py`.

## Next milestones

- **M1** process mining on the real production log (user provides it under `data/raw/`).
  Reconstruct the flow, per-step durations, rework loops, slowest transitions.
- **M3** KPI dashboard (WIP / output / utilization / cycle time over time) from `data/synthetic/`.
- **M4** bottleneck detection, validated against the synthetic ground truth, then applied to the real log.
- **M5** anomaly monitoring: inject labeled events, measure detection rate and lead time.
- **M6** decision support: capacity / scenario what-ifs and cost-per-unit impact.

## Working agreement

- Claude Code's role is **execution only**. Planning, scope, method selection, and
  interpretation are decided by the project owner and handed over as concrete
  instructions. Do not independently choose methods, redefine scope, or add analysis
  beyond what is asked.
- If something looks missing, wrong, or ambiguous, surface it as a question — do not
  self-direct or guess.
- Use plan mode to show the concrete steps of a given task before running, for approval.
- Keep modules small and documented so each method can be explained in review.

## Version control

- Commit at **every small completed step**, not only at milestone boundaries, so there is
  an auditable trail to follow.
- Push after each commit once the remote is set.
- Conventional Commits, milestone-tagged where applicable, e.g.
  `feat: process mining on real production log (M1)`.
- Generated artifacts (`data/synthetic/*`, `data/raw/*`) stay gitignored.
