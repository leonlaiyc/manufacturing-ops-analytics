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

- Use plan mode for any non-trivial milestone; present the plan before implementing.
- Keep modules small and documented so each method can be explained in review.
- For M4–M6, the method/approach is confirmed with the project owner before implementation.

## Version control

- Work on `main`. At the **end of each milestone**, commit and push.
- Commit messages: Conventional Commits, tagged with the milestone, e.g.
  `feat: KPI dashboard for WIP / cycle time / utilization (M3)`.
- Optionally mark each finished milestone with a lightweight git tag (`M3`, `M4`, ...).
- Generated artifacts (`data/synthetic/*`, `data/raw/*`) stay gitignored; the repo is
  reproducible by running the generator, so do not commit generated data.
