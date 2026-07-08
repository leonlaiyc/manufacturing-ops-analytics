# Decision Memo

## Question

Is one more litho tool worth it under 15 percent demand growth?

## Scenarios run

- `run_capacity_whatif` (station=LITHO, demand_factors=[1.15]) -> run `90894f5209e03906`

## Findings

| Tool | Metric | Value | Run |
|---|---|---|---|
| run_capacity_whatif | n_reps | 30.00 [run:90894f5209e03906] | `90894f5209e03906` |
| run_capacity_whatif | seed0 | 1000.00 [run:90894f5209e03906] | `90894f5209e03906` |
| run_capacity_whatif | 1.15.factor | 1.15 [run:90894f5209e03906] | `90894f5209e03906` |
| run_capacity_whatif | 1.15.mean_d_throughput | 0.01 [run:90894f5209e03906] | `90894f5209e03906` |
| run_capacity_whatif | 1.15.std_d_throughput | 0.01 [run:90894f5209e03906] | `90894f5209e03906` |
| run_capacity_whatif | 1.15.n | 30.00 [run:90894f5209e03906] | `90894f5209e03906` |

## Assumptions and caveats

- Rankings and deltas hold under the locked default configuration (7-station stylized wafer-fab loop, LITHO bottleneck) and CRN-paired replications (same random draw table for baseline and treatment); they are relative comparisons for decision support, not absolute forecasts.
- Cost figures use illustrative rates (see decision/cost_model.py, decision/yield_whatif.py) chosen to be defensible in review and re-ranked under a documented sensitivity sweep, never presented as real prices.
- All numbers in this memo come from a fixed-seed discrete-event simulation of a synthetic wafer fab. No real production data is used anywhere in this repository.

## Recommendation

No effect-sized field (d_* or mean_delta) was found among the supplied tool results; no automatic recommendation could be derived.

Adding one more litho tool under 15 percent demand growth increases throughput by only 0.00795 [run:90894f5209e03906] lots/hour on average. This very small gain implies that the capacity benefit of an extra litho tool at this demand level is negligible. I recommend not investing in an additional litho tool solely for this demand increase. If future evidence showed a substantially larger throughput gain or worsening cycle times without the extra tool, that would change the recommendation.
