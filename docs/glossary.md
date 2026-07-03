# Operations Glossary

Manufacturing operations vocabulary used across this project. Each term is
defined in one sentence.

## Flow & capacity

- **WIP (Work In Process)**: units released into the line but not yet completed.
- **Throughput**: completed units per unit time (the line's output rate).
- **Cycle time**: time a unit spends from release to completion (queue + process).
- **Process time**: active time a unit is being worked at a station.
- **Queue time**: time a unit waits before a station; usually the dominant part of cycle time.
- **Lead time**: time from order to delivery (broader than cycle time).
- **Takt time**: available time / customer demand; the pace the line must hit.
- **Capacity**: maximum sustainable output of a station or line (tools × rate × availability).
- **Utilization**: fraction of available capacity actually used at a station.
- **Slot utilization**: for a batch tool, work arriving per hour ÷ lot-slots servable per hour (busy time ÷ n_tools × batch_size × window); the fab-standard capacity view — raw busy-time overstates a batch tool's load.
- **X-factor**: cycle time ÷ raw process time; the headline fab flow metric — X = 1 means zero queueing, and the excess above 1 is time spent waiting.
- **Re-entrant flow**: a route that visits the same station multiple times — characteristic of semiconductor fabs.
- **Batch tool**: a tool (e.g., diffusion furnace) that processes several lots in one run; slow per run but high per-slot capacity — the classic trap for per-operation bottleneck heuristics.
- **Lot / FOUP**: a production batch unit tracked through the line; in a fab, typically a carrier holding 25 wafers.

## Bottleneck & improvement

- **Bottleneck**: the resource with the least effective capacity relative to demand; it sets line throughput.
- **Theory of Constraints (TOC)**: improve the system by managing the bottleneck; non-bottleneck improvements rarely raise throughput.
- **Little's Law**: WIP = throughput × cycle time. Used here to validate the synthetic generator.
- **Dispatch rule**: the policy deciding which waiting lot a station processes next (e.g., FIFO).
- **OEE (Overall Equipment Effectiveness)**: availability × performance × quality; a breakdown is an Availability loss, a slow-running tool a Performance loss — the two equipment anomalies injected and monitored in M5.
- **Yield**: fraction of units passing without scrap/rework.
- **Rework loop**: a unit returning to an earlier step after a failure.
- **Illustrative cost model**: a simplified cost framework used to compare scenarios under shared assumptions; it is not a real factory quote or financial forecast.
- **Holding / waiting cost**: an illustrative cost assigned to lots waiting in queue, used to represent WIP burden, longer lead time, and congestion risk.
- **Station-specific tool cost**: an illustrative assumption that the cost of adding one tool can differ by station.
- **Investment-stress scenario**: a what-if scenario that changes investment-cost assumptions to test whether the recommended option remains attractive.
- **Break-even added-tool cost**: the maximum added-tool cost at which a scenario's net cost change is zero.

## Monitoring

- **Control chart**: flags points outside mean ± k·sigma as out-of-control.
- **EWMA**: a smoothing-based monitor that accumulates small persistent shifts over time, useful for slow drift.
- **Output-only monitor**: a baseline monitor that watches daily output / throughput and alerts only when output drops outside the normal range.
- **Lead time of detection**: how early a monitor flags an injected anomaly before its full impact.
- **Clean baseline**: a baseline fitted only on known clean pre-anomaly data, without using future anomaly periods.
- **Clean twin**: a comparable no-degradation simulation run used as the counterfactual reference.
- **Avoidable cost**: the future extra degradation cost that has not yet accumulated at the alert day, assuming action is taken after detection.
- **Degradation cost**: the extra cost gap between a degraded line and its clean twin.
- **Backtest horizon**: the time window over which a detection and cost scenario is evaluated.

## Process mining

- **Event log**: records of (case, activity, timestamp) — the input to process mining.
- **Directly-follows graph (DFG)**: a map of which activities follow which, with frequencies/durations.
- **Conformance**: how far the observed flow deviates from the intended process.

## Systems

- **MES (Manufacturing Execution System)**: tracks production execution on the floor; a source of event logs.
- **SAP**: enterprise system often holding orders, materials, and cost data.
- **Scenario / capacity planning**: estimating output, cycle time, and cost under hypothetical demand or capacity changes.
