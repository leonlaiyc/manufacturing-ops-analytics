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
- **Re-entrant flow**: a route that visits the same station multiple times — characteristic of semiconductor fabs.

## Bottleneck & improvement

- **Bottleneck**: the resource with the least effective capacity relative to demand; it sets line throughput.
- **Theory of Constraints (TOC)**: improve the system by managing the bottleneck; non-bottleneck improvements rarely raise throughput.
- **Little's Law**: WIP = throughput × cycle time. Used here to validate the synthetic generator.
- **Dispatch rule**: the policy deciding which waiting lot a station processes next (e.g., FIFO).
- **OEE (Overall Equipment Effectiveness)**: availability × performance × quality.
- **Yield**: fraction of units passing without scrap/rework.
- **Rework loop**: a unit returning to an earlier step after a failure.

## Monitoring

- **Control chart**: flags points outside mean ± k·sigma as out-of-control.
- **EWMA**: exponentially weighted moving average; reacts to small sustained shifts faster than a simple mean.
- **Lead time of detection**: how early a monitor flags an injected anomaly before its full impact.

## Process mining

- **Event log**: records of (case, activity, timestamp) — the input to process mining.
- **Directly-follows graph (DFG)**: a map of which activities follow which, with frequencies/durations.
- **Conformance**: how far the observed flow deviates from the intended process.

## Systems

- **MES (Manufacturing Execution System)**: tracks production execution on the floor; a source of event logs.
- **SAP**: enterprise system often holding orders, materials, and cost data.
- **Scenario / capacity planning**: estimating output, cycle time, and cost under hypothetical demand or capacity changes.
