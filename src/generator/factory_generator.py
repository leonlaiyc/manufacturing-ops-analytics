"""
Synthetic fab-style production-line generator (Milestone M2).

A transparent, hand-written discrete-event simulation (DES) of an open
multi-server queueing network. No external DES library is used so that every
modeling step is explicit and defensible.

Design choices (each is a deliberate, defendable trade-off):

- Open queueing network with FIFO multi-server stations. FIFO is the simplest
  baseline dispatch rule. Real fabs use far more complex dispatching; that is a
  documented limitation / future work, not an oversight.

- Re-entrant route: a station can appear multiple times in the route. This is a
  defining feature of semiconductor fabs (e.g., litho/etch revisited) and is the
  structural idea borrowed from the public SMT2020 testbed. It is also what makes
  the designed bottleneck loaded.

- Lognormal processing times: positive and right-skewed, which matches real
  process-time behavior. The coefficient of variation (cv) controls variability;
  variability is what creates queueing (Kingman intuition), so WIP builds in
  front of the busiest station.

- Poisson arrivals (exponential inter-arrival times): the standard memoryless
  arrival assumption for an open line.

- A single station is engineered to have the least capacity headroom (highest
  utilization). By Theory of Constraints it sets line throughput, so it is the
  ground-truth bottleneck recorded in metadata and later used to validate the
  M4 detector.

The simulation is fully seeded for reproducibility.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StationConfig:
    """Configuration for one station (tool group)."""
    name: str
    pt_mean: float          # mean processing time per operation (hours)
    n_tools: int            # number of parallel tools (servers)
    pt_cv: float = 0.5      # coefficient of variation of processing time


@dataclass
class FactoryConfig:
    """Full factory / experiment configuration."""
    stations: dict          # name -> StationConfig
    route: list             # ordered station names; repeats = re-entrant flow
    arrival_rate: float     # lots per hour (Poisson process)
    horizon_hours: float    # total simulated time
    warmup_hours: float     # initial period excluded from steady-state stats
    seed: int = 42
    product_type: str = "P1"


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert a target mean and CV into lognormal (mu, sigma) parameters."""
    sigma2 = math.log(1.0 + cv ** 2)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, math.sqrt(sigma2)


def theoretical_utilization(cfg: FactoryConfig) -> dict:
    """
    Design-time utilization per station:
        rho_s = arrival_rate * visits_s * pt_mean_s / n_tools_s

    This is the planned load and identifies the intended bottleneck before any
    simulation runs. The DES should reproduce this ordering empirically.
    """
    visits = {s: cfg.route.count(s) for s in cfg.stations}
    rho = {}
    for s, st in cfg.stations.items():
        rho[s] = cfg.arrival_rate * visits[s] * st.pt_mean / st.n_tools
    return rho


def simulate(cfg: FactoryConfig):
    """
    Run the discrete-event simulation.

    Returns
    -------
    log : pd.DataFrame
        One row per completed operation:
        [lot_id, product_type, step_seq, station,
         queue_entry_time, process_start_time, process_complete_time]
    lifecycle : pd.DataFrame
        One row per lot: [lot_id, arrival_time, completion_time].
    meta : dict
        Configuration echo + ground-truth bottleneck.
    """
    rng = np.random.default_rng(cfg.seed)

    free = {s: st.n_tools for s, st in cfg.stations.items()}   # free tools per station
    pending = {s: [] for s in cfg.stations}                    # FIFO queues
    rows = []
    arrivals: dict[int, float] = {}
    completions: dict[int, float] = {}

    heap: list = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(heap, (t, seq, kind, payload))
        seq += 1

    def sample_pt(s):
        st = cfg.stations[s]
        mu, sigma = _lognormal_params(st.pt_mean, st.pt_cv)
        return float(rng.lognormal(mu, sigma))

    def request(lot, step, now):
        """Lot requests the station for this route step."""
        s = cfg.route[step]
        if free[s] > 0:
            free[s] -= 1
            pt = sample_pt(s)
            push(now + pt, "complete",
                 {"lot": lot, "step": step, "qentry": now, "start": now})
        else:
            pending[s].append({"lot": lot, "step": step, "qentry": now})

    # Schedule Poisson arrivals up front.
    t, lot_id = 0.0, 0
    while True:
        t += rng.exponential(1.0 / cfg.arrival_rate)
        if t >= cfg.horizon_hours:
            break
        push(t, "arrive", {"lot": lot_id})
        lot_id += 1

    # Event loop.
    while heap:
        now, _, kind, p = heapq.heappop(heap)

        if kind == "arrive":
            arrivals[p["lot"]] = now
            request(p["lot"], 0, now)
            continue

        # kind == "complete"
        s = cfg.route[p["step"]]
        rows.append({
            "lot_id": p["lot"],
            "product_type": cfg.product_type,
            "step_seq": p["step"],
            "station": s,
            "queue_entry_time": p["qentry"],
            "process_start_time": p["start"],
            "process_complete_time": now,
        })
        free[s] += 1

        # A tool just freed: pull the next waiting lot at this station (FIFO).
        if pending[s]:
            nxt = pending[s].pop(0)
            free[s] -= 1
            pt = sample_pt(s)
            push(now + pt, "complete",
                 {"lot": nxt["lot"], "step": nxt["step"],
                  "qentry": nxt["qentry"], "start": now})

        # Advance the completed lot to its next route step (or finish).
        nstep = p["step"] + 1
        if nstep < len(cfg.route):
            request(p["lot"], nstep, now)
        else:
            completions[p["lot"]] = now

    log = (pd.DataFrame(rows)
           .sort_values(["lot_id", "step_seq"])
           .reset_index(drop=True))

    lifecycle = pd.DataFrame({
        "lot_id": list(arrivals.keys()),
        "arrival_time": list(arrivals.values()),
    })
    lifecycle["completion_time"] = lifecycle["lot_id"].map(completions)

    rho = theoretical_utilization(cfg)
    bottleneck = max(rho, key=rho.get)
    meta = {
        "seed": cfg.seed,
        "arrival_rate": cfg.arrival_rate,
        "horizon_hours": cfg.horizon_hours,
        "warmup_hours": cfg.warmup_hours,
        "route": cfg.route,
        "theoretical_utilization": rho,
        "ground_truth_bottleneck": bottleneck,
    }
    return log, lifecycle, meta


def default_config(seed: int = 42) -> FactoryConfig:
    """
    The agreed M2 starting configuration:
      - 7 stations (S1..S7), single product
      - re-entrant route visiting S4 twice
      - S4 engineered as the bottleneck (highest planned utilization, ~0.85)
      - 60-day horizon (hours), 6-day warm-up
    """
    stations = {
        "S1": StationConfig("S1", pt_mean=1.0, n_tools=2),
        "S2": StationConfig("S2", pt_mean=1.2, n_tools=2),
        "S3": StationConfig("S3", pt_mean=1.3, n_tools=2),
        "S4": StationConfig("S4", pt_mean=0.85, n_tools=2),   # bottleneck
        "S5": StationConfig("S5", pt_mean=1.0, n_tools=2),
        "S6": StationConfig("S6", pt_mean=1.1, n_tools=2),
        "S7": StationConfig("S7", pt_mean=0.9, n_tools=2),
    }
    route = ["S1", "S2", "S3", "S4", "S5", "S4", "S6", "S7"]  # S4 re-entrant
    return FactoryConfig(
        stations=stations,
        route=route,
        arrival_rate=1.0,        # 1 lot/hour
        horizon_hours=60 * 24,   # 60 days
        warmup_hours=6 * 24,     # 6 days
        seed=seed,
    )
