"""
Dedicated M5 configuration (longer horizon).

M5 needs room for a clean baseline period, several spaced anomalies, and clean
recovery gaps between them (so the false-alarm rate is estimated on real clean
data and slow drift has room to develop). The M2-M4 ``default_config`` is left
untouched — its 60-day horizon and golden outputs must not change — so M5 builds
its own longer-horizon config by reusing the locked station/route definition.
"""

from __future__ import annotations

import copy

from factory_generator import default_config, FactoryConfig


def m5_config(horizon_days: int = 120, warmup_days: int = 6,
              seed: int = 42) -> FactoryConfig:
    """Return the locked 7-station S4-bottleneck config with a longer horizon.

    Reuses ``default_config`` (same stations, route, arrival rate, cv) and only
    stretches the horizon and sets the warm-up, so the line dynamics are exactly
    the validated M2 line — just observed for longer.
    """
    cfg = copy.deepcopy(default_config(seed=seed))
    cfg.horizon_hours = horizon_days * 24
    cfg.warmup_hours = warmup_days * 24
    return cfg
