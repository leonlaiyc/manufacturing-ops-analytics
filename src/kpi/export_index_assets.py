"""Export static assets used by the GitHub Pages index page.

The index page must show charts directly on GitHub Pages, so these assets use
matplotlib static images rather than relying on the retired interactive
dashboard page.

Run: py src/kpi/export_index_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FINDINGS_CACHE = ROOT / "data" / "synthetic" / "findings_cache.json"
OUT_DIR = ROOT / "docs" / "assets"
OUT_F3 = OUT_DIR / "finding03_output_cycle_time.png"


def export_finding03_static() -> Path:
    """Render the old dashboard Finding 03 chart as a static index asset."""
    with FINDINGS_CACHE.open(encoding="utf-8") as f:
        f3 = json.load(f)["finding_03"]

    day = f3["day"]
    onset = f3["onset_day"]
    alert = f3["alert_day"]

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.4), dpi=160, sharex=True)

    axes[0].plot(
        day,
        f3["deg_output"],
        color="#D32F2F",
        lw=1.15,
        label="with degradation",
    )
    axes[0].plot(
        day,
        f3["clean_output"],
        color="#607D8B",
        lw=1.15,
        label="no degradation twin",
    )
    axes[0].set_ylabel("lots/day")
    axes[0].set_title("Daily output: stays in its normal band", fontsize=11)

    axes[1].plot(day, f3["deg_cycle_time"], color="#D32F2F", lw=1.35)
    axes[1].plot(day, f3["clean_cycle_time"], color="#607D8B", lw=1.35)
    axes[1].set_ylabel("median CT (h)")
    axes[1].set_xlabel("day")
    axes[1].set_title(
        "Daily median cycle time: diverges before output does",
        fontsize=11,
    )

    for ax in axes:
        ax.axvline(onset, color="#8E24AA", ls=":", lw=1.2)
        ax.axvline(alert, color="#00695C", ls="--", lw=1.2)
        ax.grid(True, color="#90A4AE", alpha=0.22, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].annotate(
        "degradation onset (day 30)",
        xy=(onset, axes[0].get_ylim()[1]),
        xytext=(onset + 3, axes[0].get_ylim()[1] * 0.94),
        fontsize=8,
        color="#6A1B9A",
    )
    axes[0].annotate(
        "EWMA alert (day 84)",
        xy=(alert, axes[0].get_ylim()[1]),
        xytext=(alert + 3, axes[0].get_ylim()[1] * 0.86),
        fontsize=8,
        color="#00695C",
    )

    fig.suptitle(
        "Finding 03: degradation shows up in cycle time before output",
        fontsize=12,
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_F3, bbox_inches="tight")
    plt.close(fig)
    return OUT_F3


def main() -> None:
    out = export_finding03_static()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
