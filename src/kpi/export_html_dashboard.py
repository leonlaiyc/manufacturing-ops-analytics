"""
Export two standalone Plotly HTML pages: the interactive evidence page for the
three findings, and the M3 KPI baseline page.

Reads:
  - the committed synthetic artifacts (data/synthetic/event_log.csv,
    lot_lifecycle.csv, metadata.json) for the baseline 6-panel figure, and
  - data/synthetic/findings_cache.json (written by precompute_findings.py) for
    the three finding figures.

Writes (plotly.js via CDN on the first figure of each page, so files stay small):

  - docs/dashboard.html                  (evidence page: the three findings)
  - docs/baseline.html                   (M3 KPI baseline page)
  - reports/html/03_kpi_dashboard.html   (repo copy of the BASELINE page:
                                          it is the M3 artifact, so its name
                                          matches its content)

The evidence page leads with the three findings (CRN counterfactual, cost
trade-off, degradation backtest) and links to the baseline page in its footer.
Both pages carry the same honest-scope labeling (synthetic data, clearly stated).

Run:  py src/kpi/export_html_dashboard.py
      (run src/kpi/precompute_findings.py first if findings_cache.json is missing)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kpi.kpi_metrics import (          # noqa: E402
    cycle_time_stats,
    daily_median_ct,
    daily_throughput,
    station_utilization,
    wip_timeseries,
    x_factor,
)

DATA = ROOT / "data" / "synthetic"
CACHE = DATA / "findings_cache.json"
OUT_REPORT = ROOT / "reports" / "html" / "03_kpi_dashboard.html"
OUT_DOCS = ROOT / "docs" / "dashboard.html"
OUT_BASELINE_DOCS = ROOT / "docs" / "baseline.html"

STATION_COLORS = {"LITHO": "#EF5350"}
DEFAULT_STATION_COLOR = "#90A4AE"


def load_findings_cache() -> dict:
    if not CACHE.exists():
        print(
            "ERROR: data/synthetic/findings_cache.json not found.\n"
            "Run precompute first:  py src/kpi/precompute_findings.py",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Finding 01: CRN counterfactual dot-plot with 95% CI
# --------------------------------------------------------------------------- #
def build_finding01_figure(cache: dict) -> go.Figure:
    f1 = cache["finding_01"]
    stations = f1["stations"]
    mean = f1["delta_mean"]
    ci_low = f1["ci_low"]
    ci_high = f1["ci_high"]

    colors = [STATION_COLORS.get(s, DEFAULT_STATION_COLOR) for s in stations]
    err_plus = [h - m for h, m in zip(ci_high, mean)]
    err_minus = [m - lo for m, lo in zip(mean, ci_low)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mean, y=stations, orientation="h",
        marker_color=colors,
        error_x=dict(type="data", symmetric=False,
                     array=err_plus, arrayminus=err_minus,
                     color="#37474F", thickness=1.5, width=6),
        customdata=list(zip(ci_low, ci_high)),
        hovertemplate="<b>%{y}</b><br>Δ mean cycle time: %{x:.2f} h"
                      "<br>95%% CI: %{customdata[0]:.2f}–%{customdata[1]:.2f} h"
                      "<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Finding 01: Δ mean cycle time, +1 tool vs baseline "
                        f"(CRN-paired, N={f1['n_reps']})", x=0.5, font=dict(size=14)),
        xaxis_title="Mean cycle-time reduction (h), +1 tool vs baseline (CRN-paired, N=30)",
        yaxis=dict(autorange="reversed"),
        height=340, template="plotly_white", showlegend=False,
        margin=dict(t=60, l=90, r=30, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Finding 02: compare operating savings before the added-tool price against one
# illustrative quote. A price marker inside the bar means savings cover the quote.
# --------------------------------------------------------------------------- #
def _fmt_k(value_k: float) -> str:
    """Format $k values like the owner's copy: −$12k, +$8.0k, +$10.3k, −$1.7k.

    One decimal by default; the decimal is dropped only when the value is both
    ≥ $10k and rounds to a whole number (so 11.98 → −$12k but 10.28 → +$10.3k).
    """
    sign = "−" if value_k < 0 else "+"
    mag = round(abs(value_k), 1)
    body = f"{mag:.0f}" if (mag >= 10 and mag == int(mag)) else f"{mag:.1f}"
    return f"{sign}${body}k"


def build_finding02_figure(cache: dict) -> go.Figure:
    f2 = cache["finding_02"]
    stations = f2["stations"]
    stress = [v / 1000 for v in f2["stress_cost"]]
    operating_savings = [v / 1000 for v in f2["break_even"]]
    prices = [v / 1000 for v in f2["stress_tool_costs"]]
    gains = cache["finding_01"]["delta_mean"]  # same station order as finding_02

    labels = [f"{s} (-{g:.2f} h)" for s, g in zip(stations, gains)]
    colors = [STATION_COLORS.get(s, DEFAULT_STATION_COLOR) for s in stations]
    # hover shows operating savings, illustrative quote, and net cost at that quote.
    hover_data = [[p, _fmt_k(s)] for p, s in zip(prices, stress)]

    xmax = max(operating_savings + prices) * 1.12

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=operating_savings, y=labels, orientation="h", marker_color=colors,
        showlegend=False,
        customdata=hover_data,
        hovertemplate="<b>%{y}</b>"
                      "<br>Operating savings before tool price: $%{x:.1f}k"
                      "<br>Illustrative quote: $%{customdata[0]:.0f}k"
                      "<br>Net cost = quote - savings: %{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=prices, y=labels, mode="markers", name="illustrative quote",
        marker=dict(symbol="diamond", size=12, color="#EF6C00",
                    line=dict(color="#B34700", width=1)),
        customdata=hover_data,
        hovertemplate="<b>%{y}</b>"
                      "<br>Illustrative quote: $%{x:.0f}k"
                      "<br>Net cost = quote - savings: %{customdata[1]}"
                      "<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Finding 02: cycle-time gain must clear the tool quote",
                   x=0.5, font=dict(size=14)),
        xaxis=dict(title="Operating savings before tool price ($k): quote must be lower than the bar",
                   range=[0, xmax]),
        yaxis=dict(autorange="reversed"),
        height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=85, l=130, r=30, b=55),
    )
    return fig


# --------------------------------------------------------------------------- #
# Finding 03: 2-row subplot: daily output (top) vs daily median cycle time (bottom)
# --------------------------------------------------------------------------- #
def build_finding03_figure(cache: dict) -> go.Figure:
    f3 = cache["finding_03"]
    day = f3["day"]
    onset, alert = f3["onset_day"], f3["alert_day"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            "Daily output (lots/day): stays in its normal band",
            "Daily median cycle time (h): diverges long before output does",
        ),
    )
    fig.add_trace(go.Scatter(x=day, y=f3["deg_output"], mode="lines",
                             name="with degradation",
                             line=dict(color="#D32F2F", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=day, y=f3["clean_output"], mode="lines",
                             name="no degradation (twin)",
                             line=dict(color="#607D8B", width=1.4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=day, y=f3["deg_cycle_time"], mode="lines",
                             name="with degradation",
                             line=dict(color="#D32F2F", width=1.8), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=day, y=f3["clean_cycle_time"], mode="lines",
                             name="no degradation (twin)",
                             line=dict(color="#607D8B", width=1.8), showlegend=False),
                  row=2, col=1)

    # Labels on the top row only: passing annotation_text=None makes Plotly fall
    # back to its "new text" default, and the bottom row's CT curves diverge right
    # where labels would sit, so the bottom-row reference lines stay unlabeled
    # (same colors/dash as the labeled top-row lines).
    fig.add_vline(x=onset, line_dash="dot", line_color="#8E24AA", row=1, col=1,
                  annotation_text="degradation onset (day 30)",
                  annotation_position="top left")
    fig.add_vline(x=alert, line_dash="dash", line_color="#00695C", row=1, col=1,
                  annotation_text="EWMA alert (day 84)",
                  annotation_position="top right")
    fig.add_vline(x=onset, line_dash="dot", line_color="#8E24AA", row=2, col=1)
    fig.add_vline(x=alert, line_dash="dash", line_color="#00695C", row=2, col=1)

    fig.update_xaxes(title_text="day", row=2, col=1)
    fig.update_yaxes(title_text="lots/day", row=1, col=1)
    fig.update_yaxes(title_text="median CT (h)", row=2, col=1)
    fig.update_layout(
        title=dict(text="Finding 03: degradation shows up in cycle time long before output",
                  x=0.5, font=dict(size=14)),
        height=580, template="plotly_white",
        # horizontal legend BELOW the plot area so it never crowds the title
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5),
        margin=dict(t=70, l=60, r=30, b=85),
    )
    return fig


# --------------------------------------------------------------------------- #
# Baseline: the M3 6-panel figure (own page), with visible annotation labels
# on every reference line.
# --------------------------------------------------------------------------- #
def build_baseline_figure() -> go.Figure:
    event_log = pd.read_csv(DATA / "event_log.csv")
    lifecycle = pd.read_csv(DATA / "lot_lifecycle.csv")
    with open(DATA / "metadata.json") as f:
        meta = json.load(f)

    t0, t1 = meta["warmup_hours"], meta["horizon_hours"]
    stations = meta["stations"]
    theo = meta["theoretical_utilization"]

    thr = daily_throughput(lifecycle, t0, t1)
    times, wip = wip_timeseries(lifecycle)
    util = station_utilization(event_log, t0, t1, stations)
    ct_series, ct_med, ct_p90 = cycle_time_stats(lifecycle, t0, t1)
    dct = daily_median_ct(lifecycle, t0, t1)
    x_series, x_med, x_p90 = x_factor(event_log, lifecycle, t0, t1)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Daily output (lots completed)",
            "WIP over time (warm-up shaded)",
            "Slot utilization vs design target (LITHO = engineered bottleneck)",
            "Daily median cycle time (h)",
            "Cycle time distribution (h)",
            f"X-factor per lot (median {x_med:.2f}, p90 {x_p90:.2f})",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 1: daily throughput
    fig.add_trace(go.Bar(x=thr["day"], y=thr["count"], name="lots/day",
                         marker_color="#4878A8"), row=1, col=1)
    fig.add_hline(y=meta["validation"]["throughput_per_hour"] * 24,
                  line_dash="dash", line_color="darkorange", row=1, col=1,
                  annotation_text="throughput target", annotation_position="top left")

    # 2: WIP step function (thin the point cloud for page weight)
    step = max(1, len(times) // 4000)
    fig.add_trace(go.Scatter(x=times[::step], y=wip[::step], mode="lines",
                             name="WIP", line=dict(color="#4878A8", width=1)),
                  row=1, col=2)
    fig.add_vline(x=t0, line_dash="dash", line_color="crimson", row=1, col=2,
                  annotation_text="warm-up ends", annotation_position="top right")

    # 3: slot utilization
    colors = ["#EF5350" if s == meta["ground_truth_bottleneck"] else "#90A4AE"
              for s in util["station"]]
    fig.add_trace(go.Bar(x=util["station"], y=util["utilization"],
                         name="empirical", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=util["station"],
                             y=[theo[s] for s in util["station"]],
                             mode="markers", name="design target",
                             marker=dict(symbol="diamond", size=10,
                                         color="darkorange")), row=2, col=1)

    # 4: daily median cycle time
    fig.add_trace(go.Scatter(x=dct["day"], y=dct["median_ct"], mode="lines+markers",
                             name="median CT", line=dict(color="#4878A8")),
                  row=2, col=2)

    # 5: cycle time histogram
    fig.add_trace(go.Histogram(x=ct_series, nbinsx=40, name="cycle time",
                               marker_color="#4878A8"), row=3, col=1)
    fig.add_vline(x=ct_med, line_dash="dash", line_color="darkorange", row=3, col=1,
                  annotation_text="median", annotation_position="top left")
    fig.add_vline(x=ct_p90, line_dash="dot", line_color="crimson", row=3, col=1,
                  annotation_text="p90", annotation_position="top right")

    # 6: X-factor histogram
    fig.add_trace(go.Histogram(x=x_series, nbinsx=40, name="X-factor",
                               marker_color="#00897B"), row=3, col=2)
    fig.add_vline(x=x_med, line_dash="dash", line_color="darkorange", row=3, col=2,
                  annotation_text="median", annotation_position="top left")
    fig.add_vline(x=x_p90, line_dash="dot", line_color="crimson", row=3, col=2,
                  annotation_text="p90", annotation_position="top right")

    fig.update_layout(
        # Short first line + smaller <br> detail lines so the title never clips
        # at a 1280px viewport (the old single-line title overflowed the plot).
        title=dict(
            text="Shared KPI baseline"
                 "<br><span style='font-size:11px;color:#888'>SYNTHETIC fab-style "
                 "line: CLEAN → FURNACE(batch) → DEPO → LITHO → ETCH → LITHO → "
                 "IMPLANT → METRO</span>"
                 "<br><span style='font-size:11px;color:#888'>1 lot = 25-wafer "
                 "FOUP · steady-state window only</span>",
            x=0.5),
        height=1050, showlegend=False, template="plotly_white",
        # t=150 leaves room for the 3-line title above the subplot titles
        margin=dict(t=150, l=60, r=40, b=50),
    )
    fig.update_yaxes(range=[0, 1.0], row=2, col=1)
    return fig


# --------------------------------------------------------------------------- #
# Bilingual page shell (EN / 繁中) so a first-time viewer knows how to read the
# charts. The plots are language-neutral; only the surrounding copy toggles.
# --------------------------------------------------------------------------- #
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive Evidence: Manufacturing Operations Analytics</title>
<style>
  body { font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Helvetica,Arial,sans-serif;
         color:#22303C; background:#F7F9FA; margin:0; line-height:1.6; }
  .wrap { max-width:1000px; margin:0 auto; padding:0 1.1rem 3rem; }
  .topbar { display:flex; justify-content:space-between; align-items:center;
             max-width:1000px; margin:0 auto; padding:.9rem 1.1rem .2rem; }
  .back { font-size:.86rem; color:#1565C0; text-decoration:none; font-weight:600; }
  .langbtn { border:1px solid #DCE3E8; background:#fff; color:#5B7180;
              display:inline-flex; align-items:center; justify-content:center;
              min-width:3.1rem; height:1.9rem; padding:0 .7rem;
              border-radius:999px; cursor:pointer; font-size:.8rem;
              font-weight:600; letter-spacing:0; line-height:1; margin-left:.35rem; }
  .langbtn[aria-pressed="true"] { background:#1565C0; color:#fff; border-color:#1565C0; }
  .lang-zh .en { display:none; } .lang-en .zh { display:none; }
  h1 { font-size:1.35rem; margin:.7rem 0 .2rem; }
  h2 { font-size:1.1rem; margin:1.6rem 0 .3rem; }
  .lede { color:#5B7180; margin:.1rem 0 1rem; font-size:.98rem; }
  .finding-section { background:#fff; border:1px solid #DCE3E8; border-radius:12px;
                       padding:1rem 1.1rem 1.2rem; margin:1rem 0; }
  .finding-section .tag { display:inline-block; background:#E3F0FB; color:#1565C0;
                           border-radius:999px; padding:.12rem .55rem; font-size:.7rem;
                           font-weight:800; letter-spacing:.03em; margin-bottom:.5rem; }
  .finding-section h2 { margin-top:.1rem; }
  .takeaway { color:#22303C; font-size:.94rem; margin:.2rem 0 .8rem; }
  .method-note { font-size:.8rem; color:#7C8B96; margin-top:.55rem; }
  .plot { background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.4rem; }
  .note { font-size:.8rem; color:#9AA9B4; margin-top:1.2rem; }
  .note a { color:#1565C0; }
  @media (max-width:720px) {
    .topbar { align-items:flex-start; gap:.75rem; }
  }
</style>
</head>
<body class="lang-zh">
<div class="topbar">
  <a class="back" href="index.html"><span class="en">← Back to the 1-page report</span><span class="zh">← 回到一頁式報告</span></a>
  <div>
    <button class="langbtn" data-lang="zh" onclick="setLang('zh')">中文</button>
    <button class="langbtn" data-lang="en" onclick="setLang('en')">EN</button>
  </div>
</div>
<div class="wrap">
  <div class="en"><h1>Interactive evidence for the three findings</h1>
    <p class="lede">These charts ARE the evidence: hover to read exact values, zoom to inspect.
    (SYNTHETIC fab-style line; steady-state window only.)</p></div>
  <div class="zh"><h1>三個 Findings 的互動證據</h1>
    <p class="lede">這些圖表就是證據本身，把滑鼠移上去可以看到精確數值，也可以放大檢視。
    （合成資料，僅使用穩態時間窗。）</p></div>

  <!-- FINDING 01 -->
  <div class="finding-section">
    <span class="tag">FINDING 01</span>
    <div class="en"><h2>Local metrics show candidates; the what-if test gives the answer</h2>
      <p class="takeaway">Adding one LITHO tool cuts mean cycle time by 2.46 h (95% CI
      2.13–2.79). Adding one FURNACE tool adds four lot slots per run but buys only 0.70 h:
      batching-delay relief, not constraint relief. (CRN-paired what-if, N=30 replications.)</p></div>
    <div class="zh"><h2>局部指標指出候選，what-if 測試給出答案</h2>
      <p class="takeaway">新增一台 LITHO 工具可讓平均 cycle time 降低 2.46 小時（95% CI
      2.13–2.79）。新增一台 FURNACE 工具，每次運轉多增加 4 個 lot 處理槽，卻只換來 0.70 小時的改善，
      這是批次等待的緩解，不是瓶頸的緩解。（CRN 配對 what-if，N=30 次重複。）</p></div>
    <div id="fig-finding01" class="plot">{fig_finding01}</div>
    <p class="method-note en">Method: Common Random Numbers (CRN) pair baseline and +1-tool
    scenarios on the same random draw table per replication, so each bar is a paired
    difference with simulation noise cancelled. Error bars are the 95% CI across N=30 reps.</p>
    <p class="method-note zh">方法：Common Random Numbers（CRN）在每次重複中，讓 baseline 與
    +1 工具情境使用同一組隨機抽樣表，因此每個長條都是配對後的差異，模擬雜訊已互相抵銷。誤差棒為
    N=30 次重複的 95% CI。</p>
  </div>

  <!-- FINDING 02 -->
  <div class="finding-section">
    <span class="tag">FINDING 02</span>
    <div class="en"><h2>The biggest cycle-time gain still needs a price check</h2>
      <p class="takeaway">Finding 01 shows that LITHO cuts cycle time the most. Finding 02
      asks whether that improvement saves enough operating cost to pay for the extra tool.
      In this model, +1 LITHO removes about $32k of waiting, congestion, and processing cost
      before paying for the tool. With a $40k illustrative quote, net cost = $40k - $32k =
      +$8k, so the cost case does not pay back. FURNACE saves less cycle time, but removes
      about $9.7k of operating cost; with an $8k quote, net cost = $8k - $9.7k = -$1.7k.</p>
      <p class="method-note">Bar = operating cost removed before tool price. Marker =
      illustrative quote. Marker inside the bar means the savings cover the quote.</p></div>
    <div class="zh"><h2>降 cycle time 最多，仍然要過價格檢查</h2>
      <p class="takeaway">Finding 01 顯示 LITHO 最能降低 cycle time。Finding 02 只問一件事：
      這個改善省下的營運成本，是否足以付掉新增機台？在這個模型裡，+1 LITHO 在不計機台價格前，
      約可少掉 $32k 的等待、壅塞與處理成本。若示範報價是 $40k，淨成本 = $40k - $32k =
      +$8k，所以成本面不回本。FURNACE 降 cycle time 較少，但約可少掉 $9.7k；若報價 $8k，
      淨成本 = $8k - $9.7k = -$1.7k。</p>
      <p class="method-note">長條＝不計機台價格前可省下的營運成本。標記＝示範報價。標記落在長條內，
      代表節省金額足以付掉報價。</p></div>
    <div id="fig-finding02" class="plot">{fig_finding02}</div>
    <p class="method-note en">Method: net cost change = scenario total cost − baseline total
    cost (processing + waiting/holding + added-tool cost). Here the bar is the operating
    cost saved before the added-tool price, so net cost = illustrative quote - bar. Hover
    each bar for the savings, quote, and net cost.</p>
    <p class="method-note zh">方法：淨成本變化 = 情境總成本 − 基準總成本（處理成本 + 等待/持有成本 +
    新增工具成本）。這裡的長條是不計新增機台價格前省下的營運成本，所以淨成本 = 示範報價 - 長條。
    將滑鼠移到長條上可看到節省金額、報價與淨成本。</p>
  </div>

  <!-- FINDING 03 -->
  <div class="finding-section">
    <span class="tag">FINDING 03</span>
    <div class="en"><h2>Degradation shows up in cycle time long before output</h2>
      <p class="takeaway">LITHO starts degrading on day 30. Daily output never leaves its
      normal band; an output-only monitor gives no alert within the 160-day horizon, while
      EWMA on daily median cycle time alerts on day 84. About 95% of the ≈ $249k extra cost is
      still avoidable at the alert.</p>
      <p class="method-note">The "no degradation (twin)" line is the same production line
      re-run with identical random numbers but the drift switched off; any gap between the
      two lines is the pure effect of degradation. In the top panel the two lines are barely
      distinguishable: output alone never reveals the problem. In the bottom panel they
      separate within weeks.</p></div>
    <div class="zh"><h2>劣化會先反映在 cycle time，遠早於 output</h2>
      <p class="takeaway">LITHO 從第 30 天開始劣化。Daily output 始終沒有脫離正常區間，只看
      output 的監控在 160 天的觀察期內完全不會 alert；以 EWMA 監控每日 cycle time 中位數則在第
      84 天 alert。在 alert 當下，約 $249k 額外成本中仍有 95% 可被避免。</p>
      <p class="method-note">「no degradation (twin)」這條線，是同一條產線用完全相同的隨機數重跑
      一次，只是把劣化關掉；兩條線之間的任何差距，都是劣化本身的純粹效果。上圖中兩條線幾乎無法分辨：
      只看 output 永遠發現不了問題；下圖中兩條線在幾週內就分開了。</p></div>
    <div id="fig-finding03" class="plot">{fig_finding03}</div>
    <p class="method-note en">Method: reference pair = one clean twin run and one degraded run
    sharing the same CRN draw table, differing only in the LITHO degradation. EWMA (λ=0.2,
    L=3σ) is fit on a clean pre-onset baseline window, so it never "sees" the anomaly it is
    meant to catch.</p>
    <p class="method-note zh">方法：參照組為一組乾淨對照與一組劣化 run，兩者共用同一組 CRN
    抽樣表，差異僅在於 LITHO 是否劣化。EWMA（λ=0.2，L=3σ）以劣化前的乾淨基準期估計參數，因此偵測器
    從未「看過」它要偵測的異常。</p>
  </div>

  <p class="note en">KPI baseline (M3 deliverable) →
    <a href="baseline.html">baseline.html</a></p>
  <p class="note zh">KPI baseline（M3 交付物）→
    <a href="baseline.html">baseline.html</a></p>

  <p class="note en">Synthetic data, clearly labeled. Cost/rate assumptions live in the
  notebooks; the findings above test the baseline KPI signals with additional analysis.</p>
  <p class="note zh">合成資料，明確標示。成本與費率假設放在 notebook；上面的 findings 用後續分析
  測試 baseline 的 KPI 訊號。</p>
</div>
<script>
  function setLang(l){
    document.body.className = 'lang-' + l;
    document.documentElement.lang = (l === 'zh') ? 'zh-Hant' : 'en';
    try { localStorage.setItem('mfg_lang', l); } catch(e){}
    document.querySelectorAll('.langbtn').forEach(function(b){
      b.setAttribute('aria-pressed', b.dataset.lang === l);
    });
  }
  var saved; try { saved = localStorage.getItem('mfg_lang'); } catch(e){}
  setLang(saved || 'zh');
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Baseline page shell (docs/baseline.html + reports/html/03_kpi_dashboard.html)
# The M3 KPI baseline lives on its own page; the evidence page links to it.
# --------------------------------------------------------------------------- #
BASELINE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shared KPI Baseline (M3): Manufacturing Operations Analytics</title>
<style>
  body { font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Helvetica,Arial,sans-serif;
         color:#22303C; background:#F7F9FA; margin:0; line-height:1.6; }
  .wrap { max-width:1000px; margin:0 auto; padding:0 1.1rem 3rem; }
  .topbar { display:flex; justify-content:space-between; align-items:center;
             max-width:1000px; margin:0 auto; padding:.9rem 1.1rem .2rem; }
  .back { font-size:.86rem; color:#1565C0; text-decoration:none; font-weight:600; }
  .langbtn { border:1px solid #DCE3E8; background:#fff; color:#5B7180;
              display:inline-flex; align-items:center; justify-content:center;
              min-width:3.1rem; height:1.9rem; padding:0 .7rem;
              border-radius:999px; cursor:pointer; font-size:.8rem;
              font-weight:600; letter-spacing:0; line-height:1; margin-left:.35rem; }
  .langbtn[aria-pressed="true"] { background:#1565C0; color:#fff; border-color:#1565C0; }
  .lang-zh .en { display:none; } .lang-en .zh { display:none; }
  h1 { font-size:1.35rem; margin:.7rem 0 .2rem; }
  .lede { color:#5B7180; margin:.1rem 0 1rem; font-size:.98rem; }
  .plot { background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.4rem; }
  .note { font-size:.8rem; color:#9AA9B4; margin-top:1.2rem; }
  @media (max-width:720px) {
    .topbar { align-items:flex-start; gap:.75rem; }
  }
</style>
</head>
<body class="lang-zh">
<div class="topbar">
  <a class="back" href="dashboard.html"><span class="en">← Back to the findings evidence</span><span class="zh">← 回到 findings 證據頁</span></a>
  <div>
    <button class="langbtn" data-lang="zh" onclick="setLang('zh')">中文</button>
    <button class="langbtn" data-lang="en" onclick="setLang('en')">EN</button>
  </div>
</div>
<div class="wrap">
  <div class="en"><h1>Shared KPI baseline (M3)</h1>
    <p class="lede">The five baseline KPIs: output, WIP, utilization, cycle time, and X-factor.
    These sit behind the three findings. The findings test these signals with what-if simulation,
    cost modeling, and a monitoring backtest.</p></div>
  <div class="zh"><h1>Shared KPI baseline（M3）</h1>
    <p class="lede">三個 findings 背後的五個 baseline KPI：output、WIP、utilization、
    cycle time、X-factor。Findings 用 what-if simulation、成本模型與監控 backtest
    進一步測試這些訊號。</p></div>

  <div class="plot">{fig_baseline}</div>

  <p class="note en">Synthetic data, clearly labeled. The simulated line exists to give the
  methods a known ground truth. Cost/rate assumptions live in the notebooks; this page shows
  physical KPIs only.</p>
  <p class="note zh">合成資料，明確標示。模擬產線的用途是給方法一個已知的標準答案。成本與費率假設
  放在 notebook，本頁只呈現物理 KPI。</p>
</div>
<script>
  function setLang(l){
    document.body.className = 'lang-' + l;
    document.documentElement.lang = (l === 'zh') ? 'zh-Hant' : 'en';
    try { localStorage.setItem('mfg_lang', l); } catch(e){}
    document.querySelectorAll('.langbtn').forEach(function(b){
      b.setAttribute('aria-pressed', b.dataset.lang === l);
    });
  }
  var saved; try { saved = localStorage.getItem('mfg_lang'); } catch(e){}
  setLang(saved || 'zh');
</script>
</body>
</html>
"""


def main() -> None:
    t0 = time.time()
    print("Loading findings cache ...")
    cache = load_findings_cache()

    print("Building Finding 01 figure (CRN counterfactual) ...")
    fig1 = build_finding01_figure(cache)
    print("Building Finding 02 figure (cost trade-off) ...")
    fig2 = build_finding02_figure(cache)
    print("Building Finding 03 figure (degradation backtest) ...")
    fig3 = build_finding03_figure(cache)
    print("Building baseline 6-panel figure ...")
    fig_baseline = build_baseline_figure()

    print("Rendering HTML (plotly.js via CDN on the first figure of each page) ...")
    div1 = fig1.to_html(include_plotlyjs="cdn", full_html=False)
    div2 = fig2.to_html(include_plotlyjs=False, full_html=False)
    div3 = fig3.to_html(include_plotlyjs=False, full_html=False)
    div_baseline = fig_baseline.to_html(include_plotlyjs="cdn", full_html=False)

    evidence_html = PAGE_TEMPLATE.replace("{fig_finding01}", div1) \
                                 .replace("{fig_finding02}", div2) \
                                 .replace("{fig_finding03}", div3)
    baseline_html = BASELINE_TEMPLATE.replace("{fig_baseline}", div_baseline)

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.write_text(evidence_html, encoding="utf-8")
    print(f"wrote {OUT_DOCS}  ({OUT_DOCS.stat().st_size/1024:.0f} KB)  [evidence page]")

    # The M3 artifact (03_kpi_dashboard.html) is the BASELINE page; its name
    # must match its content, and docs/baseline.html is its published copy.
    for out in (OUT_BASELINE_DOCS, OUT_REPORT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(baseline_html, encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)  [baseline page]")

    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
