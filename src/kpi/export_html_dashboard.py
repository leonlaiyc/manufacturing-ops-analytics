"""
Export the interactive evidence page for the three findings, plus the demoted
KPI baseline, as a standalone HTML page.

Reads:
  - the committed synthetic artifacts (data/synthetic/event_log.csv,
    lot_lifecycle.csv, metadata.json) for the baseline 6-panel figure, and
  - data/synthetic/findings_cache.json (written by precompute_findings.py) for
    the three finding figures.

Writes one self-contained Plotly page (plotly.js via CDN on the first figure
only, so the file stays small) to:

  - reports/html/03_kpi_dashboard.html   (repo artifact)
  - docs/dashboard.html                  (published via GitHub Pages)

The page leads with the evidence for the three findings (CRN counterfactual,
cost trade-off, degradation backtest) and demotes the KPI baseline to a
"shared context" section at the bottom. Carries the same honest-scope
labeling (synthetic data, clearly stated) as before.

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
# Finding 01 — CRN counterfactual dot-plot with 95% CI
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
        title=dict(text=f"Finding 01 — Δ mean cycle time, +1 tool vs baseline "
                        f"(CRN-paired, N={f1['n_reps']})", x=0.5, font=dict(size=14)),
        xaxis_title="Mean cycle-time reduction (h), +1 tool vs baseline (CRN-paired, N=30)",
        yaxis=dict(autorange="reversed"),
        height=340, template="plotly_white", showlegend=False,
        margin=dict(t=60, l=90, r=30, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Finding 02 — grouped bar: base-case vs investment-stress net cost change
# --------------------------------------------------------------------------- #
def build_finding02_figure(cache: dict) -> go.Figure:
    f2 = cache["finding_02"]
    stations = f2["stations"]
    base = f2["base_cost"]
    stress = f2["stress_cost"]
    break_even = f2["break_even"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stations, y=[v / 1000 for v in base], name="Base case (equal $20k tool cost)",
        marker_color="#4878A8",
        customdata=break_even,
        hovertemplate="<b>%{x}</b><br>Base-case net cost: $%{y:.1f}k"
                      "<br>Break-even added-tool cost: $%{customdata:.1f}k<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=stations, y=[v / 1000 for v in stress], name="Investment-stress (station-specific cost)",
        marker_color="#EF6C00",
        customdata=break_even,
        hovertemplate="<b>%{x}</b><br>Investment-stress net cost: $%{y:.1f}k"
                      "<br>Break-even added-tool cost: $%{customdata:.1f}k<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1,
                  annotation_text="net cost = 0", annotation_position="top left")
    fig.update_layout(
        title=dict(text="Finding 02 — net cost change: base case vs investment stress",
                  x=0.5, font=dict(size=14)),
        yaxis_title="net cost change vs baseline ($k)",
        barmode="group", height=380, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, l=60, r=30, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Finding 03 — 2-row subplot: daily output (top) vs daily median cycle time (bottom)
# --------------------------------------------------------------------------- #
def build_finding03_figure(cache: dict) -> go.Figure:
    f3 = cache["finding_03"]
    day = f3["day"]
    onset, alert = f3["onset_day"], f3["alert_day"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            "Daily output (lots/day) — stays in its normal band",
            "Daily median cycle time (h) — diverges long before output does",
        ),
    )
    fig.add_trace(go.Scatter(x=day, y=f3["deg_output"], mode="lines", name="degraded run",
                             line=dict(color="#D32F2F", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=day, y=f3["clean_output"], mode="lines", name="clean twin",
                             line=dict(color="#607D8B", width=1.4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=day, y=f3["deg_cycle_time"], mode="lines", name="degraded run",
                             line=dict(color="#D32F2F", width=1.8), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=day, y=f3["clean_cycle_time"], mode="lines", name="clean twin",
                             line=dict(color="#607D8B", width=1.8), showlegend=False),
                  row=2, col=1)

    # Labels on the top row only: passing annotation_text=None makes Plotly fall
    # back to its "new text" default, and the bottom row's CT curves diverge right
    # where labels would sit — so the bottom-row reference lines stay unlabeled
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
        title=dict(text="Finding 03 — degradation shows up in cycle time long before output",
                  x=0.5, font=dict(size=14)),
        height=560, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        margin=dict(t=90, l=60, r=30, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Baseline — existing 6-panel figure, demoted; now with visible annotation
# labels on every reference line.
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

    # 1 — daily throughput
    fig.add_trace(go.Bar(x=thr["day"], y=thr["count"], name="lots/day",
                         marker_color="#4878A8"), row=1, col=1)
    fig.add_hline(y=meta["validation"]["throughput_per_hour"] * 24,
                  line_dash="dash", line_color="darkorange", row=1, col=1,
                  annotation_text="throughput target", annotation_position="top left")

    # 2 — WIP step function (thin the point cloud for page weight)
    step = max(1, len(times) // 4000)
    fig.add_trace(go.Scatter(x=times[::step], y=wip[::step], mode="lines",
                             name="WIP", line=dict(color="#4878A8", width=1)),
                  row=1, col=2)
    fig.add_vline(x=t0, line_dash="dash", line_color="crimson", row=1, col=2,
                  annotation_text="warm-up ends", annotation_position="top right")

    # 3 — slot utilization
    colors = ["#EF5350" if s == meta["ground_truth_bottleneck"] else "#90A4AE"
              for s in util["station"]]
    fig.add_trace(go.Bar(x=util["station"], y=util["utilization"],
                         name="empirical", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=util["station"],
                             y=[theo[s] for s in util["station"]],
                             mode="markers", name="design target",
                             marker=dict(symbol="diamond", size=10,
                                         color="darkorange")), row=2, col=1)

    # 4 — daily median cycle time
    fig.add_trace(go.Scatter(x=dct["day"], y=dct["median_ct"], mode="lines+markers",
                             name="median CT", line=dict(color="#4878A8")),
                  row=2, col=2)

    # 5 — cycle time histogram
    fig.add_trace(go.Histogram(x=ct_series, nbinsx=40, name="cycle time",
                               marker_color="#4878A8"), row=3, col=1)
    fig.add_vline(x=ct_med, line_dash="dash", line_color="darkorange", row=3, col=1,
                  annotation_text="median", annotation_position="top left")
    fig.add_vline(x=ct_p90, line_dash="dot", line_color="crimson", row=3, col=1,
                  annotation_text="p90", annotation_position="top right")

    # 6 — X-factor histogram
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
        margin=dict(t=115, l=60, r=40, b=50),
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
  .langbtn { border:1px solid #DCE3E8; background:#fff; color:#5B7180; font-size:.8rem;
              padding:.28rem .66rem; border-radius:999px; cursor:pointer; font-weight:600; margin-left:.35rem; }
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
  .baseline-intro { color:#5B7180; font-size:.9rem; margin:.2rem 0 .8rem; }
  .plot { background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.4rem; }
  .baseline-section { margin-top:2.2rem; }
  .note { font-size:.8rem; color:#9AA9B4; margin-top:1.2rem; }
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
    <p class="lede">These charts ARE the evidence — hover to read exact values, zoom to inspect.
    (SYNTHETIC fab-style line; steady-state window only.)</p></div>
  <div class="zh"><h1>三個 Findings 的互動證據</h1>
    <p class="lede">這些圖表就是證據本身——把滑鼠移上去可以看到精確數值，也可以放大檢視。
    （合成資料，僅使用穩態時間窗。）</p></div>

  <!-- FINDING 01 -->
  <div class="finding-section">
    <span class="tag">FINDING 01</span>
    <div class="en"><h2>Local metrics show candidates; the what-if test gives the answer</h2>
      <p class="takeaway">Adding one LITHO tool cuts mean cycle time by 2.46 h (95% CI
      2.13–2.79). Adding one FURNACE tool adds four lot slots per run but buys only 0.70 h —
      batching-delay relief, not constraint relief. (CRN-paired what-if, N=30 replications.)</p></div>
    <div class="zh"><h2>局部指標指出候選，what-if 測試給出答案</h2>
      <p class="takeaway">新增一台 LITHO 工具可讓平均 cycle time 降低 2.46 小時（95% CI
      2.13–2.79）。新增一台 FURNACE 工具，每次運轉多增加 4 個 lot 處理槽，卻只換來 0.70 小時的改善——
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
    <div class="en"><h2>Biggest operational win is not automatically the best financial case</h2>
      <p class="takeaway">Under equal tool costs ($20k for every station), LITHO wins both
      filters: largest cycle-time reduction and the only negative net cost (≈ −$12k). Under
      station-specific costs (LITHO $40k, FURNACE $8k, DEPO $5k, METRO $2k), LITHO becomes
      ≈ +$8.0k while FURNACE turns into the best financial result (≈ −$1.7k) — the business
      objective decides.</p></div>
    <div class="zh"><h2>營運效益最大的方案，不一定是財務上最划算的方案</h2>
      <p class="takeaway">在每站工具成本相同（$20k）的假設下，LITHO 同時勝出：cycle time 降幅最大，
      也是唯一淨成本為負的方案（約 −$12k）。若改用站點別成本假設（LITHO $40k、FURNACE $8k、
      DEPO $5k、METRO $2k），LITHO 的淨成本變成約 +$8.0k，而 FURNACE 反而成為財務上最佳結果
      （約 −$1.7k）——最終選擇取決於商業目標。</p></div>
    <div id="fig-finding02" class="plot">{fig_finding02}</div>
    <p class="method-note en">Method: net cost change = scenario total cost − baseline total
    cost (processing + waiting/holding + added-tool cost). Hover each bar for that station's
    break-even added-tool cost — how expensive the tool could be before the option stops
    net-saving.</p>
    <p class="method-note zh">方法：淨成本變化 = 情境總成本 − 基準總成本（處理成本 + 等待/持有成本 +
    新增工具成本）。將滑鼠移到長條上可看到該站點的損益平衡新增工具成本——工具可以多貴而不讓方案由淨節省
    轉為淨增加。</p>
  </div>

  <!-- FINDING 03 -->
  <div class="finding-section">
    <span class="tag">FINDING 03</span>
    <div class="en"><h2>Degradation shows up in cycle time long before output</h2>
      <p class="takeaway">LITHO starts degrading on day 30. Daily output never leaves its
      normal band — an output-only monitor gives no alert within the 160-day horizon — while
      EWMA on daily median cycle time alerts on day 84. About 95% of the ≈ $249k extra cost is
      still avoidable at the alert.</p></div>
    <div class="zh"><h2>劣化會先反映在 cycle time，遠早於 output</h2>
      <p class="takeaway">LITHO 從第 30 天開始劣化。Daily output 始終沒有脫離正常區間——只看
      output 的監控在 160 天的觀察期內完全不會 alert——而以 EWMA 監控每日 cycle time 中位數則在第
      84 天 alert。在 alert 當下，約 $249k 額外成本中仍有 95% 可被避免。</p></div>
    <div id="fig-finding03" class="plot">{fig_finding03}</div>
    <p class="method-note en">Method: reference pair = one clean twin run and one degraded run
    sharing the same CRN draw table, differing only in the LITHO degradation. EWMA (λ=0.2,
    L=3σ) is fit on a clean pre-onset baseline window, so it never "sees" the anomaly it is
    meant to catch.</p>
    <p class="method-note zh">方法：參照組為一組乾淨對照與一組劣化 run，兩者共用同一組 CRN
    抽樣表，差異僅在於 LITHO 是否劣化。EWMA（λ=0.2，L=3σ）以劣化前的乾淨基準期估計參數，因此偵測器
    從未「看過」它要偵測的異常。</p>
  </div>

  <!-- BASELINE (demoted) -->
  <div class="baseline-section">
    <div class="en"><h2>Shared KPI baseline (context for the findings)</h2>
      <p class="baseline-intro">Output, WIP, utilization, cycle time, and X-factor are the
      background signals the three findings above test with what-if simulation, cost
      modeling, and a monitoring backtest.</p></div>
    <div class="zh"><h2>共用 KPI Baseline（三個 findings 的背景資料）</h2>
      <p class="baseline-intro">Output、WIP、utilization、cycle time 與 X-factor 是上面三個
      findings 用 what-if simulation、成本模型與監控 backtest 進一步測試的背景訊號。</p></div>
    <div class="plot">{fig_baseline}</div>
  </div>

  <p class="note en">Synthetic data, clearly labeled. Cost/rate assumptions live in the
  notebooks; the findings above test the signals in this baseline with additional analysis.</p>
  <p class="note zh">合成資料，明確標示。成本與費率假設放在 notebook；上面的 findings 用後續分析
  測試這組 baseline 訊號。</p>
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

    print("Rendering HTML (plotly.js via CDN on the first figure only) ...")
    div1 = fig1.to_html(include_plotlyjs="cdn", full_html=False)
    div2 = fig2.to_html(include_plotlyjs=False, full_html=False)
    div3 = fig3.to_html(include_plotlyjs=False, full_html=False)
    div_baseline = fig_baseline.to_html(include_plotlyjs=False, full_html=False)

    html = PAGE_TEMPLATE.replace("{fig_finding01}", div1) \
                        .replace("{fig_finding02}", div2) \
                        .replace("{fig_finding03}", div3) \
                        .replace("{fig_baseline}", div_baseline)

    for out in (OUT_REPORT, OUT_DOCS):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
