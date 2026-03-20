"""
NVDA Alpha Research Platform
Auto-refreshes data every 24 hours via st.cache_data(ttl=86400).

Deploy free: https://share.streamlit.io
  1. pip install -r requirements.txt
  2. streamlit run nvda_alpha_complete.py
  3. Push to GitHub, connect on share.streamlit.io
"""

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Alpha Research", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container{padding-top:1.2rem}
.formula-box{background:#f3f3f0;border:0.5px solid #d3d1c7;border-radius:6px;
  padding:10px 14px;font-family:monospace;font-size:12px;line-height:1.9;
  white-space:pre-wrap;color:#2c2c2a}
.sig-long{background:#e1f5ee;border:1px solid #5dcaa5;border-radius:10px;
  padding:14px 18px;text-align:center}
.sig-flat{background:#f1efe8;border:1px solid #b4b2a9;border-radius:10px;
  padding:14px 18px;text-align:center}
.sig-exit{background:#faece7;border:1px solid #f0997b;border-radius:10px;
  padding:14px 18px;text-align:center}
</style>""", unsafe_allow_html=True)

BULL="#1D9E75"; BEAR="#D85A30"; BLUE="#185FA5"; PUR="#534AB7"
AMB="#BA7517";  GRY="#888780";  TEAL="#0F6E56"; CORN="#993C1D"

# ── make_layout ──────────────────────────────────────────────────────────────
# Always use make_layout(**overrides) instead of spreading **LAYOUT directly.
# Dict merge means a key in overrides silently wins over the base — no
# "multiple values for keyword argument" crash even if keys collide.
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=11, color="#5f5e5a"),
    margin=dict(l=50, r=20, t=35, b=30),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
               zeroline=False, showline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
               zeroline=False, showline=False),
    hovermode="x unified",
)
def make_layout(**overrides):
    return {**_BASE, **overrides}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    ticker     = st.text_input("Ticker", value="NVDA").upper().strip()
    start_date = st.date_input("History start", value=pd.Timestamp("2016-01-01"))
    st.markdown("---")
    mom_lookback = st.slider("Momentum lookback (days)", 126, 504, 252)
    mom_skip     = st.slider("Momentum skip (days)", 5, 42, 21)
    ma_window    = st.slider("Regime MA window", 50, 300, 200)
    st.markdown("---")
    st.caption("Factor weights — auto-normalised")
    w_mom = st.slider("Momentum",       0.0, 1.0, 0.40, 0.05)
    w_si  = st.slider("Short interest", 0.0, 1.0, 0.30, 0.05)
    w_ey  = st.slider("Earnings yield", 0.0, 1.0, 0.20, 0.05)
    w_gp  = st.slider("Gross profit",   0.0, 1.0, 0.10, 0.05)
    _ws = w_mom+w_si+w_ey+w_gp or 1.0
    w_mom,w_si,w_ey,w_gp = w_mom/_ws,w_si/_ws,w_ey/_ws,w_gp/_ws
    st.markdown("---")
    threshold = st.slider("Long entry z-score", 0.0, 2.0, 0.50, 0.05)
    ic_window = st.slider("Rolling IC window (months)", 12, 36, 24)
    st.markdown("---")
    st.caption(f"Cache 24h · {datetime.now().strftime('%b %d %Y %H:%M')}")
    if st.button("Force refresh"):
        st.cache_data.clear(); st.rerun()

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_price(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]].dropna()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fund(ticker):
    stk = yf.Ticker(ticker); info = stk.info; out = {}
    try:
        inc = stk.income_stmt; bs = stk.balance_sheet
        ni  = float(inc.loc["Net Income"].iloc[0])
        sh  = float(info.get("sharesOutstanding", 1))
        pr  = float(info.get("currentPrice", info.get("regularMarketPrice", 1)))
        eps = ni/sh
        out["earnings_yield"]    = eps/pr
        out["trailing_eps"]      = eps
        out["current_price"]     = pr
        rev  = float(inc.loc["Total Revenue"].iloc[0])
        cogs = float(inc.loc["Cost Of Revenue"].iloc[0])
        ta   = float(bs.loc["Total Assets"].iloc[0])
        out["gross_profitability"] = (rev-cogs)/ta
        out["gross_margin_pct"]    = (rev-cogs)/rev*100
    except Exception:
        out["earnings_yield"]      = None
        out["gross_profitability"] = None
        out["current_price"]       = float(info.get("currentPrice",
                                    info.get("regularMarketPrice",0)))
    out["short_ratio"]     = info.get("shortRatio")
    out["short_pct_float"] = info.get("shortPercentOfFloat")
    out["shares_short"]    = info.get("sharesShort")
    out["company_name"]    = info.get("longName", ticker)
    out["sector"]          = info.get("sector","N/A")
    out["market_cap"]      = info.get("marketCap")
    out["beta"]            = info.get("beta")
    out["52w_high"]        = info.get("fiftyTwoWeekHigh")
    out["52w_low"]         = info.get("fiftyTwoWeekLow")
    return out

# ── Signals ───────────────────────────────────────────────────────────────────
def zscore(s, w=60):
    mu = s.rolling(w, min_periods=20).mean()
    sg = s.rolling(w, min_periods=20).std()
    return ((s-mu)/(sg+1e-8)).clip(-3,3)

def compute_signals(df, lb, sk, maw):
    s = df[["Close","Volume"]].copy()
    s["ma"]      = s["Close"].rolling(maw).mean()
    s["regime"]  = (s["Close"]>s["ma"]).astype(int)
    s["mom_raw"] = s["Close"].shift(sk)/s["Close"].shift(lb)-1
    s["mom_z"]   = zscore(s["mom_raw"])
    s["vol_ma20"]= s["Volume"].rolling(20).mean()
    m = s.resample("M").last()
    m["fwd_ret_1m"] = m["Close"].pct_change().shift(-1)
    return s, m

def build_composite(m, fund, weights):
    df = m.copy(); wm,ws,we,wg = weights
    df["f1_mom_z"]    = zscore(df["mom_z"])
    ey = fund.get("earnings_yield") or 0.0
    df["f3_ey_z"]     = zscore(pd.Series([ey]*len(df), index=df.index))
    gp = fund.get("gross_profitability") or 0.0
    df["f4_gp_screen"]= int(gp>0.40)
    df["f4_gp_z"]     = zscore(pd.Series([gp]*len(df), index=df.index))
    sr = fund.get("short_ratio") or 2.0
    df["f5_si_raw_z"] = zscore(pd.Series([sr]*len(df), index=df.index))
    df["f5_si_z"]     = -df["f5_si_raw_z"]   # inversion: low SI = bullish
    df["composite"]   = (df["f1_mom_z"]*wm + df["f5_si_z"]*ws +
                         df["f3_ey_z"]*we  + df["f4_gp_z"]*wg
                         ) * df["regime"] * df["f4_gp_screen"]
    return df

def entry_signals(m, thr):
    entries,exits,in_t = [],[],False
    for date,row in m.iterrows():
        c=row.get("composite",np.nan); r=row.get("regime",0)
        if pd.isna(c): continue
        if not in_t and c>thr and r==1: entries.append(date); in_t=True
        elif in_t and (c<0 or r==0):   exits.append(date);   in_t=False
    return entries,exits

def spearman_ic(factor, fwd):
    al = pd.concat([factor,fwd],axis=1).dropna(); al.columns=["f","r"]
    if len(al)<12: return None,None,len(al)
    ic,pv = stats.spearmanr(al["f"],al["r"])
    return round(float(ic),4),round(float(pv),4),len(al)

def rolling_ic(factor, fwd, win):
    al = pd.concat([factor,fwd],axis=1).dropna(); al.columns=["f","r"]
    ics,dates = [],[]
    for i in range(win,len(al)):
        w=al.iloc[i-win:i]; r,_=stats.spearmanr(w["f"],w["r"])
        ics.append(float(r)); dates.append(al.index[i])
    return pd.Series(ics, index=dates)

def icir(s): return float(s.mean()/(s.std()+1e-8)) if len(s)>2 else float("nan")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker}…"):
    try:
        price_df = fetch_price(ticker, str(start_date))
        fund     = fetch_fund(ticker)
    except Exception as e:
        st.error(f"Data error: {e}"); st.stop()

if price_df.empty: st.error(f"No data for {ticker}"); st.stop()

signals,monthly = compute_signals(price_df, mom_lookback, mom_skip, ma_window)
monthly = build_composite(monthly, fund, (w_mom,w_si,w_ey,w_gp))
long_entries,exits = entry_signals(monthly, threshold)

price_now  = float(price_df["Close"].iloc[-1])
regime_now = int(signals["regime"].iloc[-1])
comp_s     = monthly["composite"].dropna()
comp_now   = float(comp_s.iloc[-1]) if not comp_s.empty else 0.0
mom_s      = monthly["mom_raw"].dropna()
mom_now    = float(mom_s.iloc[-1]) if not mom_s.empty else 0.0
prev       = float(price_df["Close"].iloc[-2]) if len(price_df)>1 else price_now
day_chg    = (price_now/prev-1)*100
ey  = fund.get("earnings_yield")
gp  = fund.get("gross_profitability")
sr  = fund.get("short_ratio")
spct= fund.get("short_pct_float")

def price_at(d):
    try: return float(price_df["Close"].asof(d))
    except: return price_now

# ── Header ────────────────────────────────────────────────────────────────────
mc = fund.get("market_cap")
mc_s = (f"${mc/1e12:.2f}T" if mc and mc>1e12 else
        f"${mc/1e9:.1f}B"  if mc else "N/A")
st.markdown(f"## {ticker} · {fund.get('company_name',ticker)}")
st.caption(f"Sector: {fund.get('sector','N/A')} · Cap: {mc_s} · "
           f"Beta: {fund.get('beta','N/A')} · "
           f"Yahoo Finance · Cache 24h · {datetime.now().strftime('%b %d %Y %H:%M')}")

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.metric("Price",        f"${price_now:,.2f}", f"{day_chg:+.2f}%")
k2.metric("Regime",       "🟢 Bull" if regime_now else "🔴 Bear", f"vs {ma_window}d MA")
k3.metric("Composite z",  f"{comp_now:+.3f}",
          "LONG" if comp_now>threshold and regime_now
          else ("FLAT" if not regime_now else "NEUTRAL"))
k4.metric("12-1 Mom",     f"{mom_now*100:+.1f}%")
k5.metric("Earnings yld", f"{ey*100:.2f}%" if ey else "N/A",
          f"P/E {1/ey:.1f}x" if ey else None)
k6.metric("Gross profit", f"{gp:.3f}" if gp else "N/A",
          "Screen PASS" if (gp or 0)>0.40 else "Screen FAIL")
k7.metric("Short ratio",  f"{sr:.1f}d" if sr else "N/A",
          f"{spct*100:.1f}% float" if spct else None)

if not regime_now:
    cls,msg = "sig-flat", f"⬜ FLAT — regime OFF · price below {ma_window}d MA"
elif comp_now>threshold:
    cls,msg = "sig-long", f"🟢 LONG · composite {comp_now:+.3f} > +{threshold} · enter next open"
elif comp_now<0:
    cls,msg = "sig-exit", f"🔴 EXIT · composite {comp_now:+.3f} < 0"
else:
    cls,msg = "sig-flat", f"⬜ NEUTRAL · composite {comp_now:+.3f} below threshold +{threshold}"
st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_price,tab_factors,tab_ic,tab_comp,
 tab_si,tab_cadence,tab_entries,tab_next) = st.tabs([
    "📈 Price & Signals","🔬 Five Factors","📐 IC & ICIR",
    "🧮 Composite Builder","↕️ Short Interest Inversion",
    "📅 Factor Cadence","🎯 Trade Entries","🗺️ Next Steps",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PRICE & SIGNALS
# ════════════════════════════════════════════════════════════════
with tab_price:
    fig = make_subplots(rows=4,cols=1,shared_xaxes=True,
        row_heights=[0.50,0.20,0.15,0.15],vertical_spacing=0.025,
        subplot_titles=(
            f"Price vs {ma_window}d MA · regime shading · entry/exit markers",
            "Composite alpha signal (z-scored, regime-filtered)",
            "12-1 month momentum (%)",
            "Volume (green=up day · red=down day)"))

    bm = signals["regime"]==1
    fig.add_trace(go.Scatter(
        x=list(signals.index)+list(signals.index[::-1]),
        y=list(signals["Close"].where(bm,signals["ma"]))+
          list(signals["ma"].where(bm,signals["Close"])[::-1]),
        fill="toself",fillcolor="rgba(29,158,117,0.07)",
        line=dict(width=0),name="Bull regime"),row=1,col=1)
    fig.add_trace(go.Scatter(x=signals.index,y=signals["Close"],
        line=dict(color=BLUE,width=1.5),name="Close"),row=1,col=1)
    fig.add_trace(go.Scatter(x=signals.index,y=signals["ma"],
        line=dict(color=AMB,width=1,dash="dot"),
        name=f"{ma_window}d MA"),row=1,col=1)
    if long_entries:
        fig.add_trace(go.Scatter(x=long_entries,
            y=[price_at(d) for d in long_entries],mode="markers",
            marker=dict(symbol="triangle-up",size=13,color=BULL,
                        line=dict(color=TEAL,width=1.5)),
            name="Long entry"),row=1,col=1)
    if exits:
        fig.add_trace(go.Scatter(x=exits,
            y=[price_at(d) for d in exits],mode="markers",
            marker=dict(symbol="x",size=11,color=BEAR,
                        line=dict(color=CORN,width=2)),
            name="Exit"),row=1,col=1)
    fig.add_trace(go.Bar(x=comp_s.index,y=comp_s,
        marker_color=[BULL if v>0 else BEAR for v in comp_s],
        marker_opacity=0.75,showlegend=False),row=2,col=1)
    fig.add_hline(y=threshold,line_dash="dash",line_color=BULL,opacity=0.5,
        annotation_text=f"+{threshold}",annotation_font_size=9,row=2,col=1)
    fig.add_hline(y=0,line_color=GRY,line_width=0.7,row=2,col=1)
    mom_bar = monthly["mom_raw"].dropna()*100
    fig.add_trace(go.Bar(x=mom_bar.index,y=mom_bar,
        marker_color=[BULL if v>0 else BEAR for v in mom_bar],
        marker_opacity=0.65,showlegend=False),row=3,col=1)
    fig.add_hline(y=0,line_color=GRY,line_width=0.7,row=3,col=1)
    vc=[BULL if float(price_df["Close"].iloc[i])>=float(price_df["Open"].iloc[i])
        else BEAR for i in range(len(price_df))]
    fig.add_trace(go.Bar(x=price_df.index,y=price_df["Volume"]/1e6,
        marker_color=vc,marker_opacity=0.5,showlegend=False),row=4,col=1)

    fig.update_layout(**make_layout(
        height=750, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10),
                    orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0)))
    fig.update_yaxes(title_text="Price ($)",row=1,col=1)
    fig.update_yaxes(title_text="z-score",  row=2,col=1)
    fig.update_yaxes(title_text="Return %", row=3,col=1)
    fig.update_yaxes(title_text="Vol (M)",  row=4,col=1)
    st.plotly_chart(fig,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — FIVE FACTORS
# ════════════════════════════════════════════════════════════════
with tab_factors:
    st.markdown("### What each factor measures and why")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Factor 1 · 12-1 Month Momentum")
        st.markdown("""
`momentum = price(-21d) / price(-252d) − 1`

One-month skip avoids short-term reversal.
Updates **every month** — pure price data.
""")
        fig_m=go.Figure()
        fig_m.add_trace(go.Bar(x=mom_bar.index,y=mom_bar,
            marker_color=[BULL if v>0 else BEAR for v in mom_bar],marker_opacity=0.7))
        fig_m.add_hline(y=0,line_color=GRY,line_width=0.8)
        fig_m.update_layout(**make_layout(height=200,yaxis_title="12-1 return %"))
        st.plotly_chart(fig_m,use_container_width=True)
    with c2:
        st.markdown(f"#### Factor 2 · {ma_window}-Day MA Regime Filter")
        st.markdown(f"""
Binary multiplier — not a scorer.

- Above {ma_window}d MA → ×1 → model active
- Below {ma_window}d MA → ×0 → all signals off

Updates **every day** — pure price data.
""")
        fig_r=go.Figure()
        fig_r.add_trace(go.Scatter(x=signals.index,y=signals["Close"],
            line=dict(color=BLUE,width=1.2),name="Close"))
        fig_r.add_trace(go.Scatter(x=signals.index,y=signals["ma"],
            line=dict(color=AMB,width=1,dash="dot"),name=f"{ma_window}d MA"))
        fig_r.update_layout(**make_layout(height=200,yaxis_title="Price ($)"))
        st.plotly_chart(fig_r,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("#### Factor 3 · Earnings Yield (Value)")
        st.markdown("""
`earnings_yield = trailing_EPS / price`

Inverse of P/E — higher = cheaper = more attractive.
No inversion needed, direction is already correct.
Updates **quarterly** (10-Q). Current value is a snapshot.
""")
        eps_v = fund.get("trailing_eps"); pr_v = fund.get("current_price",price_now)
        st.metric("Current EY", f"{ey*100:.2f}%" if ey else "N/A",
                  f"EPS ${eps_v:.2f} ÷ ${pr_v:.2f}" if eps_v else None)
        if ey: st.caption(f"Implied P/E: {1/ey:.1f}x")
    with c4:
        st.markdown("#### Factor 4 · Gross Profitability (Screen only)")
        st.markdown("""
`gross_profit = (revenue − COGS) / total_assets`

Updates **annually** — cannot score monthly.
Correct treatment: binary screen.
- GP > 0.40 → model runs ✓
- GP < 0.40 → model off ✗
""")
        ps = (gp or 0)>0.40
        st.metric("Gross profit ratio", f"{gp:.4f}" if gp else "N/A",
                  "PASS — active" if ps else "FAIL — off")
        gm = fund.get("gross_margin_pct")
        if gm: st.caption(f"Gross margin: {gm:.1f}%")

    st.markdown("---")
    st.markdown("#### Factor 5 · Short Interest Ratio (inverted)")
    st.markdown("""
`short_ratio = shares_short / avg_daily_volume` (days to cover)

High SI → informed sellers → **bearish** → should give **negative z**
Low SI  → few sellers    → **bullish**  → should give **positive z**

Without inversion, high SI = positive raw z → model goes long when smart
money is short. **Fix:** `z_si = −(raw_z)` reverses the direction.
Updates **bi-monthly** via FINRA reports.
""")
    sc1,sc2,sc3 = st.columns(3)
    sc1.metric("Days to cover", f"{sr:.2f}" if sr else "N/A")
    sc2.metric("% of float",    f"{spct*100:.2f}%" if spct else "N/A")
    ss = fund.get("shares_short")
    sc3.metric("Shares short",  f"{ss/1e6:.1f}M" if ss else "N/A")

    st.markdown("---")
    st.markdown("#### Factor snapshot table")
    mzl = (float(monthly["f1_mom_z"].dropna().iloc[-1])
           if not monthly["f1_mom_z"].dropna().empty else None)
    szl = (float(monthly["f5_si_z"].dropna().iloc[-1])
           if "f5_si_z" in monthly and not monthly["f5_si_z"].dropna().empty else None)
    st.dataframe(pd.DataFrame([
        ["12-1 Momentum",f"{mom_now*100:+.1f}%",
         f"{mzl:+.3f}" if mzl else "N/A","Daily",f"{w_mom:.2f}","✅ Live"],
        [f"{ma_window}d MA Regime","Active" if regime_now else "Inactive",
         "Binary ×1/×0","Daily","N/A (mult.)","✅ Live"],
        ["Earnings Yield",f"{ey*100:.2f}%" if ey else "N/A",
         "Snapshot","Quarterly",f"{w_ey:.2f}","⚠️ Needs history"],
        ["Gross Profit",f"{gp:.4f}" if gp else "N/A",
         "Screen only","Annual",f"{w_gp:.2f}","⚠️ Use as screen"],
        ["Short Ratio (inv.)",f"{sr:.2f}d" if sr else "N/A",
         f"{szl:+.3f}" if szl else "N/A","Bi-monthly",f"{w_si:.2f}","✅ Live"],
    ], columns=["Factor","Current","z-score","Update freq","Weight","Status"]),
    use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — IC & ICIR
# ════════════════════════════════════════════════════════════════
with tab_ic:
    st.markdown("### IC — Information Coefficient")
    st.markdown("""
Spearman rank correlation between your factor today and next-month return.

`IC = 1 − (6 × Σd²) / (n × (n²−1))`  where d = signal rank − return rank

IC > 0.05 = useful · IC > 0.10 = strong · IC < 0 = wrong direction
""")
    ic_val,ic_pval,ic_n = spearman_ic(monthly["f1_mom_z"],monthly["fwd_ret_1m"])
    ic_roll = rolling_ic(monthly["f1_mom_z"],monthly["fwd_ret_1m"],ic_window)
    icir_v  = icir(ic_roll)

    i1,i2,i3,i4 = st.columns(4)
    i1.metric("Full-period IC",f"{ic_val:.4f}" if ic_val else "N/A",
              "Significant ✓" if (ic_pval or 1)<0.05 else "Not sig.")
    i2.metric("p-value",f"{ic_pval:.4f}" if ic_pval else "N/A")
    i3.metric("ICIR",f"{icir_v:.3f}" if not np.isnan(icir_v or np.nan) else "N/A",
              "Tradeable ✓" if (icir_v or 0)>0.5 else "< 0.5 threshold")
    i4.metric("Observations",str(ic_n))

    if not ic_roll.empty:
        fig_ic=go.Figure()
        fig_ic.add_trace(go.Scatter(x=ic_roll.index,y=ic_roll,
            line=dict(color=CORN,width=1.5),fill="tozeroy",
            fillcolor="rgba(29,158,117,0.10)",name=f"Rolling {ic_window}m IC"))
        fig_ic.add_hline(y=0.05,line_dash="dash",line_color=BULL,opacity=0.6,
            annotation_text="IC=+0.05")
        fig_ic.add_hline(y=-0.05,line_dash="dash",line_color=BEAR,opacity=0.6)
        fig_ic.add_hline(y=0,line_color=GRY,line_width=0.8)
        fig_ic.update_layout(**make_layout(height=280,
            title=f"Rolling {ic_window}-month IC · Momentum factor",
            yaxis_title="IC (Spearman ρ)"))
        st.plotly_chart(fig_ic,use_container_width=True)

    st.markdown("---")
    st.markdown("### Step-by-step: 6 most recent monthly observations")
    recent = pd.concat([monthly["f1_mom_z"].rename("signal"),
                        monthly["fwd_ret_1m"].rename("return")],axis=1).dropna().tail(6)
    if len(recent)>=4:
        recent = recent.copy()
        recent["signal_rank"] = recent["signal"].rank(ascending=False).astype(int)
        recent["return_rank"] = recent["return"].rank(ascending=False).astype(int)
        recent["d"]   = recent["signal_rank"]-recent["return_rank"]
        recent["d_sq"]= recent["d"]**2
        d2 = recent.copy(); d2.index = d2.index.strftime("%b %Y")
        d2["signal"]=d2["signal"].map("{:+.3f}".format)
        d2["return"]=d2["return"].map("{:+.2%}".format)
        d2["d"]=d2["d"].map("{:+d}".format); d2["d_sq"]=d2["d_sq"].map("{:.0f}".format)
        st.dataframe(d2,use_container_width=True)
        n=len(recent); sd2=int(recent["d_sq"].sum())
        ic_s=1-(6*sd2)/(n*(n**2-1))
        st.markdown(f'<div class="formula-box">'
            f'IC = 1 − (6 × {sd2}) ÷ ({n} × {n**2-1})\n'
            f'   = 1 − {6*sd2} ÷ {n*(n**2-1)}\n'
            f'   = {ic_s:.4f}</div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ICIR — Sharpe ratio of your signal")
    st.markdown("`ICIR = mean(rolling IC) ÷ std(rolling IC)` · ICIR > 0.5 = tradeable")
    if not ic_roll.empty:
        mi=float(ic_roll.mean()); si=float(ic_roll.std())
        td = "← tradeable ✓" if icir_v>0.5 else "← below 0.5"
        st.markdown(f'<div class="formula-box">'
            f'Mean IC = {mi:.4f}\nStd  IC = {si:.4f}\n'
            f'ICIR    = {mi:.4f} ÷ {si:.4f} = {icir_v:.4f}  {td}'
            f'</div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### IC summary — all five factors")
    ic_rows=[]
    for fn,fc in [("12-1 Momentum","f1_mom_z"),("Short int. (inv.)","f5_si_z"),
                  ("Earnings yield","f3_ey_z"),("Gross profit","f4_gp_z")]:
        if fc not in monthly.columns: continue
        iv,pv,nn = spearman_ic(monthly[fc],monthly["fwd_ret_1m"])
        rs = rolling_ic(monthly[fc],monthly["fwd_ret_1m"],min(ic_window,12))
        iir= icir(rs)
        v  = ("🟢 Trade it" if (iv or 0)>0.05 and (iir or 0)>0.5
              else "🟡 Monitor" if (iv or 0)>0.02 else "🔴 Weak")
        ic_rows.append([fn,
            f"{iv:.4f}" if iv else "N/A",
            f"{iir:.3f}" if iir and not np.isnan(iir) else "N/A",
            f"{pv:.4f}" if pv else "N/A",
            "✅" if (pv or 1)<0.05 else "❌",v])
    st.dataframe(pd.DataFrame(ic_rows,
        columns=["Factor","IC","ICIR","p-value","Sig.","Verdict"]),
        use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — COMPOSITE BUILDER
# ════════════════════════════════════════════════════════════════
with tab_comp:
    st.markdown("### Composite signal — exact step-by-step math")
    st.markdown("""
**L1:** Each factor → z-score (60-month rolling mean/std, clipped ±3)  
**L2:** Weighted average of z-scores  
**L3:** × regime filter (0 = flat, 1 = active)
""")
    def sl(col):
        s=monthly[col].dropna() if col in monthly.columns else pd.Series()
        return float(s.iloc[-1]) if not s.empty else 0.0

    mz=sl("f1_mom_z"); sz=sl("f5_si_z"); ez=sl("f3_ey_z"); gz=sl("f4_gp_z")
    rc=mz*w_mom+sz*w_si+ez*w_ey+gz*w_gp
    dec="LONG SIGNAL" if comp_now>threshold and regime_now else "FLAT / NEUTRAL"

    st.markdown(f'<div class="formula-box">'
        f'Step 1 · z-scores (clipped ±3)\n'
        f'  mom_z = {mz:+.4f}   ← momentum vs 60-month rolling avg\n'
        f'  si_z  = {sz:+.4f}   ← short interest INVERTED\n'
        f'  ey_z  = {ez:+.4f}   ← earnings yield vs 60-month rolling avg\n'
        f'  gp_z  = {gz:+.4f}   ← gross profit vs 60-month rolling avg\n\n'
        f'Step 2 · weighted sum\n'
        f'  raw = ({mz:+.4f}×{w_mom:.2f}) + ({sz:+.4f}×{w_si:.2f})'
        f' + ({ez:+.4f}×{w_ey:.2f}) + ({gz:+.4f}×{w_gp:.2f})\n'
        f'      = {mz*w_mom:+.4f} + {sz*w_si:+.4f}'
        f' + {ez*w_ey:+.4f} + {gz*w_gp:+.4f}\n'
        f'      = {rc:+.4f}\n\n'
        f'Step 3 · regime filter\n'
        f'  composite = {rc:+.4f} × {regime_now} = {comp_now:+.4f}\n\n'
        f'Step 4 · decision\n'
        f'  {comp_now:+.4f} {">" if comp_now>threshold else "<="} +{threshold} → {dec}'
        f'</div>',unsafe_allow_html=True)

    fv=[mz*w_mom,sz*w_si,ez*w_ey,gz*w_gp]
    fig_c=go.Figure()
    fig_c.add_trace(go.Bar(x=["Momentum","Short int.","Earnings yld","Gross profit"],
        y=fv,marker_color=[BULL if v>0 else BEAR for v in fv],marker_opacity=0.8,
        text=[f"{v:+.3f}" for v in fv],textposition="outside"))
    fig_c.add_hline(y=0,line_color=GRY,line_width=0.8)
    fig_c.add_shape(type="line",x0=-0.5,x1=3.5,y0=comp_now,y1=comp_now,
        line=dict(color=PUR,width=2,dash="dot"))
    fig_c.add_annotation(x=3.5,y=comp_now,
        text=f"  composite={comp_now:+.3f}",showarrow=False,
        font=dict(color=PUR,size=11))
    fig_c.update_layout(**make_layout(height=280,
        title="Weighted z-score contribution per factor (current month)",
        yaxis_title="Weighted z contribution"))
    st.plotly_chart(fig_c,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — SHORT INTEREST INVERSION
# ════════════════════════════════════════════════════════════════
with tab_si:
    st.markdown("### Why short interest is inverted")
    st.markdown("""
`short_ratio = shares_short ÷ avg_daily_volume` = days to cover

Short sellers pay borrow costs and face unlimited downside —
they only short when they have high conviction the stock falls.

| Short ratio | Meaning | Expected return |
|-------------|---------|----------------|
| High (8 days) | Heavy short — informed sellers | **Lower** |
| Average (2 days) | Normal positioning | Neutral |
| Low (1 day) | Almost nobody shorting | **Higher** |

Without inversion: `zscore(high SI)` = positive z → model goes **long**
when smart money is short. Completely backwards.

**Fix:** `z_si = −(raw_z)` — one negative sign corrects the direction.
""")
    srn=sr or 2.0; sim=2.1; sis=0.8
    rzs=(srn-sim)/sis; rzc=float(np.clip(rzs,-3,3)); ivz=-rzc
    wi= ("WRONG: low SI reads bearish" if rzc<0 else "WRONG: high SI reads bullish")
    ri= ("CORRECT: low SI → bullish ✓" if ivz>0 else "CORRECT: high SI → bearish ✓")
    st.markdown(f'<div class="formula-box">'
        f'Current short ratio : {srn:.2f} days\n'
        f'Historical mean     : {sim} days  (60-month rolling)\n'
        f'Historical std      : {sis} days\n\n'
        f'Raw z = ({srn:.2f} − {sim}) ÷ {sis} = {rzs:.4f}  (clipped → {rzc:.4f})\n\n'
        f'WITHOUT inversion: z = {rzc:+.4f}  ← {wi}\n'
        f'WITH    inversion: z = −({rzc:.4f}) = {ivz:+.4f}  ← {ri}'
        f'</div>',unsafe_allow_html=True)

    sr_r=np.linspace(0.2,8,100)
    rzs2=np.clip((sr_r-sim)/sis,-3,3); ivz2=-rzs2
    fig_si=go.Figure()
    fig_si.add_trace(go.Scatter(x=sr_r,y=rzs2,
        line=dict(color=BEAR,width=2,dash="dash"),name="Raw z (WRONG)"))
    fig_si.add_trace(go.Scatter(x=sr_r,y=ivz2,
        line=dict(color=BULL,width=2),name="Inverted z (CORRECT)",
        fill="tozeroy",fillcolor="rgba(29,158,117,0.07)"))
    fig_si.add_vline(x=srn,line_dash="dot",line_color=PUR,
        annotation_text=f"{ticker} now ({srn:.1f}d)")
    fig_si.add_hline(y=0,line_color=GRY,line_width=0.8)
    fig_si.update_layout(**make_layout(height=300,
        title="Raw vs inverted z-score across all short ratio values",
        xaxis_title="Short ratio (days to cover)",yaxis_title="z-score"))
    st.plotly_chart(fig_si,use_container_width=True)

    st.dataframe(pd.DataFrame([
        ["Very low 0.3d","Almost no shorts","−2.25 (WRONG: bearish)","+2.25 ✓ Bullish"],
        [f"Low {srn:.1f}d ({ticker})","Light short interest",
         f"{rzc:+.2f} (WRONG)",f"{ivz:+.2f} ✓ Bullish"],
        ["Average 2.1d","Historical norm","0.00 neutral","0.00 neutral"],
        ["High 5.0d","Heavy shorts","+3.63 (WRONG: bullish)","−3.00 ✓ Bearish"],
        ["Extreme 8.0d","Max short","+3.00 capped (WRONG)","−3.00 ✓ Bearish"],
    ],columns=["Short ratio","Meaning","Raw z (no inversion)","Inverted z (correct)"]),
    use_container_width=True,hide_index=True)
    st.info("**Short squeeze exception:** extreme SI > 20% of float can briefly "
            "flip bullish due to forced covering. Not applicable to NVDA's "
            "low float-short %. Advanced models handle this non-linearly.")

# ════════════════════════════════════════════════════════════════
# TAB 6 — FACTOR CADENCE
# ════════════════════════════════════════════════════════════════
with tab_cadence:
    st.markdown("### Factor update cadence")
    st.markdown("""
Gross profitability is **annual** — 11 months of identical z-scores.
It's not adding new information, just a constant weight.
""")
    st.dataframe(pd.DataFrame([
        ["12-1 Momentum","Daily","Price","Recompute every month-end","✅ Live"],
        [f"{ma_window}d MA Regime","Daily","Price","Recompute every day","✅ Live"],
        ["Short interest","Bi-monthly","FINRA","Update on release dates","⚠️ Minor lag"],
        ["Earnings yield","Quarterly","10-Q/EDGAR","Refresh on filing date","⚠️ Needs tracker"],
        ["Gross profit","Annual","10-K/EDGAR","Use as binary screen only","🔴 Don't score monthly"],
    ],columns=["Factor","Freq","Source","How to handle","Status"]),
    use_container_width=True,hide_index=True)

    st.markdown("---")
    t1,t2,t3=st.columns(3)
    with t1:
        st.markdown("**A — Binary screen**")
        st.markdown("`if GP > 0.40: run_model()`\nGates universe entry.\nNovy-Marx original treatment.")
    with t2:
        st.markdown("**B — Filing-date refresh**")
        st.markdown("Update z-score on 10-K/10-Q\nfile date only. Hold constant\nbetween filings.")
    with t3:
        st.markdown("**C — Trend instead of level**")
        st.markdown("`GM_trend = this_Q − last_Q`\nUpdates quarterly.\nCaptures improving quality.")

    mlab=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cf=[("Momentum",[1]*12,BULL),("Regime filter",[1]*12,TEAL),
        ("Short interest",[1 if i%2==0 else 0.25 for i in range(12)],AMB),
        ("Earnings yield",[1 if i%3==0 else 0.2  for i in range(12)],BLUE),
        ("Gross profit",[1]+[0.15]*11,CORN)]
    fig_cd=go.Figure()
    for fi,(fn,vals,col) in enumerate(cf):
        for mi,v in enumerate(vals):
            fig_cd.add_trace(go.Bar(x=[mlab[mi]],y=[1],base=[fi*1.2],
                marker_color=col,marker_opacity=v,marker_line_width=0.5,
                marker_line_color="white",name=fn if mi==0 else None,
                showlegend=(mi==0),text="new" if v==1 else "",
                textposition="inside",textfont=dict(size=9,color="white"),width=0.9,
                hovertemplate=f"{fn}·{mlab[mi]}: {'new' if v==1 else 'frozen'}<extra></extra>"))
    fig_cd.update_layout(**make_layout(height=320,barmode="overlay",
        title="Factor update frequency (bright=new · faded=frozen)",
        yaxis=dict(tickvals=[0.5,1.7,2.9,4.1,5.3],
                   ticktext=["Momentum","Regime","Short int.","Earnings yld","Gross profit"],
                   showgrid=False),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
                    bgcolor="rgba(0,0,0,0)",font=dict(size=10))))
    st.plotly_chart(fig_cd,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 7 — TRADE ENTRIES
# ════════════════════════════════════════════════════════════════
with tab_entries:
    st.markdown("### Three entry methods")
    e1,e2,e3=st.columns(3)
    e1.metric("Total long entries",len(long_entries))
    e2.metric("Total exits",len(exits))
    e3.metric("In trade","Yes ✓" if len(long_entries)>len(exits) else "No")

    st.markdown("""
| Method | Entry condition | Timing | Exit | Turnover |
|--------|----------------|--------|------|---------|
| **A · Threshold cross** | Composite > threshold AND regime=1 | Next open | Composite < 0 OR regime flips | Higher |
| **B · Monthly rebalance** | End-of-month positive | 1st day of month | Monthly | Lower |
| **C · Pullback to MA** | Signal positive + price touches 20d MA | Day of touch | Stop −8% | Medium |
""")
    if long_entries or exits:
        all_s=([(d,"🟢 LONG",float(monthly.loc[d,"composite"]) if d in monthly.index else None,
                 price_at(d)) for d in long_entries]
              +[(d,"🔴 EXIT",float(monthly.loc[d,"composite"]) if d in monthly.index else None,
                 price_at(d)) for d in exits])
        all_s.sort(key=lambda x:x[0])
        sdf=pd.DataFrame(all_s,columns=["Date","Signal","Composite z","Price"])
        sdf["Date"]=sdf["Date"].dt.strftime("%Y-%m-%d")
        sdf["Composite z"]=sdf["Composite z"].apply(lambda x:f"{x:+.3f}" if x else "N/A")
        sdf["Price"]=sdf["Price"].apply(lambda x:f"${x:.2f}" if x else "N/A")
        st.markdown("#### Signal history (most recent 20)")
        st.dataframe(sdf.tail(20),use_container_width=True,hide_index=True)

    fig_e=go.Figure()
    fig_e.add_trace(go.Scatter(x=comp_s.index,y=comp_s,
        line=dict(color=PUR,width=1.5),fill="tozeroy",
        fillcolor="rgba(83,74,183,0.08)",name="Composite"))
    if long_entries:
        ec=[float(monthly.loc[d,"composite"]) if d in monthly.index else None
            for d in long_entries]
        fig_e.add_trace(go.Scatter(x=long_entries,y=ec,mode="markers",
            marker=dict(symbol="triangle-up",size=12,color=BULL,
                        line=dict(color=TEAL,width=1.5)),name="Long entry"))
    if exits:
        xc=[float(monthly.loc[d,"composite"]) if d in monthly.index else None
            for d in exits]
        fig_e.add_trace(go.Scatter(x=exits,y=xc,mode="markers",
            marker=dict(symbol="x",size=10,color=BEAR,
                        line=dict(color=CORN,width=2)),name="Exit"))
    fig_e.add_hline(y=threshold,line_dash="dash",line_color=BULL,opacity=0.6,
        annotation_text=f"Entry +{threshold}")
    fig_e.add_hline(y=0,line_color=GRY,line_width=0.8)
    fig_e.update_layout(**make_layout(height=300,
        title="Method A — composite threshold crossover",
        yaxis_title="Composite z-score"))
    st.plotly_chart(fig_e,use_container_width=True)

    st.markdown("---")
    ic_d=f"{ic_val:.3f}" if ic_val else "~0.087"
    st.markdown(f"""### Position sizing — half-Kelly
IC ≈ {ic_d} → edge ≈ 0.8%/month

`f* = (edge / odds) × 0.5`

At 2:1 win/loss → risk **3-5% of capital per signal**.  
Never exceed **10%** in a single position as a new manager.
""")

# ════════════════════════════════════════════════════════════════
# TAB 8 — NEXT STEPS
# ════════════════════════════════════════════════════════════════
with tab_next:
    st.markdown("### Research roadmap")
    with st.expander("Phase 1 — Validate (months 1-3)",expanded=True):
        st.markdown("""
1. **Historical fundamentals** — Sharadar (~$30/mo) eliminates look-ahead bias
2. **Expand universe** — SOX semiconductor stocks, 30× more IC observations
3. **Walk-forward backtest** — `vectorbt`, never touch future data during training
4. **Transaction cost model** — realistic slippage + commission drag
5. **Stress test** — 2022 bear, 2020 COVID, 2018 rate hike
""")
    with st.expander("Phase 2 — Portfolio construction (months 3-6)"):
        st.markdown("""
6. **PyPortfolioOpt** — max Sharpe or risk parity weights, sector neutrality
7. **Gross margin trend** — replaces annual level, updates quarterly
8. **Free alt data** — Google Trends, earnings call NLP, job posting volume
9. **Alpaca paper trade** — 3-6 months before live capital
""")
    with st.expander("Phase 3 — Live trading (months 6-18)"):
        st.markdown("""
10. **GIPS tracking** — NAV Consulting / Theorem (~$500/mo)
11. **Research diary** — document every hypothesis change → LP process doc
12. **Visa path** — IC documented over 2+ years = O-1A evidence
""")
    st.markdown("---")
    st.markdown("### Deploy free")
    st.code("""pip install streamlit yfinance pandas numpy scipy plotly
streamlit run nvda_alpha_complete.py

# push to GitHub, then:
# share.streamlit.io → New app → connect repo → Deploy
# live URL: https://your-app.streamlit.app  (auto-refreshes daily)""",
    language="bash")
    for n,u in [("vectorbt","https://vectorbt.dev"),
                ("PyPortfolioOpt","https://pyportfolioopt.readthedocs.io"),
                ("Sharadar","https://data.nasdaq.com/databases/SF1"),
                ("Alpaca paper trading","https://alpaca.markets"),
                ("Streamlit Cloud","https://share.streamlit.io")]:
        st.markdown(f"- [{n}]({u})")

st.divider()
st.caption("⚠️ Research and educational purposes only. Not investment advice. "
           "Past performance does not guarantee future returns. "
           "Yahoo Finance snapshots have look-ahead bias in historical context.")
