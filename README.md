# NVDA Alpha Research Platform

Five-factor systematic alpha dashboard with IC/ICIR diagnostics,
composite signal builder, short interest inversion explainer,
and trade entry signals. Auto-refreshes data every 24 hours.

## What's included

| Tab | Content |
|-----|---------|
| Price & Signals | Live price chart, 200-day MA, composite signal, entry/exit markers, volume |
| Five Factors | Each factor explained with live values, z-scores, update cadence |
| IC & ICIR | Rolling IC chart, step-by-step Spearman calculation, ICIR, factor IC table |
| Composite Builder | Exact weighted z-score math shown live, factor contribution chart |
| Short Interest Inversion | Why SI is negated, raw vs inverted z-score visualization |
| Factor Update Cadence | Which factors freeze and how to handle them correctly |
| Trade Entry Methods | Three entry strategies, signal history table, position sizing |
| Next Steps | Research roadmap phases 1-3, deployment guide |

## Run locally

```bash
pip install -r requirements.txt
streamlit run nvda_alpha_complete.py
```

Opens at http://localhost:8501

## Deployed website on Streamlit Cloud

[https://your-app-name.streamlit.app`](https://kmycapital.streamlit.app/)

Data auto-refreshes every 24 hours. Force refresh via the sidebar button.

## Change the ticker

In the sidebar, type any ticker — NVDA, AMD, MSFT, AAPL, etc.
The entire model recalculates instantly for that stock.

## Important notes on data quality

- **Earnings yield and gross profitability** are point-in-time snapshots from
  Yahoo Finance. They have look-ahead bias in historical context. For proper
  backtesting, replace with Sharadar: https://data.nasdaq.com/databases/SF1
- **Gross profitability** updates annually — the app uses it as a binary
  screen (GP > 0.40 = model runs) rather than a live scoring factor
- **Short interest** is inverted: `z = -(raw_z)` because high short interest
  is bearish, so the raw z-score direction is backwards without the negation
