# Conditional Dispersion on the S&P 500

Research project: does conditioning a dispersion trade (short index volatility, long constituent volatility) on the implied-vs-forecast correlation premium and on the volatility regime improve on opening a tranche every month?

- **Rules first:** see [`PREREGISTRATION.md`](PREREGISTRATION.md), committed before any result was computed.
- **Data:** OptionMetrics IvyDB US and CRSP via WRDS (1996–2025). The data are licensed and are not included; `notebooks/01_estrazione_ivydb_us.ipynb` reproduces the extraction for anyone with WRDS access.

See [`ROADMAP.md`](ROADMAP.md) for the paper plan and current status.

## Structure

```
notebooks/   01 extraction · 02 signals · 03 forecast · 04 backtest · 05 kill test
src/         reusable code (implied/realized correlation, pricing, P&L engine)
results/     figures and tables (aggregated only)
```

## Status and main results

Paper: `paper/main.pdf`.

- Correlation risk premium (3M): 8.5 correlation points on average (95% CI 6.5–10.5), positive in 80% of months.
- Basic trade on listed contracts (short SPX straddle, long top-20 straddles, delta-hedged, held to expiry): 1.35 vol pts per tranche gross, 0.31 net (not significant). Exiting 10 days before expiry turns it negative.
- Correlation-weighted long leg (x sqrt(IC)): 0.57 net (CI 0.21–0.94), ladder Sharpe 0.51, max drawdown a third of the vega-neutral version; crash losses 1.6 (2008) and 3.0 (2020) vs 25 and 24 for a short SPX straddle.
- Timing (richness vs HAR forecast net of entry cost, walk-forward) and walk-forward ML do not beat trading every month.
- Original pre-registered timing test (2011–2025): dropped (`results/kill_test_report.txt`).

Main scripts: `src/listed_engine.py` (P&L engine; `expiry` mode), `src/basic_trade.py`, `src/richness.py`, `src/ml_walkforward.py`.
