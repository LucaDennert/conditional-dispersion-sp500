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

Complete. Paper: `paper/main.pdf`.

- Correlation risk premium (3M): 8.5 correlation points on average, positive in 80% of months (t = 8.4).
- Stylized dispersion trade (held to expiry, entry spreads): 0.97 vol pts per tranche, Sharpe 0.56, alpha robust to FF5 + momentum + ΔVIX.
- Same trade on listed contracts with round-trip costs: −0.62 vol pts per tranche (gross +1.33, costs 1.94).
- Pre-registered timing rule (part 3) and final trade (part 4, ridge / gradient boosting): pass in sample (1996–2010), fail out of sample (2011–2025). Both **dropped** under the pre-registered kill test (`results/kill_test_report.txt`).
