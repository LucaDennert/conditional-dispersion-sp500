# Conditional Dispersion on the S&P 500

Research project: does conditioning a dispersion trade (short index volatility, long constituent volatility) on the implied-vs-forecast correlation premium and on the volatility regime improve on opening a tranche every month?

- **Rules first:** see [`PREREGISTRATION.md`](PREREGISTRATION.md), committed before any result was computed.
- **Data:** OptionMetrics IvyDB US and CRSP via WRDS (1996–2025). The data are licensed and are not included; `notebooks/01_estrazione_ivydb_us.ipynb` reproduces the extraction for anyone with WRDS access.

## Structure

```
notebooks/   01 extraction · 02 signals · 03 forecast · 04 backtest · 05 kill test
src/         reusable code (implied/realized correlation, pricing, P&L engine)
results/     figures and tables (aggregated only)
```

## Status

- [x] Pre-registration drafted
- [x] Data extraction (CRSP, IvyDB standardized options, SPX/VIX, zero curve)
- [ ] Listed option histories for the backtest (needs WRDS programmatic access)
- [x] Signals and forecast (estimation sample 1996–2010, frozen in `results/forecast_frozen.json`)
- [ ] Backtest and kill test (test sample 2011–2025)
