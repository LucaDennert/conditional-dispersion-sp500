# Roadmap and status (handoff notes)

*Last update: 2026-11-12. Read this first when resuming work.*

## Paper structure (agreed with Luca, all on SPX)

1. **The correlation risk premium, 1996–2025.** Implied vs subsequently realized correlation, how it varies over time and across regimes.
2. **The standard dispersion trade.** Unconditional vega-neutral straddle dispersion: performance, costs, and tests against standard risk factors.
3. **The initial trade idea.** The pre-registered conditional rule (`PREREGISTRATION.md`), evaluated once on 2011–2025 with the kill test.
4. **The final trade.** A more complex model built on what parts 1–3 show.
   - *Open point:* to stay honest, part 4 must be specified and committed **before** the part-3 test is run, so both are judged on 2011–2025 once.
   - Otherwise part 4 can only be reported as exploratory.

## Status

| Part | Done | To do |
|---|---|---|
| Data | CRSP S&P 500 members/weights/returns; IvyDB standardized options (SPX + members, 30/91d); SPX/VIX; zero curve; SPX 91d vol surface; Fama-French 5 + momentum (daily). Validation: `src/validate_data.py` | Listed option prices with real bid/ask (`opprcd`): needs Luca's personal WRDS account (Python/JupyterHub). Member vol surfaces (variance-swap strikes, Linders-Schoutens IC). Cboe COR3M for cross-check |
| 1 | Monthly IC, RC (21/63/252d), forward RC, VP, VIX: `src/signals.py`. IC ≈ 0.40 vs forward RC ≈ 0.30–0.33 | Write-up and figures; premium by regime; Cboe cross-check; Marshall (2009) replication 2005–07; Linders-Schoutens (2014) robustness; exact-vs-Taylor IC check already done (≤ 0.55 pts) |
| 2 | Exact delta-hedged straddle P&L on real paths, top-50 and all members, costs, factor regressions, SPX model-free variance strike: `src/simple_dispersion*.py`, `results/SIMPLE_DISPERSION.md` | Real bid/ask by year (decimalization 2001, penny pilot 2007); pre/post-2001 split |
| 3 | Pre-registration committed (7034421); HAR forecast estimated and frozen on 1996–2010 (`results/forecast_frozen.json`) | Cost estimate *c* from real spreads; P&L engine on listed contracts; ladder, benchmark at equal vega, variants, kill test |
| 4 | — | Design and pre-register before the part-3 test |
| Paper | — | Format and author list to be decided by Luca |

## How to rebuild the working data

The raw files are in a local `data_raw/` folder (licensed, not in git). Stage them into the session, then:

```
python -m src.build_data RAW_DIR DATA_DIR
python -m src.validate_data RAW_DIR DATA_DIR
python -m src.signals DATA_DIR
python -m src.forecast DATA_DIR results
python -m src.simple_dispersion DATA_DIR results
python -m src.simple_dispersion_report DATA_DIR RAW_DIR results
```

Intermediate files are pickles: this environment has no parquet engine.

## WRDS access notes

- **Library access.** Through the Bocconi OpenAthens proxy, in the desktop app's built-in browser. Web query forms only.
  - Code lists are limited to 4096 characters: upload a file instead.
  - Date pickers: set the end date before the start date.
  - Downloads can stay as hidden temp files in Downloads: check them with `gzip -t`.
  - The library account is shared, so other users' queries appear in the query list.
- **IvyDB US** covers 1996-01-04 to 2025-08-29. IvyDB Europe is not licensed.
- **Personal account**: active since 2026-11-13. Needed for Python/JupyterHub, i.e. for the listed-option extraction (`notebooks/01_estrazione_ivydb_us.ipynb`, two-step contract selection).
