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

| Part | Done | Open |
|---|---|---|
| Data | CRSP members/weights/returns; IvyDB standardized options, vol surfaces (SPX and members), listed option quotes for the selected contracts; SPX/VIX; zero curve; Fama-French 5 + momentum | Cboe COR3M cross-check (optional) |
| 1 | Correlation risk premium 1996–2025, by period, maturity and VIX regime: `src/premium.py` | Marshall (2009) replication; Linders-Schoutens robustness with member smiles (optional) |
| 2 | Stylized trade with measured entry spreads, factor regressions: `src/simple_dispersion*.py`, `results/SIMPLE_DISPERSION.md` | — |
| 3 | Listed-contract engine (`src/listed_engine.py`, `src/tranches.py`), cost threshold c, kill test run once on 2011–2025 (`src/kill_test.py`, `results/kill_test_report.txt`): **DROP** | — |
| 4 | Pre-registered (amendment 2), ridge + GBM estimated on 1996–2010 (`src/part4_model.py`, `results/part4_frozen.json`), tested once: **DROP** | — |
| Paper | `paper/` (LaTeX), complete draft | Review |

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
