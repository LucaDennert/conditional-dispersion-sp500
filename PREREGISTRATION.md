# Pre-registration — Conditional Dispersion on the S&P 500

**Status:** APPROVED and committed on 2026-11-12, before any signal, forecast or P&L was computed. Only raw data downloads and data-quality checks precede this commit.
**Authors:** Luca Dennert and team
**Date of draft:** 2026-11-12

This document fixes, before looking at any result, the data, the definitions, the trading rule, the way every free parameter is chosen, and the criteria that decide whether the strategy is kept or dropped. The git commit timestamp of this file is the proof that the rules came first. Any later change is logged in the *Amendments* section at the bottom, with date and reason, and results under the original rules are still reported.

---

## 1. Hypothesis

Index options price more correlation than the index constituents subsequently deliver (Driessen, Maenhout & Vilkov, 2009). A dispersion trade (short index volatility, long constituent volatility) harvests this premium but loses when correlation spikes. **Hypothesis:** opening dispersion tranches only when implied correlation is rich relative to a forecast of realized correlation over the life of the trade, and the volatility regime is calm, improves risk-adjusted returns and reduces drawdowns relative to opening a tranche every month.

## 2. Data

| Item | Source | Period |
|---|---|---|
| Index and constituent option data (standardized options, option prices, bid/ask, greeks) | OptionMetrics IvyDB US via WRDS | 1996-01-04 to 2025-08-29 |
| Index prices, VIX | IvyDB US `secprd` (SPX secid 108105, VIX) | same |
| Zero-coupon curve | IvyDB US `zerocd` | same |
| S&P 500 membership, market cap, daily returns | CRSP S&P 500 constituents (CIZ, `dsp500list_v2`) | same |
| CRSP–OptionMetrics link | WRDS `wrdsapps.opcrsphist`, score = 1 | same |

Raw data are licensed and are **not** included in the repository.

## 3. Sample split

- **Estimation sample:** 1996-01-04 to 2010-12-31. All forecast weights, cost estimates and thresholds are computed here and then frozen.
- **Test sample:** 2011-01-01 to 2025-08-29. Every Keep/Drop criterion is judged on this period only. The test sample is used once.

## 4. Definitions

**Dates.** Signals are computed and tranches opened on the first trading day of each month (entry date *t*).

**Index weights.** wᵢ,t = CRSP market cap of member *i* at the previous month end, divided by the sum over members with valid data. Members without a valid implied volatility on date *t* are dropped and weights renormalized; if the dropped members exceed 10% of index weight, the date is excluded and logged.

**Implied volatility.** 91-day ATM-forward implied volatility from IvyDB standardized options, average of call and put. σ_I for SPX, σᵢ for members.

**Implied correlation (IC).**
IC_t = (σ_I² − Σ wᵢ² σᵢ²) / (Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ), using all S&P 500 members.

**Realized correlation (RC).** Same formula with annualized realized volatilities from daily log returns (CRSP total returns for members; SPX returns from IvyDB) over a trailing window: RC21 (1 month), RC63 (3 months), RC252 (1 year, the "long-run" term).

**Holding-period realized correlation (forecast target).** RC over the 53 trading days following *t*, matching the typical life of a tranche (about 63 trading days to expiry minus the 10-day exit buffer).

**Volatility premium (VP).** 91-day ATM implied vol of SPX minus 21-day realized vol of SPX.

## 5. Correlation forecast

Fisher-z transformed HAR regression, estimated by OLS on monthly entry dates in the estimation sample:

z(RC_fwd,t) = a + b₁ z(RC21_t) + b₂ z(RC63_t) + b₃ z(RC252_t) + b₄ z(IC_t) + ε_t,  z(x) = atanh(x)

Standard errors: Newey-West with 3 lags (tranches overlap for about 2.5 months). The forecast RCfcst_t is the back-transformed fitted value. RMSE is measured in correlation units on the estimation sample. Coefficients are frozen after estimation.

Reported alongside, not used for decisions: pure RC21, pure RC63, and the regression without IC.

**Correlation premium:** CP_t = IC_t − RCfcst_t.

## 6. Trading rule (primary specification)

Open a new tranche on entry date *t* if and only if:

1. **CP_t > c + k · RMSE**, where *c* is the average round-trip cost of a tranche expressed in correlation units, estimated on the estimation sample from the bid/ask spreads of the contracts actually selected, and **k = 0.5**;
2. **VIX_t < the 80th percentile of VIX closes in the estimation sample.**

VP is **not** part of the primary rule (it is tested as a variant, section 9).

## 7. Trade construction

- **Contracts.** On each entry date: for SPX and for each of the **top 50 members by market cap at the previous month end**, the standard monthly expiry closest to 91 calendar days (between 60 and 130 days), and the strike closest to spot with both call and put quoted (bid > 0, ask > bid). If several contracts qualify, the one with the tighter spread.
- **Legs.** Short SPX straddle; long straddles on the 50 members.
- **Sizing.** Vega-neutral: total long-leg vega equals short-leg vega. Long-leg vega is split across the 50 members in proportion to their index weights (renormalized within the 50).
- **Execution prices.** Buy at the ask, sell at the bid, at entry and exit.
- **Delta hedging.** Daily at the close, using IvyDB option deltas: SPX leg hedged with the index (futures proxy), member legs with the stocks. Hedging cost: 0.5 bp of traded notional for the index, 1 bp for stocks.
- **Ladder.** At most three live tranches; each tranche uses one-third of the vega budget.
- **Exit.** 10 trading days before expiry, at bid/ask.
- **Missing quotes.** A missing option price is carried forward for at most 3 trading days; after that the leg is closed at the last available price and the event is logged.

## 8. Benchmark

The same ladder, opening a tranche **every month** with identical construction and exit, **rescaled to the same average vega exposure** as the conditional strategy over the test sample (so the drawdown comparison is not driven by lower exposure).

## 9. Variants (reported, not decisive)

1. CP filter only (no VIX filter).
2. Primary rule plus VP > 0.
3. Primary rule plus stops: exit early if VIX closes above its 95th percentile (estimation sample), or if RC21 exceeds the IC sold by 0.15.
4. Long leg weighted by √IC instead of vega-neutral.
5. Top-30 and top-100 baskets.
6. Costs doubled.
7. Continuous sizing: tranche vega ∝ max(CP − c, 0), capped at one-third of the budget.

Total number of strategy variants evaluated on the test sample: **8** (primary + 7). The deflated Sharpe ratio (Bailey & López de Prado, 2014) is reported using this count.

## 10. Kill test (decided before looking)

All criteria on the test sample (2011-01-01 to 2025-08-29), monthly P&L per unit of vega budget.

**Keep** if all hold:
1. The annualized Sharpe ratio of the primary rule exceeds the benchmark's, and the one-sided block-bootstrap p-value of the difference (blocks of 3 months, 10,000 draws) is below 0.10;
2. Maximum drawdown of the primary rule is at most two-thirds of the rescaled benchmark's;
3. Criteria 1 (point estimate only) and 2 hold for every k in {0, 0.25, 0.5, 0.75, 1.0}.

**Drop** if the primary rule adds nothing out of sample (Sharpe difference not significant and drawdown not reduced by one-third), or if the sign of the Sharpe difference flips across the values of k.

Any other outcome is reported as **inconclusive**.

## 11. Reporting

Regardless of outcome: P&L by leg (index vs members) and by component (vega, gamma/theta, costs); results by sub-period (2011–2014, 2015–2019, 2020–2025); turnover and number of tranches opened; all variants.

## 12. Known limitations, accepted in advance

- IvyDB data end on 2025-08-29; later periods (including April 2026) are not in the test sample.
- Weights use CRSP total market cap, not S&P float-adjusted cap.
- The 50-name basket does not replicate the index; tracking error is measured, not eliminated.
- Single-stock options are American; early exercise is ignored, which can matter around ex-dividend dates.
- Hedging and option execution costs are modeled, not observed fills.

---

## Amendments

### Amendment 1 — 2026-11-13, before any test-sample signal outcome or P&L is computed

Written and committed before running the listed-contract P&L engine on any date and before computing any trading signal outcome on 2011–2025. Disclosure: on 2026-11-12 the out-of-sample RMSE of the correlation *forecast* on 2011–2025 was computed (`results/forecast_estimation.txt`); no trade, P&L or rule outcome was computed on the test sample.

**A1. Cost threshold in correlation units (clarifies section 6).** Round-trip cost of a tranche per unit of tranche vega: cost_t = h_SPX + Σ wᵢ hᵢ, with h = (half-spread at entry + half-spread at exit) / straddle vega at entry, for the contracts actually selected, exit quote 10 trading days before expiry. A change dρ in the correlation priced into the index moves index vol by dσ_I = C·dρ/(2σ_I), C = (Σwσ)² − Σw²σ². Hence c_t = (cost_t/100)·2σ_I/C and **c = mean of c_t over 1996–2010 entries = 0.0937** (`results/cost_threshold_frozen.json`, `src/cost_corr.py`). Frozen.

**A2. Engine details (clarifies section 7).** Options marked daily at mid; entry and exit at ask (buy) / bid (sell). Straddle delta = IvyDB call delta + put delta, carried forward when missing. Hedge P&L per day: −Δ·S·(R − r·dt), with R the CRSP total return for stocks and the SPX price return plus the dividend yield implied by the 91-day forward for the index (futures proxy); option premium financed at the 91-day zero rate. Hedge costs as in section 7, charged on |change in delta| × notional, including the initial hedge and the unwind. The missing-quote rule applies to the straddle as a whole. P&L is expressed per unit of SPX-leg vega at entry; members with no selected contract are dropped and the long-leg weights renormalized among the names traded.

**A3. Variants (clarifies section 9).**
- Variant 3 (stops): checked on daily closes; when triggered the whole tranche is closed at that close at bid/ask. RC21 is the trailing 21-day realized correlation of the top-50 basket weights.
- Variant 4 (√IC): long-leg vega = √IC_t × short-leg vega.
- Variant 5: contracts were selected for the top 50 only, so the **top-100 basket is not feasible** and is dropped; only the top-30 basket (top 30 among the 50, weights renormalized) is reported. Trial count stays at 8 (conservative).
- Variant 7 (continuous sizing): tranche size = min(max(CP − c, 0) / (0.5·RMSE), 1) × one-third of the vega budget, i.e. full size when the premium net of costs reaches the primary threshold.

### Amendment 2 — 2026-11-13: the final trade (part 4), pre-registered before the test of part 3 is run

**Idea.** Instead of a single threshold on the correlation premium, forecast the net P&L of the tranche directly from several signals observed at entry, and size the tranche in proportion to the forecast.

- **Target.** Net P&L (vol points per unit of tranche vega) of the unconditional tranche from the listed-contract engine (section 7, amendment A2). Estimation uses tranches whose exit date is on or before 2010-12-31.
- **Predictors (8, standardized with estimation-sample mean and s.d.):** CP_t (section 5); IC 3M; IC slope (IC 3M − IC 1M); VIX; VP (SPX 91-day ATM IV − 21-day RV); SPX skew premium (91-day model-free variance-swap vol strike − ATM IV, from IvyDB `vsurfd`); cross-sectional dispersion of member implied vols (index-weighted s.d. of 91-day ATM IV across members); measured round-trip cost of the tranche cost_t (A1).
- **Primary model: ridge regression.** Penalty λ chosen on the grid 10^{−2,−1.5,…,3} by expanding-window time-series cross-validation inside 1996–2010 (first fold trains on entries up to 2001-12, then adds one year per fold; validation = next year; 3-month gap between train end and validation start to avoid overlap). Refit on all estimation tranches with the chosen λ and freeze.
- **Secondary model (reported, not decisive): gradient boosting**, scikit-learn `GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)`, same target, predictors and estimation sample.
- **Rule.** Open a tranche only if the predicted net P&L p̂_t > 0. Size = min(p̂_t / m, 1) × one-third of the vega budget, where m is the 90th percentile of the in-sample fitted values (estimation sample). No other filter.
- **Evaluation (2011–2025, used once, together with part 3).** The kill test of section 10 applied to the ridge rule against the vega-matched unconditional benchmark (criteria 1–2; the k-robustness criterion is replaced by robustness to m ∈ {75th, 90th, 100th percentile} and to λ one grid step either side). In addition, the Sharpe difference against the part-3 primary rule is reported with the same bootstrap. Deflated Sharpe ratios use **10 trials** (8 of part 3 + 2 of part 4).

