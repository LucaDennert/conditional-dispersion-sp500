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

*(none yet)*
