# The plain dispersion trade on the S&P 500, 1996–2025

*Descriptive analysis, separate from the pre-registered test. Code: `src/simple_dispersion.py`, `src/simple_dispersion_report.py`. Full output: `simple_dispersion_report.txt`.*

## The trade

On the first trading day of every month, open a 3-month tranche:

- **Short** an ATM-forward straddle on SPX.
- **Long** ATM-forward straddles on the 50 largest S&P 500 members (market cap at the previous month end), vega split by index weight.
- **Vega-neutral:** the same vega on the two legs.
- Each straddle is **delta-hedged daily** at its inception implied vol and **held to expiry** (63 trading days). Three tranches are live at any time.

**How P&L is measured.** Entry prices come from market implied vols (OptionMetrics 91-day ATM-forward). The hedge and the payoff are computed on the **actual price paths** (CRSP prices, splits, dividends; SPX), with financing at the OptionMetrics zero curve. P&L is in **vol points per unit of vega**. For example, with $100k vega per leg, 1 vol point is $100k per tranche.

**Costs.** Half bid/ask paid at entry: 0.15 vol points on SPX and 0.50 on single stocks (desk-level assumptions). Sensitivity is shown below. Real spreads will replace these once the listed-option data are extracted.

## Results

| Period | Tranches | Mean, net (vol pts) | t-stat (NW) | Sharpe (ann.) | Hit rate | Worst | Best |
|---|---|---|---|---|---|---|---|
| 1996–2007 | 143 | 2.13 | 4.55 | 1.17 | 71% | −5.7 | 12.0 |
| 2008–2009 | 24 | 1.22 | 0.83 | 0.41 | 58% | −8.8 | 16.3 |
| 2010–2019 | 120 | 0.25 | 0.97 | 0.24 | 60% | −9.2 | 7.2 |
| 2020–2025 | 65 | 1.04 | 1.56 | 0.56 | 55% | −4.3 | 18.3 |
| **Full sample** | **352** | **1.23** | **4.57** | **0.69** | **64%** | −9.2 | 18.3 |

Gross of costs, the full-sample mean is 1.88 vol points per tranche and the Sharpe ratio is 1.05. Using all ~500 members instead of the top 50 changes little (net mean 1.18, Sharpe 0.70).

![cumulative](fig_simple_dispersion_cumulative.png)

**Where the money comes from**

- **The index leg.** Short SPX straddles earn +2.1 vol points per tranche: the index variance risk premium.
- **The single-stock leg.** Long single-stock straddles lose −0.3 vol points on average, roughly break-even: single-stock implied vol is close to fair.
- **Most of the gain came before 2003.** Since then the trade has been nearly flat after costs, except for gains in the 2008 and 2020 crises.

**Costs matter.** The edge after 2010 is of the same order as the costs:

| Cost multiple (1× = assumptions above) | 0× | 0.5× | 1× | 2× | 3× |
|---|---|---|---|---|---|
| Mean (vol pts) | 1.88 | 1.55 | 1.23 | 0.58 | −0.07 |
| Sharpe | 1.05 | 0.87 | 0.69 | 0.32 | −0.04 |

## Is it just a factor bet?

The dependent variable is net P&L per tranche. Factors are summed over each tranche's life. Standard errors are Newey-West with 2 lags.

| Model | Alpha (vol pts/tranche) | t(alpha) | R² |
|---|---|---|---|
| CAPM | 1.60 | 5.84 | 0.12 |
| Fama-French 5 + momentum | 1.60 | 5.53 | 0.18 |
| FF5 + momentum + ΔVIX | 1.58 | 5.71 | 0.18 |
| Short SPX vol only | 1.33 | 3.64 | 0.01 |
| All of the above | 1.30 | 3.84 | 0.20 |

- **Alpha survives every specification.** Standard equity factors explain less than a fifth of the variance.
- **Negative market beta.** Dispersion P&L is higher when the market falls. A vega-neutral book is net long the *level* of volatility, because single-stock vols move more in points than index vol. This bias offsets the correlation spike in sell-offs, which is why 2008 and 2020 were profitable tranches rather than losses.
- **It is not short index vol in disguise.** The tranche P&L is essentially uncorrelated with the short-SPX-straddle P&L (R² 0.01).

![scatter](fig_dispersion_vs_short_vol.png)

## Index skew: straddles vs variance swaps

The model-free 3-month SPX variance-swap strike, computed from the OptionMetrics smile in the same way as the VIX, averages 20.3 against 18.6 for ATM vol. That is 1.8 vol points of skew premium.

- A short SPX variance swap struck at the true model-free strike earns +1.7 vol points per tranche.
- Struck at ATM vol, it would earn about zero.

So a variance-swap version of the dispersion trade needs proper model-free strikes on both legs, which requires the single-stock smiles. The straddle version above is not affected, because straddles are priced at ATM vol.

## Caveats

- Entry uses OptionMetrics interpolated ATM-forward vols, not traded quotes. Bid/ask spreads are assumed (see above).
- Single-stock options are American; early exercise is ignored.
- The hedge is at the close, without hedging costs.
- Exact delta-hedged P&L differs materially from the common shortcut "vega × (realized − implied vol)": the correlation between the two is 0.64.
