# SARIMAX — Seasonal ARIMA with Exogenous Regressors

> **TL;DR.** SARIMAX = **AR** (past values) + **I** (differencing to remove trend) + **MA** (past errors) + **Seasonal** versions of all three + **eXogenous** regressors. Notation `(p,d,q)(P,D,Q)[s] + X`. It's the classical, interpretable workhorse for a *single* time series with trend, one seasonal cycle, and known external drivers. The non-negotiables: make the series **stationary** first, read orders from **ACF/PACF**, only use exogenous variables whose **future values you'll have at forecast time**, and validate with **walk-forward** CV — never k-fold.

**Where it fits:** Univariate forecasting (demand, sales, traffic) with modest data and a need for interpretable coefficients + uncertainty intervals. For many series, multiple seasonalities, or nonlinear drivers, move to the alternatives in §10.
**Prereqs:** [[stationarity]], [[autocorrelation-acf-pacf]], [[linear-regression]], [[time-series-cross-validation]].

```
Running example — monthly ice-cream sales (3 yrs), exog = avg temperature (°C):
  has TREND (sales grow yearly) + SEASONALITY (summer spike) + EXOG (temp drives spikes) + NOISE.
  SARIMAX models all four at once.
```

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Residual Diagnostics](#7-residual-diagnostics)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

```
AR (p) → uses past VALUES        I (d) → DIFFERENCES away the trend     MA (q) → uses past ERRORS
+ Seasonal (P,D,Q,s): the same three, but at seasonal lags (s, 2s, 3s …)
+ eXogenous (X): external drivers added as linear regressors (temperature, holidays, promos)
```
- **AR = pattern/momentum** ("last two months predict this month"). **MA = error correction** ("I overshot by 50 last month → adjust down"). Together ARMA captures both the signal and its mistakes.
- **Why stationarity first:** AR/MA learn *fixed* weights. If the mean drifts (trend), fixed weights chase a moving target → systematically wrong. Differencing makes the structure stable enough to learn.

---

## 2. The Formal Core

```
AR(p):     yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + … + φₚyₜ₋ₚ + εₜ          (free weights via OLS — unlike SES's forced α(1−α)ᵏ decay)
MA(q):     yₜ = c + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + … + θ_qεₜ₋_q
SARIMAX:   yₜ = [SARIMA structure] + β₁X₁ₜ + β₂X₂ₜ + … + εₜ
```
- `β₁=8.5` for temperature ⇒ *"+1 °C → +8.5 sales units, holding all else constant."*
- **Notation:** `(p,d,q)` non-seasonal · `(P,D,Q)` seasonal mirrors · `s` season length · `X` exog.
- **Stationarity** = constant mean (no trend), constant variance, constant autocorrelation (no seasonality).
- **Order-reading rule (memorize):** **PACF cutoff → p** (and P at seasonal lags); **ACF cutoff → q** (and Q at seasonal lags). *Mnemonic: PACF→P→AR→p.*

---

## 3. How It Works

### Stationarity & testing (do this before anything)
```
ADF test:  H0 = NON-stationary.  p<0.05 → reject → STATIONARY ✅   p>0.05 → difference.
KPSS test: H0 = stationary (the COMPLEMENT). Run both: want ADF to reject AND KPSS to NOT reject.
Always pair the tests with a visual rolling-mean / rolling-std check.
```

### `I` — differencing (d): remove the trend
```
sales = [120,135,200,310,480]
d=1:   [15, 65, 110, 170]      → linear trend gone ✅   (statsmodels cumsums predictions back to scale)
d=2:   [50, 45, 60]            → quadratic trend gone, but d=2 is a RED FLAG → try a log/Box-Cox transform first
Rule: d is almost always 0 or 1. Choose the smallest d that passes ADF (+KPSS).
```
A **log or Box-Cox transform** stabilizes *variance* (when spread grows with level) — differencing only stabilizes the *mean*.

### Seasonal part `(P,D,Q,s)`
After `d=1` a spike remaining at lag `s` (e.g. 12) = seasonality plain ARIMA can't catch. Seasonal terms operate at lags `s,2s,3s…`: **seasonal-difference** `yₜ − yₜ₋ₛ` (D, removes the repeating cycle), seasonal-AR `P` (this Jan depends on prior Jans), seasonal-MA `Q`. Pick `s` from **domain knowledge** (monthly→12, daily→7, weekly→52), confirm with ACF spikes at `s,2s`. Keep `d+D ≤ 2`.

> **Why SARIMA beats just using a huge `p`:** capturing yearly cycle with `p=12` burns 12 AR coefficients and overfits; seasonal `(P,Q)` does it with 1–2 parameters → parsimonious, generalizes far better.

### Exogenous variables — the 5-step selection (recite all five)
```
1. Domain causality   — does X genuinely cause Y? right direction? (temp→sales ✅, stock market ❌)
2. Cross-correlation   — corr(Y_t, X_{t−lag}) across lags → which lag matters
3. Granger causality   — does X improve prediction beyond Y's own history? (≠ true causality — note this)
4. VIF                 — drop multicollinear exog (VIF>10 severe, >5 investigate)
5. FUTURE AVAILABILITY — will I KNOW X at forecast time? temp/holidays/promos ✅; competitor price ❌
                         ← THE most-missed step. If you won't have X's future value, it's useless.
```

---

## 4. Worked Example

**ACF/PACF order reading** on the differenced ice-cream series:
```
PACF: significant at lags 1,2 then cuts off  → p = 2
ACF : significant at lags 1,2,3 then cuts off → q = 3
(Read SEASONAL P,Q the same way at lags 12, 24.)
"Cuts off" = drops sharply inside the confidence band; "tails off" = decays slowly.
```
**Seasonal differencing** `(D=1, s=12)`: `Jan-Y2 − Jan-Y1 = 132 − 120 = 12`, `Feb-Y2 − Feb-Y1 = 13`, … → the repeating yearly pattern is removed, leaving fluctuations around zero.

---

## 5. Code / Implementation

```python
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("icecream.csv", parse_dates=["month"], index_col="month")
df.index.freq = "MS"                                   # set frequency — easy to forget, breaks forecasting
sales, exog = df["sales"], df[["temperature", "holiday_flag"]]

# 1) Stationarity: difference until ADF passes
print(adfuller(sales)[1])                              # >0.05 → non-stationary → d=1
print(adfuller(sales.diff().dropna())[1])             # <0.05 → stationary after d=1 ✅

# 2) Orders: PACF→p,P  ACF→q,Q  (always on the STATIONARY series)
#    (plot_pacf / plot_acf on sales.diff().dropna())

# 3) Time-ordered split (NEVER shuffle a time series)
train, test = df[:"2022"], df["2023":]

# 4) Fit
res = SARIMAX(train["sales"], exog=train[["temperature","holiday_flag"]],
              order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
print(res.summary())                                  # coefficients, p-values, AIC/BIC
res.plot_diagnostics(figsize=(15,10))                 # residuals must look like white noise (§7)

# 5) Forecast WITH future exog + uncertainty intervals
fc = res.get_forecast(steps=len(test), exog=test[["temperature","holiday_flag"]])
mean, ci = fc.predicted_mean, fc.conf_int()           # point forecast + 95% prediction interval
mae  = mean_absolute_error(test["sales"], mean)
rmse = np.sqrt(mean_squared_error(test["sales"], mean))
```
When ACF/PACF are ambiguous, **`pmdarima.auto_arima`** grid-searches orders to minimize AIC/BIC — know the manual reading to *explain why*, but reach for auto_arima in practice.

---

## 6. When It Breaks

```
❌ Linear & single-seasonality only — one s. Multiple/changing cycles (daily＋weekly＋yearly) → TBATS/Prophet (§10).
❌ Over-differencing — a sharp NEGATIVE lag-1 ACF (≈ −0.5) + rising variance ⇒ you differenced too much
   (it injects artificial MA structure). Stop at the smallest d that's stationary.
❌ Spurious regression — a trending exog vs a trending target shows high corr from shared trend, not a real link.
   Difference both / test cointegration before trusting β.
❌ Non-constant variance — differencing fixes the mean, not the variance → log/Box-Cox transform.
❌ Structural breaks / regime change (COVID, a new competitor) — fixed coefficients can't adapt → re-fit / add dummies.
❌ Reading orders off a NON-stationary series — ACF/PACF are only meaningful after differencing.
```

| Common mistake | Correct statement |
|---|---|
| "differencing auto-converts to stationary" | We manually difference; statsmodels cumsums predictions back |
| "ACF gives the AR order" | **PACF → p (AR)**, **ACF → q (MA)** |
| "use all correlated variables as exog" | Must also pass Granger **and** be available at forecast time |
| "fit model → forecast" | Always check residual diagnostics first |
| "d=1 always" | Test with ADF/KPSS; sometimes d=0 or a log transform is right |
| "optimize only AIC" | AIC for accuracy, BIC for parsimony — know the difference |
| "evaluate with k-fold CV" | Use **walk-forward** — k-fold leaks the future into the past (§8) |

---

## 7. Residual Diagnostics

After fitting, residuals **must** be white noise — otherwise the model is mis-specified and the forecast is untrustworthy.
```
White noise: ✅ no significant ACF spikes at any lag  ✅ mean ≈ 0  ✅ constant variance  ✅ ~normal
Formal test: Ljung–Box (want p > 0.05 → cannot reject "no autocorrelation").
```
| Residual symptom | Fix |
|---|---|
| ACF spike at lag 1–3 | increase **q** (or p) |
| PACF spike at lag 1–2 | increase **p** |
| Spike at seasonal lag 12 | increase **Q** or **P** |
| Variance fanning out | log/Box-Cox transform `y` |
| Mean ≠ 0 | revisit differencing `d` |
| Heavy tails / outliers | investigate events, add dummies |

---

## 8. Production & MLOps Notes

### Validation — the part that's easy to get catastrophically wrong
- **Walk-forward (rolling/expanding-window) CV, never k-fold.** `P0`. Random k-fold trains on future points to predict past ones → **leakage** → optimistic, dishonest scores. Use `sklearn.TimeSeriesSplit` or expanding-origin backtesting that always trains on the past and tests on the next block. `(certain)`
- **Forecast horizon matters:** evaluate at the horizon you actually deploy (1-step vs 12-step error are very different).

### Uncertainty & metrics
- **Ship prediction intervals, not just point forecasts** — `get_forecast().conf_int()` gives calibrated bands from the state-space model; decisions (inventory, staffing) need the range. `P0`.
- **Pick the right error metric:** RMSE (penalizes big misses), MAE (robust), **MAPE breaks near zero and is asymmetric** → prefer **sMAPE** or **MASE** (scaled vs a naïve seasonal baseline — also tells you if you even beat "repeat last season").

### Exogenous variables at inference
- You **must supply future X** to forecast. If a useful X isn't known ahead, either **forecast it first** (and propagate its uncertainty) or drop it. Forecasting on predicted exog compounds error — account for it.

### Operating the model
- **Retraining cadence + drift:** time series drift by nature; schedule refits and monitor rolling forecast error — a sustained rise signals **concept drift / structural break** → retrain or re-specify.
- **Scaling to many series:** SARIMAX is **one model per series** and doesn't share learning. For thousands of SKUs/stores this is slow and weak on short/cold-start series → use a **global model** (one [[gradient-boosting]] or deep model over all series with lag+calendar features) — see §10.
- **Reproducibility:** persist the fitted state, the exact differencing/transform pipeline, and the exog schema; set the index frequency.

---

## 9. Interview Lens

> ⚡ **Golden rule:** diagnose before solving; **lead with the stationarity → ACF/PACF → exog → residuals → walk-forward pipeline**, not a parameter dump.

**"How would you use SARIMAX?"** → 🎯 *"ADF/KPSS for stationarity (difference to get d), read p/q from PACF/ACF on the stationary series and P/Q at seasonal lags, select exog by domain→cross-corr→Granger→VIF→**future availability**, fit, confirm residuals are white noise (Ljung-Box), and validate **walk-forward**."*

**"How do you pick p and q?"** → 🎯 *"PACF cutoff → p, ACF cutoff → q on the stationary series; if the plots are noisy, grid-search/`auto_arima` on AIC (accuracy) or BIC (parsimony), then confirm via residual ACF."*

**Likely follow-ups:**
- *AR vs MA intuitively?* → AR uses past **values** (momentum); MA uses past **errors** (correction).
- *Why SARIMA over a huge p?* → seasonal `(P,Q)` capture the cycle in 1–2 params vs 12+ AR coeffs → less overfitting.
- *Exog is itself trending?* → make both stationary or test **cointegration** to avoid spurious regression.
- *SARIMAX vs regression with month/trend features?* → regression captures deterministic season/trend; SARIMAX **also** captures stochastic autocorrelation (depends on past values *and* past errors).
- *AIC vs BIC?* → BIC penalizes by `log(n)` (harder for n>7) → simpler models; AIC favors accuracy.
- *Residuals still autocorrelated?* → diagnose the lag and bump the matching p/q/P/Q; refit until white noise.
- *Missing values?* → state-space + Kalman filter handles gaps in `y`; missing **future exog** is the harder problem (impute/forecast).

---

## 10. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Single series, one season, interpretable + intervals | **SARIMAX** | Classical, transparent coefficients, principled uncertainty |
| Multiple / complex seasonality (daily+weekly+yearly) | **TBATS**, **Prophet** | Built for several cycles; Prophet adds holidays + easy trend changepoints |
| Strong nonlinear effects, many features | **Gradient boosting** (XGBoost/LightGBM) on lag+calendar features | Captures nonlinearity/interactions; same **walk-forward** validation |
| Many related series / cold-start | **Global models** (one boosting/DL model over all series) | Shares learning, scales to thousands of series |
| Large data, complex patterns, probabilistic | **DeepAR · N-BEATS · Temporal Fusion Transformer** | Learn cross-series patterns, output distributions |
| Simple trend+season baseline | **ETS / Holt-Winters** | Fast, robust, a baseline SARIMAX must beat |

🎯 **Always state your baseline:** a **seasonal-naïve** forecast (repeat last season) is the bar — if SARIMAX doesn't beat it on **MASE**, the complexity isn't earning its keep.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why must the series be stationary before fitting AR/MA?
   <details><summary>answer</summary>AR/MA learn **fixed** weights; a drifting mean (trend) makes those weights chase a moving target → systematic error. Differencing stabilizes the structure.</details>
2. Which plot gives p and which gives q?
   <details><summary>answer</summary>**PACF cutoff → p** (AR), **ACF cutoff → q** (MA); seasonal P/Q the same way at lags s, 2s. Mnemonic: PACF→P→AR→p.</details>
3. You differenced and now see a sharp negative lag-1 ACF and rising variance. What happened?
   <details><summary>answer</summary>**Over-differencing** — it injects artificial MA structure. Step back to the smallest d that's stationary; consider a log transform for the variance.</details>
4. The single most-missed step in exogenous selection?
   <details><summary>answer</summary>**Future availability** — you must have X's value at forecast time. Temp/holidays/promos ✅; competitor price ❌. Otherwise forecast X first or drop it.</details>
5. Why is k-fold CV wrong here, and what replaces it?
   <details><summary>answer</summary>K-fold trains on future to predict past → **leakage**. Use **walk-forward** (expanding/rolling origin) backtesting; `TimeSeriesSplit`.</details>
6. What must residuals look like, and how do you test it?
   <details><summary>answer</summary>**White noise** — no ACF spikes, mean≈0, constant variance, ~normal. Test with **Ljung-Box** (want p>0.05).</details>
7. SARIMAX vs a regression with month dummies + trend?
   <details><summary>answer</summary>Regression captures *deterministic* season/trend; SARIMAX **also** models *stochastic* autocorrelation (depends on past values and past errors).</details>
8. You have 5,000 store-SKU series, many short. Why is plain SARIMAX a poor fit and what do you do?
   <details><summary>answer</summary>It's **one model per series**, slow and weak on short/cold-start data. Use a **global model** (boosting or DeepAR/TFT over all series with lag+calendar features).</details>

---

*Covers: stationarity (ADF/KPSS) · ARIMA p,d,q · differencing & over-differencing · AR vs MA · SARIMA P,D,Q,s · exogenous X & the 5-step selection · ACF/PACF order reading · residual diagnostics (Ljung-Box) · AIC/BIC · auto_arima · walk-forward validation · prediction intervals · metrics (MASE/sMAPE) · drift & global models · Prophet/TBATS/boosting/DeepAR alternatives.*
