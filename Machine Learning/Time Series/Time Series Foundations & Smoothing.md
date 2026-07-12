# Time Series Foundations & Smoothing — Decomposition, Baselines, Exponential Smoothing

> **TL;DR.** A time series is data indexed by time, so **order matters and you can never shuffle it**. First understand its parts — **level + trend + seasonality + residual** (additive or multiplicative) — via **decomposition**. Then forecast, from dumbest to smartest: **baselines** (mean / naive / seasonal-naive — the bar you must beat) → **moving average** → **exponential smoothing**, which weights recent data more, decaying exponentially: **SES** (level only) → **Holt/DES** (level + trend) → **Holt-Winters/TES** (level + trend + seasonality). Evaluate with a **time-based split** (never `train_test_split` shuffle) and metrics like **MAE/RMSE/MAPE (→ sMAPE/MASE)**. When you need autocorrelation structure and exogenous drivers, graduate to [SARIMAX](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md).

**Where it fits:** The on-ramp to forecasting — decomposition + smoothing before the ARIMA family. Its ARIMA-family sequel is [SARIMAX](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md).
**Prereqs:** mean/variance, moving average, the idea that time order can't be broken; regression basics ([linear regression](../Supervised%20ML/Linear%20Regression.md)).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [Components & Decomposition](#2-components--decomposition)
3. [Preprocessing & the Time-Based Split](#3-preprocessing--the-time-based-split)
4. [Simple Baselines](#4-simple-baselines)
5. [Moving Average Forecasting](#5-moving-average-forecasting)
6. [Exponential Smoothing — SES · Holt · Holt-Winters](#6-exponential-smoothing--ses--holt--holt-winters)
7. [Forecast Error Metrics](#7-forecast-error-metrics)
8. [From Smoothing to ARIMA — the Stationarity Bridge](#8-from-smoothing-to-arima--the-stationarity-bridge)
9. [Code / Implementation](#9-code--implementation)
10. [When It Breaks & How to Choose](#10-when-it-breaks--how-to-choose)
11. [Interview Lens](#11-interview-lens)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

A time series is a sequence of observations over time (monthly sales, daily traffic). The defining constraint: **the past predicts the future, so you must respect time order** — no shuffling, no using future to predict past.

```
value │                          ╱╲          = LEVEL     (the baseline value)
      │              ╱╲    ╱╲   ╱   ╲    ↗    + TREND     (long-run direction)
      │       ╱╲   ╱   ╲ ╱    ╲╱          + SEASONALITY  (a repeating cycle, fixed period)
      │  ╱╲ ╱   ╲╱                        + RESIDUAL     (irregular noise)
      └────────────────────────── time
```

- Every forecasting method is a bet about **which of these components exist** and how to project them. SES bets "just a level," Holt adds "a trend," Holt-Winters adds "a season." `(certain)`
- **Trend** = sustained up/down direction. **Seasonality** = a cycle of *fixed, known period* (yearly, weekly). **Cyclic** = wavy but *irregular* period (business cycles) — not the same as seasonality. `(certain)`

---

## 2. Components & Decomposition

**Decomposition** splits a series into level+trend, seasonality, and residual so you can see and model each. Two forms: `(certain)`

```
Additive:        y_t = Trend_t + Seasonality_t + Residual_t     (seasonal swing is a FIXED amount, e.g. +70 units)
Multiplicative:  y_t = Trend_t × Seasonality_t × Residual_t     (seasonal swing SCALES with level, e.g. ×1.4)
```
Use **multiplicative** when the seasonal amplitude grows as the series grows (common in revenue); **additive** when the swing is roughly constant. (A `log` transform turns a multiplicative series into an additive one.) `(certain)`

**How the pieces are estimated:**
- **Trend** ← a **centered moving average** (rolling mean over one full season) smooths away the season & noise, leaving the trend.
- **Seasonality** ← after de-trending (`y − trend`), average the leftovers *by season position* (all Januaries together) → the seasonal index.
- **Residual** ← what's left: `y − trend − seasonality` (additive). If the residual still shows pattern, a component is mis-estimated.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(series, model="additive", period=12).plot()   # trend / seasonal / resid panels
```

---

## 3. Preprocessing & the Time-Based Split

- **Missing values** → **interpolate** (`series.interpolate(method="linear")`) rather than drop — you must keep the time index contiguous. `(certain)`
- **Outliers/anomalies** → **clip to quantiles** (`series.clip(lower=q02, upper=q98)`) so a spike doesn't distort trend/season estimates.
- 🎯 **The split: time-based, never shuffled.** `train_test_split()` shuffles → it would train on the future to predict the past = leakage and nonsense. **Slice by time**: earliest block = train, latest block = test (e.g. last 12 months). This generalises to **walk-forward CV** (see [SARIMAX](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md)). `(certain)`

---

## 4. Simple Baselines

Dumb forecasts you must **beat** before anything fancy earns its keep: `(certain)`

| Baseline | Forecast for all future | Captures |
|---|---|---|
| **Mean** | the historical mean → a flat line | nothing (level only, and barely) |
| **Naive** | the *last* observed value | level; changes every time new data arrives (brittle for clients) |
| **Seasonal naive** | the value from the **same season last cycle** (last year's same month) | **seasonality** (but not trend) |

🎯 *"A new model that can't beat seasonal-naive isn't worth deploying."* Baselines are the honesty check — every real method is measured against them.

---

## 5. Moving Average Forecasting

Forecast the next point as the **average of the last `k` observations**, rolling forward:

```
ŷ_{t+1} = mean(y_t, y_{t-1}, …, y_{t-k+1})      # then feed forecasts back in for multi-step
```

- Better than a flat mean (tracks the local level), but has two flaws: it **weights the last `k` points equally and ignores everything older** (why keep 18 years of data to look at only the last 3?), and multi-step forecasts **flatten to a straight line** (each new average is of previous averages → variance shrinks). `(certain)`
- The fix — weight *all* history, decaying with age → **exponential smoothing** (§6).

---

## 6. Exponential Smoothing — SES · Holt · Holt-Winters

**One idea, extended three times:** weight recent observations more, with weights decaying **exponentially** into the past. Each version adds a component. `(certain)`

```
SES  → Level                    (flat, no trend/season)
DES  → Level + Trend            (Holt's linear method)
TES  → Level + Trend + Season   (Holt-Winters)
```

### Simple Exponential Smoothing (SES) — level only
```
ŷ_{t+1} = α·y_t + (1−α)·ŷ_t          # new forecast = α·(what happened) + (1−α)·(what we predicted)
```
Expanding it shows the exponential decay: weights are `α, α(1−α), α(1−α)², …` — recent data dominates, old data fades (all weights sum to 1). **α ∈ (0,1)** trades reactivity vs stability: `α→1` ≈ naive (chases noise), `α→0` ≈ very smooth/slow. A good start is `α = 1/(2·season_length)`. **Weakness:** no trend, no season → it always *lags behind* a rising series and forecasts a flat line. `(certain)`

**Worked example** (α=0.3, init = first value):
```
y:  200  220  210  230
ŷ:  200  200  206  207.2    (206 = .3·220+.7·200 ; 207.2 = .3·210+.7·206)
next ŷ = .3·230 + .7·207.2 = 214.04
```

### Double Exponential Smoothing (DES / Holt) — + trend
Track **level** `L` and **trend** `T` separately, each smoothed: `(certain)`
```
L_t = α·y_t + (1−α)(L_{t-1} + T_{t-1})      # level, corrected toward the new observation
T_t = β(L_t − L_{t-1}) + (1−β)·T_{t-1}       # trend = smoothed recent slope
ŷ_{t+h} = L_t + h·T_t                         # project the trend h steps out
```
Intuition (a runner with noisy GPS): `L` = current **position**, `T` = current **speed**; forecast = "position + h·speed." `β` controls how fast the trend adapts. **Weakness:** no seasonality — it fits a straight line through seasonal ups/downs. `(certain)`

**Worked example** (`L₁=100, T₁=20, α=0.4, β=0.3`, series 100,120,140…):
```
t2: L₂ = .4·120 + .6·(100+20) = 120 ;  T₂ = .3·(120−100)+.7·20 = 20 ;  ŷ₃ = 120+20 = 140
forecast 3 steps ahead from t2:  ŷ = L₂ + 3·T₂ = 120 + 3·20 = 180
```

### Triple Exponential Smoothing (TES / Holt-Winters) — + seasonality
Add a **seasonal** component `S` with period `m`, smoothed by `γ`: `(certain)`
```
L_t = α(y_t − S_{t-m}) + (1−α)(L_{t-1}+T_{t-1})   # de-seasonalize before updating level
T_t = β(L_t − L_{t-1}) + (1−β)·T_{t-1}
S_t = γ(y_t − L_t) + (1−γ)·S_{t-m}                 # seasonal index vs SAME season last cycle
ŷ_{t+h} = L_t + h·T_t + S_{t-m+h}                  # level + trend + the right season's bump
```
This is the **additive** form; the **multiplicative** form multiplies/divides by `S` instead (use it when seasonal swings scale with the level). Intuition (a runner on a hilly loop): level = flat-ground position, trend = base speed, seasonality = the "hill effect" that repeats each lap. You must supply the **season length `m`** (monthly→12, weekly→7, quarterly→4). It's the strongest classical smoother — it tracks level, trend *and* season. `(certain)`

```
DES misses the summer spike:  ŷ = L + 3T          (straight line)
TES catches it:               ŷ = L + 3T + S_Q3   (adds the recurring seasonal bump)
```

`α`, `β`, `γ` are all in `(0,1)`; `statsmodels` optimizes them by minimizing training SSE if you don't fix them.

---

## 7. Forecast Error Metrics

| Metric | Formula | Read it as |
|---|---|---|
| **MAE** | `mean(\|y−ŷ\|)` | avg error in the target's units; robust to outliers |
| **RMSE** | `√mean((y−ŷ)²)` | like MAE but punishes big misses; same units |
| **MAPE** | `mean(\|y−ŷ\|/\|y\|)·100` | **% error** — intuitive & absolute (no need to compare models) |

- 🎯 **MAPE's trap:** it's **undefined/explodes when `y=0`** and is **asymmetric** (penalises over- vs under-forecast unequally). When zeros or near-zeros exist, prefer **sMAPE** (symmetric) or **MASE** (error scaled by a naive-seasonal baseline — MASE < 1 means you beat the naive benchmark). `(certain)`
- Always evaluate at the **horizon you'll actually forecast** (1-step vs 12-step errors differ a lot), on the **time-held-out** test block.

---

## 8. From Smoothing to ARIMA — the Stationarity Bridge

Smoothing models the components directly. The **ARIMA family** instead models the series' own **autocorrelation** — but it first requires the series to be **stationary** (constant mean, variance, autocorrelation — no trend or season). `(certain)`

```
Non-stationary (trend+season) ──difference──▶ stationary ──ARIMA/SARIMA/SARIMAX──▶ forecast
   de-trend:      y_t − y_{t-1}        (regular differencing, d)
   de-seasonalize: y_t − y_{t-m}       (seasonal differencing, D, m = season length)
   test:  Dickey-Fuller (ADF) → p < 0.05 ⇒ stationary
```

That's the launch point for the full ARIMA/SARIMAX treatment — stationarity testing, ACF/PACF order selection, exogenous regressors, residual diagnostics, and walk-forward validation live in **[SARIMAX — Seasonal ARIMA with Exogenous Regressors](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md)**.

---

## 9. Code / Implementation

```python
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# preprocessing
s = df["Sales"].interpolate("linear")
s = s.clip(lower=s.quantile(0.02), upper=s.quantile(0.98))
train, test = s[:-12], s[-12:]                 # TIME-BASED split — never shuffle

# baselines to beat
seasonal_naive = train.shift(12)               # same month last year

# SES (level)
SimpleExpSmoothing(train).fit(smoothing_level=1/(2*12)).forecast(12)

# Holt / DES (level + trend)
ExponentialSmoothing(train, trend="add").fit().forecast(12)

# Holt-Winters / TES (level + trend + season)
ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(12)
#   seasonal="mul" when seasonal swings scale with the level; fit() optimizes α,β,γ by default
```

---

## 10. When It Breaks & How to Choose

```
trend?  ── no ── season? ── no ──▶ SES               (stable, noisy level)
                          └ yes ─▶ Holt-Winters (damped, no trend)
  └ yes ── season? ── no ──▶ Holt / DES              (trending, no cycle)
                    └ yes ─▶ Holt-Winters (TES)      (trend + season)  → additive vs multiplicative by swing shape
need autocorrelation structure / exogenous drivers / uncertainty intervals ──▶ SARIMAX
```

**Failure modes:** SES lags any trend; DES ignores seasonality; multiplicative TES breaks if the series hits zero/negative; all of them assume the component structure is *stable* (a regime change/structural break — COVID — breaks them). Smoothing gives point forecasts but **no principled prediction intervals or exogenous regressors** — that's when you move to [SARIMAX](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md), and to [gradient boosting](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)/DeepAR for many series or nonlinear drivers.

---

## 11. Interview Lens

**"Walk me through the exponential smoothing family."** → 🎯 *"One idea — exponentially decaying weights on past observations — extended three times: SES smooths the level (α); Holt/DES adds a trend component (β) and forecasts level + h·trend; Holt-Winters/TES adds a seasonal component (γ) at period m. Pick by which components your data has."*

**"Why can't you use a normal train/test split for time series?"** → 🎯 *"It shuffles, which trains on future points to predict past ones — leakage. You split by time (past = train, future = test) and validate walk-forward."*

**Likely follow-ups:**
- *Additive vs multiplicative seasonality?* → Additive = constant-size swing; multiplicative = swing scales with the level (log makes it additive).
- *What does α do in SES?* → Reactivity: high α chases recent data (noisy), low α is smooth/slow; α=1 ≈ naive.
- *SES's weakness?* → No trend/season → lags a trend and forecasts a flat line.
- *Difference between seasonality and a cycle?* → Seasonality has a fixed known period; cycles are irregular.
- *Why is a moving-average forecast poor?* → Equal weight on a short window, ignores older data, and flattens for multi-step forecasts.
- *When is MAPE a bad metric?* → Near/at zero values (undefined/explodes) and it's asymmetric — use sMAPE or MASE.
- *How do you know smoothing isn't enough?* → When residuals still show autocorrelation, or you need exogenous variables / prediction intervals → ARIMA/SARIMAX.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Name the four components of a time series and the two decomposition forms.
   <details><summary>answer</summary>Level, trend, seasonality, residual. Additive (`y = T + S + R`, constant swing) vs multiplicative (`y = T × S × R`, swing scales with level).</details>
2. Why can't you shuffle a train/test split for time series, and what do you do instead?
   <details><summary>answer</summary>Shuffling trains on the future to predict the past (leakage). Split by time — earliest block train, latest block test — and validate walk-forward.</details>
3. What does each of SES, DES, TES model, and with which parameters?
   <details><summary>answer</summary>SES = level (α); DES/Holt = level + trend (α, β); TES/Holt-Winters = level + trend + seasonality (α, β, γ, plus season length m).</details>
4. Write the SES update and say what α controls.
   <details><summary>answer</summary>`ŷ_{t+1} = α·y_t + (1−α)·ŷ_t`. α = reactivity: high → tracks recent data (noisy), low → smooth/slow; α=1 ≈ naive.</details>
5. Why does SES forecast poorly on trending data?
   <details><summary>answer</summary>It models only a level with no trend term, so it systematically lags a rising/falling series and projects a flat line.</details>
6. In Holt-Winters, why compare against `S_{t-m}` not `S_{t-1}`?
   <details><summary>answer</summary>The seasonal index must be updated against the same season one full cycle ago (t−m), because t−1 is a different season.</details>
7. When is MAPE the wrong metric, and what replaces it?
   <details><summary>answer</summary>When actuals are zero/near-zero (undefined/explodes) and because it's asymmetric — use sMAPE or MASE (MASE < 1 beats the naive baseline).</details>

---

*Covers: time-series components (level/trend/seasonality/cyclic/residual), additive vs multiplicative decomposition (seasonal_decompose), preprocessing (interpolation, quantile clipping), the no-shuffle time-based split, baselines (mean/naive/seasonal-naive), moving-average forecasting, exponential smoothing (SES/Holt-DES/Holt-Winters-TES, additive & multiplicative, α/β/γ), forecast metrics (MAE/RMSE/MAPE→sMAPE/MASE), and the stationarity bridge to [SARIMAX](Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md). Sourced from Scaler Time Series lectures 1–2.*
