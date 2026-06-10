# SARIMAX — Interview Revision Notes
### Everything you need to answer confidently in any ML/DS interview

---

## ⚡ The Golden Rule Before Answering Any Interview Question

> **Spend 15 seconds re-reading the question. Diagnose first. Then solve.**
>
> Interviewers give most marks for:
> 1. Correct reasoning and mechanism (the WHY)
> 2. Then the solution (the WHAT)

---

## Table of Contents
1. [Stationarity — The Foundation](#1-stationarity--the-foundation)
2. [ARIMA — p, d, q](#2-arima--p-d-q)
3. [SARIMA — P, D, Q, s](#3-sarima--p-d-q-s)
4. [SARIMAX — Exogenous Variables](#4-sarimax--exogenous-variables-x)
5. [How to Select p, q (ACF & PACF)](#5-how-to-select-p-and-q--acf--pacf)
6. [How to Select Exogenous Variables](#6-how-to-select-exogenous-variables)
7. [Complete Implementation](#7-complete-implementation)
8. [Residual Diagnostics](#8-residual-diagnostics)
9. [Full Parameter Reference](#9-full-parameter-reference)
10. [Interview Answer Templates](#10-interview-answer-templates)
11. [Likely Follow-Up Questions & Answers](#11-likely-follow-up-questions--answers)
12. [Pre-Interview Checklist](#12-pre-interview-checklist)

---

## The Running Example (Used Throughout)

```
Monthly ice cream sales over 3 years.
Exogenous variable: average monthly temperature (°C)

Month     Sales    Temp
Jan-Y1     120      12
Feb-Y1     135      14
Mar-Y1     200      20
Apr-Y1     310      26
May-Y1     480      32    ← summer peak
Jun-Y1     520      35
Jul-Y1     495      34
Aug-Y1     410      30
Sep-Y1     280      24
Oct-Y1     180      18
Nov-Y1     140      14
Dec-Y1     125      11
... (repeats with slight upward trend each year)

This data has:
  → Trend         (sales growing year over year)
  → Seasonality   (summer spike every year)
  → Exogenous     (temperature drives the spikes)
  → Noise         (random fluctuation each month)

SARIMAX handles all four simultaneously.
```

---

# 1. Stationarity — The Foundation

## What Is Stationarity?

A time series is **stationary** when its statistical properties do not change over time:

```
Stationary requires:
  ✅ Constant mean        (no trend — not drifting up or down)
  ✅ Constant variance    (not getting wider/noisier over time)
  ✅ Constant autocorrelation structure (no seasonality)

Non-stationary means:
  ❌ Mean is changing      (trend present)
  ❌ Variance is changing  (variance grows with level)
  ❌ Seasonal patterns     (repeating cycles)
```

## Why ARIMA Needs Stationarity

```
AR and MA components learn FIXED WEIGHTS from data.

If mean keeps changing (trend):
  → Fixed weights become meaningless
  → Model is trying to hit a moving target
  → Forecasts will be systematically wrong

Stationary data:
  → Stable structure the model can reliably learn from
  → Fixed weights remain valid over time
```

## Visually

```
NON-STATIONARY (raw ice cream data):
Sales
 600|          *
 500|       *     *
 400|     *          *
 300|   *               *
 200| *                    *  *
 100|*                          *
    |________________________________ Time
    Trend going up + seasonal spikes = NOT stationary ❌

STATIONARY (after differencing):
Value
  50|  *  *        *   *
   0|*    *  *  *    *    *  *
 -50|         *              *  *
    |________________________________ Time
    Constant mean ≈ 0, constant variance = stationary ✅
```

## How to Test for Stationarity — ADF Test

```python
from statsmodels.tsa.stattools import adfuller

result  = adfuller(sales)
p_value = result[1]

if p_value < 0.05:
    print("STATIONARY ✅")
else:
    print("NON-STATIONARY ❌ — apply differencing")
```

**ADF Test Logic:**
```
Null hypothesis: series IS non-stationary

p-value < 0.05 → reject null → series IS stationary     ✅
p-value > 0.05 → fail to reject → series NOT stationary ❌
```

**Visual check (always do alongside ADF):**
```python
import matplotlib.pyplot as plt

rolling_mean = sales.rolling(window=12).mean()
rolling_std  = sales.rolling(window=12).std()

plt.plot(sales,        label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std,  label='Rolling Std')
plt.legend()
plt.show()
# Flat rolling mean + flat rolling std = stationary ✅
```

---

# 2. ARIMA — p, d, q

**ARIMA = AutoRegressive Integrated Moving Average**

Each letter solves a specific problem:

```
AR (p) → uses past VALUES      to forecast
I  (d) → uses DIFFERENCING     to remove trend
MA (q) → uses past ERRORS      to correct forecast
```

---

## The "I" — Integrated, Differencing (d)

**What it actually does:** Subtract each value from the previous one to remove trend.

```
Original sales: [120, 135, 200, 310, 480]

First difference (d=1):
  135-120=15,  200-135=65,  310-200=110,  480-310=170
  Result:       [15,   65,    110,          170]
  → Linear trend removed ✅

If still non-stationary → second difference (d=2):
  65-15=50,  110-65=45,  170-110=60
  Result:     [50,   45,    60]
  → Quadratic trend removed ✅
```

**Important precision for interviews:**
```
d does NOT "automatically convert and convert back."

What actually happens:
  1. We manually difference the series (remove trend)
  2. Fit ARMA on the stationary differenced series
  3. statsmodels automatically cumsum the predictions
     back to the original scale for you

d=0 → no differencing needed (already stationary)
d=1 → removes linear trend (most common)
d=2 → removes quadratic trend (rarely needed — red flag)
      if you need d=2, try log-transform first
```

**How to choose d:**
```
Step 1: ADF test on original series
  Stationary? → d=0 ✅

Step 2: If not stationary → difference once → ADF test
  Stationary? → d=1 ✅

Step 3: If still not → difference again → ADF test
  Stationary? → d=2 ✅

Rule: d is almost always 0 or 1.
```

---

## The "AR" — AutoRegressive Component (p)

**Core idea:** Today's value is a linear combination of the last p observations.

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t$$

**For ice cream sales with p=2:**
```
Sales_today = constant
            + φ₁ × Sales_last_month
            + φ₂ × Sales_2_months_ago
            + noise

Unlike SES where weights = α(1-α)^k (forced exponential decay),
AR learns φ₁, φ₂, ..., φₚ FREELY via OLS.
These can be ANY values — even negative.
AR is strictly more flexible than SES.
```

**AR vs SES — key distinction:**
```
SES:  weights = 0.3, 0.21, 0.147...  (forced exponential decay, one param α)
AR:   weights = φ₁, φ₂, ..., φₚ     (freely learned, any pattern, p params)

SES can only represent exponentially decaying memory.
AR can represent any linear combination of past values.
```

---

## The "MA" — Moving Average Component (q)

**Core idea:** Today's value depends on the last q **forecast errors**, not past observations.

$$y_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \ldots + \theta_q\epsilon_{t-q}$$

**Intuition:**
```
"If I overestimated last month's sales by 50 units,
 I should adjust this month's forecast DOWN by θ₁ × 50."

AR asks: "What was the actual value last period?"
MA asks: "How wrong was my forecast last period,
          and how should I correct for that?"

MA = error correction mechanism
AR = pattern learning mechanism
Together: ARMA captures both past values AND past mistakes
```

---

# 3. SARIMA — P, D, Q, s

## The Problem with Plain ARIMA

After d=1 differencing, ARIMA removes trend. But look at what remains:

```
ACF after first differencing:
1.0 |█
0.5 |          █           ← SPIKE AT LAG 12!
0.0 |  · · · ·   · · · · ·
-0.5|
     1  2  3  4  5  6  7  8  9 10 11 12

Significant spike at lag 12 = yearly seasonality still present.
Plain ARIMA cannot capture this. We need SARIMA.
```

## SARIMA = ARIMA + Seasonal ARIMA

**Full notation: SARIMA(p, d, q)(P, D, Q)[s]**

```
(p, d, q)  →  non-seasonal part  (trend + short-term patterns)
(P, D, Q)  →  seasonal part      (repeating seasonal patterns)
[s]         →  season length

Common values of s:
  s=12  → monthly data with yearly seasonality
  s=4   → quarterly data with yearly seasonality
  s=7   → daily data with weekly seasonality
  s=52  → weekly data with yearly seasonality
```

**Seasonal part works exactly like ARIMA but at seasonal lags:**

```
Seasonal AR (P):
  Today depends on values from s, 2s, 3s... periods ago
  e.g., s=12: Jan this year depends on Jan last year,
                                       Jan 2 years ago

Seasonal I (D):
  Seasonal differencing: y_t - y_{t-s}
  e.g., Jan this year MINUS Jan last year
  Removes seasonal trend

Seasonal MA (Q):
  Today depends on forecast errors from s, 2s... periods ago
  Error correction at seasonal frequency
```

## Seasonal Differencing Example

```
Original monthly data:
  Jan-Y1=120, Feb-Y1=135, ..., Jan-Y2=132, Feb-Y2=148...

Seasonal difference (D=1, s=12):
  Jan-Y2 - Jan-Y1 = 132 - 120 = 12
  Feb-Y2 - Feb-Y1 = 148 - 135 = 13
  ...

Result: removes the repeating yearly seasonal pattern.
What remains: short-term fluctuations around zero.
```

## How to Choose P, D, Q, s

```
s:   domain knowledge FIRST, then validate with ACF
     Monthly + yearly cycle → s=12
     Check ACF for spikes at exactly lag s, 2s, 3s

D:   same logic as d but for seasonal component
     Seasonal ADF test or visual inspection
     D=0 or D=1 (almost always)

P:   look at PACF at SEASONAL lags (lag s, 2s, 3s)
     PACF significant at lag 12 and 24 → P=2
     PACF significant at lag 12 only   → P=1

Q:   look at ACF at SEASONAL lags
     ACF significant at lag 12 only    → Q=1
     ACF significant at lag 12 and 24  → Q=2
```

---

# 4. SARIMAX — Exogenous Variables (X)

## What "Exogenous" Means

```
Endogenous variable:  the series we're FORECASTING (ice cream sales)
                      → determined within the system we're modeling

Exogenous variable:   external variables that INFLUENCE the series
                      but are NOT influenced by it
                      → temperature influences sales ✅
                      → sales do NOT influence temperature ✅
```

## How Exogenous Variables Enter the Model

SARIMAX adds exogenous variables as **additional linear regressors:**

$$y_t = \underbrace{\text{SARIMA terms}}_{\text{time series structure}} + \underbrace{\beta_1 X_{1t} + \beta_2 X_{2t} + \ldots}_{\text{exogenous regressors}} + \epsilon_t$$

```
The model learns:
  SARIMA part → captures trend, seasonality, autocorrelation
  β coefficients → how much each exogenous variable moves the forecast

β₁ for temperature = 8.5 means:
  "For every 1°C increase in temperature,
   expected sales increase by 8.5 units,
   holding everything else constant."
```

## Critical Rule — Future Values Must Be Available

```
At forecast time, you need FUTURE values of exogenous variables.

Temperature    → weather forecasts available 7-14 days ahead ✅
Holiday flags  → known in advance (calendar) ✅
Promotion flag → planned in advance ✅
Competitor price → may not know in advance ❌
                   → can't use for live forecasting

If future values unavailable:
  Option 1: Forecast the exogenous variable separately first
  Option 2: Remove it from the model

THIS IS THE MOST COMMON MISTAKE IN SARIMAX INTERVIEWS.
Always ask: "Will I have this variable's future values at inference time?"
```

---

# 5. How to Select p and q — ACF & PACF

## ACF — AutoCorrelation Function

```
ACF(lag=k) = correlation between y_t and y_{t-k}

Measures: "How correlated is the series with its own past?"
Includes: BOTH direct and indirect effects of lag k
```

## PACF — Partial AutoCorrelation Function

```
PACF(lag=k) = correlation between y_t and y_{t-k}
              AFTER removing the effect of all shorter lags

Measures: "Pure" direct effect of lag k on current value
Removes:  The indirect influence passing through intermediate lags
```

## The Rules — Memorise These

```
┌──────────────────────────────────────────────────────┐
│         ACF                    PACF                  │
├──────────────────────────────────────────────────────┤
│  Cuts off after lag q    │  Cuts off after lag p     │
│  → tells you q (MA)      │  → tells you p (AR)       │
│                          │                           │
│  Tails off slowly        │  Tails off slowly         │
│  → AR process present    │  → MA process present     │
└──────────────────────────────────────────────────────┘

Memory trick:
  PACF → P → AR → p
  ACF  → (remaining) → MA → q

"Cuts off" = drops sharply inside confidence band (blue shaded region)
"Tails off" = decays slowly, stays marginally outside band
```

## Visual Example

```
PACF plot (after differencing our ice cream data):
Correlation
1.0 |█
0.6 |  █
0.2 |    ─ ─ ─ ─ ─ ─ ─ ─     ← confidence band
0.0 |       · · · · · · ·
   -0.2|
       1  2  3  4  5  6  7    Lag

PACF: significant at lags 1 and 2, then cuts off → p = 2

ACF plot:
1.0 |█
0.7 |  █
0.4 |    █
0.1 |      ─ ─ ─ ─ ─ ─       ← confidence band
0.0 |         · · · · ·
   -0.2|
       1  2  3  4  5  6  7    Lag

ACF: significant at lags 1, 2, 3, then cuts off → q = 3
```

## In Python

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Always plot on the STATIONARY (differenced) series
sales_diff = sales.diff().dropna()    # d=1 differencing

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_pacf(sales_diff, lags=24, ax=axes[0])   # → read p and P
plot_acf(sales_diff,  lags=24, ax=axes[1])   # → read q and Q
axes[0].set_title("PACF — use to select p (non-seasonal) and P (seasonal)")
axes[1].set_title("ACF  — use to select q (non-seasonal) and Q (seasonal)")
plt.tight_layout()
plt.show()
```

## When ACF/PACF Are Ambiguous — Use AIC/BIC

In practice, plots are often noisy. Use information criteria:

```python
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

best_aic    = float('inf')
best_params = None

for p, q in itertools.product(range(4), range(4)):
    try:
        model  = SARIMAX(sales, order=(p, 1, q),
                         seasonal_order=(1, 1, 1, 12))
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic    = result.aic
            best_params = (p, 1, q)
    except:
        continue

print(f"Best ARIMA order: {best_params}")
print(f"Best AIC:         {best_aic:.2f}")
```

**AIC vs BIC:**
```
AIC = model fit quality - penalty for complexity
BIC = same but STRONGER penalty for complexity

Lower AIC/BIC = better model

AIC:  use when forecasting accuracy is the primary goal
BIC:  use when parsimony / interpretability matters more
      BIC tends to select simpler models than AIC
```

---

# 6. How to Select Exogenous Variables

Follow this 5-step process every time. Mention all steps in an interview.

## Step 1 — Domain Knowledge First (Always Start Here)

```
Ask: "Does this variable causally influence my target?"
     "Is the causal direction correct?"

Ice cream sales:
  ✅ Temperature       → heat causes ice cream purchases
  ✅ Public holidays   → more footfall drives sales
  ✅ Promotions        → discounts drive purchases
  ❌ Stock market      → no causal link to ice cream
  ❌ Competitor's HR   → completely irrelevant
```

## Step 2 — Cross-Correlation at the Right Lag

```python
# Does temperature at lag k predict sales today?
for lag in range(0, 5):
    corr = df['sales'].corr(df['temperature'].shift(lag))
    print(f"Lag {lag}: correlation = {corr:.3f}")

# Output:
# Lag 0: 0.87   ← temperature THIS month → sales THIS month
# Lag 1: 0.61   ← temperature LAST month still has effect
# Lag 2: 0.31
# Lag 3: 0.12   ← negligible

# → include temperature at lag 0, possibly lag 1
```

## Step 3 — Granger Causality Test

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Tests: "does temperature help predict sales
#         beyond what sales' own history can predict?"
result = grangercausalitytests(
    df[['sales', 'temperature']], maxlag=4
)

# p-value < 0.05 for any lag:
#   → temperature Granger-causes sales → include it ✅
# p-value > 0.05 for all lags:
#   → temperature adds no predictive power → exclude it ❌
```

**Important caveat to mention in interview:**
```
Granger causality ≠ true causality.
It only means: "X helps predict Y beyond Y's own history."
Domain knowledge must confirm causal direction.
```

## Step 4 — Check for Multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X   = df[['temperature', 'holiday_flag', 'promotion_flag']]
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF"    : [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]
})
print(vif)

# VIF > 10 → severe multicollinearity → drop or combine
# VIF > 5  → moderate → investigate
# VIF < 5  → acceptable ✅
```

## Step 5 — Verify Future Availability

```
Before including ANY exogenous variable, ask:
"Will I have this variable's value at forecast time?"

Available in advance     ✅ Include
  → temperature (weather forecast)
  → holiday flags (calendar)
  → planned promotions

NOT available in advance ❌ Exclude
  → real-time competitor price
  → daily footfall (if forecasting weekly)
  → any variable that depends on the outcome itself
```

---

# 7. Complete Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Step 1: Load data ─────────────────────────────────────────────
df = pd.read_csv('icecream.csv', parse_dates=['month'],
                 index_col='month')
df.index.freq = 'MS'   # monthly start frequency — critical!

sales = df['sales']
exog  = df[['temperature', 'holiday_flag']]

# ── Step 2: Check stationarity ────────────────────────────────────
adf_p = adfuller(sales)[1]
print(f"ADF p-value (original): {adf_p:.4f}")
# p > 0.05 → non-stationary → need d=1

sales_diff = sales.diff().dropna()
adf_p_diff = adfuller(sales_diff)[1]
print(f"ADF p-value (differenced): {adf_p_diff:.4f}")
# p < 0.05 → stationary after d=1 ✅

# ── Step 3: ACF / PACF to select p, q, P, Q ──────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_pacf(sales_diff, lags=24, ax=axes[0])
plot_acf(sales_diff,  lags=24, ax=axes[1])
plt.tight_layout()
plt.show()
# Read: PACF cutoff → p and P | ACF cutoff → q and Q

# ── Step 4: Train / test split ────────────────────────────────────
train = df[:'2022']
test  = df['2023':]

# ── Step 5: Fit SARIMAX ───────────────────────────────────────────
model = SARIMAX(
    endog          = train['sales'],
    exog           = train[['temperature', 'holiday_flag']],
    order          = (1, 1, 1),      # (p, d, q)
    seasonal_order = (1, 1, 1, 12)   # (P, D, Q, s)
)

result = model.fit(disp=False)
print(result.summary())
# Check: coefficients, p-values, AIC/BIC

# ── Step 6: Residual diagnostics ─────────────────────────────────
result.plot_diagnostics(figsize=(15, 10))
plt.show()
# Must see: no ACF spikes, normal distribution, constant variance

# ── Step 7: Forecast ──────────────────────────────────────────────
forecast = result.forecast(
    steps = len(test),
    exog  = test[['temperature', 'holiday_flag']]  # future exog REQUIRED
)

# ── Step 8: Evaluate ──────────────────────────────────────────────
mae  = mean_absolute_error(test['sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
mape = np.mean(np.abs((test['sales'] - forecast) / test['sales'])) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# ── Step 9: Plot forecast vs actual ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'],  label='Actual')
plt.plot(forecast,       label='Forecast', linestyle='--')
plt.legend()
plt.title('SARIMAX Forecast vs Actual')
plt.show()
```

---

# 8. Residual Diagnostics

**After fitting, residuals MUST look like white noise. Always check this.**

```
White noise means:
  ✅ ACF of residuals: no significant spikes at any lag
  ✅ Mean ≈ 0
  ✅ Constant variance (not fanning out over time)
  ✅ Roughly normally distributed

If residuals still have structure → model is mis-specified
```

## What Remaining Structure Tells You

```
Problem                        Fix
─────────────────────────────────────────────────────
ACF spike at lag 1, 2, 3    → increase q (MA order)
PACF spike at lag 1, 2      → increase p (AR order)
Spike at seasonal lag 12    → increase Q or P
Variance increasing over time → try log transform of y
Mean not zero               → check differencing (increase d)
Non-normal distribution     → outliers present, investigate
```

```python
# Quick residual check
residuals = result.resid

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF of residuals — should show NO significant spikes
plot_acf(residuals, lags=24, ax=axes[0])
axes[0].set_title("ACF of Residuals (should be white noise)")

# Distribution
axes[1].hist(residuals, bins=20, edgecolor='black')
axes[1].set_title("Residual Distribution (should be ~normal)")

plt.tight_layout()
plt.show()

print(f"Mean of residuals: {residuals.mean():.4f}")   # should be ≈ 0
print(f"Std of residuals:  {residuals.std():.4f}")
```

---

# 9. Full Parameter Reference

## SARIMAX(p, d, q)(P, D, Q)[s] — All Parameters

| Parameter | What It Controls | How to Choose | Typical Range |
|---|---|---|---|
| **p** | AR order — lags of y used | PACF cutoff (non-seasonal) | 0–3 |
| **d** | Differencing — removes trend | ADF test on original series | 0–2 |
| **q** | MA order — lags of error used | ACF cutoff (non-seasonal) | 0–3 |
| **P** | Seasonal AR order | PACF at seasonal lags (s, 2s) | 0–2 |
| **D** | Seasonal differencing | Seasonal ADF or visual check | 0–1 |
| **Q** | Seasonal MA order | ACF at seasonal lags (s, 2s) | 0–2 |
| **s** | Season length | Domain knowledge + ACF spikes | 4, 7, 12, 52 |
| **exog** | External regressors | 5-step selection process | Varies |

## Quick Decision Guide

```
Step 1: Is series stationary? (ADF test)
  No  → d=1, difference and retest
  Yes → d=0

Step 2: After differencing, is there seasonal pattern? (ACF spike at lag s)
  Yes → need seasonal component (P, D, Q, s)
  No  → plain ARIMA(p, d, q) is sufficient

Step 3: PACF of stationary series → read p (and P at seasonal lags)
Step 4: ACF  of stationary series → read q (and Q at seasonal lags)

Step 5: Do you have external variables that:
  (a) causally influence the target?
  (b) pass Granger causality test?
  (c) have future values available at forecast time?
  All yes → include as exog
```

## Common Mistakes in Interviews

| Mistake | Correct Statement |
|---|---|
| "I differencing converts to stationary automatically" | "We manually difference the series; statsmodels cumsum predictions back" |
| "ACF gives AR order" | "PACF gives p (AR), ACF gives q (MA)" |
| "Use all correlated variables as exog" | "Must also pass Granger test AND be available at forecast time" |
| "Fit model, generate forecast" | "Always check residuals before trusting forecasts" |
| "d=1 always works" | "Test with ADF; sometimes d=0 or log-transform is better" |
| "Optimise only AIC" | "AIC for accuracy, BIC for parsimony — know the difference" |

---

# 10. Interview Answer Templates

## "How would you use SARIMAX to forecast with exogenous variables?"

> *"I'd follow a structured pipeline. First, test stationarity with ADF — if non-stationary, apply differencing (d=1) and retest to confirm. Then plot ACF and PACF on the stationary series: PACF cutoff tells me p, ACF cutoff tells me q. For seasonal parameters, I look at ACF/PACF at seasonal lags — spikes at lag 12 and 24 suggest P=2 and Q=1 with s=12. For exogenous variables, I start with domain knowledge to identify variables with genuine causal relationships, validate with Granger causality tests and cross-correlation analysis, check for multicollinearity with VIF, and critically verify future values are available at forecast time. After fitting, I always check residual diagnostics — residuals should be white noise with no remaining ACF structure. If structure remains, I increase p or q accordingly. Finally I forecast by providing future exogenous values alongside the model."*

## "How do you choose p and q?"

> *"Three approaches in order: First, plot PACF on the stationary series — where it cuts off inside the confidence band tells me p. Plot ACF — where it cuts off tells me q. Second, if plots are ambiguous or noisy, I grid search over p and q combinations and select using AIC for accuracy or BIC for parsimony — whichever fits the goal. Third, I always validate by checking residual ACF — if it still shows significant spikes, the model is under-specified and I increase p or q accordingly."*

## "Explain all the parameters in SARIMAX"

> *"SARIMAX has two groups of parameters. The non-seasonal group (p, d, q): p is the AR order — how many past values to use; d is the differencing order — how many times we difference to achieve stationarity; q is the MA order — how many past forecast errors to use. The seasonal group (P, D, Q, s): P, D, Q are the seasonal equivalents of p, d, q but operating at seasonal lags; s is the season length — 12 for monthly data. Finally X represents exogenous variables — external regressors that influence the target but are not influenced by it. Their coefficients are learned alongside the time series structure."*

## "How do you select exogenous variables?"

> *"Five steps: First, domain knowledge — the variable must have a genuine causal relationship with the target. Second, cross-correlation analysis at different lags to find if and when the variable influences the target. Third, Granger causality test to confirm the variable adds predictive power beyond the series' own history. Fourth, VIF check to catch multicollinearity between exogenous variables. Fifth — and most importantly — verify the variable's future values are available at forecast time. If I can't know the value of an exogenous variable in the future, it's useless for forecasting regardless of how correlated it is."*

---

# 11. Likely Follow-Up Questions & Answers

**Q: What is the difference between AR and MA components intuitively?**
> *"AR uses past actual values — it says 'sales last month and the month before are predictive of sales this month.' MA uses past forecast errors — it says 'I overestimated last month by 50 units, so I should adjust my forecast down this month.' AR captures momentum and level persistence. MA captures error correction and short-term shocks. Together they handle both the signal and the noise correction."*

**Q: Why is SARIMA better than just increasing p to capture seasonality?**
> *"You could technically use a very high p (e.g., p=12 or p=24) to capture yearly seasonality. But this requires estimating 12-24 AR coefficients from the same data, uses enormous degrees of freedom, and risks overfitting. SARIMA captures seasonality with just 1-2 seasonal parameters (P, Q), is far more parsimonious, and generalises much better. It's the difference between learning a special-purpose parameter vs overloading a general one."*

**Q: What if your exogenous variable is also a time series with its own trend?**
> *"Both series need to be stationary before including in the model, otherwise you risk spurious regression — finding high correlation that exists only because both series trend upward over time, not because they're genuinely related. I'd difference the exogenous variable alongside the target, or test for cointegration. If they're cointegrated, they share a long-run relationship and can be modeled together without differencing via an error correction model."*

**Q: How is SARIMAX different from a regression model with time features?**
> *"A regression model with month dummies and trend features is purely deterministic — it captures fixed seasonal patterns and trends. SARIMAX additionally captures autocorrelation structure — the fact that today's value depends on yesterday's value and yesterday's error in a probabilistic way. SARIMAX can also adapt to changing patterns over time through the AR and MA components. In practice, SARIMAX tends to perform better when autocorrelation is strong, while regression with features works better when external drivers dominate."*

**Q: What is the difference between AIC and BIC?**
> *"Both measure model quality penalised for complexity — lower is better. AIC penalises each additional parameter by 2. BIC penalises by log(n), where n is sample size — for n > 7, BIC penalises more than AIC, favouring simpler models. Use AIC when forecast accuracy is the priority. Use BIC when you want the most parsimonious model — useful for interpretability or when you have a small sample and risk overfitting."*

**Q: What do you do if residuals still show autocorrelation after fitting?**
> *"The model has not captured all the structure in the data. Diagnose which lag shows the spike: non-seasonal spike at lag 1-3 → increase p or q; seasonal spike at lag 12 → increase P or Q; spike at an unusual lag → investigate if there's a special event or structural break. I'd incrementally increase the relevant parameter, refit, and recheck residuals until they're white noise."*

**Q: How does SARIMAX handle missing values?**
> *"Statsmodels' SARIMAX uses a state space representation with the Kalman filter, which can handle missing values gracefully — it simply treats them as unobserved states and marginalises over them. However, for exogenous variables, missing future values are a harder problem — you either impute them or forecast them separately before passing to SARIMAX."*

---

# 12. Pre-Interview Checklist

```
□ Can I explain stationarity in 2 sentences?
□ Do I know how to run and interpret the ADF test?
□ Can I explain what d does precisely (not "auto-converts")?
□ Can I explain the difference between AR and MA clearly?
□ Do I know: PACF → p (AR), ACF → q (MA)?
□ Can I read ACF/PACF plots and identify cutoff points?
□ Can I explain what seasonal parameters P, D, Q, s each do?
□ Do I know how to choose s from domain knowledge + ACF?
□ Can I explain what exogenous variables are and how they enter?
□ Can I name all 5 steps for exogenous variable selection?
□ Do I know the critical rule: future exog values must be available?
□ Can I explain residual diagnostics and white noise?
□ Do I know what remaining ACF spikes in residuals mean?
□ Do I know AIC vs BIC difference and when to use each?
□ Can I explain Granger causality and its limitation?
□ Can I write the complete SARIMAX pipeline in Python from memory?
```

---

*Interview revision notes — Stationarity · ARIMA · SARIMA · SARIMAX · ACF/PACF · Exogenous Variable Selection · Residual Diagnostics*