# Linear Regression — Fitting the Best Line Through Your Data

> **TL;DR.** Linear regression models a continuous target as a weighted sum of features, `ŷ = w·x + b`, and finds the weights that minimise **mean squared error**. Two ways to solve it: the **closed-form normal equation** (exact, but `O(d³)`) or **gradient descent** (iterative, scales to many features). It's the *first model you reach for* on tabular regression — fast, cheap, and fully interpretable (each weight = "effect of this feature, holding others fixed"). Reach for it when the relationship is roughly linear and you need to *explain* the model; drop it when relationships are strongly non-linear or features are highly collinear (fix the latter with **regularization**: Ridge/Lasso).

**Where it fits:** The foundation of supervised **regression** (continuous `y`). Everything downstream — [Logistic Regression](Logistic%20Regression.md) (swap the output through a sigmoid), regularized models, even a single neuron in a [[Neural Network Fundamentals]] — is this idea with one twist added.
**Prereqs:** basic linear algebra (dot product, matrix inverse), [[Gradient Descent]], mean/variance, train/test split.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works — the end-to-end pipeline](#3-how-it-works--the-end-to-end-pipeline)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [Evaluating the Fit — R², Adjusted R², and friends](#6-evaluating-the-fit--r-adjusted-r-and-friends)
7. [The Five Assumptions — When It Breaks](#7-the-five-assumptions--when-it-breaks)
8. [Regularization — Ridge, Lasso, ElasticNet](#8-regularization--ridge-lasso-elasticnet)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

You have points scattered on a graph (experience → salary, area → house price). You want **one line** that passes "through the middle" of them, so that for a new `x` you can read off a predicted `y`.

```
price │              ×
      │           ×      ╱──  ŷ = w·x + b   (the line we learn)
      │        ×    ╱ ×
      │     ×   ╱ ×          residual eᵢ = yᵢ − ŷᵢ  (vertical gap)
      │   × ╱ ×
      │ ╱ ×
      └────────────────────── max_power
```

- Infinitely many lines pass through a cloud of points. **Which is "best"?** The one whose **total squared vertical error** is smallest — that's *Ordinary Least Squares (OLS)*.
- Why *vertical* distance and not perpendicular? Because we're predicting `y` from `x`; the error we care about is "how wrong is my predicted price," which is measured along the `y` axis.
- Every ML model here has the **same three parts**: a **model** (the line `w·x+b`), a **cost function** (MSE — how wrong we are), and an **optimizer** (normal equation or gradient descent — how we improve). Linear regression is the cleanest place to see all three. `(certain)`

---

## 2. The Formal Core

**The model.** For `d` features `x = (x₁,…,x_d)`:

```
ŷ = w₁x₁ + w₂x₂ + … + w_d x_d + b   =   wᵀx + b
```

- `wⱼ` = weight/coefficient of feature `j` (its *slope*); `b` = intercept/bias (value when all `x = 0`).
- Fold `b` into `w` by adding a constant `x₀ = 1` column → `ŷ = Xw`, with `X` the `n×(d+1)` design matrix.

**The cost — Mean Squared Error (MSE).**

```
J(w) = (1/n) · Σᵢ (yᵢ − ŷᵢ)²        # average squared residual
```

Why *squared* error (not the raw sum, not absolute, not cubic)? `(certain)`
- **Raw signed errors cancel** — a line miles off can sum to zero error (big +ve and −ve residuals cancel). Squaring removes the sign.
- **Absolute error `|e|`** also removes sign but is **not differentiable at 0** → nastier to optimise.
- **Cubed error** brings the sign problem back (odd power).
- Squared error is **convex + smooth** (one global minimum, clean gradient) and is the **maximum-likelihood** estimate if noise is Gaussian. It does pay a price: squaring makes it **sensitive to outliers** (a residual of 10 contributes 100).

**Closed-form solution — the Normal Equation.** Setting `∂J/∂w = 0` solves it exactly in one shot — `w* = (XᵀX)⁻¹ Xᵀ y` — no iteration, no learning rate. The catch: inverting that matrix is **`O(d³)`** and fails when `XᵀX` is singular (collinear features) — which is exactly *why* gradient descent (§3) takes over once `d` is large. `(certain)`

**Interpreting a weight.** For `ŷ = 3 + 2·TV + 3·SM`: *holding TV fixed, a one-unit rise in SM raises predicted sales by 3.* Sign = direction of effect, magnitude = strength — **but only comparable across features if the features are on the same scale** (§7 multicollinearity & §3 scaling).

---

## 3. How It Works — the end-to-end pipeline

Fitting weights is one step; a real regression run is a pipeline:

```
1. Split         train_test_split (e.g. 70/30) BEFORE anything else — no peeking at test
2. Encode        categoricals → one-hot / dummies (drop_first to dodge the dummy trap, below)
3. Impute        fill missing (median is robust) — fit on TRAIN, apply to test
4. Scale         standardize features (needed for GD, regularization, coef comparison)
5. Fit           normal equation OR gradient descent → learn w, b
6. Evaluate      R²/RMSE on the HELD-OUT test set, not train
7. Diagnose      residual plots, VIF, assumption checks (§7)
```

**Gradient descent (step 5, the scalable optimizer).** Start at random `w`, repeatedly step *downhill* on `J`:

```
wⱼ  :=  wⱼ − α · ∂J/∂wⱼ ,   where   ∂J/∂wⱼ = −(2/n) Σᵢ (yᵢ − ŷᵢ) xᵢⱼ
```

- `α` = **learning rate**. Too large → overshoot/diverge; too small → crawls. 
- **Variants:** *Batch* GD (all rows per step — stable, slow), *Stochastic* GD (one row — noisy, fast, escapes shallow spots), *Mini-batch* (a few hundred rows — the practical default).
- **Normal equation vs GD:** exact & no tuning vs scales to huge `d` and streams data. Rule of thumb: `d ≲ 10⁴` → normal equation is fine; bigger → GD. `(likely)`

**Why scaling matters here.** Feature scaling does **not** change the OLS *fit* or `R²` for the plain closed-form (it's scale-equivariant), but it is essential for three things: `(certain)`
1. **GD convergence** — unscaled features make the loss surface a stretched valley; GD zig-zags. Scaling → round bowl → fast, direct descent.
2. **Regularization** (§8) — the penalty `Σwⱼ²` is scale-sensitive; without scaling, large-unit features are unfairly penalised.
3. **Coefficient comparison** — you can only say "SM matters more than TV" if both are standardized.

**One-hot & the dummy-variable trap.** Encode a `k`-category column as `k` binary columns, then **drop one** (`drop_first=True`). Keeping all `k` makes them sum to the intercept's constant column → perfect collinearity → `XᵀX` singular. Ordinal integer coding (America=1, Europe=2, Asia=3) is *wrong* — it invents a fake ordering/spacing. `(certain)`

---

## 4. Worked Example

Fit a line to three points by hand: `(x, y) = (1, 2), (2, 2), (3, 4)`. Model `ŷ = wx + b`.

**Closed form via the least-squares slope/intercept formulas** (`x̄ = 2`, `ȳ = 8/3 ≈ 2.667`):

```
w = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
  = [(-1)(-0.667) + (0)(-0.667) + (1)(1.333)] / [1 + 0 + 1]
  = (0.667 + 0 + 1.333) / 2  =  2.0 / 2  =  1.0

b = ȳ − w·x̄ = 2.667 − 1.0·2 = 0.667
```

So `ŷ = 1.0·x + 0.667`. Predictions: `[1.667, 2.667, 3.667]`, residuals `[+0.333, −0.667, +0.333]`.

**Now score it with R²:**

```
SS_res = 0.333² + (−0.667)² + 0.333²  = 0.111 + 0.444 + 0.111 = 0.667
SS_tot = Σ(yᵢ − ȳ)² = (−0.667)² + (−0.667)² + (1.333)²        = 2.667
R²     = 1 − SS_res/SS_tot = 1 − 0.667/2.667 = 1 − 0.25 = 0.75
```

Our line explains **75%** of the variance in `y` versus the dumb "always predict the mean" baseline. That baseline is the anchor of R² — see §6.

---

## 5. Code / Implementation

**sklearn — the 5-line baseline:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)     # solves the normal equation internally

print(model.coef_, model.intercept_)           # learned w, b
print(model.score(X_te, y_te))                 # R² on HELD-OUT data (the number that counts)
```

**statsmodels — when you need *inference*, not just prediction.** sklearn gives you `coef_`; it does **not** give p-values or confidence intervals. Interviewers love this gap: to know whether a coefficient is *statistically significant*, use statsmodels. `(certain)`

```python
import statsmodels.api as sm
X_sm = sm.add_constant(X_tr)                    # explicit intercept column
ols  = sm.OLS(y_tr, X_sm).fit()
print(ols.summary())    # per-coef std err, t-stat, p-value (H0: wⱼ=0), 95% CI,
                        # R², Adjusted R², F-statistic (is the whole model useful?)
```

**From scratch — gradient descent (this is the part that teaches):**

```python
def fit_gd(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        y_hat = X @ w + b
        err   = y_hat - y                      # (ŷ − y)
        w -= lr * (2/n) * (X.T @ err)          # ∂J/∂w
        b -= lr * (2/n) * err.sum()            # ∂J/∂b
    return w, b
# NOTE: standardize X first, or GD zig-zags on unscaled features.
```

**Regularized + tuned (the real-world version):** standardize, then grid-search `λ` (`alpha`) with cross-validation — never tune it on the test set.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), Ridge())          # scale INSIDE the CV to avoid leakage
grid = GridSearchCV(pipe, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
                    scoring="r2", cv=5)
grid.fit(X_tr, y_tr)
print(grid.best_params_, grid.score(X_te, y_te))
```

---

## 6. Evaluating the Fit — R², Adjusted R², and friends

**R² (coefficient of determination).** *How much better are you than always predicting the mean?*

```
R² = 1 − SS_res / SS_tot ,   SS_res = Σ(yᵢ−ŷᵢ)²  ,  SS_tot = Σ(yᵢ−ȳ)²
```

- `R² = 1`: perfect. `R² = 0`: you're no better than the mean model (the "dumbest reasonable baseline"). **`R² < 0` (down to −∞): you're *worse* than the mean** — genuinely possible on test data with a bad model. `(certain)`
- 🎯 **The trap:** R² **never decreases when you add a feature**, even a random one — so a high train R² can be pure overfitting. That's what Adjusted R² fixes.

**Adjusted R² — R² with a complexity tax** (`n` samples, `p` features):

```
R²_adj = 1 − (1 − R²) · (n − 1)/(n − p − 1)
```

It only rises if a new feature helps *more than chance*. Use it to **compare models with different feature counts**; report plain R² for a single fixed model. `(certain)`

**Error metrics (same units as `y`, so more tangible for stakeholders):**

| Metric | Formula | Character |
|---|---|---|
| **MAE** | `mean(|y−ŷ|)` | robust to outliers; every error weighted equally |
| **MSE** | `mean((y−ŷ)²)` | punishes big errors hard; units are `y²` |
| **RMSE** | `√MSE` | MSE back in `y`'s units — the usual reporting metric |
| **MAPE** | `mean(|y−ŷ|/|y|)·100` | % error, scale-free; explodes when `y≈0` |

Pick MAE when outliers are noise you don't want to chase; RMSE/MSE when large misses are genuinely costly.

---

## 7. The Five Assumptions — When It Breaks

Remember them as **L.I.N.E. + no collinearity**. Each has a *diagnostic* and a *fix* — knowing the fix is what separates a junior answer from a senior one.

| #     | Assumption                                                          | How to check                                         | If violated                                                                                              |                                                                |
| ----- | ------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **L** | **Linearity** — `E[y                                                | x]` is linear in the features                        | residual-vs-fitted plot shows no curve                                                                   | add polynomial/interaction features, or transform (`log`, `√`) |
| **I** | **Independent errors** — no autocorrelation                         | Durbin–Watson ≈ 2; residuals vs time show no pattern | common in [Time Series](../Time%20Series/Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md) → use ARIMA/add lags                                                           |                                                                |
| **N** | **Normality of residuals**                                          | Q–Q plot, histogram of residuals                     | usually from outliers/skew → transform `y` (e.g. `log(price)`); large-n CLT often rescues inference      |                                                                |
| **E** | **Equal variance (homoscedasticity)**                               | residual-vs-fitted "funnel" = bad                    | transform `y`, use **Weighted Least Squares**, or robust (heteroscedasticity-consistent) standard errors |                                                                |
| **—** | **No multicollinearity** — features not linear combos of each other | **VIF**                                              | drop/combine features, PCA, or use **Ridge**                                                             |                                                                |

**Multicollinearity & VIF (the one that gets probed most).** Regress feature `xⱼ` on all *other* features; if that `R²ⱼ` is high, `xⱼ` is redundant. `(certain)`

```
VIF_j = 1 / (1 − R²ⱼ)      VIF < 5  ok  ·  5–10 high  ·  >10 severe
```

Why it hurts: collinearity doesn't wreck *predictions* much, but it makes **coefficients unstable and their standard errors balloon** — signs can flip with a tiny data change, so you can't trust the interpretation. That's the whole reason Ridge (§8) exists: it stabilises coefficients under collinearity.

**Outliers.** OLS minimises *squared* error, so a single far point drags the whole line toward it (leverage). Detect with residual/leverage plots; fix by removing genuine errors, capping, transforming `y`, or switching to **robust regression** (Huber loss, which is squared near zero but linear in the tails).

---

## 8. Regularization — Ridge, Lasso, ElasticNet

**The problem it solves.** To drive training error down, the optimizer inflates weights — big `wⱼ` = a wiggly, high-variance model that overfits (especially with polynomial features). Regularization adds a **penalty on weight size** so the model is torn between fitting the data and staying simple:

```
minimize   MSE  +  λ · (weight penalty)
                   └─ λ ≥ 0 is the knob: λ=0 → plain OLS (overfit risk);
                      λ→∞ → all weights → 0 (underfit). Tune λ by cross-validation.
```

| | Penalty | Effect | Use when |
|---|---|---|---|
| **Ridge (L2)** | `λ Σ wⱼ²` | shrinks weights smoothly toward 0, **never exactly 0** | many correlated features; you want to keep them all but tame them |
| **Lasso (L1)** | `λ Σ \|wⱼ\|` | drives some weights **exactly to 0** → automatic **feature selection**, sparse model | you suspect many features are useless |
| **ElasticNet** | `λ(α Σ\|wⱼ\| + (1−α) Σwⱼ²)` | blend of both | many features, some correlated *and* some useless |

🎯 **Why does Lasso zero out weights but Ridge doesn't?** L2's gradient shrinkage is `2λwⱼ` — *proportional to the weight*, so as `wⱼ→0` the push vanishes and it stalls just short of zero. L1's is `λ·sign(wⱼ)` — a **constant push regardless of size**, so it drives weights clean through to exactly 0. Geometrically, the L1 constraint region is a **diamond** whose corners lie on the axes, and the loss contours tend to touch it *at a corner* (a zero coordinate); L2's circle has no corners. `(certain)`

**Always standardize before regularizing** — otherwise the penalty punishes features for their units, not their importance (§3).

This is the **bias–variance tradeoff** made tunable: increasing `λ` adds bias but cuts variance. (Same tradeoff that ensembles attack differently — see [Ensemble Methods](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md).)

---

## 9. Production & MLOps Notes

- **Inference is nearly free** — a prediction is one dot product. Linear regression is the cheapest, lowest-latency model you can ship, and trivially explainable (each coefficient is a story a regulator/PM accepts). Often the right *baseline*, sometimes the right *final* model.
- **Extrapolation is dangerous.** A line keeps going forever — predict outside the training range of `x` and you get confident nonsense (negative prices, absurd extremes). Guard the input domain in production.
- **Skewed targets:** for right-skewed money/count targets, model `log(y)` — it stabilises variance (helps homoscedasticity + normality) and turns multiplicative effects additive. Remember to `exp()` predictions back (with a bias correction if you need unbiased means).
- **Retraining & coefficient drift:** monitor feature drift (PSI/KS) and watch for coefficient sign flips over retrains — a flip usually signals creeping multicollinearity or data-quality change, not a real reversal of effect.
- **Leakage discipline:** fit the scaler/imputer on **train only** and apply to test (put them in the same `Pipeline` so CV can't cheat). Never select `λ` on the test set — that's using test data for training.
- **Scaling for interpretation vs prediction:** if a stakeholder asks "which feature matters most," you must report coefficients from *standardized* features; raw-unit coefficients aren't comparable.

---

## 10. Interview Lens

> The question behind most linear-regression questions: *do you understand the model, its cost, its optimizer, and the assumptions that make its coefficients trustworthy?*

**"Why do we square the errors?"** → 🎯 *"To stop positive and negative residuals cancelling, to get a smooth convex loss with a clean gradient (unlike `|e|`, which isn't differentiable at 0), and because squared loss is the MLE under Gaussian noise — at the cost of outlier sensitivity."*

**"Normal equation or gradient descent?"** → 🎯 *"Normal equation `w=(XᵀX)⁻¹Xᵀy` is exact and tuning-free but `O(d³)` and fails on singular `XᵀX`; gradient descent scales to large `d` and streaming data but needs a learning rate. Small d → normal equation, big d → GD."*

**Likely follow-ups:**
- *Can R² be negative?* → Yes, on test data a model worse than predicting the mean gives `R² < 0`. `(certain)`
- *R² vs Adjusted R²?* → R² never drops when you add features; Adjusted R² penalises feature count, so use it to compare models of different size.
- *What does multicollinearity break?* → Not prediction much — it inflates coefficient variance/standard errors and makes signs unstable, so *interpretation* becomes unreliable. Detect with VIF (>10 severe); fix with Ridge / dropping features / PCA.
- *Ridge vs Lasso?* → Both penalise weight size; Lasso (L1) zeroes weights → feature selection & sparsity; Ridge (L2) shrinks smoothly → better with correlated features you want to keep.
- *Do I need to scale features?* → Not for plain-OLS fit or R², but yes for GD convergence, for regularization fairness, and to compare coefficient magnitudes.
- *List the assumptions.* → **L.I.N.E.**: Linearity, Independent errors, Normal residuals, Equal variance — plus no multicollinearity.
- *Why is squared error sensitive to outliers, and the fix?* → A residual of 10 contributes 100; a lone outlier dominates. Fix with robust/Huber loss, or transform/cap.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Roughly linear, need interpretability/speed | **Linear Regression** | cheap, explainable, strong baseline |
| Curved relationship | **Polynomial features + LR** | keep linear machinery, add `x², x³…` (watch overfit) |
| Many/correlated features, overfitting | **Ridge / Lasso / ElasticNet** | shrink or select weights |
| Outlier-heavy target | **Huber / RANSAC regression** | linear-in-tails loss resists outliers |
| Strong non-linearity, interactions, tabular | **[Gradient-boosted trees](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)** | capture non-linearity automatically, usually win on tabular accuracy |
| Predicting a **class**, not a number | **[Logistic Regression](Logistic%20Regression.md)** | same linear core, squashed by a sigmoid into a probability |
| Errors autocorrelated over time | **[Time Series](../Time%20Series/Seasonal%20ARIMA%20with%20Exogenous%20Regressors.md) (ARIMA/SARIMAX)** | linear regression assumes independent errors |

**Decision rule:** start with linear regression as the baseline you must beat. If residual plots show curvature → add polynomial features or move to trees. If coefficients are unstable → regularize. If you need probabilities/classes → logistic. Only accept a complex model if it beats the linear baseline by enough to justify the loss of interpretability (**Occam's razor**).

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. Write the OLS closed-form solution and say when you'd use gradient descent instead.
   <details><summary>answer</summary>`w = (XᵀX)⁻¹Xᵀy`. It's exact but `O(d³)` and fails when `XᵀX` is singular (collinear features). Use GD when `d` is large, data streams, or `XᵀX` is ill-conditioned.</details>
2. Your model gets R² = 0.92 on train and −0.1 on test. What happened, and what does a negative R² mean?
   <details><summary>answer</summary>Overfitting. A negative test R² means the model does *worse than predicting the mean* on unseen data — it has learned train noise. Regularize / reduce features / get more data.</details>
3. Why does Lasso produce exactly-zero weights while Ridge only shrinks them?
   <details><summary>answer</summary>L1's gradient push is constant (`λ·sign(w)`), driving weights fully to 0; L2's push is proportional (`2λw`) and vanishes near 0. Geometrically, the L1 diamond's corners sit on the axes.</details>
4. Name the five assumptions and one diagnostic for each.
   <details><summary>answer</summary>Linearity (residual-vs-fitted), Independent errors (Durbin–Watson), Normal residuals (Q–Q plot), Equal variance/homoscedasticity (funnel in residual plot), No multicollinearity (VIF>10).</details>
5. When does feature scaling change your linear-regression results, and when does it not?
   <details><summary>answer</summary>It does **not** change plain-OLS fit or R². It **does** matter for gradient-descent convergence, for fair regularization penalties, and for comparing coefficient magnitudes.</details>
6. You one-hot encode a 3-category feature into 3 columns and the fit blows up. Why?
   <details><summary>answer</summary>The dummy-variable trap: the 3 dummies sum to the constant/intercept column → perfect collinearity → `XᵀX` singular. Drop one dummy (`drop_first=True`).</details>
7. Why prefer squared error over absolute error, and what does that cost you?
   <details><summary>answer</summary>Squared error is smooth/differentiable everywhere and convex with a clean gradient (absolute error isn't differentiable at 0). Cost: heightened sensitivity to outliers (residuals are squared).</details>

---

*Covers: OLS & the normal equation, MSE & why squared, gradient descent (batch/SGD/mini-batch), feature scaling, one-hot & the dummy trap, R²/Adjusted R²/MAE/RMSE/MAPE, the five assumptions + VIF & fixes, outliers/robust regression, Ridge/Lasso/ElasticNet & the bias–variance tradeoff, cross-validation for λ, statsmodels inference, and production concerns. Sourced from Scaler Linear Regression Lectures 1–4.*
