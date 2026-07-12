# Logistic Regression — Linear Models for Classification

> **TL;DR.** Take [Linear Regression](Linear%20Regression.md)'s linear score `z = w·x + b`, squash it through a **sigmoid** into a probability in `(0, 1)`, and threshold that to a class. Train it by minimising **log-loss** (cross-entropy), *not* MSE — because MSE-with-sigmoid is non-convex. The output is a genuine, well-calibrated probability and each weight has a clean **odds-ratio** interpretation (`exp(wⱼ)`), which is why it's the default in credit scoring and medicine. Reach for it as your classification baseline when you want speed, probabilities, and interpretability; drop it when the decision boundary is strongly non-linear (it can only draw a hyperplane).

**Where it fits:** The workhorse of **binary (and multiclass) classification**. It's [Linear Regression](Linear%20Regression.md) + a sigmoid + a new loss — master that note first. Its evaluation lives in [Classification Metrics](Classification%20Metrics.md) (confusion matrix, precision/recall, ROC).
**Prereqs:** [Linear Regression](Linear%20Regression.md) (hyperplanes, weights, [[Gradient Descent]], regularization), sigmoid, logs, basic probability.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works — score → probability → class](#3-how-it-works--score--probability--class)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [Multiclass — One-vs-Rest vs Softmax](#6-multiclass--one-vs-rest-vs-softmax)
7. [Interpretation — Log-Odds & Odds Ratios](#7-interpretation--log-odds--odds-ratios)
8. [When It Breaks](#8-when-it-breaks)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

You want to separate two classes (churn vs no-churn). Draw a **line/hyperplane** between them. A new point's **signed distance** from that line tells you *how confident* you are and *which side* it's on — far on one side = very likely class 1, far on the other = very likely class 0, right on the line = a coin flip.

```
        class 1 (×)                    the raw score z = w·x + b
    ×  ×   ×                           is that signed distance:
  ×  ×  ╱────────  decision boundary      z ≫ 0  → deep in class-1 territory
      ╱   ∘  ∘                            z = 0  → on the boundary (p = 0.5)
     ╱ ∘  ∘   ∘  class 0 (∘)              z ≪ 0  → deep in class-0 territory

 problem: z ranges (−∞, +∞), but we need a probability in (0, 1).
 fix:     squash z through the sigmoid  σ(z) = 1/(1+e^−z)
```

- The sigmoid turns "distance from the boundary" into "probability of class 1." `σ(0)=0.5`, `σ(+∞)→1`, `σ(−∞)→0`. It's steepest at the boundary and **saturates** (flattens) far away — that flattening matters later for outliers (§8).
- 🎯 Why is it called *regression* if it classifies? Under the hood it **regresses a real number** (the log-odds), then a sigmoid + threshold turns that number into a class. The model is linear in the log-odds. `(certain)`

---

## 2. The Formal Core

**Model — linear score, then sigmoid:**

```
z  = w·x + b                     # linear score (the hyperplane), range (−∞, ∞)
p̂ = σ(z) = 1 / (1 + e^−z)        # probability of class 1,   range (0, 1)
ŷ  = 1 if p̂ ≥ threshold else 0   # threshold (default 0.5) → class
```

**Loss — Log-Loss / Binary Cross-Entropy.** For one sample with true label `y ∈ {0,1}`:

```
L = −[ y·log(p̂) + (1−y)·log(1−p̂) ]
```

Read it as two half-losses stitched together by the label: `(certain)`
- If `y = 1`, only `−log(p̂)` survives → huge loss when `p̂→0`, ~0 loss when `p̂→1`.
- If `y = 0`, only `−log(1−p̂)` survives → huge loss when `p̂→1`, ~0 when `p̂→0`.

The full objective averages over `n` samples (plus a regularizer — see below):

```
J(w) = (1/n) Σᵢ −[ yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ) ]   +   λ‖w‖²
```

**Where log-loss comes from:** it's the **negative log-likelihood** of the labels under a Bernoulli model, so minimising log-loss is exactly **maximum-likelihood estimation**. `(certain)`

**Why not MSE?** With the sigmoid inside, `MSE = (y − σ(z))²` is **non-convex** in `w` — gradient descent can stall in local minima. Log-loss with sigmoid is **convex**, guaranteeing the global optimum. (In linear regression MSE was convex because there was no sigmoid.) `(certain)`

**The gradient is the punchline — it's identical in form to linear regression:** `(certain)`

```
∂J/∂wⱼ = (1/n) Σᵢ (p̂ᵢ − yᵢ) · xᵢⱼ
```

`(prediction − target) × feature`. That elegant collapse (the sigmoid's derivative cancels the log's) is why the same gradient-descent machinery drives both models.

---

## 3. How It Works — score → probability → class

```
1. Encode/scale   one-hot categoricals; STANDARDIZE features (needed for GD + regularization)
2. Fit            gradient descent minimises log-loss → learns w, b   (no closed form!)
3. Score          z = w·x + b  →  p̂ = σ(z)
4. Threshold      ŷ = 1 if p̂ ≥ t else 0     (t is a business choice, not always 0.5)
5. Evaluate       confusion matrix / precision-recall / ROC — see [Classification Metrics](Classification%20Metrics.md)
```

**No normal equation.** Unlike linear regression, there's no closed-form `(XᵀX)⁻¹Xᵀy`; the sigmoid makes it non-linear in `w`, so you *must* iterate (gradient descent / Newton's method / L-BFGS). `(certain)`

**The decision boundary is linear.** `p̂ = 0.5 ⟺ z = 0 ⟺ w·x + b = 0` — a hyperplane. Logistic regression can *only* draw straight boundaries; for curved ones, add polynomial/interaction features (then the boundary is linear in the expanded space) or switch models. `(certain)`

**Thresholding is a lever, not a constant.** Default `0.5` assumes false positives and false negatives cost the same. They rarely do:
- **Cancer screening** — missing a case (FN) is catastrophic, so **lower** the threshold (e.g. 0.3) to catch more positives at the cost of more false alarms.
- **Spam-to-inbox** — a false positive (real mail → spam) is costly, so **raise** the threshold.

Threshold choice is downstream of the *probability*, so you tune it on validation data against a cost matrix — see [Classification Metrics](Classification%20Metrics.md).

---

## 4. Worked Example

A churn model learned `b = −1`, weights `w = [2, −1]` over standardized features `[CustServ_calls, account_length]`. Score a customer `x = [1.5, 0.5]`:

```
z  = w·x + b = 2(1.5) + (−1)(0.5) + (−1) = 3 − 0.5 − 1 = 1.5
p̂ = σ(1.5) = 1/(1 + e^−1.5) = 1/(1 + 0.223) = 0.818     → 82% likely to churn
ŷ  = 1  (since 0.818 ≥ 0.5)
```

**Its log-loss if the customer truly churned (`y = 1`):**

```
L = −[1·log(0.818) + 0] = −log(0.818) = 0.201     # small loss, good prediction
```

**If instead they did *not* churn (`y = 0`) — a confident mistake:**

```
L = −[0 + 1·log(1 − 0.818)] = −log(0.182) = 1.70   # ~8× bigger — confident & wrong is punished hard
```

**Odds-ratio reading of `w₁ = 2`:** `exp(2) ≈ 7.4` → each +1 SD in customer-service calls multiplies the **odds** of churning by ~7.4×, holding the other feature fixed (see §7).

---

## 5. Code / Implementation

**sklearn — note the `C` gotcha.** sklearn parametrises regularization by `C = 1/λ`, so **small C = strong regularization** (the opposite direction of `alpha` in Ridge/Lasso). L2 is on by default. `(certain)`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(),                 # scale inside the pipeline → no leakage
                    LogisticRegression(C=1.0,          # C = 1/λ  (smaller C ⇒ MORE regularization)
                                       penalty="l2",
                                       class_weight="balanced"))  # counter class imbalance
clf.fit(X_tr, y_tr)

clf.predict(X_te)               # hard classes (uses 0.5 threshold)
clf.predict_proba(X_te)[:, 1]   # calibrated P(class 1) — threshold this yourself for custom cost
```

**From scratch — the whole model in a gradient loop:**

```python
def fit_logreg(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))     # σ(z)
        w -= lr * (X.T @ (p - y)) / n          # ∂J/∂w = Xᵀ(p̂ − y)/n  — same shape as linear reg
        b -= lr * (p - y).mean()
    return w, b
# Standardize X first, or GD crawls.
```

**statsmodels — for coefficients you can defend.** Gives p-values (is this feature significant?) and, after `np.exp(...)`, odds ratios with confidence intervals — the reporting format regulators expect.

```python
import statsmodels.api as sm
logit = sm.Logit(y_tr, sm.add_constant(X_tr)).fit()
print(logit.summary())          # coef, std err, z, p-value, CI per feature
print(np.exp(logit.params))     # odds ratios
```

**Tuning C** is exactly the Ridge/Lasso cross-validation story: grid-search `C` on validation log-loss (or use `LogisticRegressionCV`). As regularization strengthens (C↓), accuracy first rises then falls as the model underfits.

---

## 6. Multiclass — One-vs-Rest vs Softmax

Logistic regression is binary at heart; two ways to reach `K` classes:

**One-vs-Rest (OvR).** Train `K` binary classifiers, each "class k vs everything else." Predict the class whose model gives the highest `p̂` — **argmax, even if none exceeds 0.5**. Simple, parallelisable; the go-to Scaler taught.

```
20 car brands  →  20 binary logistic models  →  predict = argmax over the 20 probabilities
```

**Softmax / Multinomial (usually better).** One model with a weight vector per class; convert all `K` scores to probabilities *jointly* so they sum to 1, and minimise categorical cross-entropy: `(certain)`

```
p̂_k = e^{z_k} / Σⱼ e^{z_j}          # softmax: normalizes K scores into one distribution
```

Softmax models the classes together (calibrated joint probabilities), whereas OvR trains each in isolation (probabilities need renormalising). In sklearn, `LogisticRegression` does multinomial softmax by default with the `lbfgs`/`saga` solvers; `OneVsRestClassifier` forces OvR. 🎯 *"OvR = K independent one-vs-all models + argmax; softmax = one joint model with normalised probabilities — prefer softmax when classes are mutually exclusive."*

---

## 7. Interpretation — Log-Odds & Odds Ratios

This is logistic regression's superpower and a guaranteed interview probe. Invert the sigmoid: `(certain)`

```
p = σ(z)   ⟺   z = log( p / (1−p) ) = w·x + b
                    └─ log-odds (the "logit"). Logit and sigmoid are inverses.
```

So **the model is linear in the log-odds.** A one-unit rise in `xⱼ` adds `wⱼ` to the log-odds → multiplies the **odds** by `exp(wⱼ)`:

- `wⱼ = 0` → `exp(0)=1` → feature has no effect on the odds.
- `wⱼ > 0` → `exp(wⱼ) > 1` → increases the odds of class 1 (odds ratio).
- `wⱼ < 0` → `exp(wⱼ) < 1` → decreases them.

Example: `wⱼ = 0.7` → `exp(0.7) ≈ 2.0` → "each extra unit **doubles the odds**." This clean, monotonic, per-feature story is why medicine (risk factors) and finance (credit scorecards) love logistic regression. Sign = direction, `exp(coef)` = multiplicative effect on odds — but only comparable across features if they're **standardized**.

---

## 8. When It Breaks

- **Non-linear boundaries.** It draws only hyperplanes. If classes interleave in a circle/XOR, it underfits — add polynomial features, or use trees/SVM-RBF/neural nets.
- **Outliers — a nuanced story.** Because the sigmoid **saturates**, an outlier *far on the correct side* contributes ~0 loss and ~0 gradient → logistic regression is actually **more robust** to those than linear regression. But an outlier on the **wrong side** (or a mislabeled point) sits where the loss `−log(p̂)` blows up → it drags the boundary. So: correct-side outliers, mostly harmless; wrong-side/mislabeled, damaging → find and fix. `(certain)`
- **Perfect separation** — the classic gotcha. If a feature (or combination) perfectly splits the classes, MLE pushes the weights to **±∞** to make `p̂` exactly 0/1; training won't converge and coefficients explode. **Regularization (L2) is the fix** — it caps the weights. `(certain)`
- **Class imbalance.** With 99% negatives, it can learn "always predict negative" (99% accuracy, useless). Use `class_weight='balanced'`, resampling, threshold tuning, and judge with PR-AUC/F1 — **not accuracy** ([Classification Metrics](Classification%20Metrics.md)).
- **Multicollinearity.** Like linear regression, correlated features make coefficients (and their odds-ratio interpretations) unstable — drop/combine features or lean on L2.
- **Assumptions** it *does* make: linearity **of the log-odds** in the features, independent observations, little multicollinearity, and a reasonably large sample. It does **not** assume normally-distributed features or residuals (a common confusion with linear regression). `(likely)`

---

## 9. Production & MLOps Notes

- **Probabilities are (largely) calibrated** out of the box — unlike SVMs or plain trees, `predict_proba` means what it says, so you can threshold by expected cost. Verify with a reliability curve; if off, apply Platt/isotonic calibration.
- **Choose the threshold by cost, not habit.** Convert the confusion-matrix costs into an operating point on the PR/ROC curve; ship that threshold, and re-check it when class balance drifts.
- **Solvers:** pick one that supports your penalty — `lbfgs` (default; L2/multinomial), `saga` (large/sparse; L1/ElasticNet), `liblinear` (small data). `(likely)`
- **Cheap, interpretable, auditable.** Inference is one dot product + a sigmoid; coefficients map to odds ratios a regulator will accept (credit scorecards are literally logistic-regression weights). Often the right *production* choice, not just a baseline.
- **Monitoring.** Track feature drift (PSI/KS), score/probability drift, and calibration over time; watch for coefficient sign flips across retrains (usually creeping multicollinearity or data-quality change).
- **Leakage discipline.** Fit scaler/encoder on train only, inside a `Pipeline`; tune `C` on validation, never on test.

---

## 10. Interview Lens

**"Why log-loss and not MSE for logistic regression?"** → 🎯 *"MSE composed with the sigmoid is non-convex, so gradient descent can get stuck; log-loss (the negative Bernoulli log-likelihood) is convex, so we reach the global optimum — and it penalises confident wrong predictions much harder."*

**"What does a logistic regression coefficient mean?"** → 🎯 *"It's the change in the log-odds per unit feature; `exp(coef)` is the odds ratio — the multiplicative effect on the odds of the positive class, holding others fixed."*

**Likely follow-ups:**
- *Why 'regression' in the name?* → It regresses a continuous quantity — the log-odds — which is linear in the features; the sigmoid + threshold turn it into a class.
- *Sigmoid vs logit?* → Inverses. Sigmoid maps score→probability; logit maps probability→score (log-odds).
- *How does it handle >2 classes?* → OvR (K binary models + argmax) or softmax/multinomial (one joint model, probabilities sum to 1); prefer softmax for mutually-exclusive classes.
- *Is the decision boundary linear?* → Yes, `w·x+b=0` is a hyperplane; add polynomial features for curved boundaries.
- *What is the C parameter?* → Inverse regularization strength (`C = 1/λ`); small C = strong regularization. Opposite direction to Ridge's `alpha`.
- *When does training blow up?* → Perfect separation → weights → ∞; fix with L2 regularization.
- *Is it robust to outliers?* → Correct-side outliers barely matter (sigmoid saturates); wrong-side/mislabeled ones hurt.
- *Why not evaluate with accuracy?* → Under imbalance a trivial model scores high; use precision/recall/F1/ROC-AUC — see [Classification Metrics](Classification%20Metrics.md).

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Linear-ish boundary, need probabilities + interpretability | **Logistic Regression** | calibrated, explainable (odds ratios), fast |
| Curved boundary, still want a linear model | **LogReg + polynomial features** | boundary becomes linear in expanded space |
| Strong non-linearity / interactions on tabular data | **[Gradient-boosted trees](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)** | capture non-linearity, usually top tabular accuracy |
| Max-margin separation, high-dim (e.g. text) | **[[SVM]]** | margin objective; kernels for non-linearity |
| Simple, fast, high-dim text baseline | **[Naive Bayes](Naive%20Bayes.md)** | trains in one pass, strong on bag-of-words |
| Predicting a **number**, not a class | **[Linear Regression](Linear%20Regression.md)** | wrong tool for continuous targets |
| Deep feature learning (images/text) | **[[Neural Network Fundamentals]]** | a logistic unit is literally the last layer |

**Decision rule:** logistic regression is the classification baseline you must beat. Keep it if the boundary is roughly linear and you value probabilities/interpretability; move to trees when residual analysis or CV shows it underfitting the non-linear structure. A single logistic unit is also the final layer of most classification neural nets — you're learning the atom of deep learning.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. Why do we use log-loss instead of MSE, and what property does that guarantee?
   <details><summary>answer</summary>MSE + sigmoid is non-convex (local minima); log-loss (negative Bernoulli log-likelihood) is convex → gradient descent finds the global optimum. It also punishes confident-and-wrong predictions heavily.</details>
2. Write the gradient of the log-loss w.r.t. `wⱼ`. What's surprising about it?
   <details><summary>answer</summary>`∂J/∂wⱼ = (1/n)Σ(p̂ᵢ − yᵢ)xᵢⱼ` — identical in form to linear regression's gradient; the sigmoid derivative cancels the log, giving `(prediction − target)·feature`.</details>
3. A coefficient is `w = 1.1`. Translate it for a stakeholder.
   <details><summary>answer</summary>`exp(1.1) ≈ 3.0` → each one-unit increase in that (standardized) feature roughly **triples the odds** of the positive class, holding others fixed.</details>
4. Your logistic model won't converge and weights are exploding. Diagnose and fix.
   <details><summary>answer</summary>Likely **perfect separation** — a feature perfectly splits the classes, so MLE drives weights to ±∞. Add L2 regularization (lower `C`).</details>
5. OvR vs softmax for 5 mutually-exclusive classes — which and why?
   <details><summary>answer</summary>Softmax/multinomial: it models all classes jointly so probabilities sum to 1 and are better calibrated; OvR trains 5 isolated one-vs-all models and takes the argmax.</details>
6. Is logistic regression sensitive to outliers? Be precise.
   <details><summary>answer</summary>Depends on side: outliers far on the *correct* side barely matter (sigmoid saturates → ~0 loss/gradient); outliers on the *wrong* side (or mislabeled) blow up `−log(p̂)` and shift the boundary.</details>
7. Why is accuracy a poor metric here, and where do you go instead?
   <details><summary>answer</summary>Under class imbalance a trivial "predict majority" model scores high but is useless. Use the confusion matrix, precision/recall/F1, PR-AUC, ROC-AUC — see [Classification Metrics](Classification%20Metrics.md).</details>

---

*Covers: sigmoid & the linear score, log-loss/cross-entropy & its MLE origin, why-not-MSE (convexity), the gradient, decision boundary & thresholding, C = 1/λ regularization & perfect separation, OvR vs softmax multiclass, log-odds & odds-ratio interpretation, outlier nuance, calibration, solvers, and class imbalance. Sourced from the Scaler Logistic Regression lecture; evaluation depth continues in [Classification Metrics](Classification%20Metrics.md).*
