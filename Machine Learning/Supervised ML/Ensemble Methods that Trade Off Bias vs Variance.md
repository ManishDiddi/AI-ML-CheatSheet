# Bagging & Boosting — Ensemble Methods that Trade Off Bias vs Variance

> **TL;DR.** Both combine many weak learners into one strong model, but they attack opposite problems. **Bagging** trains learners *in parallel* on bootstrap samples and averages them → kills **variance** (use when your model overfits; Random Forest is the flagship). **Boosting** trains learners *sequentially*, each fixing the last one's errors → kills **bias** (use for max accuracy on clean data; XGBoost / LightGBM / CatBoost are the flagships). Reach for bagging when data is noisy/outlier-heavy, boosting when data is clean and you need every last point of accuracy.

**Where it fits:** Supervised tabular ML — still the *default winning approach* on structured/tabular data (boosting wins most Kaggle tabular competitions; deep learning rarely beats it there).
**Prereqs:** [decision trees](Decision%20Trees.md), [[bias-variance-tradeoff]], [[gradient-descent]], [[cross-validation]].

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Interpretability — SHAP & the Importance Trap](#7-interpretability--shap--the-importance-trap)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

> *"Both Bagging and Boosting are ensemble methods — they combine multiple weak learners into a stronger model. They differ fundamentally in HOW they combine them and WHAT problem they solve."*

```
Bagging  →  solves HIGH VARIANCE  (model overfits)  → train in PARALLEL, then AVERAGE
Boosting →  solves HIGH BIAS      (model underfits) → train SEQUENTIALLY, each fixes prior errors
```

- **Bagging** = *"ask 100 independent experts and average their votes."* Each expert overfits in a different direction; averaging cancels the noise.
- **Boosting** = *"a single expert who keeps studying their own past mistakes."* Each round it focuses on what the ensemble still gets wrong.

**The four ensemble families** (be able to name all four; this note dives into the first two, the bias/variance workhorses):
- **Bagging** — parallel learners on bootstrap samples, then averaged → cuts **variance** (Random Forest).
- **Boosting** — sequential learners, each correcting the last's errors → cuts **bias** (GBDT/XGBoost).
- **Stacking** — train several *diverse* base models, then a **meta-model** learns how best to combine their (out-of-fold) predictions.
- **Cascading** — chain models by **confidence**: pass only the low-confidence cases on to the next, costlier stage (often a human) — common in fraud/medical screening.

---

## 2. The Formal Core

**Bagging — why averaging reduces variance.** For `N` learners each with variance `σ²` and pairwise correlation `ρ`:

```
Var(average) = ρσ²  +  (1 − ρ)·(σ²/N)
                ▲                ▲
        correlation floor   averaged-away part
```

- `ρ = 0` (uncorrelated) → `Var = σ²/N`  → variance drops by a factor of N.
- `ρ = 1` (identical) → `Var = σ²` → **no reduction at all**.
- **Key insight:** once N is large the second term vanishes, so the *floor* `ρσ²` dominates. **The whole game is lowering ρ** — which is exactly what Random Forest's feature subsampling does. `(certain)`

**Boosting — additive model fit by gradient descent in function space.**

```
F_m(x) = F_{m-1}(x) + η · h_m(x)          η = learning rate (shrinkage)

h_m  is fit to the NEGATIVE GRADIENT of the loss at F_{m-1}:
     For MSE loss:  −∂L/∂F = (y − ŷ) = the residual
```

Each new tree `h_m` points in the direction of steepest loss decrease. Swap in *any* differentiable loss and the same framework works — fit the negative gradient of **log-loss** for classification, of **pinball loss** for quantile regression, and so on. That generality is the power of GBDT. `(certain)`

---

## 3. How It Works

### Bagging (parallel)

```
Original dataset D (n samples)
         │
         ├── Bootstrap sample D₁ → Model 1
         ├── Bootstrap sample D₂ → Model 2
         ├── Bootstrap sample D₃ → Model 3
         │         ...
         └── Bootstrap sample Dₖ → Model K
                                      │
                    Final = AVERAGE (regression)
                         or MAJORITY VOTE (classification)
```

**Bootstrap = sampling with replacement.** Each `Dₖ` has the same size n as the original; some rows repeat, **~37%** never appear (`(1 − 1/n)ⁿ → 1/e ≈ 0.368`). Those never-seen rows are the **Out-of-Bag (OOB)** set → a free, built-in validation set, no separate split needed.

**Random Forest = Bagging + extra decorrelation:**

```
Bagging:        each tree sees different ROWS (bootstrap)
Random Forest:  each tree sees different ROWS + a random COLUMN subset at each split
                → column subsampling pushes ρ toward 0
                → more variance reduction than pure bagging
                → slight bias increase (weaker trees), but variance win dominates
```

### Boosting (sequential)

```
F₀ = initial prediction (mean of y for regression)
  ↓  compute residuals/errors on F₀
Train Model 1 on those errors →  F₁ = F₀ + η × Model1
  ↓  compute NEW residuals on F₁
Train Model 2 on those errors →  F₂ = F₁ + η × Model2
  ↓  ...repeat N times...
Final = F₀ + η×M1 + η×M2 + ... + η×Mₙ
```

```
A single stump (depth-1 tree):   simple, underfits (high bias, low variance), but stable
100 stumps boosted:              collectively learn complex patterns
                                 → bias drops dramatically → low bias + moderate variance ✅
```

---

## 4. Worked Example

**(a) Variance reduction — why ρ is everything.** Say each tree has `σ² = 1.0` and we ensemble `N = 100`:

```
Uncorrelated (ρ=0):     Var = 0·1 + 1·(1/100)            = 0.010   (100× reduction)
Plain bagging (ρ=0.20): Var = 0.20·1 + 0.80·(1/100)      = 0.208   (correlation floor dominates!)
Random Forest (ρ=0.05): Var = 0.05·1 + 0.95·(1/100)      = 0.0595  (≈3.5× better than bagging)
```

→ Going from 100 to 1000 trees barely helps (`σ²/N` already tiny); **lowering ρ from 0.20 → 0.05 helps a lot.** That is the entire reason Random Forest beats vanilla bagging.

**(b) One boosting round — residuals shrink.** Targets `y = [10, 20, 30]`, learning rate `η = 0.5`:

```
F₀ = mean(y) = 20            → residuals r₀ = y − F₀ = [−10,  0, +10]
Tree₁ fits r₀, predicts        h₁ = [−8, 0, +8]
F₁ = F₀ + η·h₁ = 20 + 0.5·[−8,0,8] = [16, 20, 24]
                              → residuals r₁ = [−6, 0, +6]   ← smaller than r₀ ✅
```

Each round nudges predictions toward the targets; `η` controls how big each nudge is.

---

## 5. Code / Implementation

**Random Forest baseline (robust first model):**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators = 300,
    max_features = "sqrt",   # column subsampling per split → lowers ρ (the whole point)
    n_jobs       = -1,       # trees are independent → embarrassingly parallel
    oob_score    = True,     # free validation estimate from the ~37% OOB rows
)
rf.fit(X_train, y_train)
print(rf.oob_score_)          # generalization estimate without a held-out split
```

**XGBoost with early stopping (the right way to set n_estimators):**

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators         = 5000,    # set high — early stopping finds the right number
    learning_rate        = 0.01,    # low LR + many trees generalizes better
    max_depth            = 5,       # shallow trees = weak learners (boosting wants these)
    subsample            = 0.8,     # row subsampling → decorrelates, regularizes
    colsample_bytree     = 0.8,     # column subsampling
    reg_lambda           = 1.0,     # L2 on leaf weights (standard start is 1, NOT 0.01)
    gamma                = 0.1,     # min loss reduction to make a split
    min_child_weight     = 5,       # min summed instance weight in a leaf (anti-memorization)
    early_stopping_rounds= 50,      # stop when val metric stalls for 50 rounds
    eval_metric          = "logloss",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print(f"Optimal trees: {model.best_iteration}")
```

---

## 6. When It Breaks

### Bagging limitations
```
❌ Does NOT fix high bias — averaging 100 underfit models = an underfit model.
❌ Memory/latency heavy at inference — stores & queries all K trees.
❌ Less interpretable — 500 trees can't be explained to a stakeholder (see §7).
❌ Random Forest struggles with: high-dim sparse text · extrapolation beyond the
   training range (time series trends) · extreme imbalance without explicit handling.
```

### Boosting limitations
```
❌ SENSITIVE TO OUTLIERS (its most famous weakness) — each round chases hard
   examples; outliers ARE hard examples → it keeps boosting noise → overfit risk.
❌ Sequential = slower to train (XGB/LGBM parallelize WITHIN a tree, not across trees).
❌ More hyperparameters — get them wrong → severe over/underfit.
❌ Overfits without early stopping — too many estimators memorize training noise.
❌ Less robust to distribution shift in production — tuned tightly to training patterns.
```

### XGBoost overfitting — diagnose FIRST, then fix in order
```
Symptom: Train=98%, Val=79% → 19% gap = SEVERE overfit.
Root-cause checklist: max_depth=10, n_estimators=1000, subsample=1.0, colsample=1.0,
                      lambda=0, gamma=0, min_child_weight=1  → every knob at MAX complexity.

Fix in THIS order (not all at once):
  1. Early stopping       → sets n_estimators from the val signal (no guessing)
  2. ↓ max_depth          → 10 → 4–6 (biggest single lever)
  3. subsample/colsample  → 1.0 → 0.8 (decorrelate trees)
  4. regularize           → reg_lambda 0→1.0, gamma 0→0.1, min_child_weight 1→5
  5. learning_rate LAST   → ↓ LR with ↑ n_estimators (they always move together)
```

| Parameter | Controls | Overfit-fix direction |
|---|---|---|
| `max_depth` | Tree depth | Reduce (10 → 4-6) |
| `n_estimators` | Number of trees | Use early stopping |
| `learning_rate` | Step size per tree | Reduce (pair with more trees) |
| `subsample` | Row fraction per tree | Reduce (1.0 → 0.8) |
| `colsample_bytree` | Feature fraction per tree | Reduce (1.0 → 0.8) |
| `reg_lambda` | L2 on leaf weights | Increase (0 → 1.0) |
| `gamma` | Min gain to split | Increase (0 → 0.1-1.0) |
| `min_child_weight` | Min weight in a leaf | Increase (1 → 5-10) |
| `scale_pos_weight` | Class imbalance weight | = negative/positive count |

---

## 7. Interpretability — SHAP & the Importance Trap

Tree ensembles are black boxes, but you almost always must explain them — to debug, to convince a stakeholder, or to satisfy regulation (credit, insurance, healthcare). Three importance methods, in increasing trustworthiness:

- **Built-in `feature_importances_` (gain / split-count)** — fast but **biased**: it inflates high-cardinality and continuous features (they offer more places to split) and arbitrarily splits credit among correlated features. Never ship a decision based on this alone. `(certain)`
- **Permutation importance** — shuffle one feature, measure the validation-score drop. Model-agnostic and honest about *predictive value*, but misleads under correlation (a correlated proxy survives the shuffle) and costs one full eval per feature.
- **SHAP (SHapley Additive exPlanations)** — the production standard. It attributes each *individual prediction* to its features using game-theoretic Shapley values: a feature's contribution is its average marginal effect across all orderings of the features. Why it wins:
    - **Local + global:** explains a single prediction (*"this loan was denied: DTI +0.4, income −0.2"*) **and** aggregates to a global ranking.
    - **Additive & exact:** `prediction = base_value + Σ shap_values` — contributions sum exactly to the output.
    - **TreeSHAP** computes this *exactly* for trees in polynomial time (general SHAP is exponential), so it's fast enough for production.

```python
import shap
explainer   = shap.TreeExplainer(model)        # exact & fast for tree ensembles
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)          # GLOBAL: features ranked by impact + direction
shap.force_plot(explainer.expected_value,      # LOCAL: why THIS row got its prediction
                shap_values[0], X_val.iloc[0])
```

🎯 **Interview line:** *"Default gain importance is biased toward high-cardinality features, so for anything that matters I use SHAP — it's local + global, additive, and TreeSHAP makes it exact and fast on trees."*

---

## 8. Production & MLOps Notes

### Imbalanced classification (e.g. fraud) — the part interviews + real systems both probe

**The accuracy trap:** at 999 genuine : 1 fraud, predicting "all genuine" scores **99.9% accuracy** and is useless. Accuracy is the wrong metric under imbalance.

```
Precision = of all flagged fraud, how many are real?      (false-alarm rate)
Recall    = of all real fraud, how many did we catch?     (miss rate)
F1        = harmonic mean of precision & recall
F2        = weights RECALL 2× — right for fraud (a miss costs more than a false alarm)
PR-AUC    = area under precision–recall curve → honest under imbalance
WHY NOT ROC-AUC? Inflated by the huge true-negative count; looks great on mediocre models.
```

```python
from sklearn.metrics import fbeta_score, average_precision_score

model = XGBClassifier(
    scale_pos_weight = 999/1,        # = count(neg)/count(pos): re-weights the gradient
    eval_metric      = "aucpr",      # optimize PR-AUC during training
    early_stopping_rounds = 50, subsample = 0.8, min_child_weight = 5,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Threshold tuning — default 0.5 is WRONG under severe imbalance (single biggest practical win)
y_proba = model.predict_proba(X_val)[:, 1]
best_t, best_f2 = 0.5, 0
for t in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    f2 = fbeta_score(y_val, (y_proba >= t).astype(int), beta=2)
    if f2 > best_f2: best_f2, best_t = f2, t
```

**Full imbalance toolkit:** `scale_pos_weight` · **threshold tuning** (most missed) · SMOTE (train-only, never val/test) · `eval_metric='aucpr'` · majority undersampling.

**Outlier paradox:** generally outliers = noise → don't boost them; but in fraud/anomaly detection the outliers *are the signal*, so boosting's focus on hard examples is an **advantage** — control it with `min_child_weight` and `subsample` so it learns outlier-*like patterns*, not specific outliers.

### Data leakage in cross-validation — the silent killer
```
WRONG: CV-tune on 800 rows → RETRAIN on all 1000 (incl. the 200 test) → "evaluate" on those 200.
       Model trained on test data → test score is an optimistic lie.

SUBTLER: even without test leakage, trying 100 hyperparam combos → some look good by CHANCE
         on the val folds → "best" params are overfit to the folds → prod is worse than CV says.
FIX:     Nested CV — OUTER loop estimates true generalization, INNER loop tunes hyperparameters.
```
Simple 5-fold CV is fine when: dataset is large (>10K), search space is small, and the test set is *never* touched during tuning. Nested CV matters when: small data (<1K), large search space, or you need an unbiased published estimate.

### Serving & monitoring
- **Inference cost:** RF queries all K trees (latency ∝ trees × depth); boosting ditto but trees are shallow. Cap `n_estimators`/`max_depth` for latency-SLO endpoints; export to **ONNX** or use a batch scoring service.
- **Monitor in prod:** track **PR-AUC / F2 on labeled feedback**, not accuracy; watch **feature drift** (PSI/KS) and **prediction-score drift**. Boosting degrades faster under distribution shift → set drift-triggered **retraining**.
- **Reproducibility:** pin `random_state`, library versions, and the exact feature pipeline; tree models are sensitive to feature-engineering changes.

---

## 9. Interview Lens

> ⚡ **Golden rule:** spend 15 seconds re-reading the question, identify exactly what's asked, answer THAT first. Marks go to *diagnosis (the WHY)* before *solution (the WHAT)*. Never jump to a solution without diagnosing.

**"Explain Bagging and Boosting"** → 🎯 *"Both combine weak learners, but solve opposite problems: bagging trains in parallel on bootstrap samples and averages to cut **variance** (Random Forest); boosting trains sequentially, each model correcting the last's errors, to cut **bias** (XGBoost/LightGBM)."*

**"Which would you choose?"** → 🎯 *"I start from the data: outliers/noisy labels → Random Forest (boosting amplifies them); clean data optimizing accuracy → XGBoost/LightGBM; >500K rows → LightGBM for speed. In practice I build an RF baseline first, then move to boosting to push accuracy."*

**Likely follow-ups:**
- *Why does RF slightly increase bias vs a single deep tree?* → Feature subsampling makes each tree weaker, but the variance reduction from averaging far outweighs it — a deliberate trade. `(certain)`
- *AdaBoost vs GBDT?* → AdaBoost **re-weights samples** (misclassified get higher weight); GBDT **fits residuals = negative gradient**, works with any differentiable loss, more robust to noise. `(certain)`
- *Why is LightGBM faster than XGBoost?* → **Histogram binning** + **GOSS** (keep high-gradient samples, drop easy ones) + **EFB** (bundle exclusive sparse features) + **leaf-wise** growth (vs XGB level-wise) → fewer splits to converge. `(likely)`
- *How do you explain a tree ensemble?* → SHAP (local + global, additive, exact via TreeSHAP); default gain importance is biased toward high-cardinality features — see §7.
- *When would you NOT use ensembles?* → Strict interpretability, tiny data (<100 rows), or high-dim sparse text where linear/NN win.
- *What is OOB error?* → The ~37% never-bootstrapped rows act as a per-tree validation set → free unbiased generalization estimate, great on small data.
- *Can boosting overfit? Prevent it?* → Yes; early stopping + `lambda/gamma/min_child_weight` + subsample/colsample + low LR with more trees.

---

## 10. Alternatives & How to Choose

| Dataset Property | Recommended | Reason |
|---|---|---|
| Heavy outliers | Random Forest | Bootstrap dilutes outlier influence |
| Noisy labels | Random Forest | Averaging smooths noise; boosting amplifies it |
| Clean, curated data | XGBoost / LightGBM | Exploits clean signal for max accuracy |
| Small data (<1K) | Random Forest | Boosting overfits easily |
| Large data (>1M) | LightGBM | GOSS + EFB → 10-20× faster |
| Moderate imbalance | XGBoost + `scale_pos_weight` | Focuses on minority errors |
| Severe imbalance (99:1) | XGBoost + SMOTE + threshold | Weights alone insufficient |
| Many categorical features | **CatBoost** / LightGBM | Native categorical handling, ordered boosting |
| Missing values | XGBoost / LightGBM | Native missing-value routing |
| Ranking (search/ads) | LightGBM (LambdaMART) | Industry standard |
| Interpretability needed | Single/shallow tree + SHAP | Deep ensembles are black boxes (§7) |
| High-dim sparse text | Linear model / NN | Trees weak on sparse high-dim |

### Beyond bagging vs boosting
- **CatBoost** — best default when categorical features dominate. Two ideas: **ordered target statistics** (encode a category using only rows seen *before* it in a random permutation → prevents the target leakage that naïve mean-encoding causes) and **ordered boosting** (the same anti-leakage trick applied to gradient estimation). Strong out-of-the-box with little tuning.
- **Histogram-based split finding** — XGBoost (`tree_method='hist'`) and LightGBM bin continuous features into ~255 buckets, so split search is O(bins) not O(samples). This (plus GOSS/EFB/leaf-wise growth) is *why* they scale to millions of rows; `tree_method='gpu_hist'` / `device='gpu'` moves the histogram build onto the GPU for another big speedup.
- **Quantile / probabilistic boosting** — swap MSE for **pinball (quantile) loss** to predict intervals (P10/P50/P90) instead of a point estimate — standard for demand and risk forecasting. Same additive framework, different loss (see §2).

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. Why does averaging reduce variance, and what's the *floor* you can't average away?
   <details><summary>answer</summary>`Var = ρσ² + (1−ρ)σ²/N`; the `ρσ²` correlation floor survives no matter how many trees. Lower ρ (RF feature subsampling) is the real lever.</details>
2. What does the "gradient" in gradient boosting refer to, concretely, for MSE loss?
   <details><summary>answer</summary>Each tree fits the **negative gradient** of the loss w.r.t. the current prediction; for MSE that's exactly the residual `y − ŷ`. It's gradient descent in function space — add a tree instead of updating weights.</details>
3. Train=98%, Val=79%. Name the first fix and why it's first.
   <details><summary>answer</summary>**Early stopping** — it sets `n_estimators` from the validation signal with no guessing; then reduce `max_depth`, add subsampling, then regularize, tune LR last.</details>
4. Why is ROC-AUC misleading for fraud, and what replaces it?
   <details><summary>answer</summary>ROC-AUC is inflated by the huge true-negative count, so it looks good on mediocre models. Use **PR-AUC** and tune the decision **threshold** (and optimize **F2**, weighting recall).</details>
5. A colleague retrains on "all the data" after CV before evaluating. When is that a bug?
   <details><summary>answer</summary>Fine if "all data" = the full *training* set; a catastrophic **leak** if it includes the test set. Also beware hyperparameter overfitting to the val folds → use **nested CV** when data is small / search space large.</details>
6. Why shouldn't you trust XGBoost's default `feature_importances_`, and what do you use instead?
   <details><summary>answer</summary>Gain/split importance is **biased toward high-cardinality / continuous features** and splits credit among correlated features. Use **SHAP** (local + global, additive, exact via TreeSHAP) — or permutation importance.</details>
7. Your model's `predict_proba` says 0.8 but only ~50% of those cases are positive. What's wrong and the fix?
   <details><summary>answer</summary>The scores are **uncalibrated** (boosting optimizes log-loss/AUC, not calibration). Fit **Platt scaling** or **isotonic regression** on held-out data (`CalibratedClassifierCV`); check with a reliability diagram / Brier score.</details>
8. Why is LightGBM faster than XGBoost (name the tricks)?
   <details><summary>answer</summary>**Histogram binning** of features, **GOSS** (drop low-gradient samples), **EFB** (bundle exclusive sparse features), and **leaf-wise** growth (split highest-gain leaf → fewer splits to converge).</details>

---

*Covers: the four ensemble families (bagging/boosting/stacking/cascading), Random Forest, Boosting, GBDT, XGBoost, LightGBM, CatBoost, SHAP/interpretability, calibration, imbalanced classification, cross-validation & leakage.*
