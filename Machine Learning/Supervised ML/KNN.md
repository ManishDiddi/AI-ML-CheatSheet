# K-Nearest Neighbors — Classify by Who Your Neighbors Are

> **TL;DR.** KNN predicts a point's label by looking at its **K closest training points** and taking a **majority vote** (classification) or **mean** (regression). It's a **lazy, non-parametric** learner — "training" is just storing the data; all the work happens at inference, where it computes the distance to every training point. Dead simple and boundary-flexible, but it **must have scaled features** (it's distance-based), **collapses in high dimensions** (curse of dimensionality), and is **expensive to serve** (all data, every query). `K` is the one knob: small `K` overfits, large `K` underfits.

**Where it fits:** Supervised **classification & regression**, plus **missing-value imputation** and **similarity search / recommendations**. The intuitive counterpart to [Logistic Regression](Logistic%20Regression.md) — no learned boundary, just local neighborhoods. Its "store everything" cost is exactly why [Decision Trees](Decision%20Trees.md) exist.
**Prereqs:** distance/norms, feature scaling, [bias–variance](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md), [Classification Metrics](Classification%20Metrics.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — algorithm & distances](#2-the-formal-core--algorithm--distances)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [Choosing K & the Bias–Variance Tradeoff](#6-choosing-k--the-biasvariance-tradeoff)
7. [When It Breaks](#7-when-it-breaks)
8. [Handling Imbalanced Data](#8-handling-imbalanced-data)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

"You are the company you keep." To label a new point, look at the labelled points nearest to it and let them vote.

```
       ×  ×                   x_q = ? 
     ×  × ×  ← class A          │  its 3 nearest neighbours:  A, B, B
           × x_q  ∘ ∘          │  majority vote → B
              ∘  ∘  ∘  ← class B
   K controls how big the "neighbourhood" is.
```

- **Lazy / instance-based:** there's no model to fit — KNN just **memorises** the training set. All computation is deferred to prediction time (compute distances → sort → vote). `(certain)`
- **Non-parametric:** it assumes no functional form for the boundary, so it can trace arbitrarily wiggly, non-linear boundaries that a linear model can't. `(certain)`
- **The one assumption:** *similar points are close together* — neighbourhoods are locally homogeneous. When that holds (and features are scaled), KNN works; when distance stops being meaningful (high dimensions), it fails.

---

## 2. The Formal Core — algorithm & distances

**The algorithm** (to predict for a query `x_q`):

```
1. for each training point xᵢ:  compute distance(x_q, xᵢ)
2. sort the distances ascending
3. take the K nearest points
4. CLASSIFY → majority class among them   |   REGRESS → mean (or median) of their targets
```

**Distance metrics** (the "closeness" measure — itself a hyperparameter):

```
Euclidean (L2):  d = sqrt( Σⱼ (x₁ⱼ − x₂ⱼ)² )      # straight-line; the default
Manhattan (L1):  d = Σⱼ |x₁ⱼ − x₂ⱼ|               # grid/city-block; robuster to outliers
Minkowski(p):    d = ( Σⱼ |x₁ⱼ − x₂ⱼ|ᵖ )^(1/p)    # p=2→Euclidean, p=1→Manhattan (sklearn default p=2)
Cosine:          angle between vectors            # for text/high-dim where magnitude ≠ meaning
```

Euclidean and Manhattan are the same idea at different powers (`|x|² = x²`, so the `abs` is implicit in L2). `(certain)`

**Weighted KNN — fix the "far majority" problem.** Plain voting treats all K neighbours equally, so 3 far positives outvote 2 near negatives. Weight each vote by `1/distance` so **closer neighbours count more**:

```
plain:    3 far +ve  vs  2 near −ve  → +ve wins on count
weighted: +ve weight 3×(1/2)=1.5  vs  −ve weight 2×(1/0.5)=4  → −ve wins  (the near points dominate)
```

**Ties:** use an **odd K** for binary problems to avoid even splits; if points are exactly equidistant, KNN breaks the tie arbitrarily (or shift K). `(certain)`

---

## 3. How It Works

```
1. SCALE      standardize features — mandatory; distance is dominated by large-scale features otherwise
2. Choose     K and a distance metric (both hyperparameters, tuned on validation)
3. "Train"    just store X_train, y_train           → train time O(1)
4. Predict    distance to all n points, sort, vote  → test time O(n·d) per query
```

🎯 The defining asymmetry: **KNN does nothing at training time and everything at test time** — the opposite of most models. Storing points is `O(1)`; each prediction costs `O(n·d)` (distance to all `n` points in `d` dims) plus `O(n log n)` to sort. That's why it's cheap to "train" but painful to serve at scale. `(certain)`

---

## 4. Worked Example

`K = 5`, binary. For query `x_q`, the five nearest neighbours come out as:

```
neighbour   distance   label
  n1          0.5        −
  n2          0.5        −
  n3          2.0        +
  n4          2.0        +
  n5          2.0        +
```

**Plain KNN** (count): 3 positives vs 2 negatives → **predict +ve**.

**Weighted KNN** (weight = 1/distance):

```
−ve weight = 1/0.5 + 1/0.5           = 2 + 2   = 4.0
+ve weight = 1/2.0 + 1/2.0 + 1/2.0   = 0.5×3   = 1.5
```

→ **predict −ve** — the two *very close* negatives outweigh the three *distant* positives. This is exactly why `weights='distance'` matters when the vote is close.

---

## 5. Code / Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(),                          # MANDATORY — KNN is distance-based
    KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",                    # closer neighbours vote more (vs "uniform")
        metric="minkowski", p=2,               # p=2 Euclidean, p=1 Manhattan
        algorithm="auto",                      # "kd_tree"/"ball_tree" speed up neighbour search
    ),
).fit(X_train, y_train)
clf.predict(X_test)          # KNeighborsRegressor is the regression twin (mean of neighbours)
```

**KNN as a missing-value imputer** — a genuinely useful side application: fill a missing cell with the (distance-weighted) average of that column among the row's nearest complete neighbours. Scale first (distance-based), and sanity-check that per-feature means/histograms are preserved after imputation. `(certain)`

```python
from sklearn.impute import KNNImputer
X_filled = KNNImputer(n_neighbors=5).fit_transform(X_scaled)  # encode categoricals to numeric first
```

---

## 6. Choosing K & the Bias–Variance Tradeoff

`K` is the *only* real lever, and it directly controls bias vs variance: `(certain)`

```
K = 1        → decision follows the single nearest point → fits noise/outliers
             → LOW bias, HIGH variance  (OVERFIT; 100% train accuracy, poor test)
K → n        → every query returns the overall majority class
             → HIGH bias, LOW variance  (UNDERFIT)
K = K_best   → tune on validation: plot error vs K, pick the minimum
```

- Rule of thumb: start near `K ≈ √n`, keep `K` **odd** for binary classification, tune with cross-validation. `(likely)`
- 🎯 *"Increasing K increases bias and decreases variance"* — the reverse of most 'more-complex-model' intuitions, because here **small K = more complex (wigglier) boundary**.

---

## 7. When It Breaks

- **Curse of dimensionality — the fatal one.** As `d` grows, points spread out until *all* pairwise distances become nearly equal; "nearest" stops meaning "similar," and KNN degrades to random. Fix by **reducing dimensions first** (PCA/feature selection) or using a metric suited to high-d (cosine). 🎯 This is the #1 reason KNN fails on raw high-dimensional data (text, images). `(certain)`
- **Feature scaling is mandatory.** Distance is dominated by large-range features — an unscaled "income" (10⁵) drowns out "age" (10¹). Always standardize/normalize first; forgetting this silently ruins KNN. `(certain)`
- **Slow, heavy inference.** `O(n·d)` per query and it stores the whole training set in memory — the reason you *can't productionise vanilla KNN* on large data. Mitigate with **KD-trees / Ball-trees** (fast exact search in low-to-moderate `d`) or approximate nearest-neighbour indexes (see §9). `(certain)`
- **Outliers & small K.** With `K=1` an outlier neighbour flips the prediction; larger `K` averages it out.
- **Class imbalance.** With large `K`, the majority class swamps every neighbourhood → always predicts majority (see §8).
- **No interpretability / feature importance.** KNN gives you a prediction, not a reason — unlike a [Decision Tree](Decision%20Trees.md).

---

## 8. Handling Imbalanced Data

(Scaler teaches this alongside KNN because both KNN and [Logistic Regression](Logistic%20Regression.md) are hurt by imbalance — KNN when `K` is large, logistic regression because the majority class dominates the summed log-loss and shoves the boundary toward the minority.) First, **stop using accuracy** — a 90/10 split scores 90% by predicting only the majority; judge with F1/PR-AUC ([Classification Metrics](Classification%20Metrics.md)). Then rebalance:

| Technique | What it does | Watch out for |
|---|---|---|
| **Class weights / weighted loss** | multiply minority errors by `w = majority/minority` so they cost more (`class_weight='balanced'`) | model-level, no data thrown away or duplicated — often the **best first move** |
| **Random undersampling** | drop majority rows until balanced | **loses information** — only for very large data |
| **Random oversampling** | duplicate minority rows | exact copies → risk of **overfitting** the minority |
| **SMOTE** | synthesise *new* minority points by interpolating between a minority point and its KNN: `x_new = x₁ + λ·(x₂ − x₁)`, `λ∈[0,1]` | smarter than copying, but can create unrealistic points near class borders |

🎯 SMOTE is oversampling done with KNN interpolation instead of duplication — but it isn't always best: in Scaler's churn case, **class weights beat SMOTE for logistic regression**. Try weighting first, resample only if needed, and always rebalance **train only**, never validation/test. `(certain)`

---

## 9. Production & MLOps Notes

- **Serving is the hard part.** Exact KNN needs the full training set in memory and an `O(n·d)` scan per request — untenable at scale/low-latency. In production you use **approximate nearest neighbour (ANN)** indexes — FAISS, ScaNN, HNSW — that trade a little recall for orders-of-magnitude speed. This is exactly how **vector search / semantic retrieval** works (KNN over embeddings). `(likely)`
- **KD-tree / Ball-tree** give exact speedups when `d` is small-to-moderate; they stop helping in high `d` (curse of dimensionality again), where brute force or ANN wins.
- **The scaler is part of the model** — persist the fitted `StandardScaler` and apply identical scaling at inference, or distances (and predictions) are meaningless.
- **Retraining is "free"** (just add data), but memory and latency grow with every new point — monitor index size and query latency, not just accuracy.
- **Great for similarity/recommendation & imputation**, less so as a raw classifier at scale — pick the use case to its strengths.

---

## 10. Interview Lens

**"Why is KNN called a lazy learner?"** → 🎯 *"Because it does no work at training time — it just stores the data — and defers all computation to inference, where it computes distances to every training point. Train is O(1), test is O(n·d)."*

**"Why does KNN struggle in high dimensions?"** → 🎯 *"The curse of dimensionality: as dimensions grow, all points become roughly equidistant, so 'nearest neighbour' loses meaning and the model degrades. You reduce dimensions (PCA) or use cosine distance."*

**Likely follow-ups:**
- *How does K affect bias/variance?* → Small K = low bias/high variance (overfit); large K = high bias/low variance (underfit); tune K on validation.
- *Why must you scale features?* → Distance is dominated by large-scale features; without scaling one feature drowns out the rest.
- *Euclidean vs Manhattan?* → Both are Minkowski (p=2 vs p=1); Manhattan is a bit more outlier-robust; cosine for high-dim/text.
- *Why odd K / what about ties?* → Odd K avoids even vote splits in binary; exact ties are broken arbitrarily or by changing K.
- *KNN for regression?* → Predict the mean (or median) of the K neighbours' targets.
- *Can KNN handle imbalance?* → Large K biases toward the majority; use class weights / resampling / SMOTE and F1, not accuracy.
- *How do you serve KNN at scale?* → Approximate nearest-neighbour indexes (FAISS/HNSW) — the basis of vector search.
- *What is SMOTE?* → Synthetic minority oversampling: interpolate new minority points between a point and its neighbours, rather than duplicating.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Small data, non-linear boundary, latency OK | **KNN** | zero training, flexible boundary |
| Same shape but must serve fast / at scale | **[Decision Trees](Decision%20Trees.md) / ensembles** | learn compact rules → cheap inference |
| Linear-ish boundary, want probabilities | **[Logistic Regression](Logistic%20Regression.md)** | parametric, calibrated, fast |
| High-dimensional data (text, images) | **linear models / [[Neural Network Fundamentals]]** | KNN's distances collapse in high-d |
| Similarity search / recommendations / RAG | **KNN via ANN index (FAISS/HNSW)** | KNN *is* the right tool — just approximate it |
| Missing-value imputation | **KNNImputer** | fills from similar rows, distribution-preserving |

**Decision rule:** KNN shines for **small, low-dimensional, scaled** data and for **similarity/imputation** tasks; the moment you have high dimensions or a latency SLO, either reduce dimensions + use an ANN index, or switch to a model that does its work at training time (trees, logistic regression).

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. What are KNN's train and test time complexities, and why?
   <details><summary>answer</summary>Train O(1) — it just stores the data (lazy). Test O(n·d) per query (distance to all n points in d dims) plus O(n log n) to sort. All work is at inference.</details>
2. How does K trade off bias and variance?
   <details><summary>answer</summary>Small K (=1) → low bias, high variance (fits noise, overfits). Large K → high bias, low variance (underfits). Increasing K increases bias, decreases variance.</details>
3. Why must features be scaled for KNN, and what happens if you skip it?
   <details><summary>answer</summary>Distance is dominated by large-magnitude features, so an unscaled feature (e.g. income ~10⁵) drowns out others (age ~10¹) — the neighbourhood, and predictions, become meaningless.</details>
4. Explain the curse of dimensionality's effect on KNN.
   <details><summary>answer</summary>As dimensions grow, all pairwise distances converge, so "nearest" ≈ "farthest" — neighbours are no longer meaningfully similar and KNN degrades toward random. Fix with dimensionality reduction or cosine distance.</details>
5. Plain KNN predicts +ve but weighted KNN predicts −ve for the same query. How?
   <details><summary>answer</summary>Weighted KNN weights votes by 1/distance, so a few very-close negatives can outweigh more-but-distant positives; plain KNN only counts.</details>
6. What's the difference between SMOTE and random oversampling?
   <details><summary>answer</summary>Random oversampling duplicates minority rows (overfit risk); SMOTE synthesises new minority points by interpolating between a point and its KNN — more diverse, but can create borderline-unrealistic points.</details>
7. How do you deploy KNN at scale despite O(n·d) inference?
   <details><summary>answer</summary>Approximate nearest-neighbour indexes (FAISS, HNSW, ScaN) — trade a little recall for huge speedups; KD/Ball-trees help only in low dimensions. This is how vector search works.</details>

---

*Covers: lazy/non-parametric learning, the KNN algorithm, Euclidean/Manhattan/Minkowski/cosine distances, weighted KNN, train O(1)/test O(n·d) complexity, choosing K & its bias-variance effect, the curse of dimensionality, mandatory scaling, KD/Ball-trees & ANN serving, KNN imputation, and imbalanced-data handling (class weights, under/oversampling, SMOTE). Sourced from the Scaler KNN lecture.*
