# PCA & t-SNE — Dimensionality Reduction

> **TL;DR.** High-dimensional data is hard to model and impossible to see, so we reduce dimensions while keeping what matters. **PCA** is a *linear* method: it finds new orthogonal axes (**principal components**) ordered by how much **variance** they capture, drops the low-variance ones, and gives a **reusable transform** — great for compression, decorrelation, speed-ups, and preprocessing. **t-SNE** is a *non-linear* method built only for **visualization**: it matches neighbor **probabilities** between high-D (Gaussian) and low-D (heavy-tailed Student-t) so nearby points stay nearby — it reveals clusters beautifully but distorts global distances, is slow, non-deterministic, and can't transform new points. Rule of thumb: **PCA to compress/preprocess, t-SNE (or UMAP) to eyeball structure**.

**Where it fits:** Unsupervised **dimensionality reduction** — the fix for the curse of dimensionality that hurts [KNN](../Supervised%20ML/KNN.md), [Clustering](Clustering.md), and [GMM](GMM.md). Often a preprocessing step *before* clustering or supervised models.
**Prereqs:** variance/covariance, the normal distribution, [Clustering](Clustering.md) (for the "reduce dims before clustering" workflow), feature scaling.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Curse of Dimensionality](#2-the-curse-of-dimensionality)
3. [PCA — Maximizing Variance](#3-pca--maximizing-variance)
4. [PCA — Choosing k & the Scree Plot](#4-pca--choosing-k--the-scree-plot)
5. [PCA — Limitations](#5-pca--limitations)
6. [t-SNE — Matching Neighbor Probabilities](#6-t-sne--matching-neighbor-probabilities)
7. [t-SNE — Limitations & Reading Traps](#7-t-sne--limitations--reading-traps)
8. [Code / Implementation](#8-code--implementation)
9. [PCA vs t-SNE — How to Choose](#9-pca-vs-t-sne--how-to-choose)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

You have a 1000-page book (1000 features). Reading it all is impossible; a good **2-page summary** keeps the big picture and drops the rest. Dimensionality reduction is that summary — trade a little information for a huge gain in simplicity and visibility.

```
  PCA (linear, global)                          t-SNE (non-linear, local)
  rotate axes to point along max variance       keep near points near; let far points scatter
  ● ●                 →   project onto PC1       ●●●   ▲▲▲        →   ●●●     ▲▲▲
   ● ● ●   (a tilted cloud)                       (clusters in high-D)     (same clusters, made visible)
  keeps global structure & is reusable          keeps LOCAL structure; global distances distorted
```

- **PCA = rotate + keep the high-variance directions.** New features (PCs) are linear combinations of the old, ordered so PC1 explains the most variance. Drop the tail. `(certain)`
- **t-SNE = preserve who's-near-whom.** It doesn't rotate axes; it re-lays-out points in 2-D so neighbors stay neighbors, purely to *look* at cluster structure. `(certain)`
- 🎯 The one-liner: **PCA preserves variance (global, linear, reusable); t-SNE preserves neighborhoods (local, non-linear, visualization-only).**

---

## 2. The Curse of Dimensionality

Why bother reducing? As features grow: `(certain)`
- **Distances stop meaning anything** — in high-D, all points become roughly **equidistant**, so "nearest neighbor" is meaningless (this is what breaks [KNN](../Supervised%20ML/KNN.md), clustering, density methods).
- **Data gets sparse** — the volume explodes, so you'd need exponentially more data to fill it; models learn noise and **overfit**, generalizing poorly.
- **Redundancy** — many features are correlated, so the *true* structure lives in far fewer dimensions than the raw count.
- **You can't visualize** >3 dimensions.

Dimensionality reduction keeps as much signal as possible in far fewer dimensions.

---

## 3. PCA — Maximizing Variance

**The idea:** find a new axis such that projecting the data onto it **preserves the most variance** (spread) — because variance = information. That axis is **Principal Component 1**. PC2 is the direction of next-most variance **orthogonal to PC1**, and so on. `(certain)`

**Why maximize projected variance = "best-fit line":** for a point, its distance from the origin `A` is fixed. Project it onto a candidate axis; by Pythagoras `A² = (projection)² + (residual)²`. Maximizing the **squared projection distance** (spread along the line) is the same as minimizing the residual — so PCA's max-variance axis is exactly the line the points project onto most spread-out. `(certain)`

**The math Scaler flagged as "out of scope" — but is the standard interview answer:** PCA is the **eigendecomposition of the covariance matrix** `Σ` (on standardized data). `(likely)`
```
principal components  = eigenvectors of Σ   (the new orthogonal axes)
variance each explains = eigenvalues        (bigger eigenvalue ⇒ more variance ⇒ more important PC)
```
Sort eigenvectors by eigenvalue, keep the top `k`, project the data onto them. (Computed in practice via **SVD** of the data matrix for numerical stability.)

**Variation explained** for a component:
```
variation(PC) = SS(projected distance) / (n − 1)          # its eigenvalue
% explained    = variation(PCᵢ) / Σ variation(all PCs)     # explained_variance_ratio_
```

---

## 4. PCA — Choosing k & the Scree Plot

Each PC explains less than the one before, so you keep the first few that cover "enough" variance. `(certain)`

**Worked example** (Scaler's): with `SS(distance)` = 300 for PC1 and 60 for PC2, `n = 31`:
```
variation PC1 = 300/30 = 10      variation PC2 = 60/30 = 2      total = 12
PC1 explains 10/12 = 83%    ·    PC2 explains 2/12 = 17%
→ keep PC1 alone → retain 83% of the information in ONE dimension instead of two.
```

**Scree plot** — bars = variance per PC, red line = cumulative. Pick `k` at the "elbow," or where the **cumulative** hits your target (e.g. 90–95%): `(certain)`
```
%var │█
     │█ ▄
     │█ █ ▂ ▁ ▁ ▁          cumulative ●───●───●───●  → e.g. first 3 PCs ≈ 71%
     └PC1 PC2 …            keep as many PCs as your business tolerates losing (say 5–10%)
```
```python
pca.explained_variance_ratio_            # variance per PC
np.cumsum(pca.explained_variance_ratio_) # cumulative → pick k at your threshold
```

---

## 5. PCA — Limitations

- **PCs are hard to interpret** — each is a linear blend of *all* original features, so "PC1" has no clean business meaning (unlike selecting real features). `(certain)`
- **Linear only** — it captures linear structure; for curved/manifold structure it underperforms (that's t-SNE/UMAP territory).
- **Scale-sensitive** — a feature in the thousands will dominate PC1 purely by units. **Always standardize first** (this is non-negotiable). `(certain)`
- **Not robust to outliers** — variance is squared, so outliers can hijack the top components (like OLS).
- **Maximizes variance, not class separability** — high variance ≠ useful for a downstream label; if you have labels and want discriminative axes, use **LDA** instead.

---

## 6. t-SNE — Matching Neighbor Probabilities

**t-distributed Stochastic Neighbor Embedding** turns distances into **probabilities of being neighbors** and lays points out in 2-D so those probabilities match. `(certain)`

```
HIGH-D:  for each point, put a GAUSSIAN around it → pᵢⱼ = normalized similarity (close ⇒ high prob)
LOW-D:   place points randomly, use a heavy-tailed STUDENT-t (df=1) → qᵢⱼ
OPTIMIZE: move low-D points to make qᵢⱼ match pᵢⱼ, minimizing KL divergence:
             C = Σ pᵢⱼ · log(pᵢⱼ / qᵢⱼ)
gradient pulls/pushes:  Δyᵢ ∝ Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)
   pᵢⱼ > qᵢⱼ → PULL true neighbors together   ·   qᵢⱼ > pᵢⱼ → PUSH non-neighbors apart
```

**Why Gaussian in high-D but Student-t in low-D — the crowding problem:** you can't fit all the high-D neighborhood distances into 2-D (map a square's corners to a line and one pair must break). Low-D space is cramped, so moderately-distant points would collapse into a crowd. The **heavy tails of the Student-t** give far points more "room" (their `q` stays non-zero → real repulsion), which counteracts crowding. 🎯 *"Mismatched tails for mismatched dimensions: Gaussian picks neighbors in high-D, heavy-tailed t pushes everyone else away in low-D."* `(certain)`

**Perplexity** — the one key hyperparameter, ≈ **the number of neighbors** each point considers (it sets each Gaussian's width `σᵢ`). Low perplexity → local detail; high → big-picture. Typical **5–50**; never set it near `n`. `(certain)`

---

## 7. t-SNE — Limitations & Reading Traps

- **No reusable transform** — unlike PCA, t-SNE gives no mapping; a new data point means **re-running the whole optimization**. It has no `.transform()`. `(certain)`
- **Slow** — builds `n×n` similarity matrices and gradients → `O(n²)`; impractical for very large `n` (subsample, or use **[[UMAP]]**, which is faster and keeps more global structure). `(likely)`
- **Non-deterministic** — different runs (and different perplexities) give different pictures.
- 🎯 **Reading traps (a classic interview point):** t-SNE preserves *local* neighborhoods only — so **cluster *sizes* are meaningless, gaps/distances *between* clusters are meaningless**, and apparent clusters can be artifacts of perplexity. Use it to *see* that structure exists, **never** to measure it or to cluster on its output.
- **Visualization only** — it's not a general-purpose feature reducer for models (that's PCA/UMAP).

---

## 8. Code / Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X = StandardScaler().fit_transform(X_raw)     # MANDATORY for PCA (scale-sensitive); good for t-SNE too

# --- PCA: reusable linear transform ---
pca = PCA(n_components=0.95)                   # keep enough PCs to retain 95% variance (or an int k)
X_pca = pca.fit_transform(X)                  # .transform() works on NEW data later
print(pca.explained_variance_ratio_.cumsum())

# --- t-SNE: visualization only, fit_transform once ---
X_2d = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)   # no .transform for new points
# Common pattern: PCA to ~50 dims first (denoise + speed), THEN t-SNE for the 2-D plot.
```

---

## 9. PCA vs t-SNE — How to Choose

| | PCA | t-SNE |
|---|---|---|
| Type | **linear** | **non-linear** |
| Preserves | **global** variance/structure | **local** neighborhoods |
| Reusable transform? | **yes** (`.transform`) | **no** (re-run per dataset) |
| Deterministic? | yes | no |
| Speed | fast | slow `O(n²)` |
| Main use | compress, denoise, decorrelate, **preprocess**, speed up models | **visualize** cluster structure |
| Interpretable output? | PCs (weakly) | not at all (viz only) |
| Key hyperparameter | # components (variance %) | perplexity |

**Decision rule:** need to *compress features / speed up a model / preprocess* → **PCA**. Need to *see* whether high-D data has clusters → **t-SNE** (or **UMAP** for large/faster/more-global). Common combo: **PCA → ~50 dims, then t-SNE → 2-D** (PCA denoises and accelerates t-SNE). If you have labels and want discriminative axes, use **LDA**, not PCA.

---

## 10. Production & MLOps Notes

- **PCA is part of the model pipeline** — `fit` the PCA (and scaler) on **train only**, then `.transform` val/test/serving with the *same* components; refitting per split leaks and shifts axes. `(certain)`
- **PCA for real wins:** shrink feature count to cut training/inference cost and memory, decorrelate inputs for linear models, and denoise (drop low-variance PCs = drop noise). It's a genuine production preprocessing step.
- **t-SNE stays in the notebook** — it's an analysis/EDA tool, not a serving transform (no reusable mapping, non-deterministic). Don't build features or clusters off a t-SNE embedding.
- **Watch information loss** — track cumulative explained variance; retaining 95% is a common default, but validate that downstream metrics don't drop.
- **High-D + distance methods** — run PCA (or UMAP) *before* [KNN](../Supervised%20ML/KNN.md)/[clustering](Clustering.md) to dodge the curse of dimensionality.

---

## 11. Interview Lens

**"PCA vs t-SNE?"** → 🎯 *"PCA is a linear method that finds orthogonal axes of maximum variance — reusable, fast, good for compression and preprocessing, preserves global structure. t-SNE is a non-linear, visualization-only method that matches neighbor probabilities to preserve local structure — it reveals clusters but distorts global distances, is slow, non-deterministic, and can't transform new points."*

**"How does PCA actually work?"** → 🎯 *"Standardize, take the covariance matrix, and eigendecompose it — the eigenvectors are the principal components (new axes) and the eigenvalues are the variance each explains. Keep the top-k by eigenvalue and project. In practice it's done via SVD."*

**Likely follow-ups:**
- *Why standardize before PCA?* → It's variance-based; unscaled large-unit features dominate the top PCs regardless of true importance.
- *How many components to keep?* → Enough cumulative explained variance (scree plot / e.g. 95%), traded against tolerable information loss.
- *What does perplexity do in t-SNE?* → Sets the effective number of neighbors (each point's Gaussian width); typically 5–50, never near n.
- *Why Student-t in low-D?* → Its heavy tails give distant points room, fixing the crowding problem that a Gaussian would cause.
- *Can you read cluster sizes/distances off a t-SNE plot?* → No — only local neighborhoods are preserved; sizes and inter-cluster gaps are not meaningful.
- *PCA limitation for classification?* → It maximizes variance, not class separability; use LDA when you have labels and want discriminative axes.
- *t-SNE for a production feature transform?* → No — no reusable mapping and non-deterministic; use PCA/UMAP.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. In one line each, what does PCA preserve vs what does t-SNE preserve?
   <details><summary>answer</summary>PCA preserves global variance/structure (linear, reusable); t-SNE preserves local neighborhoods (non-linear, visualization-only, distorts global distances).</details>
2. What are the principal components and how much variance they explain, in linear-algebra terms?
   <details><summary>answer</summary>PCs are the eigenvectors of the (standardized) data's covariance matrix; the eigenvalues give the variance each explains. Keep the top-k by eigenvalue (computed via SVD in practice).</details>
3. Why must you standardize before PCA?
   <details><summary>answer</summary>PCA maximizes variance, so a large-unit feature would dominate the top components purely due to scale, regardless of real importance.</details>
4. How do you choose how many components to keep?
   <details><summary>answer</summary>Via the scree plot / cumulative explained-variance ratio — keep enough PCs to reach a target (e.g. 90–95%) at acceptable information loss.</details>
5. Why does t-SNE use a Gaussian in high-D but a heavy-tailed Student-t in low-D?
   <details><summary>answer</summary>The crowding problem: low-D can't hold all high-D neighborhood distances, so moderately-distant points would collapse. The Student-t's heavy tails keep far points' similarity non-zero → real repulsion → less crowding.</details>
6. Name two things you must NOT read off a t-SNE plot.
   <details><summary>answer</summary>Cluster sizes and inter-cluster distances/gaps — only local neighborhoods are preserved, so both are meaningless (and can be perplexity artifacts).</details>
7. Which technique gives a reusable transform for new data, and why does that matter?
   <details><summary>answer</summary>PCA (`.transform`) — fit on train, apply to test/production. t-SNE has no mapping and must be re-run, so it can't serve new points.</details>

---

*Covers: curse of dimensionality, PCA (max-variance projection, eigenvectors/eigenvalues of the covariance matrix, SVD, explained variance & scree plot, choosing k, limitations, LDA contrast), t-SNE (Gaussian→Student-t neighbor probabilities, KL divergence, pull/push gradient, crowding problem, perplexity, non-determinism, reading traps, UMAP), the PCA→t-SNE combo, and production usage. Sourced from the Scaler PCA & t-SNE lecture (incl. its t-SNE math notes).*
