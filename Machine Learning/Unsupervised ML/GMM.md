# Gaussian Mixture Models — Soft, Probabilistic Clustering

> **TL;DR.** A GMM models data as a **mixture of K Gaussians** and gives each point a **probability** of belonging to each cluster (soft assignment) instead of a hard label. Each Gaussian has a **mean vector μ** (where it sits) and a **covariance matrix Σ** (its shape/orientation), so GMM fits **elliptical, correlated** clusters that [K-Means](Clustering.md) can't. It's trained by **Expectation-Maximization (EM)** — alternate computing cluster responsibilities (E) and re-estimating μ/Σ by a probability-weighted average (M) — which climbs the data likelihood to a **local** optimum. Reach for it when clusters overlap, aren't spherical, or you want membership *probabilities*; but it still needs `K`, assumes the data really is Gaussian, and often lands close to K-Means.

**Where it fits:** The **soft/probabilistic** cousin of [K-Means](Clustering.md) in unsupervised learning — same "find K groups," but a generative density model underneath. Also a density estimator for [anomaly detection](Anomaly%20Detection.md) (low-likelihood points = outliers).
**Prereqs:** [Clustering](Clustering.md) (K-Means, Lloyd's algorithm), the normal distribution (mean/variance), covariance, [[Maximum Likelihood Estimation]].

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — Gaussians, μ & Σ, the mixture](#2-the-formal-core--gaussians-μ--σ-the-mixture)
3. [How It Works — the EM Algorithm](#3-how-it-works--the-em-algorithm)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [GMM vs K-Means](#6-gmm-vs-k-means)
7. [When It Breaks](#7-when-it-breaks)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

K-Means forces every customer into exactly one bucket. But a real customer can be **40% "wealthy" and 60% "price-conscious"** — they don't belong wholly to one group. GMM captures that: it says each point is *generated* by one of `K` bell-curves, and reports the **probability** it came from each.

```
 density │        ╱‾╲        ╱‾╲          data = a blend of K Gaussians
         │      ╱    ╲    ╱     ╲          each point gets P(cluster 1), P(cluster 2), …  (sum = 1)
         │    ╱   G₁   ╲╱   G₂    ╲        a point at the overlap → ~50/50 "soft" membership
         └──────────────┼─────────────
                     overlap = ambiguous point
```

- **Soft vs hard.** K-Means: "you're in cluster 2." GMM: "you're 82% cluster 1, 18% cluster 2." The soft membership is the whole point — it quantifies *uncertainty*. `(certain)`
- **Generative story.** GMM assumes the data was *produced* by picking a Gaussian (with some prior weight) and sampling from it. Clustering then inverts that: given a point, which Gaussian likely produced it?
- **Shape flexibility.** Because each Gaussian carries a full covariance matrix, clusters can be **stretched and tilted ellipses**, not just K-Means' round blobs.

---

## 2. The Formal Core — Gaussians, μ & Σ, the mixture

**One multivariate Gaussian** over `d` features is `N_d(μ, Σ)`:
- **μ** — a `d`-vector of per-feature means → *where* the cluster sits.
- **Σ** — a `d×d` covariance matrix → the cluster's *shape and orientation*: `(certain)`
```
Σ diagonal, equal entries      → circle              (features uncorrelated, same spread)
Σ diagonal, unequal entries    → axis-aligned ellipse (different spread per axis)
Σ with off-diagonal terms      → tilted/rotated ellipse (features correlated)
```

**The mixture** of `K` Gaussians. Each has a mixing weight `πⱼ` (its prior share, `Σπⱼ=1`). The density of a point is the weighted sum:

```
p(x) = Σⱼ πⱼ · N(x | μⱼ, Σⱼ)
```

**Responsibility** — the E-step quantity: the posterior probability that cluster `j` generated point `xᵢ` (Bayes' rule): `(certain)`

```
γᵢⱼ = πⱼ · N(xᵢ | μⱼ, Σⱼ)  /  Σₖ πₖ · N(xᵢ | μₖ, Σₖ)      # normalized → Σⱼ γᵢⱼ = 1
```

Fitting means finding `θ = {πⱼ, μⱼ, Σⱼ}` that **maximise the data's likelihood**. You can't do it in closed form, and the log-likelihood is riddled with local optima (so gradient descent struggles) — the `Σ⁻¹` and `|Σ|` terms in the Gaussian PDF are expensive and non-convex. That's why we use **EM**. `(certain)`

---

## 3. How It Works — the EM Algorithm

EM is **coordinate ascent** on the log-likelihood: alternately fix one set of parameters and optimize the other. `(certain)`

```
0. INIT      pick K, initialize μ, Σ, π   (often by running K-Means first)
1. E-STEP    fix μ, Σ, π → compute responsibilities γᵢⱼ for every point   ("soft assign")
2. M-STEP    fix γ → re-estimate each Gaussian by a RESPONSIBILITY-WEIGHTED average:
                μⱼ  = Σᵢ γᵢⱼ xᵢ / Σᵢ γᵢⱼ                      (weighted mean)
                Σⱼ  = Σᵢ γᵢⱼ (xᵢ−μⱼ)(xᵢ−μⱼ)ᵀ / Σᵢ γᵢⱼ        (weighted covariance)
                πⱼ  = (Σᵢ γᵢⱼ) / n                            (effective share of points)
3. REPEAT    1–2 until the log-likelihood stops improving
```

- The parallel to K-Means is exact: **E-step = assign, M-step = update** — but with *soft* weights (responsibilities) instead of *hard* 0/1 memberships, and updating a full `Σ` (not just a centroid). `(certain)`
- **EM never decreases the likelihood** each iteration, so it converges — but only to a **local** optimum, so it's **initialization-sensitive** (run several inits, keep the best; K-Means init helps). `(likely)`

---

## 4. Worked Example

1-D, two Gaussians, equal weights `π₁=π₂=0.5`: `G₁ = N(μ=0, σ=1)`, `G₂ = N(μ=4, σ=1)`. What's the responsibility of each for the point `x = 1`?

Using `N(x) ∝ exp(−(x−μ)²/2)` (the shared `1/√(2π)` cancels):
```
G₁ at x=1:  exp(−(1−0)²/2) = exp(−0.5)  = 0.6065
G₂ at x=1:  exp(−(1−4)²/2) = exp(−4.5)  = 0.0111

γ₁ = (0.5·0.6065) / (0.5·0.6065 + 0.5·0.0111) = 0.6065 / 0.6176 = 0.982
γ₂ = 0.0111 / 0.6176                                          = 0.018
```

So `x=1` is **98.2% cluster 1, 1.8% cluster 2** — a soft assignment. K-Means would just snap it to cluster 1 and throw away that 1.8% of uncertainty. The M-step would then update `μ₁` using `x=1` with weight 0.982 and `μ₂` with weight 0.018.

---

## 5. Code / Implementation

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,            # K — still required
    covariance_type="full",   # full=tilted ellipses; diag/tied/spherical trade flexibility for fewer params
    n_init=10,                 # multiple EM restarts → escape bad local optima
    random_state=0,
).fit(X)                       # scale X first (Gaussians are distance-like)

gmm.predict(X)                 # hard labels (argmax responsibility)
gmm.predict_proba(X)           # the SOFT memberships (responsibilities) — the reason to use GMM
gmm.score_samples(X)           # per-point log-likelihood → low = anomaly (see Anomaly Detection)
```

**Choosing K — use BIC/AIC, not the elbow.** GMM is a likelihood model, so pick `K` (and `covariance_type`) that **minimise BIC** (or AIC) — an information criterion balancing fit vs parameter count. This is cleaner than K-Means' elbow. `(certain)`

```python
ks = range(1, 8)
bic = [GaussianMixture(k, covariance_type="full", random_state=0).fit(X).bic(X) for k in ks]
best_k = ks[int(np.argmin(bic))]
```

**`covariance_type`** controls the shape/parameter trade-off: `spherical` (round, ≈ K-Means) → `diag` (axis-aligned ellipse) → `tied` (all clusters share one Σ) → `full` (each its own tilted ellipse; most flexible, most parameters). `(likely)`

---

## 6. GMM vs K-Means

They're the same skeleton (assign ↔ update), and 🎯 **K-Means is literally a special case of GMM** — a "hard" GMM with spherical, equal covariances and 0/1 memberships. `(certain)`

| | K-Means | GMM |
|---|---|---|
| Assignment | **hard** (one cluster) | **soft** (probabilities / responsibilities) |
| Cluster shape | spheres (Euclidean) | **ellipses** via full Σ (captures correlation) |
| What it optimizes | WCSS (inertia) | data **log-likelihood** |
| Algorithm | Lloyd's (assign/update) | **EM** (E-step/M-step) |
| Extra output | — | per-point membership probabilities + a density `p(x)` |
| Choosing K | elbow / silhouette | **BIC / AIC** |
| Cost | cheap | pricier (covariances, `Σ⁻¹`) |

Use GMM when clusters **overlap**, are **elliptical/correlated**, or you need **membership probabilities** or a **density**. Use K-Means when clusters are roughly round and you just want fast, hard labels.

---

## 7. When It Breaks

- **Still needs K** — despite soft memberships, you must set `n_components` (a common misconception is that GMM finds K itself; it doesn't — use BIC). `(certain)`
- **Assumes Gaussian components** — if the true clusters aren't blob/ellipse-shaped (rings, crescents), GMM misfits; density-based [DBSCAN](Clustering.md) handles those. 🎯 Scaler's own takeaway: GMM often ends up *close to K-Means*, so people just use K-Means unless they specifically need soft/elliptical clusters. `(certain)`
- **Singularity collapse** — a Gaussian can shrink onto a single point: its variance → 0 and the likelihood → ∞ (degenerate solution). Fix with covariance **regularization** (sklearn's `reg_covar`) or restarting. `(likely)`
- **Local optima / init-sensitivity** — EM only finds a local max; use multiple inits (`n_init`) and K-Means initialization.
- **`full` covariance is parameter-hungry** — `O(K·d²)` covariance parameters; in high `d` it overfits or turns singular → drop to `diag`/`spherical` or reduce dims with [PCA](PCA%20%26%20t-SNE.md) first.

---

## 8. Production & MLOps Notes

- **Ship the probabilities, not just labels.** `predict_proba` lets you set confidence thresholds — e.g. route only high-certainty customers to an automated campaign and hold ambiguous ones for review.
- **GMM as a density model** — `score_samples` gives `log p(x)`; points with low likelihood are **anomalies/novelties**, which is one of the cleaner unsupervised outlier detectors (see [Anomaly Detection](Anomaly%20Detection.md)). It's also a generative model — you can *sample* new synthetic points.
- **Persist the fitted scaler + model**; apply identical scaling at inference (same discipline as all distance/density methods).
- **Pick `covariance_type` for your `d`/`n`** — `full` needs enough data per cluster to estimate a stable `Σ`; with many features or few points, `diag`/`spherical` generalise better and won't go singular.
- **Monitor** cluster weights `πⱼ` and log-likelihood over retrains; a collapsing component or a plunging likelihood signals drift or a bad restart.

---

## 9. Interview Lens

**"GMM vs K-Means?"** → 🎯 *"K-Means does hard assignment to spherical clusters by minimizing within-cluster distance; GMM does soft, probabilistic assignment to elliptical Gaussians by maximizing likelihood via EM. K-Means is actually a special case of GMM — hard memberships with equal spherical covariances."*

**"What does EM do, step by step?"** → 🎯 *"E-step: fix the Gaussians, compute each point's responsibility (posterior probability) for each cluster. M-step: fix responsibilities, re-estimate each Gaussian's mean and covariance as a responsibility-weighted average. Repeat — it monotonically increases the likelihood to a local optimum."*

**Likely follow-ups:**
- *Why EM instead of gradient descent?* → The GMM log-likelihood is non-convex with many local optima and awkward `Σ⁻¹`/`|Σ|` terms; EM gives a clean, monotone-improving update in closed form each step.
- *What's a "responsibility"?* → The normalized posterior `γᵢⱼ = P(cluster j | xᵢ)`; the soft membership from the E-step.
- *Does GMM choose K?* → No — set `n_components`; select it with BIC/AIC.
- *What does the covariance matrix control?* → Cluster shape/orientation: diagonal-equal → circle, diagonal-unequal → axis-aligned ellipse, off-diagonal → tilted ellipse.
- *What can go catastrophically wrong in EM?* → A component collapses onto one point (variance→0, likelihood→∞); regularize the covariance.
- *When is GMM worth it over K-Means?* → Overlapping/elliptical/correlated clusters, or when you need membership probabilities or a density (e.g. anomaly detection).

---

## 10. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Round, well-separated clusters, want speed | **[K-Means++](Clustering.md)** | cheaper; GMM ≈ K-Means here anyway |
| Overlapping / elliptical / correlated clusters | **GMM (full covariance)** | soft memberships + shape via Σ |
| Need membership *probabilities* / uncertainty | **GMM** | `predict_proba` is the whole point |
| Arbitrary (non-Gaussian) shapes, outliers | **[DBSCAN](Clustering.md)** | density-based, no Gaussian assumption |
| Density estimation / outlier scoring | **GMM** | `p(x)` low ⇒ [anomaly](Anomaly%20Detection.md) |
| No idea of K, want to *see* structure | **[Hierarchical](Clustering.md)** | dendrogram, cut later |

**Decision rule:** default to K-Means; upgrade to GMM only when you need soft memberships, elliptical/correlated clusters, or a probabilistic density — and pick `K` and `covariance_type` by BIC. If the clusters aren't Gaussian at all, leave the mixture family for DBSCAN.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. What makes GMM "soft," and what does each Gaussian's Σ encode?
   <details><summary>answer</summary>Each point gets a probability (responsibility) of belonging to every cluster, not a single hard label. Σ encodes cluster shape/orientation — circle (diagonal-equal), axis-aligned ellipse (diagonal-unequal), tilted ellipse (off-diagonal correlation).</details>
2. Describe the E-step and M-step of EM.
   <details><summary>answer</summary>E-step: with μ/Σ/π fixed, compute each point's responsibility `γᵢⱼ` (posterior prob of cluster j). M-step: with γ fixed, re-estimate each cluster's μ and Σ as responsibility-weighted averages, and πⱼ as the effective share. Repeat.</details>
3. Why use EM rather than gradient descent to fit a GMM?
   <details><summary>answer</summary>The log-likelihood is non-convex with many local optima and expensive `Σ⁻¹`/`|Σ|` terms; EM gives closed-form, monotonically-improving updates each iteration.</details>
4. In what precise sense is K-Means a special case of GMM?
   <details><summary>answer</summary>K-Means = a hard GMM with spherical, equal covariances and 0/1 (hard) responsibilities instead of soft probabilities.</details>
5. How do you choose K for a GMM, and does GMM find K on its own?
   <details><summary>answer</summary>No, you must set `n_components`. Choose it (and covariance_type) by minimizing BIC or AIC — not the elbow.</details>
6. A GMM's log-likelihood shoots to infinity during training. What happened and the fix?
   <details><summary>answer</summary>A component collapsed onto a single point (variance → 0), a degenerate singularity. Fix with covariance regularization (`reg_covar`) or restarting with a better init.</details>
7. When is GMM genuinely better than K-Means?
   <details><summary>answer</summary>When clusters overlap, are elliptical/correlated, or you need membership probabilities or a density (e.g. anomaly scoring). For round, separated clusters GMM ≈ K-Means, so K-Means is simpler.</details>

---

*Covers: soft vs hard clustering, mixture of Gaussians, multivariate μ & Σ (shape/orientation), responsibilities, EM (E-step/M-step, coordinate ascent, monotone likelihood, local optima), choosing K via BIC/AIC, covariance_type, GMM-as-a-special-case-of-K-Means, singularity collapse, and GMM as a density model for anomaly detection. Sourced from the Scaler GMM lecture.*
