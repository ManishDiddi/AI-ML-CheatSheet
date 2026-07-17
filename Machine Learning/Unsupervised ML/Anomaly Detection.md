# Anomaly Detection — Finding the Points That Don't Belong

> **TL;DR.** Anomaly detection flags the rare points that differ from "normal" — fraud, defects, intrusions, sensor faults — usually **without labels** (anomalies are rare and you can't enumerate them). Four families: **statistical/distribution-based** (fit a Gaussian *robustly*, flag low-probability points — Elliptic Envelope); **Isolation Forest** (random trees isolate outliers at *shallow* depth — fast, scales); **One-Class SVM** (wrap the normal data in the smallest kernelized boundary, outsiders are anomalies); and **Local Outlier Factor** (compare a point's *local density* to its neighbors'). All return `−1` (outlier) / `+1` (inlier) in sklearn and take a **`contamination`** estimate of the outlier fraction. In practice **Isolation Forest and LOF** are the industry defaults.

**Where it fits:** Unsupervised (or semi-supervised) outlier/novelty detection — the flip side of [Clustering](Clustering.md). Builds on density ([DBSCAN](Clustering.md)), distance ([KNN](../Supervised%20ML/KNN.md) for LOF), random trees ([ensembles](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md) for iForest), and likelihood ([GMM](GMM.md)).
**Prereqs:** [Clustering](Clustering.md) (density, DBSCAN noise), the Gaussian & covariance ([GMM](GMM.md)), [KNN](../Supervised%20ML/KNN.md), [class imbalance & PR-AUC](../Supervised%20ML/Classification%20Metrics.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Setup — unlabeled, rare, contamination](#2-the-setup--unlabeled-rare-contamination)
3. [Statistical / Distribution-Based](#3-statistical--distribution-based)
4. [Isolation Forest](#4-isolation-forest)
5. [One-Class SVM](#5-one-class-svm)
6. [Local Outlier Factor (LOF)](#6-local-outlier-factor-lof)
7. [Code / Implementation](#7-code--implementation)
8. [When It Breaks & How to Choose](#8-when-it-breaks--how-to-choose)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

An anomaly is a point that **doesn't fit the pattern of the rest** — the transaction that looks nothing like your normal ones, the sensor reading off the charts. The challenge: anomalies are **rare and unlabeled**, so you can't just train a classifier on "anomaly vs normal."

```
     ●●●●●                    ● = normal (dense, the "pattern")
   ●●●●●●●●        ★          ★ = anomaly (rare, far from the crowd / in a sparse region)
     ●●●●●
                    the four methods differ in HOW they define "doesn't fit":
   distribution: low probability · isolation: easy to isolate · one-class SVM: outside the boundary · LOF: locally sparse
```

- **Terminology:** *outlier* / *anomaly* = a point outside normal behavior; *novelty* = a genuinely new pattern not seen in training. sklearn splits these: **outlier detection** trains on data that *already contains* outliers; **novelty detection** trains on *clean* data and flags new anomalies at inference. `(certain)`
- **Why not supervised?** You rarely have labeled anomalies, and they're extremely imbalanced. If you *do* have some labels, treat it as an [imbalanced classification](../Supervised%20ML/Classification%20Metrics.md) problem (PR-AUC, not accuracy) — otherwise it's unsupervised.

---

## 2. The Setup — unlabeled, rare, contamination

Every sklearn detector shares a contract: `(certain)`
- **`contamination`** — your estimate of the outlier fraction (0–0.5); it sets the decision threshold on the anomaly score.
- **`fit_predict` → `−1` (outlier) / `+1` (inlier)**; `score_samples`/`decision_function` give the underlying anomaly score for custom thresholds.
- **Scale features first** — all four are distance/density/geometry based.
- **Evaluation is hard** — usually no labels, so validate against domain knowledge; with partial labels use precision@k or PR-AUC.

---

## 3. Statistical / Distribution-Based

**Idea:** model "normal" as a probability distribution and flag **low-probability** points. In 1-D that's a **Z-score** (how many σ from the mean); in `d`-D, fit a **multivariate Gaussian** `N(μ, Σ)` and flag points far from the center (high Mahalanobis distance). `(certain)`

**The catch — outliers corrupt the very parameters you're estimating** (a few extreme points inflate `μ`/`Σ`). So estimate **robustly**: `(certain)`
- **RANSAC-style** — fit on a random subset, flag outliers by Z-score, drop them, refit, repeat until stable.
- **Elliptic Envelope** (sklearn) — assumes the data is one **unimodal multivariate Gaussian** and fits `(μ, Σ)` robustly via **MCD (Minimum Covariance Determinant / FastMCD)**, then flags points outside the fitted ellipse. `contamination` sets how much you cut.

**Limitation:** strong assumption of a **single Gaussian blob** — it breaks on multimodal or non-Gaussian data. (For multimodal, use a [GMM](GMM.md) and flag low `p(x)`.)

---

## 4. Isolation Forest

**Idea:** anomalies are **easy to isolate.** Build many totally **random trees** (iForest ≈ a [Random Forest](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md) cousin), where each split picks a **random feature** and a **random threshold**, growing until every leaf holds one point. `(certain)`

```
outlier (far, sparse)  → a few random cuts isolate it   → SHALLOW average depth
inlier  (dense region) → many cuts needed to isolate it → DEEP average depth
anomaly score ∝ 1 / average isolation depth across trees
```

- **No distance/density computed** — just tree depth, so it's **fast and scales to large/high-dim data** (`max_samples` sub-samples rows per tree, `n_estimators` trees). 🎯 That speed is why it's an industry default. `(certain)`
- **Splits use random thresholds** — *not* Gini/entropy/information gain (it's unsupervised; there's no label to optimize). `(certain)`
- **Weakness — axis-parallel bias:** every split is parallel to an axis, so the decision boundary is blocky ("banding"). Two equidistant points — one on-axis, one off-axis — can be scored differently purely due to orientation. `(certain)`

---

## 5. One-Class SVM

**Idea:** learn a boundary around the **normal** data; anything outside is an anomaly. It finds the **smallest hypersphere** (center `C`, radius `r`) that contains most points, allowing **slack** `ζᵢ ≥ 0` for points that fall outside: `(certain)`

```
minimize   r² + λ · Σ ζᵢ      subject to   ‖xᵢ − C‖² ≤ r² + ζᵢ ,  ζᵢ ≥ 0
           └shrink sphere┘  └penalize points left outside┘
```

Shrinking `r` and minimizing slack pull against each other — `λ` (in sklearn, `nu`) tunes the trade-off and roughly bounds the outlier fraction. Points forced outside (`ζᵢ > 0`) are anomalies. `(certain)`

- **Kernel trick:** the data appears only through distances, so use an **RBF kernel** to wrap non-spherical, complex normal regions (a sphere in high-D = a curvy boundary in the original space). `(certain)`
- **Weaknesses:** kernel/`gamma`/`nu` selection is fiddly, it **scales poorly with `n`** (all SVM costs), and it's sensitive to outliers in the "normal" training set.

---

## 6. Local Outlier Factor (LOF)

**Idea:** density is **relative.** A point isn't an outlier by absolute distance — it's an outlier if its **local density is much lower than its neighbors'** density. LOF fuses [KNN](../Supervised%20ML/KNN.md) + density. `(certain)`

Built up in three steps: `(likely)`
```
k-distance(A)          = distance from A to its k-th nearest neighbor
reachability(A→B)      = max( k-distance(B), dist(A,B) )      # smooths jitter near dense B
lrd(A) = local density = 1 / ( average reachability of A to its k neighbors )   # inverse distance = density
LOF(A) = average lrd of A's neighbors  /  lrd(A)             # ratio of neighbor density to own density
```

**Interpretation:** `(certain)`
- `LOF ≈ 1` → A is as dense as its neighbors (normal).
- `LOF < 1` → A is *denser* than its neighbors (deep inside a cluster).
- `LOF ≫ 1` → A is much *sparser* than its neighbors → **outlier** (e.g. 2.7 is clearly anomalous; ~1.05 is a benign border point).

- **Strength:** catches **local** outliers that global methods miss — a point that's "normal" globally but sparse *relative to its own dense neighborhood*.
- **Weaknesses:** picking `k` and the LOF threshold is unclear (domain-dependent), it **struggles in high dimensions**, and it's `O(n²)`-ish (KNN over all points). Like KNN, LOF gives no reusable model by default (`novelty=True` mode enables scoring new points).

---

## 7. Code / Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

X = StandardScaler().fit_transform(X_raw)      # scale first — all are distance/geometry based

EllipticEnvelope(contamination=0.08).fit_predict(X)                    # robust Gaussian ellipse
IsolationForest(n_estimators=100, contamination=0.08, random_state=0).fit_predict(X)  # fast, scalable
OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05).fit_predict(X)          # flexible boundary, slow on big n
LocalOutlierFactor(n_neighbors=20, contamination=0.065).fit_predict(X) # local density; novelty=True to score new pts
#  every one returns  -1 = outlier ,  +1 = inlier
```

Deep-learning alternative worth knowing: an **[autoencoder](../Neural%20Networks/Autoencoders.md)** trained on normal data flags points with high **reconstruction error** as anomalies — strong for images/sequences.

---

## 8. When It Breaks & How to Choose

| Method | Defines anomaly by | Shines when | Struggles with |
|---|---|---|---|
| **Elliptic Envelope** | low Gaussian probability | one clean Gaussian blob | multimodal / non-Gaussian data |
| **Isolation Forest** | shallow isolation depth | **large, high-dim** data; speed | axis-parallel bias (banding) |
| **One-Class SVM** | outside kernelized boundary | complex-shaped normal region | large `n`; kernel/`nu` tuning |
| **LOF** | low density vs neighbors | **local** outliers, varying density | high dims; choosing k & threshold |
| **[DBSCAN](Clustering.md)** | noise points (label −1) | clustering + outliers together | varying density, eps tuning |
| **[GMM](GMM.md)** | low mixture likelihood | multimodal Gaussian data | non-Gaussian shapes |

**Decision rule:** big/high-dimensional and you want a fast default → **Isolation Forest**; outliers that are only anomalous *locally* → **LOF**; clean unimodal Gaussian data → **Elliptic Envelope**; a complex but single normal region → **One-Class SVM**; already clustering → **DBSCAN** noise for free. Scaler's takeaway from the sklearn comparison: **Isolation Forest and LOF generalize best across cases**, which is why industry leans on them. `(likely)`

---

## 9. Production & MLOps Notes

- **The threshold is the product decision.** `contamination` (or a score cutoff) trades **false alarms vs missed anomalies** — in fraud a miss is costly, so lean toward recall; in alerting, too many false positives cause alert fatigue. Tune it to the cost, exactly like [threshold tuning in classification](../Supervised%20ML/Classification%20Metrics.md).
- **Semi-supervised is the realistic setup:** train (novelty mode) on data you *believe* is clean/normal, then flag deviations — safer than fitting on contaminated data.
- **Evaluate with what labels you have** — precision@k, PR-AUC on a labeled sample; pure-unsupervised means validating flagged points with domain experts.
- **Drift is the core risk** — "normal" shifts over time (spending patterns, seasonality), so a static detector rots. Monitor the anomaly-rate and **retrain** as the baseline moves; a spike in flagged rate is itself a signal (real event *or* drift).
- **Reduce dimensions first** — high-D wrecks distance/density methods (LOF, One-Class SVM); run [PCA](PCA%20%26%20t-SNE.md) before them. Isolation Forest tolerates dimensionality better.

---

## 10. Interview Lens

**"How does Isolation Forest detect anomalies?"** → 🎯 *"It builds random trees with random feature/threshold splits until each point is isolated. Outliers sit in sparse regions, so a few random cuts isolate them — they end up at shallow average depth. The anomaly score is inversely related to isolation depth; no distances needed, so it's fast and scales."*

**"What does LOF add over a global outlier detector?"** → 🎯 *"It's *local*: it compares a point's density to its k-neighbors' density (LOF ≫ 1 means much sparser than neighbors), so it flags points that are anomalous relative to their own neighborhood even if they look normal globally."*

**Likely follow-ups:**
- *Why is anomaly detection usually unsupervised?* → Anomalies are rare and unlabeled; you can't enumerate them. With labels it becomes extreme imbalanced classification.
- *Outlier vs novelty detection?* → Outlier detection trains on contaminated data; novelty detection trains on clean data and flags new anomalies.
- *What splits does Isolation Forest use?* → Random feature + random threshold (unsupervised — no Gini/entropy).
- *Isolation Forest's main weakness?* → Axis-parallel splits → blocky boundary ("banding"), can misjudge off-axis points.
- *What does One-Class SVM optimize?* → Smallest (kernelized) boundary around normal data with slack for outliers; `nu` bounds the outlier fraction.
- *How do you read a LOF value?* → ≈1 same density as neighbors, <1 denser, ≫1 outlier.
- *Which methods scale / handle high-D?* → Isolation Forest scales best; LOF/One-Class SVM degrade with n and dimensions — reduce dims (PCA) first.
- *How do you evaluate with no labels?* → Domain validation; with partial labels, precision@k / PR-AUC — never accuracy.

---

## 11. Alternatives & How to Choose

*(Folded into §8's comparison table and decision rule.)* In short: **Isolation Forest** = fast default at scale; **LOF** = local/variable-density outliers; **Elliptic Envelope/GMM** = distributional; **One-Class SVM** = complex single normal region; **DBSCAN** = free with clustering; **autoencoders** = deep/high-structure data.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. Why is anomaly detection usually done unsupervised, and what changes if you have some labels?
   <details><summary>answer</summary>Anomalies are rare and unlabeled, so you can't train a normal classifier. With some labels it becomes extreme imbalanced classification — evaluate with PR-AUC/precision@k, not accuracy.</details>
2. Explain the core idea of Isolation Forest and its anomaly score.
   <details><summary>answer</summary>Random trees (random feature + threshold) isolate points; outliers need few cuts → shallow average depth. Anomaly score ∝ 1/average depth. Fast, no distances, scales.</details>
3. What splitting criterion does Isolation Forest use, and what's its main limitation?
   <details><summary>answer</summary>Random splits (no Gini/entropy — it's unsupervised). Limitation: axis-parallel splits → blocky boundary/banding, can misclassify off-axis points.</details>
4. In distribution-based detection, why must parameter estimation be robust?
   <details><summary>answer</summary>Outliers corrupt μ and Σ (the very parameters defining "normal"). Robust methods (RANSAC, MCD in Elliptic Envelope) estimate them from clean subsets so outliers don't hijack the model.</details>
5. What does One-Class SVM optimize, and what role does the kernel play?
   <details><summary>answer</summary>The smallest boundary (hypersphere, radius r) enclosing normal data with slack ζ for outliers: min r² + λΣζ. The RBF kernel lets that boundary wrap non-spherical normal regions.</details>
6. How do you interpret LOF values 0.9, 1.05, and 2.7?
   <details><summary>answer</summary>0.9 → denser than neighbors (inside a cluster); 1.05 → about the same density (benign border); 2.7 → far sparser than neighbors → a clear outlier.</details>
7. Which detectors handle large, high-dimensional data best, and which need dimensionality reduction first?
   <details><summary>answer</summary>Isolation Forest scales and tolerates dimensions best. LOF and One-Class SVM degrade with n and dimensions — run PCA first.</details>

---

*Covers: outlier vs novelty detection, the unlabeled/contamination setup, distribution-based detection (Z-score, robust Gaussian via RANSAC/MCD, Elliptic Envelope), Isolation Forest (random-split trees, isolation depth, axis-parallel bias), One-Class SVM (kernelized enclosing boundary + slack), Local Outlier Factor (k-distance, reachability, local reachability density, LOF interpretation), DBSCAN/GMM as detectors, autoencoder alternative, and how to choose. Sourced from the Scaler Anomaly Detection lecture and the DBSCAN lecture's distribution-based section.*
