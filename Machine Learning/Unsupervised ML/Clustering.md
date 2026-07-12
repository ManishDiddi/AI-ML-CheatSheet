# Clustering — Grouping Unlabeled Data (K-Means, K-Means++, Hierarchical, DBSCAN)

> **TL;DR.** Clustering is **unsupervised** — no labels, no test set — so the goal is to group points that are similar (small **intra-cluster** distance) and keep groups apart (large **inter-cluster** distance). Four workhorses: **K-Means** (fast, centroid-based, but you must pick `K`, it's init-sensitive, and it only finds round, equal-ish blobs); **K-Means++** (the smart initialization that fixes K-Means' random-start problem); **Hierarchical/Agglomerative** (builds a dendrogram you cut at any `K`, no `K` upfront, but `O(n²)`); and **DBSCAN** (density-based — finds arbitrary shapes and labels outliers as noise, no `K`, but sensitive to its `eps`/`minPts` and fails on varying density). All are distance-based, so **scale your features first**.

**Where it fits:** The core of **unsupervised learning** — customer segmentation, anomaly pre-screening, feature discovery. Distance intuition comes straight from [KNN](../Supervised%20ML/KNN.md); [GMM](GMM.md) is the soft/probabilistic cousin of K-Means.
**Prereqs:** distance metrics & feature scaling ([KNN](../Supervised%20ML/KNN.md)), mean/variance, the idea that there's *no ground truth* to check against.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [K-Means & Lloyd's Algorithm](#3-k-means--lloyds-algorithm)
4. [Choosing K — Elbow & Silhouette](#4-choosing-k--elbow--silhouette)
5. [K-Means++ and K-Means' Limitations](#5-k-means-and-k-means-limitations)
6. [Hierarchical / Agglomerative Clustering](#6-hierarchical--agglomerative-clustering)
7. [DBSCAN — Density-Based Clustering](#7-dbscan--density-based-clustering)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks & How to Choose](#9-when-it-breaks--how-to-choose)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

You have customers plotted by *spend* and *visits*, no labels. Clustering finds the natural groups — "big spenders," "browsers," "loyal regulars" — so you can act on each. There's **no right answer**: quality is judged by whether the groups are tight, separated, and make *business sense*.

```
   spend │  ● ●            ▲ ▲            good clustering:
         │  ● ● ●          ▲ ▲ ▲            • INTRA-cluster distance small  (points in a group are close)
         │   ● ●            ▲ ▲             • INTER-cluster distance large  (groups are far apart)
         │        ■ ■ ■
         │        ■ ■              ← 3 clusters that "make sense" for the business
         └──────────────────── visits
```

- **No labels, no test set.** You can't compute accuracy — you optimise geometry (compactness/separation) and validate with domain sense. `(certain)`
- **Similarity = distance.** "Similar" means "close" under some metric (Euclidean in low-d, cosine in high-d) — so the metric and **feature scaling** are decisive.
- The four algorithms differ in *what shape of cluster they assume*: K-Means/K-Means++ → round blobs; Hierarchical → a nested tree; DBSCAN → dense regions of any shape.

---

## 2. The Formal Core

**The objective** (implicitly): minimise **intra-cluster** distance, maximise **inter-cluster** distance. K-Means makes this concrete as **WCSS** (Within-Cluster Sum of Squares, a.k.a. inertia):

```
WCSS = Σ_clusters Σ_{x in cluster} ‖x − centroid‖²        # total squared distance to own centroid
centroid Cᵢ = (1/|Sᵢ|) Σ_{x ∈ Sᵢ} x                        # cluster mean
```

Lower WCSS = tighter clusters — but it's minimised trivially (=0) when every point is its own cluster, so WCSS alone can't pick `K` (§4). `(certain)`

**Distance metrics** (a hyperparameter, application-dependent): Euclidean (low-d, the default), Manhattan (low–mid d), **cosine** (high-d, where magnitude ≠ meaning). All are dominated by large-scale features → **standardize first**.

**The families:**
```
CENTROID-BASED   K-Means / K-Means++  → assign points to nearest of K means
HIERARCHICAL     Agglomerative        → merge closest clusters into a tree (dendrogram)
DENSITY-BASED    DBSCAN               → grow clusters through dense regions, flag sparse points as noise
(DISTRIBUTION-BASED → GMM, its own note)
```

---

## 3. K-Means & Lloyd's Algorithm

K-Means finds `K` centroids and assigns each point to the nearest one. It's solved (approximately) by **Lloyd's algorithm** — alternate *assign* and *update* until stable: `(certain)`

```
1. INITIALIZE   pick K centroids (randomly, or via K-Means++ §5)
2. ASSIGN       assign each point to its nearest centroid          → forms K clusters
3. UPDATE       move each centroid to the mean of its assigned points
4. REPEAT       2–3 until centroids stop moving (or move negligibly)
```

It's **coordinate descent on WCSS**: the assign step and the update step each can only lower (never raise) WCSS, so it converges — but only to a **local** minimum, which is why initialization matters (§5). `(certain)`

**Worked example** (1-D, `K=2`, points `[1, 2, 10, 11]`, start centroids `c₁=1, c₂=2`):
```
Assign:  1→c₁, 2→c₂, 10→c₂, 11→c₂        Update: c₁=1,  c₂=(2+10+11)/3=7.67
Assign:  1,2→c₁ ; 10,11→c₂                Update: c₁=1.5, c₂=10.5
Assign:  1,2→c₁ ; 10,11→c₂  (stable ✓)    WCSS = (.5²+.5²)+(.5²+.5²) = 1.0
```
Complexity: `O(n·K·d·iterations)` — fast and scalable, the reason K-Means is the default. `(likely)`

---

## 4. Choosing K — Elbow & Silhouette

`K` is a hyperparameter. Three ways to set it: `(certain)`

- **Domain knowledge** — shirt sizes → K=3 (S/M/L). Often the best answer.
- **Elbow method** — plot WCSS vs `K`; it always decreases, but the drop flattens past the "true" `K`. Pick the **elbow** (the knee where extra clusters stop helping). Don't pick the lowest WCSS — that's `K=n` (every point its own cluster, WCSS=0), which is useless.
```
WCSS │●
     │ ●
     │   ●          ← elbow at K=3: after here, adding clusters barely helps
     │     ●_______●_______●
     └─────┴───────┴───────┴──── K
         2    3       4
```
- **Silhouette score** — sharper than the elbow. For each point: `s = (b − a) / max(a, b)`, where `a` = mean intra-cluster distance (to its own cluster) and `b` = mean distance to the nearest *other* cluster. Range `[−1, 1]`: near **+1** = well-clustered, **0** = on a boundary, **−1** = probably in the wrong cluster. Pick the `K` with the highest average silhouette. `(certain)`

---

## 5. K-Means++ and K-Means' Limitations

**The initialization problem.** K-Means is **initialization-dependent** — random starting centroids can converge to a bad (sub-optimal) local minimum, and the *same data* gives *different clusters* on different runs. `(certain)`

**K-Means++** fixes this by spreading the initial centroids out: `(certain)`
```
1. pick the first centroid at random from the data points
2. pick each next centroid FARTHER from the existing ones (probability ∝ distance² to nearest chosen centroid)
3. repeat until K centroids → then run normal Lloyd's algorithm
```
This gives far more consistent, better clusters (it's sklearn's **default** `init='k-means++'`). Costs: init is a bit slower, and it can be **thrown off by outliers** (a far outlier looks like a great "spread-out" seed). Also run several seeds (`n_init`) and keep the lowest-WCSS result.

**What K-Means fundamentally can't do** (it assumes round, similar-sized, similar-density blobs): `(certain)`
- **Different sizes** — a big loose cluster next to a small tight one confuses the equal-weight centroids.
- **Different densities** — sparse and dense clusters get mis-split.
- **Non-globular shapes** — concentric rings, crescents, elongated blobs → K-Means draws straight (Voronoi) boundaries and fails.
- **Outliers** — every point *must* join a cluster, so outliers drag centroids. (DBSCAN §7 fixes this by allowing "noise.")
- You must **specify K** in advance.

---

## 6. Hierarchical / Agglomerative Clustering

Builds a **tree of merges** instead of committing to `K` upfront. Two directions — **Agglomerative** (bottom-up, popular) and **Divisive** (top-down). Agglomerative: `(certain)`

```
1. start with every point as its own cluster (n points → n clusters)
2. compute the n×n proximity (distance) matrix
3. repeat until one cluster remains:
      merge the two CLOSEST clusters
      update the proximity matrix
   → the record of merges is a DENDROGRAM
```

**Linkage — how you measure "closest" between two clusters** (this *is* the key hyperparameter): `(certain)`

| Linkage | Cluster-to-cluster distance | Character |
|---|---|---|
| **Single** | min distance between any two points | finds stringy/non-globular clusters; prone to "chaining" |
| **Complete** | max distance between any two points | compact, similar-diameter clusters |
| **Average** | mean over all cross-pairs | a balance of the two |
| **Centroid** | distance between the two centroids | simple, can invert |
| **Ward** | merge that minimises the *increase in total variance* | tight, spherical clusters — the common default |

**The dendrogram** shows merge distances on the y-axis; **cut it horizontally** to get your clusters — a low cut = many tight clusters, a high cut = few loose ones. Choosing the cut is more intuitive than guessing `K` blind. `(certain)`

- **Pros:** no `K` upfront; the dendrogram is genuinely interpretable; deterministic (no random init).
- **Cons:** the proximity matrix makes it **`O(n²)` memory and ~`O(n²–n³)` time** → doesn't scale to large data; merges are greedy and irreversible. **Standardize before computing distances.**

---

## 7. DBSCAN — Density-Based Clustering

**Density-Based Spatial Clustering of Applications with Noise.** Instead of `K`, it grows clusters through **dense neighborhoods** and labels sparse points as **noise** — so it finds arbitrary shapes *and* detects outliers. Two hyperparameters: `(certain)`

```
eps (ε)   = radius of the neighborhood around a point
minPts    = min points within eps to call a region "dense"

CORE point   : has ≥ minPts within eps (sits in a dense region)
BORDER point : within eps of a core point, but not itself core
NOISE point  : neither core nor border  → an outlier (label −1)
```

**Algorithm:** `(likely)`
```
1. label every point core / border / noise (range query within eps)
2. drop noise points
3. for each unassigned core point: start a cluster, add all DENSITY-CONNECTED core points
   (density-connected = linked through a chain of core points each within eps)
4. attach each border point to its nearest core point's cluster
```

**Tuning:** `minPts ≈ 2·d` (or `≥ d+1`), larger for noisy data. Pick `eps` from the **k-distance plot**: for each point compute distance to its `minPts`-th nearest neighbor, sort ascending, and take the **elbow** where the curve shoots up (points past it are outliers). `(certain)`

- **Pros:** finds arbitrary shapes, **robust to outliers** (they become noise), **no `K` needed**, only two params.
- **Cons:** very **sensitive to `eps`/`minPts`** (small changes → different clusters); **fails on clusters of varying density** (one global `eps` can't fit both dense and sparse groups); degrades in **high dimensions** (distance concentration). 🎯 The modern fix for varying density is **[[HDBSCAN]]**, which builds a hierarchy over densities so you don't pick a single `eps`. `(likely)`

---

## 8. Code / Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(df)          # MANDATORY — all three are distance-based

# --- K-Means (++ init by default) + pick K by silhouette ---
for k in range(2, 8):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
    print(k, km.inertia_, silhouette_score(X, km.labels_))   # inertia = WCSS

# --- Hierarchical: view the dendrogram, then cut ---
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X, method="ward", metric="euclidean"); dendrogram(Z)   # cut visually → choose K
agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)

# --- DBSCAN: no K; noise points get label -1 ---
db = DBSCAN(eps=0.5, min_samples=2*X.shape[1]).fit(X)
#  db.labels_ == -1  ->  outliers/noise
```

---

## 9. When It Breaks & How to Choose

| Property | K-Means / ++ | Hierarchical | DBSCAN |
|---|---|---|---|
| Need to specify K? | **Yes** | No (cut dendrogram) | No |
| Cluster shape | round/convex only | any (linkage-dependent) | **arbitrary** |
| Handles outliers? | no (forces every point in) | somewhat | **yes → noise** |
| Varying density? | no | partly | **no** (single eps) |
| Scales to big data? | **yes** `O(nKd·i)` | no `O(n²)` | medium (`O(n log n)` w/ index) |
| Deterministic? | no (init) | yes | yes (given params) |
| Key hyperparameters | K, init | linkage, cut | eps, minPts |

**Decision rule:** big, roughly-spherical data with a known/guessable `K` → **K-Means++**. Small data where you want to *see* the structure and choose `K` after → **Hierarchical**. Arbitrary shapes and/or you need outliers flagged → **DBSCAN** (→ HDBSCAN for varying density). Overlapping, probabilistic ("soft") clusters where a point can partly belong to several → **[GMM](GMM.md)**.

**Universal gotchas:** always **scale features**; all distance-based methods suffer the **curse of dimensionality** (reduce dims with [PCA](PCA%20%26%20t-SNE.md) first in high-d); and there's **no ground truth**, so validate clusters against business meaning, not just silhouette.

---

## 10. Production & MLOps Notes

- **Scaling is part of the model** — persist the fitted scaler; unscaled inference distances are meaningless (same discipline as [KNN](../Supervised%20ML/KNN.md)).
- **Assigning new points:** K-Means predicts by nearest centroid (cheap, streaming-friendly). Hierarchical and DBSCAN have **no native `predict`** — you re-cluster or train a classifier on the cluster labels to serve new points. `(likely)`
- **Make clusters actionable** — raw cluster IDs mean nothing to stakeholders; profile them (e.g. **RFM**: Recency/Frequency/Monetary) and name the segments. Clustering's output is often a *feature* for a downstream supervised model.
- **Stability & drift** — re-run with several seeds and check clusters are stable; monitor cluster sizes/centroids over time, since customer behavior drifts and yesterday's segments rot.
- **Evaluation:** internal metrics (silhouette, Davies–Bouldin) when unlabeled; external metrics (ARI, NMI) only if you have some ground-truth labels to compare against.

---

## 11. Interview Lens

**"Walk me through K-Means and its main weakness."** → 🎯 *"Lloyd's algorithm alternates assigning points to the nearest of K centroids and moving centroids to their cluster mean, minimising within-cluster sum of squares — but it only reaches a local optimum, so it's initialization-sensitive (fixed by K-Means++) and it assumes round, equal-sized clusters."*

**"K-Means vs DBSCAN?"** → 🎯 *"K-Means needs K, makes convex blobs, and forces every point into a cluster; DBSCAN needs no K, finds arbitrary shapes, and labels sparse points as noise — but it's sensitive to eps/minPts and struggles with varying density."*

**Likely follow-ups:**
- *How do you choose K?* → domain knowledge, elbow (WCSS knee), or silhouette (highest average). Not lowest WCSS — that's K=n.
- *What does K-Means++ do?* → seeds centroids spread out (each new one far from existing, ∝ distance²) → better, more consistent local optima.
- *Silhouette formula/meaning?* → `(b−a)/max(a,b)`; +1 tight & separated, 0 on a boundary, −1 misassigned.
- *Linkage methods?* → single (min), complete (max), average, centroid, Ward (min variance increase); Ward is the usual default.
- *DBSCAN point types?* → core (≥minPts in eps), border (near a core, not core), noise (neither).
- *Why scale features first?* → distance is dominated by large-magnitude features otherwise.
- *When would you NOT use K-Means?* → non-globular shapes, varying sizes/densities, unknown K, or when outliers must be isolated.
- *Hierarchical downside?* → `O(n²)` memory/time from the proximity matrix → doesn't scale.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. State the two things good clustering optimises, and why you can't just report accuracy.
   <details><summary>answer</summary>Minimise intra-cluster distance (tight groups) and maximise inter-cluster distance (separated groups). There are no labels/ground truth, so accuracy is undefined — you use internal metrics + business sense.</details>
2. Why can't you pick K by minimising WCSS, and what do you use instead?
   <details><summary>answer</summary>WCSS hits 0 when every point is its own cluster (K=n), so it always prefers more clusters. Use the elbow (knee of the WCSS curve) or the silhouette score.</details>
3. What problem does K-Means++ solve and how?
   <details><summary>answer</summary>K-Means' sensitivity to random initialization (bad local optima). K-Means++ seeds centroids spread apart — each new centroid is chosen far from existing ones with probability ∝ squared distance.</details>
4. Name three cluster geometries K-Means handles poorly.
   <details><summary>answer</summary>Different sizes, different densities, and non-globular (non-convex) shapes — plus it can't isolate outliers and needs K specified.</details>
5. In DBSCAN, define core/border/noise and how eps is tuned.
   <details><summary>answer</summary>Core = ≥minPts within eps; border = within eps of a core but not core; noise = neither (outlier, label −1). Tune eps via the k-distance plot elbow (distance to the minPts-th neighbor, sorted).</details>
6. Hierarchical clustering's main advantage and main disadvantage?
   <details><summary>answer</summary>Advantage: no K upfront — build a dendrogram and cut it; interpretable, deterministic. Disadvantage: O(n²) proximity matrix → doesn't scale to large data.</details>
7. You have concentric-ring data with outliers. Which algorithm and why?
   <details><summary>answer</summary>DBSCAN — it finds arbitrary (non-convex) shapes via density and labels the outliers as noise, whereas K-Means would slice the rings with straight Voronoi boundaries.</details>

---

*Covers: unsupervised learning & clustering intuition (intra/inter distance), WCSS/inertia, K-Means & Lloyd's algorithm, choosing K (elbow, silhouette), K-Means++ initialization & K-Means' shape/size/density/outlier limits, hierarchical/agglomerative clustering (linkages, dendrogram, divisive), DBSCAN (eps/minPts, core/border/noise, density-connected, k-distance tuning, HDBSCAN), scaling, and how to choose between them. Sourced from the Scaler Clustering, K-Means++/Hierarchical, and DBSCAN lectures.*
