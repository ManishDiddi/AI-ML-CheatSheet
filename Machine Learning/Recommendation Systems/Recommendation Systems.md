# Recommendation Systems — Predicting What a User Will Like

> **TL;DR.** A recommender predicts how a user would rate/engage with items and returns the top ones. The data is a huge, **>99% sparse** user×item matrix `A` (missing ≠ dislike). Three classical families: **collaborative filtering** (recommend from the interaction matrix — *item-item* similarity, robust; *user-user*, drifts), **content-based** (use item/user **metadata** — solves cold start), and **hybrid**. The scalable winner is **matrix factorization**: learn a small **latent vector per user and per item** so `A_ij ≈ Uᵢ · Vⱼ`, training **only on observed entries** (SGD/ALS + regularization) — this both compresses and **fills the missing cells** (matrix completion, the Netflix-Prize idea). Its exact linear-algebra cousin is **SVD / truncated SVD** (keep the top themes = denoise). The perennial headache is **cold start** (new users/items have no history).

**Where it fits:** Personalization for Netflix/Amazon/Spotify/YouTube — the scalable answer to what [Association Rules](Association%20Rules%20%26%20Market%20Basket%20Analysis.md) can't do at e-commerce scale. Matrix factorization is [dimensionality reduction](../Unsupervised%20ML/PCA%20%26%20t-SNE.md) applied to interactions.
**Prereqs:** [KNN](../Supervised%20ML/KNN.md)/cosine similarity, [gradient descent & regularization](../Supervised%20ML/Linear%20Regression.md), [PCA/SVD](../Unsupervised%20ML/PCA%20%26%20t-SNE.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Setup — the User–Item Matrix](#2-the-setup--the-useritem-matrix)
3. [Collaborative Filtering — Item-Item & User-User](#3-collaborative-filtering--item-item--user-user)
4. [Content-Based Filtering](#4-content-based-filtering)
5. [The Cold Start Problem](#5-the-cold-start-problem)
6. [Matrix Factorization](#6-matrix-factorization)
7. [SVD & Truncated SVD](#7-svd--truncated-svd)
8. [Evaluation & Pitfalls](#8-evaluation--pitfalls)
9. [Code / Implementation](#9-code--implementation)
10. [When It Breaks & How to Choose](#10-when-it-breaks--how-to-choose)
11. [Interview Lens](#11-interview-lens)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

The goal: **predict the rating a user would give an unseen item**, rank the predictions, and surface the top few — to keep the user engaged. Netflix, Amazon, Spotify, YouTube all run on this.

```
             Inception  RRR  Interstellar  KGF  Bahubali
   Manish  [    5        ?        4         ?      ?    ]   ← predict the ?s, recommend the highest
   Rahul   [    ?        5        ?         4      5    ]
   Priya   [    4        ?        5         ?      ?    ]
```

- **Evolution:** *pre-2007* similarity/content/collaborative filtering → *2007–15* **matrix factorization** (the Netflix Prize) → *post-2015* **deep learning** (two-tower / neural CF). `(certain)`
- Two big ideas recur: **similarity** ("people/items like this one") and **latent factors** ("hidden taste dimensions" learned from data).

---

## 2. The Setup — the User–Item Matrix

Represent everything as a matrix `A` (`n` users × `m` items); `A_ij` = user `i`'s interaction with item `j` — a rating, a like (binary), % of song played (real), a purchase. `(certain)`

- **`A` is enormous and extremely sparse.** With `~10⁹` users and `~10⁸` items, that's `10¹⁷` cells, but each user touches maybe `~10³` items → **sparsity ≈ 10⁻⁵** (1 in 100,000 filled). `(certain)`
- 🎯 **Missing ≠ dislike.** An empty cell means "no data," not "rated zero." Treating blanks as zero teaches the model that users hate everything they haven't seen — the single most important modeling rule here. `(certain)`
- A **user vector** = a row of `A`; an **item vector** = a column. Recommenders operate on these rows/columns.

---

## 3. Collaborative Filtering — Item-Item & User-User

**Collaborative filtering (CF)** uses *only* the interaction matrix `A` (no metadata) — "collaborate" via the crowd's behavior. Two flavors, both **nearest-neighbor over cosine similarity**: `(certain)`

**Item-item CF** (Amazon, 1998): for items the user liked, find *similar items* and recommend those.
```
sim(Iᵢ, Iⱼ) = cosine(Iᵢ, Iⱼ) = (Iᵢ·Iⱼ)/(‖Iᵢ‖‖Iⱼ‖)     # each item = its column of ratings (n-dim)
→ build an m×m item-item similarity matrix Sⁱ ; recommend items most similar to the user's liked ones
```

**User-user CF:** find *similar users* (cosine over their rows), then recommend what those neighbors liked that this user hasn't seen (frequency vote among neighbors).

🎯 **Prefer item-item when `n > m`** (more users than items): it's cheaper (smaller similarity matrix) and **more stable** — *item* characteristics barely change, whereas **user preferences drift over time**, which destabilizes user-user CF. Similarity matrices are expensive (`m×m` or `n×n`), so they're precomputed **offline** (nightly) and served instantly. `(certain)`

- **Pros:** no domain knowledge / feature engineering (it learns from behavior); **serendipity** (can surface interests you'd never have tagged).
- **Cons:** **cold start** (new user/item has no interactions) and sparsity hurt it.

---

## 4. Content-Based Filtering

Instead of the interaction matrix, use **metadata** — features of the users and items themselves: `(certain)`
- **User metadata:** location, age, gender, device, card type → build a user feature vector → user-user similarity on *content*.
- **Item metadata:** description (bag-of-words), price, category → item feature vector → item-item similarity on *content*.

Because it needs no interaction history, **content-based filtering solves cold start** (recommend a brand-new item to users who liked similar-*content* items). `(certain)`

- **Pros:** works per-user with no other users' data (**scales**), can recommend **niche items**, and handles new items/users.
- **Cons:** needs **hand-engineered features / domain knowledge** (only as good as the features), and it **over-specialises** — it keeps recommending the same category, never surprising the user (no serendipity).

**Hybrid / recommendation-as-regression:** concatenate a user's and an item's feature vectors, label with the known rating `A_ij`, and train a regression/classifier to predict ratings — mixing metadata *and* interactions. Downside: to recommend top-k you must score *all* items per user → expensive at scale.

---

## 5. The Cold Start Problem

🎯 The recurring failure of CF: with **no interaction history you can't compute similarity or a latent vector.** Three forms: `(certain)`
- **New user** — empty row → no similar users, no `Uᵢ`.
- **New item** — empty column → no similar items, no `Vⱼ`.
- **New community** — a fresh platform with almost no data at all.

**Fixes:** fall back to **content-based** filtering (use metadata), recommend **popular items** as a stopgap, or **ask** the user for a few preferences on signup. Cold start is *the* reason pure CF is rarely deployed alone.

---

## 6. Matrix Factorization

The scalable heart of modern classical recommenders. **Assume each rating is the dot product of a user's and an item's latent vectors:** `(certain)`

```
A (n×m)  ≈  U (n×b)  ·  Vᵀ (b×m)          b = # latent factors (hidden taste dims), b << m
A_ij ≈ Uᵢ · Vⱼ         Uᵢ = user i's taste vector,  Vⱼ = item j's theme vector
```

**Latent factors are discovered, not given** — e.g. an "action vs cerebral" axis emerges from the data:
```
Manish=[0.2, 0.9]  RRR=[0.9, 0.1]  → 0.2·0.9+0.9·0.1 = 0.27   (won't like it)
Manish=[0.2, 0.9]  Interstellar=[0.2, 0.9] → 0.04+0.81 = 0.85  (will love it ✅)
```

**The objective — fit only observed entries** (Ω = known ratings), with L2 regularization: `(certain)`
```
min_{U,V}  Σ_(i,j)∈Ω  (A_ij − Uᵢ·Vⱼ)²  +  λ(‖Uᵢ‖² + ‖Vⱼ‖²)
```
Missing cells are **excluded** from the loss (never treated as 0). Solve by: `(certain)`
- **SGD** — per observed `(i,j)`: `Uᵢ += lr·(err·Vⱼ − λ·Uᵢ)`, `Vⱼ += lr·(err·Uᵢ − λ·Vⱼ)`. Scales to huge sparse data.
- **ALS (Alternating Least Squares)** — fix `V`, solve `U` in closed form; fix `U`, solve `V`; repeat. Parallelizes well.

**Matrix completion:** once `U`, `V` are learned, **every** missing `A_ij = Uᵢ·Vⱼ` — that filled value *is* the predicted rating. So recommendation = matrix completion via factorization. `(certain)`

- **Why it wins:** handles sparsity naturally, learns global structure, generalises to unseen pairs, scales — it's why it won the **Netflix Prize** and powers Spotify/Amazon.
- **Explicit vs implicit feedback:** explicit = star ratings (minimise squared error); implicit = clicks/plays where *missing ≠ dislike* → use confidence weighting and a **ranking** loss (e.g. BPR), not exact-value regression. `(likely)`
- Still hits **cold start** (a user/item with zero ratings has no vector to learn).

---

## 7. SVD & Truncated SVD

Matrix factorization done by **exact linear algebra** instead of gradient descent. **SVD** decomposes any matrix into ranked "theme" layers: `(certain)`

```
A = U · Σ · Vᵀ  =  σ₁(u₁v₁ᵀ) + σ₂(u₂v₂ᵀ) + …      σ₁ ≥ σ₂ ≥ … (theme strength)
U = user↔theme,  Σ = theme importance (singular values),  Vᵀ = item↔theme
```

**Truncated SVD** keeps only the **top-k** singular values (the strong themes) and drops the rest (noise): `A ≈ Uₖ Σₖ Vₖᵀ`. This **denoises + compresses**, and reconstructing fills missing cells → recommendations. Choose `k` by cumulative **explained variance** (`σₖ²/Σσ²`, e.g. 95%) or the scree-plot elbow. `(certain)`

**How the three relate** — this is a classic interview thread: `(certain)`
```
Matrix Factorization  = the general "A ≈ P·Qᵀ" idea
  ├─ Learned MF (SGD/ALS) = fits ONLY observed entries → best for sparse RecSys
  └─ SVD = exact factorization of a COMPLETE matrix (needs imputation if sparse)
       └─ Truncated SVD = top-k themes → denoise/compress/complete
       └─ PCA = SVD of mean-centered data, using V as principal components → dimensionality reduction
```
So **PCA is SVD on centered data**, and **learned MF is the sparse/approximate version of SVD** — see [PCA & t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md) for the eigen/variance details.

---

## 8. Evaluation & Pitfalls

- **Ranking metrics, not RMSE alone.** Users see a *ranked list*, so evaluate the top-k: **Precision@k / Recall@k** (relevant items in the top k), **MAP**, and **NDCG** (rewards putting the most relevant items highest). RMSE on held-out ratings is a weak proxy. `(likely)`
- **Split by time or by held-out interactions**, never randomly leak a user's future into training.
- **Popularity bias & filter bubbles:** models over-recommend already-popular items and narrow users into a niche — inject diversity/exploration. Serendipity (useful surprise) is a feature to design for.
- **Missing-as-zero** (§2) and **treating implicit as explicit** are the two most common correctness bugs.

---

## 9. Code / Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# --- Item-item collaborative filtering ---
item_sim = cosine_similarity(A.T)          # m×m; recommend items most similar to a user's liked items

# --- Matrix factorization by SGD (only observed entries) ---
def mf(A, b=2, lr=0.01, lam=0.1, epochs=1000):
    n, m = A.shape
    U, V = np.random.normal(0, .1, (n, b)), np.random.normal(0, .1, (m, b))
    obs = [(i, j) for i in range(n) for j in range(m) if A[i, j] > 0]   # 0 = missing
    for _ in range(epochs):
        for i, j in obs:
            err = A[i, j] - U[i] @ V[j]
            U[i] += lr * (err * V[j] - lam * U[i])       # gradient step on user vector
            V[j] += lr * (err * U[i] - lam * V[j])       # gradient step on item vector
    return U, V                                  # A_hat = U @ V.T  fills every missing cell

# --- Truncated SVD (denoise + reduce) ---
A_reduced = TruncatedSVD(n_components=20).fit_transform(A)   # scipy/np.linalg.svd for U,Σ,Vᵀ

# Production: the `surprise` / `implicit` / `LightFM` libraries implement SVD, ALS, BPR, and hybrids.
```

---

## 10. When It Breaks & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Rich interaction history, want behavior-driven recs | **Collaborative filtering (item-item)** | no features needed; stable, serendipitous |
| **Cold start** (new user/item) | **Content-based** (metadata) | needs no interaction history |
| Large, sparse matrix; best accuracy | **Matrix factorization (SGD/ALS)** | latent factors, scales, fills missing cells |
| Complete/dense matrix, want exact themes | **(Truncated) SVD** | one-shot linear algebra, denoise/compress |
| Both signals available | **Hybrid** | combine metadata + interactions |
| Huge scale, complex patterns, side info | **Deep learning** (two-tower / neural CF) | learns non-linear interactions, uses rich features |

**Universal cautions:** treat missing as unknown (not 0); precompute expensive similarity matrices offline; design against popularity bias; and remember **cold start** dogs every interaction-only method → keep a content-based/popularity fallback.

---

## 11. Interview Lens

**"Item-item vs user-user collaborative filtering — which and why?"** → 🎯 *"Both are cosine-similarity nearest-neighbor over the rating matrix. Prefer item-item when users outnumber items: the similarity matrix is smaller and item characteristics are stable, whereas user tastes drift, making user-user recommendations unstable."*

**"How does matrix factorization actually recommend?"** → 🎯 *"It learns a low-dimensional latent vector per user and per item by minimizing squared error **only on observed ratings** (plus L2), via SGD or ALS. Then any missing `A_ij` is the dot product `Uᵢ·Vⱼ` — that completed value is the predicted rating you rank on."*

**Likely follow-ups:**
- *Why not treat missing entries as 0?* → It teaches the model users dislike everything unseen; only train on observed entries.
- *How are MF, SVD, truncated SVD, and PCA related?* → SVD = exact factorization of a complete matrix; truncated SVD = top-k themes (denoise); PCA = SVD on centered data (uses V as PCs); learned MF = the sparse/approximate version fit on observed entries.
- *What is cold start and how do you handle it?* → New user/item/community with no history → fall back to content-based, popularity, or ask preferences.
- *Content-based vs collaborative — trade-offs?* → Content needs features/domain knowledge and over-specialises but solves cold start; collaborative needs no features and is serendipitous but suffers cold start/sparsity.
- *Explicit vs implicit feedback?* → Explicit = ratings (squared-error); implicit = clicks/plays (missing≠dislike → ranking loss like BPR + confidence weighting).
- *How do you evaluate a recommender?* → Ranking metrics on the top-k (Precision@k, Recall@k, NDCG, MAP), time-based split — not RMSE alone.
- *What does the number of latent factors control?* → Model capacity: too few underfits, too many overfits/needs more regularization (choose via validation / explained variance).

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why is "missing ≠ 0" the cardinal rule of recommender data?
   <details><summary>answer</summary>A blank means "no data," not a zero rating. Treating blanks as 0 makes the model learn that users dislike everything they haven't interacted with — and MF must therefore train only on observed entries.</details>
2. Item-item vs user-user CF — when do you pick item-item?
   <details><summary>answer</summary>When users > items: smaller (m×m) similarity matrix and item characteristics are stable, whereas user preferences drift over time and destabilize user-user recs.</details>
3. Write the matrix-factorization objective and say what's special about the sum.
   <details><summary>answer</summary>`min Σ_(i,j)∈Ω (A_ij − Uᵢ·Vⱼ)² + λ(‖Uᵢ‖²+‖Vⱼ‖²)`. The sum is over **observed** entries only; missing cells are excluded, then predicted as `Uᵢ·Vⱼ`.</details>
4. How do MF, SVD, truncated SVD, and PCA relate?
   <details><summary>answer</summary>SVD = exact factorization `A=UΣVᵀ` of a complete matrix; truncated SVD keeps top-k themes (denoise/complete); PCA = SVD of mean-centered data using V as principal components; learned MF = the sparse, gradient-descent version fit on observed entries.</details>
5. What is the cold start problem and one fix?
   <details><summary>answer</summary>A new user/item/community has no interaction history, so CF/MF can't form a vector or similarity. Fix: content-based filtering (metadata), popularity fallback, or ask the user for preferences.</details>
6. Why evaluate with Precision@k / NDCG rather than RMSE?
   <details><summary>answer</summary>Users consume a ranked top-k list, so ranking quality (relevant items ranked high) matters more than exact rating error; NDCG rewards putting the most relevant items highest.</details>
7. Explicit vs implicit feedback — how does the model change?
   <details><summary>answer</summary>Explicit (star ratings) → minimise squared error on values. Implicit (clicks/plays) → missing isn't dislike, so use confidence weighting and a ranking objective (e.g. BPR) rather than fitting exact values.</details>

---

*Covers: the sparse user-item matrix (missing≠dislike), the recsys evolution, collaborative filtering (item-item & user-user, cosine similarity, item-item stability), content-based filtering (metadata, cold-start fix, over-specialisation), the cold start problem, matrix factorization (latent factors, observed-only loss, SGD/ALS, matrix completion, Netflix, explicit vs implicit), SVD & truncated SVD (themes, denoising, and the MF/SVD/PCA relationship), ranking evaluation (Precision@k/NDCG/MAP), popularity bias, and deep-learning recommenders. Sourced from the Scaler Recommendation Systems lecture; SVD/PCA linear algebra in [PCA & t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md).*
