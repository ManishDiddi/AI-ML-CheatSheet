# Five Interview Questions — Answer Knowledge + Delivery Notes

> Each answer is written in your 6-step template: **1. Direct answer → 2. Formula → 3. Mechanism → 4. Example → 5. Limitation → 6. Alternative.** The 🎯 line in each is the sentence that actually wins the question. If you cut everything else under pressure, keep that. Confidence tags: (certain) hard fact · (likely) strong inference · (guessing) gap-fill.

---

## Q1 — XGBoost on extracted features vs. fine-tuning ResNet/EfficientNet

**This is your flagged HIGH-PRIORITY gap. The interviewer is not asking "which is better." They're testing whether you have a decision framework.**

**1. Direct answer.** There's no universal winner — the choice is driven by data volume, domain distance from ImageNet, compute/latency budget, and interpretability needs. With little labeled data, frozen pretrained embeddings + XGBoost is the strong baseline; with enough data, fine-tuning wins.

**2. Formula / mechanism axis (the deciding variable).**

- **Feature extraction** = freeze a pretrained CNN, take the penultimate-layer embedding (e.g. ResNet50 → 2048-d vector), train XGBoost on those vectors. The conv filters never change.
- **Fine-tuning** = unfreeze some/all conv layers and continue training on your labels, so the filters _adapt_ to your domain.

**3. Why fine-tuning > feature extraction with enough data (your exact documented gap).** 🎯 _Frozen embeddings are capped at how well ImageNet's filters happen to transfer to your domain; fine-tuning removes that cap by letting the mid/late filters re-specialize to your data's distribution._ (certain) The more your domain differs from ImageNet (medical scans, satellite, microscopy), the bigger the gap fine-tuning closes — and the more labeled data you have, the more layers you can safely unfreeze without overfitting.

**4. Example.** 500 product images, classes separable by color/texture → handcrafted features (color histogram + LBP) or frozen embeddings + XGBoost; fast, no GPU, interpretable feature importance. 50k images of a domain unlike ImageNet → fine-tune EfficientNet end-to-end.

**5. Limitation.** XGBoost-on-features loses end-to-end optimization (the features were never optimized for _your_ task) and handcrafted features (color/texture) miss semantic structure. Fine-tuning needs a GPU, more data, and is overfitting-prone on small sets.

**6. Alternative / nuance to drop in.** XGBoost-on-embeddings shines when you must **fuse image features with tabular metadata** (price, category, sensor readings) — concatenate and feed one model. Deep learning needs a multimodal architecture for that. Also mention the practical reality: on a laptop, the XGBoost path trains in minutes on CPU; fine-tuning ResNet does not.

**The 3-question framework to recite:** _How much labeled data? How far is my domain from ImageNet? What's my compute and latency budget?_ That sentence alone is a 5/5.

### Q1 deep-dive — training cost, resources, and latency

**The assumption to kill first.** People call "XGBoost on embeddings" the lightweight option across the board. 🎯 _At inference it usually isn't — if the features are CNN embeddings, you still run the same ResNet forward pass the fine-tuned model would, then add a near-free XGBoost predict. The two are nearly identical in inference latency. The real asymmetry is in training._

**Training speed.** The XGBoost path is two stages, and the cost is mislabeled:

- Stage 1 — a **single forward pass** of every image through ResNet to get embeddings. This is the expensive part.
- Stage 2 — fitting XGBoost on the N × 2048 vectors. Cheap: no backprop, CPU-friendly, minutes.
- Fine-tuning does forward **+ backward** over all images, repeated for **every epoch** (≈10–50). Rough compute ratio: fine-tuning ≈ **30–150× the embedding-extraction pass** (backward ≈ 2× forward, done many times instead of once). (likely)

**Which is more resource-expensive, and why — fine-tuning, for four stacking reasons:** (certain)

1. **Backprop over ~25M params** (ResNet50) vs. trees on a fixed 2048-d vector. XGBoost's cost is independent of image resolution once embeddings exist.
2. **Multiple epochs** vs. one pass.
3. **Optimizer state** — Adam stores ~2 extra copies of every parameter (momentum + variance).
4. **Activation memory** — backprop retains layer activations for the backward pass; inference doesn't. This is what blows up GPU memory.

On an M3 Pro / 18GB: the embedding pass runs batch-by-batch with a tiny footprint, even CPU-only. Fine-tuning wants the GPU (MPS), and 18GB unified memory caps batch size → slower training or gradient accumulation. (likely)

**Latency (inference) — the part the intuition gets backwards:**

|Path|Inference cost|Notes|
|---|---|---|
|Handcrafted features + XGBoost|~1–10 ms, CPU-only|Genuinely light. No neural net.|
|CNN embeddings + XGBoost|CNN forward pass + <1 ms|≈ same as fine-tuned ResNet|
|Fine-tuned ResNet|one forward pass|~5–15 ms GPU, ~50–200 ms CPU|

🎯 _The only way the XGBoost path is genuinely lower-latency at inference is if you use **handcrafted** features (color histogram, LBP, HOG) instead of CNN embeddings. On embeddings you've already paid for the CNN, so the XGBoost head adds and saves essentially nothing._ (Millisecond figures are order-of-magnitude — they swing with batch size, image size, and CPU vs MPS.)

**Choose by criterion:**

- Little labeled data → embeddings + XGBoost (fine-tuning overfits).
- Large domain shift from ImageNet → fine-tune (overrides the data-size rule).
- No GPU / fast iteration → XGBoost path.
- Ultra-low CPU/edge latency → **handcrafted** features + XGBoost, not embeddings.
- Tabular metadata to fuse → XGBoost on concatenated features.
- Interpretability → XGBoost (feature importance).
- Max accuracy with enough data → fine-tuning.

**One-line interview kill shot:** _"The training cost gap is large and real; the inference latency gap is mostly an illusion unless I drop the CNN and use handcrafted features."_

---

## Q2 — Silhouette + WCSS to choose optimal k

**Trap: the question asks for BOTH metrics. The point is that they can disagree. Describing each in isolation = 3/5. Naming the conflict = 5/5.**

**1. Direct answer.** Use WCSS for the elbow as a coarse first cut, then use Silhouette (which rewards separation, not just compactness) to disambiguate, and finally validate the survivors against business interpretability.

**2. Formula (your "formula not shown" gap — write these on the board).**

- **WCSS / inertia:** `WCSS = Σ_clusters Σ_{x in cluster} ||x − μ_cluster||²` — sum of squared distances of points to their centroid.
- **Silhouette for one point:** `s = (b − a) / max(a, b)`, where `a` = mean distance to points in its _own_ cluster (cohesion), `b` = mean distance to points in the _nearest other_ cluster (separation). Range **[−1, 1]**. Report the **mean s over all points**; pick k that **maximizes** it.

**3. Mechanism.** WCSS **decreases monotonically** as k grows (at k = n it's 0), so you can't minimize it — you look for the **elbow**, the k after which the marginal drop flattens. Silhouette has an actual optimum because it _penalizes_ clusters that are compact but sitting on top of each other (high cohesion, low separation).

**4. Example.** 🎯 _WCSS elbow says k = 4, Silhouette peaks at k = 6 — this is the real interview moment._ WCSS only saw "tighter clusters"; Silhouette saw that at k = 6 the clusters are also better _separated_. Trust the separation signal, but only ship the k whose clusters map to **actionable customer segments** — a statistically optimal k that the business can't act on is useless.

**5. Limitation.** Both assume roughly spherical/convex clusters (the K-Means bias) and **mislead on density-based or elongated shapes**. Silhouette is O(n²) — expensive on large datasets.

**6. Alternatives (your flagged missing ones).** **Davies–Bouldin index** (lower = better), **Calinski–Harabasz** (higher = better), and the **Gap statistic** (compares WCSS to a random-uniform null). For non-spherical clusters, switch to **DBSCAN/HDBSCAN** and validate differently.

---

## Q3 — Pooling types + convolutional filters

**You're strong here. The 5/5 differentiator is the "0 parameters" trick and the modern note that pooling is being replaced.**

**1. Direct answer.** Filters are _learnable_ feature detectors; pooling is a _fixed_ downsampling op that shrinks spatial size, adds local translation invariance, and cuts computation.

**2. Formula.** Conv output size: `floor((W − F + 2P) / S) + 1`. Conv parameters: `(F × F × C_in + 1) × C_out`. 🎯 **Pooling has zero learnable parameters** — common trick question, say it unprompted.

**3. Mechanism + pooling types.**

- **Filters** detect patterns hierarchically — edges → textures → parts → objects — and **weight sharing** (same filter slid everywhere) gives translation invariance and slashes parameters.
- **Max pooling:** keeps the strongest activation in the window → best for detecting _presence_ of a feature; ideal for sparse, strong activations. Most common.
- **Average pooling:** mean of the window → captures _overall intensity / texture_; smooths.
- **Min pooling:** keeps weakest signal → _absence_ detection; rarely used.

**4. Example.** **Global Average Pooling (GAP)** averages each entire feature map to one number, replacing the dense FC layers in modern nets (ResNet, etc.) — fewer parameters, acts as a regularizer.

**5. Limitation.** Max pooling discards location/magnitude info — **not acceptable** for fine-grained recognition, segmentation, or medical imaging where exact spatial detail matters.

**6. Alternative (modern note that signals you're current).** Many recent architectures **drop pooling for strided convolutions** (learnable downsampling) and use GAP only at the head — so you get adaptive downsampling instead of a fixed rule.

---

## Q4 — SARIMAX for weekly ad-impression forecasting

**Two traps: (a) the granularity contradiction, (b) the future-availability of exogenous variables. Both are the kind of thing that separates a senior answer.**

**1. Direct answer (lead with the clarifying question — your documented "ask what questions" gap).** 🎯 _First I'd confirm the granularity: the scenario mentions weekend spikes, but "weekly impressions" would aggregate those away — so if the data is daily I'd set seasonal period s = 7; if it's truly weekly aggregates, the weekend effect is gone and the relevant seasonality is annual (s ≈ 52)._ Stating this is the single highest-value move in the answer.

**2. Formula / spec.** SARIMAX = Seasonal ARIMA + eXogenous regressors: `(p, d, q)(P, D, Q, s)` plus a regression term on external variables X. `d/D` = differencing, `p/P` = AR order, `q/Q` = MA order, `s` = seasonal period.

**3. Mechanism / pipeline.**

1. **Stationarity** — ADF test + visual rolling mean/std; difference to get `d` (and seasonal-difference for `D`).
2. **Orders** — `p` from PACF cutoff, `q` from ACF cutoff; seasonal `P/Q` from ACF/PACF at lags that are multiples of `s`.
3. **Exogenous selection (your 5-step):** domain causality → cross-correlation at lags → Granger causality → VIF for multicollinearity → **future availability at forecast time**.
4. Fit, select by **AIC/BIC**, then **residual diagnostics**: residuals must be white noise (Ljung-Box).
5. Validate with **walk-forward** (expanding/rolling window), NOT k-fold — k-fold leaks future into past.

**4. Example — the exogenous insight that wins it.** 🎯 _Marketing budget is usable (you set it in advance). Competitor ad spend is NOT — you won't know it at forecast time, so you'd have to forecast it first or drop it. Major events are usable (calendar-known)._ The "future availability" filter is your documented most-missed step — name it explicitly.

**5. Limitation.** SARIMAX is **linear** and assumes one stable seasonal cycle; it struggles with multiple/changing seasonality and nonlinear interactions.

**6. Alternative.** For complex or multiple seasonality: **TBATS / Prophet**; for nonlinear effects and rich exogenous features: **gradient boosting (XGBoost/LightGBM) on lag + calendar features**, validated with the same walk-forward scheme.

### Q4 deep-dive — parameters, the waterfall, and ACF/PACF cutoff

**The premise to kill:** the seven parameters are NOT found in parallel. There's a strict dependency order. 🎯 _You cannot read ACF/PACF for p, q, P, Q until d and D are fixed, because ACF/PACF are only meaningful on a stationary series — on a trending series the plots are dominated by the trend and every lag looks significant._

**The waterfall:** d/D first → p,q,P,Q on the differenced series → exogenous → AIC selection → residual check → walk-forward.

**Parameters:**

||Meaning|Read from|
|---|---|---|
|p|non-seasonal AR — past _values_|PACF cutoff|
|d|non-seasonal differencing — removes trend|stationarity tests|
|q|non-seasonal MA — past _errors_|ACF cutoff|
|P|seasonal AR — values at lags s, 2s…|PACF at seasonal lags|
|D|seasonal differencing — removes seasonality|seasonal stationarity|
|Q|seasonal MA — errors at seasonal lags|ACF at seasonal lags|
|s|season length|domain knowledge|
|X|exogenous regressors|5-step selection|

**Step 1 — d and D (everything depends on this):**

- **d:** smallest differencing making the mean stationary. ADF (null = non-stationary; p > 0.05 → difference) + KPSS (null = stationary; complement) — want ADF to reject AND KPSS to not reject. Almost always 0–1, rarely 2.
- 🎯 **Over-differencing trap:** sharp _negative_ lag-1 ACF (≈ −0.5) + _rising_ variance = you differenced too much; it injects artificial MA structure. Stop at the smallest d.
- **D:** seasonal spike in ACF at s, 2s… ; formal test OCSB / Canova–Hansen (`nsdiffs`). Seasonal-difference `y_t − y_{t−s}`. Keep `d + D ≤ 2`; apply seasonal difference first.

**Step 2 — cutoff vs tails off (the conceptual heart):**

- **Cutoff** = drops _abruptly_ inside the ±1.96/√N band after some lag and stays there.
- **Tails off** = decays _gradually_, never snaps dead.

|Process|ACF|PACF|
|---|---|---|
|AR(p)|tails off|**cuts off after p**|
|MA(q)|**cuts off after q**|tails off|
|ARMA(p,q)|tails off|tails off|

_Why AR(p) cuts off in PACF:_ PACF = the **direct** correlation at lag k with intermediate lags partialled out. AR(p) depends only on the first p lags directly; beyond p any link runs _through_ intermediates, which PACF removes → zero past p. _Why MA(q) cuts off in ACF:_ two points correlate only if they **share white-noise terms**; they share terms only when `k ≤ q` → covariance zero past q. _Why the other curve tails off:_ AR has infinite memory so correlation propagates and decays; MA(q) = AR(∞) so its PACF inherits infinitely many decaying terms.

**Step 3 — p, q, P, Q on the stationary series:** p = PACF cutoff, q = ACF cutoff; P/Q = same at seasonal lags (s, 2s…). **Honest disclosure that signals seniority:** real plots are ambiguous — confirm with `auto_arima` (pmdarima) grid-searching to minimize AIC/BIC. Know the hand-reading to _explain why_, but say you'd validate with an AIC search.

**Step 4 — selection + validation:** AIC (prediction) vs BIC (parsimony, penalizes complexity harder), pick lowest. Residuals must be **white noise** — Ljung–Box (want p > 0.05), residual-ACF plot. Validate **walk-forward**, never k-fold.

**Framing each follow-up (avoid the verbosity trap):**

- _"How would you use SARIMAX?"_ → lead with the waterfall, not a parameter list.
- _"What are the parameters / how to find each?"_ → group: 3 non-seasonal (p,d,q), 3 seasonal mirrors (P,D,Q), season length s, exogenous X — d/D from stationarity tests, p/q from ACF/PACF _after_ differencing.
- _"What does cutoff mean?"_ → cutoff vs tails-off in one line each, the table, then one why-sentence.
- _"How to find d/D?"_ → smallest differencing passing ADF/KPSS (d) and seasonal tests (D); stop early to avoid over-differencing (negative lag-1 ACF + rising variance).

**The sentence that proves you've built one:** _"I fix d and D first, because ACF and PACF only mean anything on a stationary series — reading orders off a trending series is the classic mistake."_

---

## Q5 — TF-IDF from scratch (NumPy/Pandas only)

**You've already implemented this, so I'm giving you the knowledge + a skeleton, not the finished function — write the body yourself, that's the rep that matters.**

**1. Direct answer.** TF-IDF weights a word high when it's frequent _in a document_ but rare _across the corpus_, so common words like "the" get suppressed and distinctive words get boosted.

**2. Formula (state the variant — interviewers probe this).**

- `TF(t, d)` = count of term t in doc d, optionally normalized by doc length: `count / total_terms_in_d`.
- `IDF(t)` = `log(N / df(t))`, where `N` = number of documents, `df(t)` = number of documents containing t. Smoothed variant: `log((1 + N) / (1 + df(t))) + 1` to avoid divide-by-zero and zero-IDF.
- `TF-IDF(t, d) = TF(t, d) × IDF(t)`.

**3. Algorithm (skeleton — fill in the bodies).**

```text
1. tokens = each review split on spaces                # already in starter code
2. vocab  = sorted set of all unique tokens             # column order
3. TF matrix (n_docs × n_vocab):
      for each doc: count each vocab word (Counter), optionally / len(doc)
4. df vector (per vocab word):
      number of docs whose count > 0  -> sum over docs of (TF_count > 0)
5. idf  = np.log(N / df)            # or smoothed variant
6. tfidf = TF * idf                 # broadcast idf across rows
7. wrap in pd.DataFrame(tfidf, columns=vocab)
```

**4. Example intuition.** "fantastic" appears in 2 of 4 reviews → moderate IDF; a word in all 4 → IDF ≈ 0 (log(4/4)=0) → killed. That's the mechanism doing its job.

**5. Limitation.** Bag-of-words: no word order, no semantics ("not good" ≈ "good not"); sparse high-dim matrix.

**6. Alternative.** Word/sentence embeddings (Word2Vec, or sentence-transformers) capture semantics and word order; for your sentiment project, that's the upgrade path.

🎯 **Interview tell for the code question:** before writing, _say which TF and IDF variant you're using and why_ — "raw count TF, smoothed IDF to avoid zero weights." That one sentence shows you know there are choices, which is what the "from scratch" framing is testing.

---

## Cross-cutting delivery rules (your real bottleneck)

1. **Sentence one is the direct answer.** No "there are several ways to look at this."
2. **Write the formula on the board** whenever one exists. Your feedback flagged this 5×.
3. **Name the trade-off the question is testing** — every one of these 5 has a hidden "when does it break / what do they disagree on" core.
4. **Diagnose or clarify before solving** — Q1 and Q4 above both reward asking a question first.
5. **6–8 sentences, then stop.** Verbosity was your #1 documented issue.