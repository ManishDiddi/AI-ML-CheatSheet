# Siamese Networks & Image Similarity — embeddings + distance = similarity

> **TL;DR.** Represent every image as an **embedding** (a feature vector); "similar" then just means "close in that vector space." Two regimes: **(1) off-the-shelf** — take a pretrained CNN's penultimate features *as-is* for reverse-image-search / retrieval (k-NN → PCA → approximate-NN/vector-DB to scale to millions); **(2) learned** — when you need *verification* or *one-shot* recognition (signature, face) with ~1 example per class, plain classification fails, so you **train the embedding** with a **Siamese network** + **contrastive** or **triplet** loss so same-class pairs land close and different-class pairs land far. Triplet loss (anchor/positive/negative + margin) learns *relative* similarity and generalizes far better than contrastive. The killer property: adding a new identity is just adding one embedding — **no retraining**.

**Where it fits:** the "compare two images" task (verification, retrieval, dedup, clustering, one-shot/few-shot), distinct from "classify one image." Powers face ID, signature/fraud verification, Pinterest/Google reverse-image search, and near-duplicate detection.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (GAP, ResNet/Inception backbones), [Transfer Learning](Transfer%20Learning.md) (the backbone-as-feature-extractor idea), [PCA & t-SNE](../Unsupervised%20ML/PCA%20&%20t-SNE.md) (compress + visualize embeddings), [KNN](../Supervised%20ML/KNN.md) (the search), and [Classification Metrics](../Supervised%20ML/Classification%20Metrics.md) (TP/FP threshold tuning for verification).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — embeddings, distance, contrastive & triplet loss](#2-the-formal-core--embeddings-distance-contrastive--triplet-loss)
3. [How It Works — the two regimes](#3-how-it-works--the-two-regimes)
4. [Worked Examples — losses, triplet types, search speed](#4-worked-examples--losses-triplet-types-search-speed)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Production & MLOps Notes](#7-production--mlops-notes)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**Everything hinges on the embedding.** A CNN's layers before the classifier already encode an image as a rich vector (ResNet-50's global-average-pooled output is 2048-D). That vector *is* the image's meaning in numbers. Compare two images by comparing their vectors:

```
image ─▶ CNN backbone ─▶ [0.12, −0.44, …, 0.09]  (embedding, e.g. 2048-D)
                              │
similar images → nearby vectors     different images → distant vectors
similarity(A, B) = 1 / distance(f(A), f(B))
```
Why not compare raw pixels? Because image meaning lives in the *interactions* between pixels, not per-pixel values — a 2-pixel shift changes every pixel but not the content. Embeddings capture content; pixels don't.

**Why classification can't do verification / one-shot.** Suppose you build face-attendance for 1000 employees with a 1000-way softmax CNN:
```
❌ Data: you need many labeled photos PER person — often you have just 1.
❌ Scalability: a new hire = 1001 classes → add an output neuron → RETRAIN the whole net.
```
A Siamese network sidesteps both: it doesn't classify *who* — it learns a *distance* ("are these two the same person?"). A new identity is just a new stored embedding; **no retraining, no new class**. That's one-shot learning.

**Identification vs verification (say both):** *verification* is 1:1 ("is this the person they claim?" — compare to one reference, threshold the distance); *identification* is 1:K ("who is this?" — nearest neighbour among K enrolled embeddings).

🎯 *"For verification/one-shot I don't classify — I learn an embedding with a Siamese network so same-identity images are close and different ones far, then threshold the distance. Enrolling a new identity is just storing one embedding, no retraining."*

---

## 2. The Formal Core — embeddings, distance, contrastive & triplet loss

**Embedding & distance.** `f(x) ∈ ℝᵈ` is the learned (or extracted) vector. Similarity is a distance:
```
Euclidean:  D(a,b) = ‖f(a) − f(b)‖₂           Cosine: cos(a,b) = (a·b)/(‖a‖‖b‖)
```
**L2-normalize** embeddings (`f/‖f‖`) so magnitude doesn't dominate — after that, Euclidean and cosine rank neighbours identically. Normalization is why "distance thresholds" transfer across images.

**Siamese network.** Two (or three) **identical subnetworks that share weights** ("twins") map each input through the *same* function `f`, then a distance layer compares the embeddings. Weight sharing is essential — both inputs must be embedded by the same function for distances to be meaningful. It can't be trained with cross-entropy (there's no fixed class set); you use a **metric-learning** loss:

**Contrastive loss** (pairwise; label `Y=1` same, `Y=0` different, margin `m`):
```
L = Y · D²  +  (1 − Y) · max(m − D, 0)²
   ├── same pair:      pull together (minimize D²)
   └── different pair: push apart until D ≥ m, then stop (max(...)=0 → no more force)
```
The margin caps how hard easy negatives are penalized — once two different images are `m` apart, they're "different enough."

**Triplet loss** (three inputs — **A**nchor, **P**ositive same-class, **N**egative different-class; margin `α`):
```
L(A,P,N) = max( 0,  d(A,P) − d(A,N) + α )
   want:  d(A,P) + α  ≤  d(A,N)     "anchor closer to positive than to negative, by margin α"
```
The margin also prevents the trivial collapse `f(x)=0` (which would make all distances 0). Triplets fall into three regimes — **this classification drives training**:
```
easy      : d(A,N) > d(A,P) + α   → loss = 0  → NO gradient (already correct) → skip
hard      : d(A,N) < d(A,P)       → loss large (negative closer than positive!) → learn a lot
semi-hard : d(A,P) < d(A,N) < d(A,P)+α → small positive loss → useful, stable
```
Training only on random triplets wastes most batches on zero-loss easy triplets → you must **mine** hard/semi-hard triplets.

**Why triplet > contrastive.** Contrastive judges pairs in *absolute* terms — it shoves every negative to distance `m` with no view of the wider space, so pushing class 1 from class 2 may accidentally collide class 1 with class 3. Triplet is *relative* — it only insists the positive is closer than the negative by `α`, which shapes a globally coherent space better suited to ranking/retrieval. The cost: triplet needs good sampling and is fiddlier to train.

---

## 3. How It Works — the two regimes

### Regime A — off-the-shelf embeddings (no training): image similarity / retrieval
1. **Extract** features: run a pretrained backbone (ResNet-50, `include_top=False, pooling='avg'`) → a 2048-D vector per image. **L2-normalize** it.
2. **Index** the dataset's vectors once (precompute, cache to disk).
3. **Search**: for a query image, embed it and find nearest neighbours. Baseline = **brute-force k-NN** (sklearn `NearestNeighbors`, Euclidean) — "training" is instant (lazy learning); all cost is at query time, `O(N·d)`.
4. **Scale down the vectors**: [PCA](../Unsupervised%20ML/PCA%20&%20t-SNE.md) 2048 → ~150 dims keeps almost all accuracy (diminishing returns past the elbow) and gives ~**20× less data + ~20× faster** search — and lets the whole index sit in RAM.
5. **Scale up the search**: swap brute force for **Approximate Nearest Neighbours (ANN)** — Annoy (trees), Faiss, ScaNN, HNSW — trading a little recall for orders-of-magnitude speed. Above millions of vectors, use a **vector database** (Pinecone, Milvus, Weaviate, pgvector).
6. **Visualize/sanity-check** with **PCA → t-SNE** to 2-D (t-SNE alone doesn't scale, so PCA-reduce first) — clean class clusters mean the embedding space is good.

### Regime B — learned embeddings (Siamese metric learning): verification / one-shot
1. **Embedding tower**: a shared backbone (pretrained ResNet-50, frozen or fine-tuned) → GAP → `Dense` → the embedding. *One* tower, reused for every input.
2. **Contrastive path**: build **pairs** `(anchor, positive, 1)` and `(anchor, negative, 0)`, feed both through the shared tower, compute the Euclidean distance layer, optimize **contrastive loss**.
   **Triplet path**: build **triplets** `(anchor, positive, negative)`, embed all three, optimize **triplet loss** (usually via a custom `train_step` with `GradientTape`). Mine hard/semi-hard triplets for signal.
3. **Pick a decision threshold**: on a validation set, sweep the distance and choose the threshold that maximizes accuracy (`(TP+TN)/total`); at inference `distance ≤ threshold ⇒ "same/genuine"`.
4. **Deploy just the embedding tower** (drop the twin structure): embed each new image once, store the vector, and do all verification/retrieval as distance comparisons — so **enrolling a new identity = storing one embedding, zero retraining**.

```
TRAIN (twin, shared weights)                 INFERENCE (single tower + threshold)
 A ─▶[f]─┐                                    new ─▶[f]─▶ e_new
 P ─▶[f]─┼─▶ loss (pull A–P, push A–N)         ref ─▶[f]─▶ e_ref
 N ─▶[f]─┘                                    same? ⇔ ‖e_new − e_ref‖ ≤ threshold
```

---

## 4. Worked Examples — losses, triplet types, search speed

**Contrastive loss** (margin `m=1`):
```
positive pair (Y=1), D=0.3 → L = 1·0.3² + 0            = 0.09   (pull closer)
negative pair (Y=0), D=0.4 → L = 0 + max(1−0.4,0)²      = 0.36   (still too close → push)
negative pair (Y=0), D=1.2 → L = 0 + max(1−1.2,0)²      = 0      (already ≥ margin → done)
```

**Triplet loss** (`α=0.5`, `d(A,P)=0.4`):
```
d(A,N)=1.0 → easy      : 1.0 > 0.4+0.5 → L = max(0, 0.4−1.0+0.5) = 0     (no gradient — skip)
d(A,N)=0.6 → semi-hard : 0.4<0.6<0.9   → L = max(0, 0.4−0.6+0.5) = 0.3   (useful)
d(A,N)=0.3 → hard      : 0.3 < 0.4      → L = max(0, 0.4−0.3+0.5) = 0.6   (learn a lot)
```

**Search speed** (ResNet-50 features on Caltech-101, per query):
```
brute-force + 2048-D          → 59.8  ms      (baseline)
brute-force + PCA(150)        →  5.3  ms      (~11× — smaller vectors)
Annoy (ANN) + PCA(150)        → 19    µs      (~3000× — approximate index)
```

**Contrastive vs triplet on the signature task** — the generalization tell:
```
Contrastive:  train 92.6%   test 63.8%   ← big gap = overfit (absolute pushing)
Triplet:      train 82.7%   test 82.5%   ← train ≈ test = generalizes (relative structure)
```
Lower *train* accuracy but matching *test* accuracy is exactly why triplet is preferred for retrieval/verification.

---

## 5. Code / Implementation

**Shared embedding tower + Siamese distance (Keras):**
```python
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import resnet
import tensorflow.keras.backend as K

def make_embedding(dim=(180,180,3)):                     # the SHARED tower
    base = resnet.ResNet50(weights="imagenet", include_top=False, input_shape=dim)
    base.trainable = False                               # transfer learning: freeze backbone
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    return Model(base.input, layers.Dense(128)(x), name="embedding")

embedding = make_embedding()
a, b = Input((180,180,3)), Input((180,180,3))
dist = layers.Lambda(lambda v: K.sqrt(K.sum(K.square(v[0]-v[1]), axis=1, keepdims=True))
                     )([embedding(a), embedding(b)])     # same tower on BOTH inputs (shared weights)
siamese = Model([a, b], dist)
```

**Contrastive & triplet losses:**
```python
def contrastive_loss(y, D, margin=1.0):                  # y=1 same, y=0 different
    y = tf.cast(y, D.dtype)
    return K.mean(y * K.square(D) + (1 - y) * K.square(K.maximum(margin - D, 0)))

def triplet_loss(anchor, positive, negative, alpha=0.5):
    pos = tf.reduce_sum(tf.square(anchor - positive), -1)  # d(A,P)
    neg = tf.reduce_sum(tf.square(anchor - negative), -1)  # d(A,N)
    return tf.reduce_mean(tf.maximum(pos - neg + alpha, 0.0))  # max(0, d(A,P)−d(A,N)+α)
```

**Triplet training via a custom `train_step` (three towers share `embedding`):**
```python
class SiameseModel(tf.keras.Model):
    def __init__(self, net, margin=0.5): super().__init__(); self.net, self.margin = net, margin
    def train_step(self, data):
        with tf.GradientTape() as tape:
            a, p, n = self.net(data)                      # embeddings of anchor/pos/neg
            loss = tf.maximum(tf.reduce_sum((a-p)**2,-1) - tf.reduce_sum((a-n)**2,-1) + self.margin, 0.)
        g = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(g, self.net.trainable_weights))
        return {"loss": tf.reduce_mean(loss)}
```

**Regime A — off-the-shelf retrieval with ANN:**
```python
import numpy as np; from annoy import AnnoyIndex
feats = np.stack([embedding.predict(x)[0] for x in images])   # or a pretrained ResNet50(pooling='avg')
feats /= np.linalg.norm(feats, axis=1, keepdims=True)         # L2-normalize
idx = AnnoyIndex(feats.shape[1], "euclidean")                 # after optional PCA(150)
for i, v in enumerate(feats): idx.add_item(i, v)
idx.build(40)                                                 # 40 trees
neighbors = idx.get_nns_by_vector(query_vec, 5)              # ~microseconds at scale
```
(sklearn `NearestNeighbors(algorithm='brute')` is the exact baseline; `faiss` / `scann` are the production-grade ANN engines.)

---

## 6. When It Breaks

```
❌ Embedding collapse f(x)=const. Without the margin, the net drives all distances to 0 (loss 0, useless).
   The margin (contrastive m / triplet α) is what forbids the trivial solution — never drop it.

❌ Easy-triplet starvation. Random triplets are mostly "easy" (loss 0, no gradient) → training stalls.
   FIX: hard / semi-hard triplet mining (batch-hard: within each minibatch pick the hardest P and N).

❌ Contrastive's absolute pushing distorts the space. It shoves negatives to a fixed margin without global
   context → classes collide elsewhere → poor retrieval. Triplet (relative) usually fixes this.

❌ Overfitting looks like a train/test gap (contrastive 92→64 here). Watch train≈test, not just train.

❌ Threshold drift. The verification threshold is data/model-specific; re-tune it whenever you retrain or
   the input distribution shifts. A single global threshold rarely holds across domains.

❌ Cosine vs Euclidean mismatch. If you index with cosine but train/threshold with Euclidean (or forget to
   L2-normalize), neighbours and thresholds silently misbehave. Pick one and normalize.

❌ Approximate-NN recall loss. ANN trades exactness for speed → it can miss true neighbours. Tune trees/
   ef/nprobe and validate recall against the brute-force baseline.

❌ Curse of dimensionality. In raw 2048-D, distances concentrate and search is slow/ambiguous → PCA/whiten
   to ~128–256 dims first (also speeds search massively).

❌ Class imbalance / bias in pairs. Verification systems (face) can encode demographic bias — audit FPR/FNR
   across subgroups, not just overall accuracy.
```

---

## 7. Production & MLOps Notes

**Search infrastructure is the real system.** Brute force is only a baseline. In production:
- **ANN engines:** **Faiss** (incl. GPU + Product Quantization for billion-scale), **ScaNN**, **Annoy**, **HNSW** (graph-based, the common default in vector DBs).
- **Vector databases:** Pinecone, Milvus, Weaviate, Qdrant, **pgvector** — they handle indexing, filtering, sharding, and updates so you don't hand-roll it.
- **Compression:** PCA + **quantization (PQ/int8)** shrink vectors 10–30× for RAM-resident billion-scale indexes.

**Modern face recognition moved past triplets.** **FaceNet** popularized triplet loss, but **margin-based softmax** losses — **ArcFace** (additive *angular* margin), CosFace, SphereFace — train a classification head with a margin on the hypersphere and reach higher accuracy *without* triplet mining (no sampling headache). Reach for ArcFace-style losses for large-scale face/metric learning; triplet/contrastive still shine when you truly have pairs/one-shot. `(likely)`

**The cold-start superpower.** Enrolling a new identity = compute and store one embedding — **no retraining, no new class, instant**. This is *the* production reason to choose metric learning over a softmax classifier for open-set problems (face, product, fraud).

**Embedding versioning is a footgun.** Embeddings from model v1 and v2 are **not comparable** — if you retrain/upgrade the tower, you must **re-embed and re-index the entire corpus**, or new and old vectors live in different spaces. Version the embedding model with the index; migrate atomically.

**Related: self-supervised contrastive learning.** The same pull-together/push-apart idea powers **SimCLR/MoCo** (InfoNCE loss over augmented views) to pretrain backbones *without labels* — a scalable cousin of contrastive/triplet worth knowing as the modern representation-learning direction. `(likely)`

**Monitoring.** Track retrieval recall@k / verification FPR–FNR on labeled feedback, index freshness/latency, and per-subgroup error (bias). Watch for **embedding drift** as the input distribution moves away from training.

---

## 8. Interview Lens

> ⚡ The tell they want: *classification identifies among a fixed K; Siamese/metric learning measures distance for open-set verification/one-shot, so a new class needs no retraining.*

**"Face attendance for 1000 employees with 1 photo each — how?"** → 🎯 *"Not a 1000-way classifier — I'd have too little data per person and would have to retrain for every new hire. I train a Siamese network with triplet loss to embed faces so same-person images are close, enroll each employee as one stored embedding, and verify by thresholding the distance. New employee = store one more embedding, no retraining."*

**Likely follow-ups:**
- *Why not cross-entropy?* → No fixed class set; open-set + one-shot. Metric learning learns distance, not class scores. `(certain)`
- *Contrastive vs triplet, and why triplet usually wins?* → Contrastive pulls same / pushes different to an absolute margin (can distort the global space); triplet enforces *relative* order (A closer to P than N by α) → better-structured space for retrieval; but needs triplet mining. `(certain)`
- *What are easy/hard/semi-hard triplets and which do you train on?* → Easy (loss 0, skip), hard (N closer than P), semi-hard (N between P and P+α). Train on hard/semi-hard; mine them.
- *Why the margin?* → Prevents the collapse `f(x)=0` and caps effort on already-separated pairs.
- *Identification vs verification?* → 1:K (who is this? nearest neighbour among enrolled) vs 1:1 (are they who they claim? threshold one distance).
- *How do you scale similarity search to millions?* → Compress with PCA/quantization, replace brute force with ANN (Faiss/HNSW/ScaNN) / a vector DB; validate recall vs brute force. `(certain)`
- *Euclidean or cosine?* → Either after L2-normalization (they rank the same); normalize so magnitude doesn't dominate.
- *Beyond triplets for faces?* → ArcFace/CosFace (angular-margin softmax) — SOTA without triplet mining.
- *Add a new identity — what changes?* → Just store its embedding. No retraining. (The whole point.)

---

## 9. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Reverse-image search / retrieval, no labels to train | **Pretrained embeddings + k-NN/ANN** | features transfer; zero training |
| Verification / one-shot, ~1 example per class | **Siamese + triplet (or contrastive) loss** | learns distance; new class = new embedding |
| Large-scale face recognition | **ArcFace / CosFace** (angular-margin softmax) | higher accuracy, no triplet mining |
| Fixed, closed set of many-example classes | **Plain softmax classifier** | simpler when classes are fixed & data-rich |
| Millions–billions of vectors | **ANN (Faiss/HNSW/ScaNN) + vector DB** | brute force doesn't scale |
| Shrink/speed the index | **PCA + quantization (PQ)** | ~20× smaller/faster, RAM-resident |
| Label-free backbone pretraining | **SimCLR / MoCo (contrastive SSL)** | same idea, self-supervised at scale |
| Just visualize the space | **PCA → t-SNE to 2-D** | inspect cluster quality |

**Decision rule:** *closed set + lots of data per class → classifier; open set / verification / one-shot → metric learning (triplet, or ArcFace at scale); retrieval → pretrained embeddings + ANN.* Always L2-normalize, and re-index whenever the embedding model changes.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why can't a softmax classifier do face verification with one photo per person?
   <details><summary>answer</summary>Too little data per class, and it's open-set: a new person adds a class → retrain the whole net. Siamese/metric learning learns a *distance* instead, so a new identity is just a stored embedding — no retraining.</details>
2. What makes a network "Siamese," and why must the towers share weights?
   <details><summary>answer</summary>Two/three identical subnetworks applying the *same* function `f` to each input, then comparing embeddings by distance. Shared weights ensure both inputs are embedded by the same mapping, so distances are meaningful.</details>
3. Write contrastive and triplet loss and say what each optimizes.
   <details><summary>answer</summary>Contrastive: `Y·D² + (1−Y)·max(m−D,0)²` — pull same pairs together, push different pairs to ≥ m. Triplet: `max(0, d(A,P) − d(A,N) + α)` — anchor closer to positive than negative by margin α.</details>
4. Define easy / hard / semi-hard triplets and which you train on.
   <details><summary>answer</summary>Easy: `d(A,N) > d(A,P)+α` (loss 0, no gradient). Hard: `d(A,N) < d(A,P)` (negative closer than positive). Semi-hard: `d(A,P) < d(A,N) < d(A,P)+α`. Train on hard/semi-hard (mine them); easy triplets waste compute.</details>
5. Why is triplet loss usually preferred over contrastive?
   <details><summary>answer</summary>Contrastive pushes negatives to an *absolute* margin with no global view (can collide other classes); triplet enforces *relative* order → better-structured embedding space for ranking/retrieval. Cost: needs triplet mining. (Signature task: contrastive 92/64 overfit vs triplet 82/82.)</details>
6. Trace how you'd scale reverse-image search from 10k to 100M images.
   <details><summary>answer</summary>Precompute + L2-normalize embeddings; PCA-compress (2048→~150) for ~20× speed/size; replace brute-force k-NN with ANN (Faiss/HNSW/ScaNN) or a vector DB; quantize (PQ) for RAM-resident billion-scale; validate recall vs the brute-force baseline.</details>
7. You upgrade the embedding model. What must you do to the search index, and why?
   <details><summary>answer</summary>Re-embed and re-index the *entire* corpus. Embeddings from different model versions live in different spaces and aren't comparable — mixing them silently corrupts neighbours. Version the model with the index.</details>
8. What's the modern alternative to triplet loss for large-scale face recognition?
   <details><summary>answer</summary>Margin-based softmax — **ArcFace** (additive angular margin), CosFace, SphereFace — trains a classifier with a hypersphere margin, reaching SOTA without triplet sampling.</details>

---

*Covers: embeddings & why pixel distance fails · Euclidean/cosine + L2-normalization · identification vs verification · why classification fails at one-shot · Siamese/shared-weight towers · contrastive vs triplet loss (formulas, margin, collapse) · easy/hard/semi-hard triplets & mining · feature-extraction retrieval (k-NN → PCA → ANN → vector DB) · search benchmarks · PCA→t-SNE visualization · threshold selection · ArcFace/CosFace · self-supervised contrastive (SimCLR/MoCo) · cold-start enrollment, embedding versioning/re-indexing, bias monitoring.*
