# Embeddings — Dense Vectors, Bi- & Cross-Encoders, Fine-Tuning & Evaluation

> **TL;DR.** An **embedding** is a learned function `E: text → ℝ^d` that turns text into a dense vector where **semantic closeness = geometric closeness** — so "how do I reset my password" lands near "account recovery steps" with *zero shared words*. Two model shapes do the work: a **bi-encoder** encodes query and document *separately* into comparable vectors (fast, precomputable, scales to millions — the **retriever**), and a **cross-encoder** feeds `[query ⊕ document]` through one transformer for a single relevance score (slow, accurate — the **reranker**). Generic embeddings under-retrieve on your jargon, so you **fine-tune** them on in-domain `(anchor, positive[, negative])` pairs with a contrastive-family loss (Triplet, **Multiple Negatives Ranking**, Contrastive; BCE for the cross-encoder). You **prove** the gain with offline IR metrics — **Recall@K, MRR@K, NDCG@K**. Cosine similarity rules because it measures *direction* (meaning), not magnitude — and on normalized vectors it *is* the dot product.

**Where it fits:** Embeddings are the substrate under semantic search, [RAG](RAG.md) retrieval, clustering, dedup, classification, and recommendation. This note is the **model-level deep dive** that [RAG](RAG.md) forward-points to: it owns the *encoders, how you train/choose/evaluate them*; [RAG](RAG.md) owns the *pipeline around them* (chunking, ANN/vector DBs, hybrid merge, generation). It's the middle rung of the adaptation ladder — **prompt → RAG for knowledge → fine-tune for behavior** — but here we fine-tune the *retriever*, not the generator.
**Prereqs:** [RNN · LSTM · Transformers](../Machine%20Learning/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) (BERT, attention, `[CLS]`/`[SEP]`, mean-pooling), [RAG](RAG.md) (where retrieval sits), and vectors/cosine from linear algebra. The metric-learning machinery here (triplet/contrastive loss, Siamese towers) is the *same idea* as [Siamese Networks & Image Similarity](../Machine%20Learning/Computer%20Vision/Siamese%20Networks%20&%20Image%20Similarity.md) — text instead of images.

> ⚙️ *Format note: this adapts the vault's standard topic skeleton. The "Formal Core" is split into **similarity geometry (§2)** and the **architectures (§4)**; a dedicated **§3 (BERT→SBERT)** explains why sentence embeddings need their own model, and **§5 (model choice)** + **§6 (fine-tuning)** + **§7 (evaluation)** are the lecture's three deliverables. Retrieval *pipeline* internals (HNSW, RRF) stay in [RAG](RAG.md); end-to-end RAG answer-quality eval stays deferred to [[RAG Evaluation]].*

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — Similarity & Distance Metrics](#2-the-formal-core--similarity--distance-metrics)
3. [From BERT to SBERT — Why Sentence Embeddings Need Their Own Model](#3-from-bert-to-sbert--why-sentence-embeddings-need-their-own-model)
4. [Bi-Encoder vs Cross-Encoder — The Two Architectures](#4-bi-encoder-vs-cross-encoder--the-two-architectures)
5. [Choosing an Embedding Model](#5-choosing-an-embedding-model)
6. [Fine-Tuning Embedding Models](#6-fine-tuning-embedding-models)
7. [Evaluating Retrieval — Precision, Recall, MRR, NDCG](#7-evaluating-retrieval--precision-recall-mrr-ndcg)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks](#9-when-it-breaks)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose](#12-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

A computer can't compare *meaning* directly — it compares numbers. An embedding is the bridge: **text → a point in a high-dimensional space**, trained so that points which *mean* similar things sit close together and unrelated things sit far apart.

```
        "puppy"        ← close (both young pets)
   "kitten" •  • "cat"
              \   
               • "dog"        far →   • "invoice"   • "tax return"
                                        (a totally different neighbourhood)
```

The classic proof that the geometry *encodes* meaning is the word2vec analogy — relationships become **directions** you can do arithmetic on:

```
vec("King") − vec("Man") + vec("Woman") ≈ vec("Queen")
```

The vector *difference* King→Man is (roughly) the "royal→commoner" direction; the *difference* Man→Woman is the "gender" direction. Embeddings **preserve the relationships between words**, not just their identities. That's the whole reason they beat one-hot/keyword representations, which treat every word as equidistant from every other.

**Two levels of embedding — know which one you're holding:**

```
Token embedding     : one vector PER token/word-piece. Contextual in BERT
                      ("bank" of a river ≠ "bank" for money). Building block.
Sentence / chunk    : ONE vector for a whole sentence/passage. What you index
   embedding          and search in RAG. Built by POOLING token vectors (§3).
```

`(certain)` A vanilla BERT gives you *token* embeddings; getting a good *sentence* embedding out of it is exactly the problem SBERT solved (§3).

**Why we need embeddings — two jobs, one representation:**

```
Semantic SEARCH      : given a query, find the closest documents.   (RAG retrieval)
Semantic SIMILARITY  : given two texts, score how alike they are.   (dedup, STS, clustering)
```

Both reduce to the same operation: embed everything, then measure closeness with a **similarity metric** — which is §2, and the first place people quietly get it wrong.

🎯 **Kill-shot:** *"An embedding maps text into a space where distance = dissimilarity of meaning, so semantic search becomes nearest-neighbour geometry — and `King − Man + Woman ≈ Queen` is the proof the space encodes relationships, not just words."*

---

## 2. The Formal Core — Similarity & Distance Metrics

Once text is a vector, "similar?" becomes "close?" — and there are four common ways to measure it. They **do not always agree**, and picking the wrong one silently wrecks retrieval.

```
Metric              Formula                          Feels like            Higher/Lower = closer
────────────────────────────────────────────────────────────────────────────────────────────
Cosine similarity   (A·B) / (‖A‖‖B‖)   ∈ [−1,1]      angle between them    HIGHER
Dot product         A·B = Σ Aᵢ Bᵢ                    angle + magnitude     HIGHER
Euclidean (L2)      √Σ (Aᵢ − Bᵢ)²                    straight-line gap     LOWER
Manhattan (L1)      Σ |Aᵢ − Bᵢ|                      city-block gap        LOWER
```
(Vector DBs often store **squared L2** to skip the square root, and **cosine distance** = `1 − cosine similarity` so that "lower = closer" like the distances.)

### Cosine vs Dot — the one that flips rankings

Cosine cares only about **direction** (meaning); dot product cares about direction **and magnitude**. Here's the example that makes it click. Compare `A = [1, 1]` against two candidates:

```
            A=[1,1] points at 45°
                ↗
   B₂=[400,1] ↗   (almost along the x-axis, but a hair above)
   B₁=[800,0] →   (exactly on the x-axis)

Dot product:   A·B₁ = 800·1 + 0·1 = 800      A·B₂ = 400·1 + 1·1 = 401
   → dot says  B₁ (800) is MORE similar than B₂ (401)      [magnitude wins]

Cosine:        cos(A,B₁) = 800 / (√2 · 800)   = 0.7071
               cos(A,B₂) = 401 / (√2 · 400.001)= 0.7089
   → cosine says B₂ is MORE similar than B₁                [direction wins]
```

🎯 **The rankings literally swap.** `B₁` looks closer to dot product *only because it's a bigger vector*; by **angle**, `B₂` is a touch nearer to `A`'s 45° heading. For **semantic** embeddings we care about meaning = direction, so **cosine is the default**. Magnitude in an embedding is mostly an artifact of length/frequency, not meaning — which is why you normalize it away.

**The identity worth memorizing:** on **L2-normalized** vectors (‖A‖ = ‖B‖ = 1),

```
cosine(A, B) = A·B          ← cosine IS the dot product once you normalize
```

So production systems normalize all vectors at index time and then use a **fast dot-product** (`IndexFlatIP` in FAISS) to get cosine ranking for free. `(certain)`

**When would you *want* dot product (keep magnitude)?** Recommendation / matrix-factorization systems, where a longer vector deliberately encodes **popularity / confidence / importance** — you *want* the blockbuster to outrank the obscure indie film of the same genre. Semantic retrieval, SBERT, and OpenAI-style embeddings → **cosine**. Recsys → often **dot**. 🎯 *Use the metric the embedding model was trained with — a cosine-trained model queried by dot (or vice-versa) quietly loses recall.*

### Sign of the dot product = relation type

The instructor's `Strawberry = [4,0,1]`, `Blueberry = [3,0,1]` toy shows a second use of dot product — its **sign**:

```
Strawberry · Blueberry   = 4·3 + 0 + 1·1  =  13   (positive, large → RELATED)
Strawberry · [−3,0,−1]   = −12 + 0 − 1    = −13   (negative → OPPOSITE meaning)
```

This maps straight onto how you **label training data** (§6): `+1` relevant/similar, `0` neutral/unrelated, `−1` opposite/contradictory.

### Euclidean, Manhattan, and the curse of dimensionality

- **Euclidean (L2)** is magnitude-sensitive (straight-line distance) and intuitive in low dimensions; it's the natural metric when the raw coordinate values matter.
- **Manhattan (L1)** sums per-axis gaps — cheaper to compute and more robust in very high dimensions (each dimension contributes independently), which is why some ANN indexes offer it.
- **Curse of dimensionality** `(likely)`: in a *random* high-`d` space, all pairwise distances concentrate to nearly the same value, so "nearest neighbour" becomes meaningless. Embeddings dodge this because a **trained** space isn't random — it's a low-dimensional *meaning manifold* embedded in `ℝ^d`, so structure survives. Still, the practical takeaways are: **normalize**, and prefer cosine/dot for semantic vectors.

---

## 3. From BERT to SBERT — Why Sentence Embeddings Need Their Own Model

BERT is a phenomenal *token-level* model (masked-language-modelling), but it was **not built to emit a sentence vector**. Before SBERT (Reimers & Gurevych, 2019), you had two bad options for "how similar are these two sentences?":

```
Option A — CROSS-ENCODE with BERT           Option B — POOL BERT's token vectors
[CLS] sentA [SEP] sentB → classifier        mean-pool (or take [CLS]) → one vector each
✔ very accurate                             ✔ fast: encode once, cosine-compare
✗ ONE forward pass PER PAIR                 ✗ embeddings NOT semantically meaningful
✗ 10k sentences → ~50M pair comparisons     ✗ often WORSE than averaged GloVe (!)
✗ ≈ 65 hours for a similarity search
```

You were forced to choose **accuracy (cross-encoder, unusably slow)** or **speed (pooled BERT, poor quality)**. There was no method that was *both* scalable and accurate.

**SBERT's fix — make pooling meaningful by training for it:**

```
Step 1  SIAMESE architecture: two sentences go through the SAME encoder
        (shared weights) → two independent pooled embeddings.
                sentA → [BERT] → mean-pool → u
                sentB → [BERT] → mean-pool → v      (same weights)
Step 2  TRAIN on sentence pairs (NLI: SNLI ~570K, MultiNLI ~430K) with a
        similarity objective, so the geometry gets shaped:
             entailment  → pull u,v together
             contradiction→ push u,v apart
Step 3  Now mean-pooling produces a GENUINE sentence embedding — cosine over
        it actually tracks meaning.
```

The payoff is the whole reason embeddings are usable at scale:

```
Cross-encoder search over 10,000 sentences : ~65 hours
SBERT: encode all 10,000 once              : ~5 seconds
        then cosine-compare embeddings      : ~0.01 seconds
→ massive speed-up, minimal accuracy loss.  THIS is what makes RAG possible.
```

`(certain)` SBERT didn't invent a new architecture so much as **fine-tune BERT with a Siamese, similarity-shaped objective** — the same metric-learning trick as [Siamese Networks & Image Similarity](../Machine%20Learning/Computer%20Vision/Siamese%20Networks%20&%20Image%20Similarity.md), applied to text. That objective is exactly what §6 generalizes when you fine-tune on *your* domain.

🎯 **Kill-shot:** *"Vanilla BERT is token-level: cross-encoding every pair is accurate but O(n²)-slow, and naïvely pooling its tokens gives embeddings worse than GloVe. SBERT trains a Siamese BERT on NLI pairs so that mean-pooling becomes semantically meaningful — turning a 65-hour search into 5 seconds."*

---

## 4. Bi-Encoder vs Cross-Encoder — The Two Architectures

This is the central mental model of the whole topic — and the single most-asked interview question here. Both take text pairs; they differ in **when** the query and document meet.

```
        BI-ENCODER  (the RETRIEVER)                 CROSS-ENCODER  (the RERANKER)
   ┌─────────────┐   ┌─────────────┐            ┌───────────────────────────────┐
   │  Query      │   │  Document   │            │ [CLS] Query [SEP] Document [SEP]│
   └──────┬──────┘   └──────┬──────┘            └────────────────┬──────────────┘
      [encoder]         [encoder]  (shared)            [ ONE transformer ]
          │                 │                     full cross-attention: every query
        vec_q             vec_d                     token attends to every doc token
          └───── cosine ────┘                                    │
             ONE score                                    [CLS] → FC → σ
                                                        ONE relevance score ∈ (0,1)
   • docs encoded OFFLINE, once, stored           • CANNOT precompute — score needs the PAIR
   • query = 1 forward pass + fast ANN            • ONE forward pass PER candidate document
   • scales to MILLIONS                           • too slow for the full corpus
   • weakness: q & d never "see" each other       • strength: token-level interaction → precise
                → coarser relevance                             → far more accurate
```

**Side-by-side:**

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Input | query and doc **separately** | `[CLS] query [SEP] doc` **together** |
| Output | two vectors → cosine | one relevance score |
| Precompute doc reps? | **Yes** (index offline) | **No** (depends on the pair) |
| Cost at query time | 1 encode + ANN over millions | 1 forward pass **per candidate** |
| Speed / Scale | fast, scales to millions | slow, only ~top 10–100 |
| Accuracy | good (coarse) | **best** (fine-grained) |
| Role | **retrieval** (semantic search) | **reranking** |
| Loss to train (§6) | Triplet / MNR / Contrastive | BCE / log-loss (relevant vs not) |

**You don't choose one — you stage them.** The bi-encoder casts a wide, cheap net for **recall**; the cross-encoder does expensive, precise **ranking** on the shortlist:

```
query ─► BI-ENCODER retrieve top-N (e.g. 20–100)  ─►  CROSS-ENCODER rerank ─► top-k (e.g. 3–5)
         └ fast, recall-oriented, over millions      └ slow, precision-oriented, over the shortlist
        (optionally merge with BM25 keyword hits, dedup, THEN rerank — see RAG §5)
```

This **two-stage retrieve→rerank** split is the backbone of every serious retriever; the pipeline plumbing (hybrid merge/RRF, dedup, vector DB) lives in [RAG §5](RAG.md#5-stage-2--retrieval--generation-online) and [RAG §6](RAG.md#6-vector-databases--ann-indexing-hnsw). `(certain)`

🎯 **Kill-shot:** *"Bi-encoder for recall at scale — precompute document vectors and ANN-search millions; cross-encoder for precision on the shortlist — joint attention scores each pair but can't be precomputed. Retrieve-then-rerank is the backbone."*

---

## 5. Choosing an Embedding Model

Before you can fine-tune (§6) you pick a base. Three axes decide it.

### Open-source / local vs API / hosted

```
OPEN-SOURCE (SentenceTransformers, run locally)   API / HOSTED (OpenAI, Cohere, Voyage, Jina)
✔ documents NEVER leave your infra (privacy,      ✔ zero ops, strong general quality out of the box
   compliance — legal/health/finance)             ✔ big models you couldn't host cheaply
✔ free at inference (own the GPU/CPU)             ✗ your text is SENT to a third party (security review)
✔ FULLY fine-tunable (own the weights)           ✗ per-token cost, rate limits, network latency
✗ you run/scale/monitor it yourself              ✗ can't truly fine-tune the weights (adapter-only)
```

🎯 The decision usually reduces to **data sensitivity** and **whether you need to fine-tune**: sensitive corpus or domain fine-tuning → **open-source, local**; general content and you want it working today → **API**. The instructor's blunt version: with an API model *"embedding may not work well on your domain and fine-tuning is ~impossible."* `(likely)`

### Domain-specific vs general — the "SWIFT" problem

A general model embeds by *general* usage — which mis-fires on domain jargon. Word sense is **domain-dependent**:

```
"SWIFT"  in a GENERAL corpus  → bird / fast / a car model / a translation system
"SWIFT"  in a BANKING corpus  → the interbank financial-messaging / transaction system
```

If your corpus is banking and your embedding model thinks SWIFT is a bird, retrieval fails. Fix: **pick a domain-tuned model** (legal/bio/code embeddings) **or fine-tune** a general one on your data (§6) so the geometry disambiguates *your* senses.

### The model zoo (and dimensions)

```
Model                              dim    notes
──────────────────────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2                   384    fast, tiny, great DEFAULT; general-purpose
all-mpnet-base-v2                  768    higher quality, slower; general-purpose
multi-qa-mpnet-base-dot-v1         768    tuned for QA RETRIEVAL, ASYMMETRIC, dot-product
bge / e5 / gte (base/large)      768/1024 strong open MTEB models; e5/bge use query/passage prefixes
OpenAI text-embedding-3-small/large 1536/3072  API; Matryoshka-truncatable dims
```

- **Dimension trade-off:** bigger `d` = generally better recall but **more storage + slower search** (1M × 1536 × 4 bytes ≈ 6 GB). Pick the smallest `d` that hits your recall target.
- **Symmetric vs asymmetric** (a real trap, expanded in §9): STS models (`all-mpnet`) assume both sides are *similar-length, same-nature* text; QA-retrieval models (`multi-qa-*`, `e5`) are trained for **short query ↔ long passage** and often need `"query:"` / `"passage:"` prefixes. Match the model to your query shape.
- **How to actually pick:** consult the **MTEB** leaderboard (Massive Text Embedding Benchmark) for your task family (retrieval vs STS vs clustering), then validate on *your* data with §7 metrics — leaderboard rank ≠ your-corpus rank. `(likely)`

---

## 6. Fine-Tuning Embedding Models

**Why fine-tune at all?** A generic model places *generic* neighbours close; on your corpus, the queries your users actually type may land nearer to the wrong documents (the SWIFT problem). Fine-tuning **re-shapes the geometry** so *your* queries sit next to *your* relevant docs. You can fine-tune the **bi-encoder** (better retrieval), the **cross-encoder** (better reranking), or **both**.

### Supervised setup — the data you need

Fine-tuning embeddings is **supervised metric learning**. Your training signal is pairs/triplets built from a query (the *anchor* / *candidate*) and documents labelled by relation:

```
anchor (query) ──► positive  : a relevant document           (pull CLOSER)
               ──► negative  : an irrelevant document        (push APART)
               ──► neutral   : weakly/unrelated (optional)
```

**Hard negatives** (plausible-but-wrong docs — e.g. BM25's top non-answers) teach far more than random negatives; random negatives are so obviously wrong the model learns nothing from them.

### Bi-encoder losses (pick one)

**① Triplet loss** — pull anchor→positive closer than anchor→negative by a `margin`:

```
L = max( 0 ,  d(a,p) − d(a,n) + margin )
      loss is 0 once the negative is at least `margin` farther than the positive.
```

Worked example (instructor's cat sentences, cosine *distance*, `margin = 0.3`):

```
a = "The cat sat on the rug"    p = "The cat resting on the mat"   (positive)
                                 n = "The dog playing outside"      (negative)
d(a,p) = 0.3 ,  d(a,n) = 0.5
L = max(0, 0.3 − 0.5 + 0.3) = max(0, 0.1) = 0.1   → still some loss; not separated enough

If instead d(a,n) = 0.9 (negative pushed well away):
L = max(0, 0.3 − 0.9 + 0.3) = max(0, −0.3) = 0    → satisfied, zero loss ✓
```

**② Softmax loss** (original SBERT) — classify the NLI label from `[u ; v ; |u−v|]` (entailment / neutral / contradiction). Good when you have labelled NLI-style pairs.

**③ Multiple Negatives Ranking (MNR) loss** — **current best practice.** You supply only `(anchor, positive)` pairs; **every other positive in the batch automatically becomes a negative** for this anchor. So a batch of size B gives each anchor 1 positive and B−1 free negatives:

```
batch: (a₁,p₁) (a₂,p₂) … (a_B,p_B)
for a₁:  p₁ is the positive; p₂…p_B are in-batch NEGATIVES (no mining needed)
bigger batch → more negatives per step → sharper contrast → better model
```

🎯 MNR is why modern embedding training barely needs explicit negative mining — it's cheap, and **larger batch = better**. `(certain)`

**④ Contrastive loss** — pairwise: increase similarity for positive pairs, decrease it for negative pairs (with a margin). The general form the other three specialize.

### Cross-encoder loss

The cross-encoder outputs a single relevance score, so it's trained as **binary classification** — Binary Cross-Entropy / log-loss over `query→positive` (label 1) and `query→negative` (label 0):

```
L = −(1/n) Σ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]     yᵢ ∈ {0,1},  ŷᵢ = σ(FC([CLS]))
```

(You can also regress to a **graded** label — e.g. 0.99 relevant / 0 neutral / −0.99 opposite — if you have graded judgments.)

### What actually moves after fine-tuning

You embed the corpus with the fine-tuned bi-encoder, rebuild the index, and the *same query* now pulls the right documents into the shortlist — then the fine-tuned cross-encoder ranks them better. You **measure** the lift with §7 (default vs fine-tuned), which is the entire point of the exercise.

---

## 7. Evaluating Retrieval — Precision, Recall, MRR, NDCG

You have a labelled set of `query → relevant-doc(s)`. Retrieve the top-K and score how good the ranking is. These are **offline IR metrics** for the *retriever/embedding model* — distinct from end-to-end RAG answer quality (faithfulness/groundedness), which is [[RAG Evaluation]].

```
Top-K retrieved (✓ = relevant, ✗ = not), K=5, and say there are 5 relevant docs total:
   rank1 ✓   rank2 ✗   rank3 ✓   rank4 ✓   rank5 ✗
```

**Precision@K** — of what you *showed*, how much was relevant:
```
Precision@5 = (relevant in top-5) / K = 3 / 5 = 0.60
```

**Recall@K** — of all the *relevant* docs, how many you *found* in top-K. This is the retriever's key metric — did the right doc even make the shortlist the reranker sees?
```
Recall@5  = (relevant in top-5) / (total relevant) = 3 / 5 = 0.60
Recall@10 = 5 / 5 = 1.0     ← widen K and you catch them all
```
🎯 For a two-stage system you tune the bi-encoder for **high Recall@N** (N = shortlist size, e.g. 50–100); anything it misses, the cross-encoder can *never* recover.

**MRR@K (Mean Reciprocal Rank)** — how *high up* the **first** relevant hit lands; "how fast is the retrieval." Average `1/rank` of the first relevant doc across queries:
```
q1: first relevant at rank 1 → 1/1 = 1.0
q2: first relevant at rank 2 → 1/2 = 0.5
q3: first relevant at rank 3 → 1/3 = 0.33
q4: none in top-K            → 0
MRR@4 = (1 + 0.5 + 0.33 + 0) / 4 = 0.458
```
Great for "one right answer" tasks (QA, "I'm feeling lucky"). Blind to everything after the first hit.

**NDCG@K (Normalized Discounted Cumulative Gain)** — the richest: handles **graded** relevance (0/1/2…) *and* discounts by position. `DCG` sums `relᵢ / log₂(i+1)`; divide by the **ideal** ordering's `IDCG` to normalize to [0,1]:
```
DCG@K  = Σ  relᵢ / log₂(i + 1)          (i = rank position, 1-based)
NDCG@K = DCG@K / IDCG@K                  IDCG = DCG of the perfectly-sorted list

Bakery example, query "goes well with jam", graded rel ∈ {0,1,2}, K=4:
   your order  → DCG@4  = 2.193
   ideal order → IDCG@4 = 4.693
   NDCG@4 = 2.193 / 4.693 = 0.467
```
🎯 Use **NDCG** when relevance is graded and *order matters* (the default for ranking quality); **Recall@K** to size your retriever's net; **MRR** for single-answer tasks; **Precision@K** when you show users a fixed-length list.

**Log the runs.** In the notebook these are tracked with **Opik** so you can compare *default vs fine-tuned* bi-/cross-encoders on the same query set — the comparison table is the deliverable that proves the fine-tune helped. Evaluate on a **held-out** query set, never the pairs you trained on.

---

## 8. Code / Implementation

`sentence-transformers` is the workhorse. The arc: encode → similarity → two-stage retrieve/rerank → fine-tune → evaluate.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np, faiss

# ── Bi-encoder: independent encoding + cosine ──────────────────────────────
bi = SentenceTransformer("all-MiniLM-L6-v2")          # 384-dim, fast default
q_vec = bi.encode("What are fundamental human rights?", convert_to_tensor=True)
d_vec = bi.encode("Human rights are inherent to all people…", convert_to_tensor=True)
print(util.cos_sim(q_vec, d_vec).item())               # one cosine score

# ── Index a corpus with FAISS (normalize → inner product == cosine) ────────
docs = [...]                                            # your chunks
emb  = bi.encode(docs, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)                                 # unit vectors ⇒ IP = cosine
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

def retrieve(query, k=20):                              # STAGE 1 — bi-encoder recall
    qv = bi.encode([query], convert_to_numpy=True); faiss.normalize_L2(qv)
    scores, idx = index.search(qv, k)
    return [(docs[i], float(s)) for s, i in zip(scores[0], idx[0])]

# ── Cross-encoder rerank the shortlist ─────────────────────────────────────
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(query, candidates, top_k=5):                 # STAGE 2 — precision
    pairs  = [[query, d] for d, _ in candidates]
    scores = ce.predict(pairs)                          # ONE pass per pair
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for (d, _), s in ranked[:top_k]]

shortlist = retrieve("How are human rights protected?", k=20)
top5      = rerank("How are human rights protected?", shortlist)   # the two-stage retriever
```

```python
# ── Fine-tune the BI-ENCODER: Triplet loss OR Multiple-Negatives-Ranking ───
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

# Triplet: (anchor, positive, negative)
triplets = [InputExample(texts=[q, pos, neg]) for q, pos, neg in my_triplets]
loss = losses.TripletLoss(model=bi,
        distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)

# MNR (best practice): only (anchor, positive) — batch supplies the negatives
pairs = [InputExample(texts=[q, pos]) for q, pos in my_pairs]
loss  = losses.MultipleNegativesRankingLoss(model=bi)   # bigger batch = more negatives

dl = DataLoader(pairs, shuffle=True, batch_size=32)
bi.fit(train_objectives=[(dl, loss)], epochs=3,
       warmup_steps=int(0.1*len(dl)*3), output_path="./bi_finetuned")

# ── Fine-tune the CROSS-ENCODER: binary relevance (BCE) ────────────────────
ce_train = ([InputExample(texts=[q, pos], label=1.0) for q, pos in pos_pairs] +
            [InputExample(texts=[q, neg], label=0.0) for q, neg in neg_pairs])
ce.fit(train_dataloader=DataLoader(ce_train, shuffle=True, batch_size=16), epochs=3)
```

```python
# ── Evaluate: Recall@K + MRR on a held-out query set ───────────────────────
def recall_at_k(retrieved_idx, gold_idx, k):  return float(gold_idx in retrieved_idx[:k])
def rr(retrieved_idx, gold_idx):
    return 1.0/(retrieved_idx.index(gold_idx)+1) if gold_idx in retrieved_idx else 0.0
# sentence-transformers ships this for you: evaluation.InformationRetrievalEvaluator
```

`(certain)` The prebuilt `evaluation.InformationRetrievalEvaluator` computes Recall@K / MRR / NDCG / MAP over a corpus + query→relevant map — reach for it instead of hand-rolling, and log to Opik/TensorBoard to compare default vs fine-tuned.

---

## 9. When It Breaks

- **You forgot to normalize / anisotropy.** Raw transformer embeddings occupy a narrow **cone**, so *everything* looks ~0.8 cosine-similar and ranking is mush. SBERT training + **L2-normalizing** fixes it. If your cosine scores are all suspiciously high and close, this is why.
- **Metric mismatch.** Querying a **dot-product-trained** index (`multi-qa-*-dot`) with cosine, or vice-versa, silently degrades recall. Match the metric the model was trained with.
- **Symmetric vs asymmetric mixup.** An STS model (`all-mpnet`, both sides similar text) used for **short-query → long-passage** retrieval under-performs a QA-retrieval model (`multi-qa`, `e5` with `query:`/`passage:` prefixes). Wrong shape = quiet recall loss.
- **Domain shift.** General model on legal/medical/code/banking → the SWIFT problem. Symptom: high scores for *lexically* similar but *semantically* wrong docs. Fix: domain model or fine-tune (§6).
- **Max-sequence-length truncation.** Most models cap at **256–512 tokens** and **silently truncate** the rest — a long document's tail is invisible. Chunk *before* embedding (see [RAG](RAG.md) chunking); never embed a 10-page PDF as one vector.
- **Swapping models without re-embedding.** Embeddings from two different models live in **incompatible spaces** — you can't compare them. Change or upgrade the model ⇒ **re-embed the entire corpus** and rebuild the index. Version your embeddings.
- **Random (easy) negatives.** Training with random negatives plateaus fast — they're too obviously wrong. **Mine hard negatives** (BM25's top non-answers) for real gains. This is the #1 lever on fine-tune quality after data volume.
- **Over-fitting a tiny fine-tune set.** A few hundred pairs can *worsen* a strong base model (catastrophic forgetting of general semantics). Fine-tune with enough data, a held-out eval, warmup, and few epochs — and **always** compare against the un-tuned baseline before shipping.
- **Storage/latency blow-up.** 1536-d float32 × millions of docs = many GB of RAM and slow search. Reduce `d` (Matryoshka), **quantize** (int8/binary), or use a disk-backed ANN index (§10).

---

## 10. Production & MLOps Notes

- **Two cost regimes.** Document embedding is a **batch, offline** job (embed once at index time, amortized); query embedding is **online** (one forward pass on the hot path — cache repeated queries). The cross-encoder reranker is the expensive online piece: cost = `candidates × forward-pass`, so cap the shortlist (~50–100) and rerank only that.
- **Dimension vs cost — Matryoshka (MRL).** Matryoshka-trained embeddings (OpenAI `text-embedding-3`, `nomic`, some `bge`) pack the most information in the **early dimensions**, so you can **truncate** `1536 → 256` and keep most of the quality — huge storage/latency win. Store full, search truncated, or store truncated outright.
- **Quantization.** `int8` (~4× smaller) or **binary** embeddings (~32× smaller, Hamming distance) trade a little recall for massive scale; often paired with a float re-scoring pass on the top candidates.
- **Re-embedding & versioning.** A model upgrade = **full re-index**. Do it **blue-green**: build the new index alongside the old, validate §7 metrics on a golden query set, then cut over. Tag every vector with its `model@version` so you never mix spaces.
- **Monitoring & drift.** Track Recall@K / MRR / NDCG on a **fixed golden query set** over time; a drop signals corpus drift (new topics/jargon) or a broken embedding step. Retrain/refresh when it slips. Watch embedding-service latency and the truncation rate (docs exceeding max-seq-len).
- **The vector DB is downstream.** ANN index choice (HNSW/IVF-PQ), filtering, and hybrid merge (RRF) belong to the retrieval layer — see [RAG §6](RAG.md#6-vector-databases--ann-indexing-hnsw). This note's job is to hand it good, correctly-normalized, version-tagged vectors.

---

## 11. Interview Lens

The trade-offs these questions really test: **speed vs accuracy** (bi vs cross), **direction vs magnitude** (cosine vs dot), **generic vs adapted** (why fine-tune), and **which number proves it** (metrics).

- 🎯 **Bi vs cross:** *"Bi-encoder encodes query and doc separately so doc vectors precompute and ANN-search scales to millions — that's recall. Cross-encoder jointly attends over `[query ⊕ doc]` for a precise score but can't precompute, so it only reranks the shortlist — that's precision. Retrieve-then-rerank stages them."*
- 🎯 **Cosine vs dot:** *"Cosine is magnitude-invariant — it scores direction, i.e. meaning — and on L2-normalized vectors it equals the dot product, which is why DBs normalize and use inner product. Keep magnitude (dot) only when it encodes something real, like popularity in recsys."*
- 🎯 **Why fine-tune embeddings:** *"Generic embeddings encode generic word sense, so on domain jargon (SWIFT = bird, not bank) queries land near the wrong docs. Fine-tuning on in-domain (anchor, positive) pairs — usually MNR with in-batch negatives — re-shapes the space so your queries neighbour your relevant docs."*
- 🎯 **Which metric:** *"Recall@K for whether the retriever puts the answer on the reranker's shortlist; NDCG when relevance is graded and order matters; MRR for single-answer tasks; Precision@K for a fixed user-facing list."*
- **Likely follow-ups:**
  - *"What's Multiple Negatives Ranking loss?"* → in-batch negatives: each anchor's positive is the target, all other positives in the batch are negatives; bigger batch = more negatives = better, no mining needed. `(certain)`
  - *"Why is naïve BERT bad at sentence similarity?"* → token-level; pooling isn't trained to be meaningful (worse than GloVe); cross-encoding every pair is O(n²). SBERT's Siamese NLI training fixes pooling. `(certain)`
  - *"Symmetric vs asymmetric search?"* → STS (query≈doc) vs QA retrieval (short query ↔ long passage); pick the matching model / use query/passage prefixes. `(likely)`
  - *"Do you need to re-embed when you change the model?"* → yes, always — different models are incompatible spaces; full re-index, blue-green. `(certain)`
  - *"Hard negatives?"* → plausible-but-wrong docs (BM25 top non-answers); the biggest quality lever after data volume. `(likely)`

---

## 12. Alternatives & How to Choose

Dense embeddings aren't the only way to represent text for retrieval:

```
Approach          What it is                          Strength                 Weakness
──────────────────────────────────────────────────────────────────────────────────────────
Sparse / lexical  BM25, TF-IDF — vector over the      exact terms, rare words, no semantics; blind
   (BM25)         VOCABULARY, mostly zeros            IDs/codes/names; no training  to paraphrase
Learned sparse    SPLADE — neural, but sparse over    lexical + some semantics; keeps  heavier index
   (SPLADE)       vocab with learned term weights     interpretable term matches
Dense (this note) bi-encoder embedding, cosine/dot    paraphrase, meaning, cross-      blind to exact
                                                       lingual; scales via ANN          unseen tokens
Late interaction  ColBERT — per-TOKEN embeddings +    near cross-encoder quality at    big index (many
   (ColBERT)      MaxSim over tokens                  much lower cost; middle ground   vectors per doc)
Cross-encoder     joint [q⊕d] scoring                 most accurate ranking            can't scale — rerank only
```

**How to choose:**
- **Default RAG retriever:** dense bi-encoder + cross-encoder rerank (§4). Add **BM25 in a hybrid** (merge with RRF — [RAG §5](RAG.md#5-stage-2--retrieval--generation-online)) when exact terms matter (codes, names, acronyms) — hybrid beats either alone.
- **Tight latency/quality budget in the middle:** consider **ColBERT** (late interaction) instead of a full cross-encoder.
- **Don't fine-tune when:** a strong API/general model already hits your Recall@K on a general corpus, or you lack labelled in-domain pairs. **Do fine-tune when:** generic embeddings under-retrieve on your jargon *and* you have (or can mine) `(query, relevant)` pairs — start with **MNR**, add **hard negatives**, prove it with §7.
- **Sensitive data / must own weights:** open-source local model (§5), fine-tuned in-house.

🎯 **Kill-shot:** *"Sparse matches words, dense matches meaning, hybrid does both, ColBERT and cross-encoders buy accuracy for compute — the production default is dense retrieve + cross-encode rerank, with BM25 hybrid when exact tokens matter."*

---

## 🧠 Self-Test

1. Why can `dot product` and `cosine similarity` rank the *same* two candidates in opposite orders, and which do you default to for semantic search?
   <details><summary>answer</summary> Dot product rewards magnitude *and* direction; cosine rewards direction only. A longer vector can win on dot product while being at a wider *angle* — e.g. `A=[1,1]`: dot picks `[800,0]` (800 > 401) but cosine picks `[400,1]` (0.7089 > 0.7071). Default to **cosine** for semantic search (meaning = direction), and note it *equals* dot product once vectors are L2-normalized. </details>

2. What exactly was wrong with using vanilla BERT for sentence similarity, and how did SBERT fix it?
   <details><summary>answer</summary> BERT is token-level. Cross-encoding every pair is accurate but O(n²) (~65h for 10k sentences); naïvely mean-pooling its tokens is fast but gives embeddings *worse than averaged GloVe*. SBERT fine-tunes a **Siamese** BERT (shared weights) on **NLI pairs** with a similarity objective so mean-pooling becomes meaningful — encode 10k once (~5s), then cosine-compare (~0.01s). </details>

3. Contrast bi-encoder and cross-encoder on: what they take in, whether doc reps precompute, and their role in a retriever.
   <details><summary>answer</summary> Bi-encoder encodes query and doc **separately** → two vectors → cosine; doc vectors **precompute** at index time → ANN over millions → the **retriever** (recall). Cross-encoder takes `[CLS] q [SEP] d` **together** → one relevance score; **can't precompute** (needs the pair) → one forward pass per candidate → the **reranker** (precision) on the shortlist. Stage them: retrieve top-N, rerank to top-k. </details>

4. You have 500 labelled `(query, relevant-doc)` pairs and want a better retriever. Which loss, and what makes the negatives?
   <details><summary>answer</summary> **Multiple Negatives Ranking (MNR)** loss on the bi-encoder: supply only `(anchor, positive)`; every **other positive in the batch** becomes an in-batch negative for this anchor, so bigger batches give more negatives — no explicit mining required. Add **hard negatives** (e.g. BM25 top non-answers) to push quality further. Fine-tune the cross-encoder separately with **BCE** on relevant(1)/not(0). </details>

5. Your retriever must feed a cross-encoder that reranks the top 50. Which single metric do you optimize the bi-encoder for, and why not Precision@5?
   <details><summary>answer</summary> **Recall@50** — the bi-encoder's only job is to get every relevant doc *into* the 50-candidate shortlist; anything it misses, the reranker can never recover. Precision@5 measures the *final* user-facing list (a reranking metric), not whether the retriever's net was wide enough. </details>

6. You upgrade your embedding model from MiniLM (384-d) to a 768-d model. What must you do before queries work, and what breaks if you skip it?
   <details><summary>answer</summary> **Re-embed the entire corpus** and rebuild the index — embeddings from different models live in incompatible spaces and different dimensions, so old and new vectors aren't comparable. Skip it and cosine scores are meaningless (mixed spaces / shape mismatch). Do it blue-green: build the new index, validate Recall@K/MRR/NDCG on a golden query set, then cut over; tag vectors with `model@version`. </details>
