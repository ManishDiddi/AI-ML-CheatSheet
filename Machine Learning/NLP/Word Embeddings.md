# Word Embeddings — Dense Vectors that Capture Word Meaning

> **TL;DR.** One-hot / bag-of-words vectors are huge, sparse, and **orthogonal** — every pair of distinct words is equidistant, so the representation knows nothing about meaning. **Word embeddings** map each word to a short **dense** vector (~100–300 dims) where geometric closeness = semantic/syntactic similarity, learned from *the company a word keeps* (the **distributional hypothesis**). Two families: **count-based** (co-occurrence matrix + SVD) and **prediction-based** (**Word2Vec** — CBoW predicts a word from its context, Skip-gram predicts context from a word; trained cheaply with **negative sampling**). The payoff is transferable features and semantic algebra (`king − man + woman ≈ queen`). The catch: **one static vector per word**, so "bank" (river) and "bank" (money) collapse into one point — exactly what **contextual** embeddings ([BERT](BERT.md)) fix.

**Where it fits:** the **representation** stage of NLP — it turns the clean tokens from [Text Preprocessing](Text%20Preprocessing.md) into vectors that models can reason over. Static word vectors are the ancestor of the sentence/contextual embeddings in [Embeddings](../../AI%20Engineering/Embeddings.md) and [BERT](BERT.md), and the input layer of every classic text neural net ([Text Classification with Neural Networks](../Neural%20Networks/Text%20Classification%20with%20Neural%20Networks.md), [RNN · LSTM · Transformers](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)).
**Prereqs:** [Text Preprocessing](Text%20Preprocessing.md) (tokenization, vocabulary), one-hot/BoW & TF-IDF, softmax + cross-entropy, basic linear algebra (dot products, SVD), cosine similarity.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [The Embedding Zoo — GloVe, fastText, Contextual](#7-the-embedding-zoo--glove-fasttext-contextual)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**Distributional hypothesis** — *"You shall know a word by the company it keeps"* (Firth). Words that appear in similar contexts (`coffee` and `tea` both near *drink, cup, hot, morning*) mean similar things, so their vectors should sit close together. Embeddings are just a way to **place every word in a shared space so that neighbors are semantically related**.

```
Discrete (one-hot)                     Distributed (embedding)
 cat  = [0,0,1,0,…,0]  |V|-dim          cat   = [ 0.21, -0.9,  0.4, …]  d≈300
 dog  = [0,1,0,0,…,0]                   dog   = [ 0.19, -0.8,  0.5, …]  ← near cat
 king = [1,0,0,0,…,0]                   king  = [ 0.7,   0.1, -0.3, …]
   • every pair orthogonal               • distance encodes meaning
   • cosine(cat,dog)=0 (no similarity)   • cosine(cat,dog) high
   • dimension = vocab (10⁴–10⁶)         • dimension = small & dense
```

The famous proof that this space is *structured*, not just clustered: **vector arithmetic** works.
```
vec("king") − vec("man") + vec("woman") ≈ vec("queen")
vec("Paris") − vec("France") + vec("Italy") ≈ vec("Rome")
```
The offset "man→woman" is roughly the same direction as "king→queen" — gender is a *direction* in the space. That linear structure is why embeddings became the foundation of modern NLP. 🎯

---

## 2. The Formal Core

**The problem with one-hot / BoW.** A word is a basis vector in `ℝ^|V|`. Any two distinct words have dot product 0 → cosine 0 → the model must learn every word's behavior independently, with no sharing. Embeddings fix this by learning a low-rank map `E ∈ ℝ^{|V|×d}` (`d ≪ |V|`); row `E[w]` is word `w`'s vector.

### Count-based (SVD) methods
Build a **co-occurrence matrix** `X` (`X_ij` = how often word *i* co-occurs with word *j* within a window — an *affinity matrix*; or a word–document matrix). Then factor with **SVD**: `X = U Σ Vᵀ`, with `UᵀU = VᵀV = I`. Keep the top-`k` singular directions and use `U[:, :k]` (scaled by `Σ`) as the embeddings — `k` chosen to capture a target fraction of variance (`Σ`²). Dense, semantic — but SVD is **O(|V|²·k)+**, the matrix is huge and sparse, and adding new words means **recomputing from scratch**. This is why prediction-based methods took over.

### Prediction-based: Word2Vec
A **shallow, log-bilinear model** with two weight matrices: input/embedding `W ∈ ℝ^{|V|×d}` and output/context `W' ∈ ℝ^{d×|V|}`. Crucially the projection is **linear — there is no hidden nonlinearity** (a lookup + linear layer + softmax). That is *why the learned `W` rows can be read off directly as the embeddings*. `(certain)` *(Some course slides describe a ReLU hidden layer; the canonical Word2Vec uses a linear projection.)*

- **CBoW** — predict the center word from the averaged context:
  `P(w_center | context) = softmax(Wᵀ · avg(one-hots of context))`. Objective: minimize cross-entropy `J = −Σ_k y_k log ŷ_k`.
- **Skip-gram** — predict each context word from the center word `c`:
  `P(o | c) = exp(u_o · v_c) / Σ_{w∈V} exp(u_w · v_c)` where `v_c` = input (center) vector, `u_o` = output (context) vector. Maximize the log-likelihood over the window: `Σ_t Σ_{−m ≤ j ≤ m, j≠0} log P(w_{t+j} | w_t)`.

**The softmax denominator is the killer** — it sums over the entire vocabulary every step. Two fixes:

- **Negative sampling (SGNS)** — replace the `|V|`-way softmax with `k` small **binary** problems: push the true (center, context) pair together, push `k` random "noise" words apart.
  `log σ(u_o · v_c) + Σ_{i=1..k} E_{w_i∼P_n}[ log σ(−u_{w_i} · v_c) ]`
  `k ≈ 5–20` for small data, `2–5` for large; noise drawn from the **unigram distribution raised to 3/4** (`P_n(w) ∝ freq(w)^{0.75}`, which up-samples rare words). `(certain)`
- **Hierarchical softmax** — arrange the vocabulary as a **Huffman binary tree** with words at the leaves; a word's probability is the product of left/right decisions along the root→leaf path → cost drops from `O(|V|)` to `O(log|V|)`.

**Cosine similarity** is the metric of the space: `cos(a,b) = (a·b)/(‖a‖‖b‖)`. Use **cosine, not Euclidean/dot**, because a vector's *magnitude* tracks word frequency while its *direction* carries meaning — deep dive in [Embeddings §2](../../AI%20Engineering/Embeddings.md). `(certain)`

---

## 3. How It Works

Skip-gram-with-negative-sampling, end to end:

1. **Build vocabulary** from the corpus (drop words below `min_count`). Optionally **subsample** very frequent words (the/of/is) with probability tied to their frequency — denoises and speeds training.
2. **Slide a context window** of size `m` over each sentence; each center word + a neighbor forms a positive `(center, context)` training pair.
3. **Look up** the center word's row in `W` (`v_c`) — this *is* the "forward pass" of the linear projection.
4. **Score** against the true context word `u_o` and `k` sampled negatives; compute the negative-sampling loss.
5. **Backprop** and update **both** `W` and `W'` by gradient descent. Repeat over many epochs.
6. **After training, keep `W`** as the embedding matrix; `W'` is usually discarded (or the two are averaged).

**CBoW vs Skip-gram — the trade-off to memorize:**

| | CBoW | Skip-gram |
|---|---|---|
| Predicts | center **from** context | context **from** center |
| Speed | faster (context averaged) | slower (one prediction per context word) |
| Rare words | weak (averaging drowns them) | **strong** — each pair is a training signal |
| Frequent words | can **overfit** them | handles better |
| Best when | large corpus, frequent words matter | smaller corpus, rare words matter |

🎯 *"CBoW smooths over context and is fast; Skip-gram makes a separate prediction for every context word, so it learns rare words better — at higher compute. Skip-gram + negative sampling is the usual default."*

---

## 4. Worked Example

**Count matrix (SVD path).** Corpus, window = 1:
`"I enjoy flying." · "I like NLP." · "I like deep learning."`

Counting how often each word sits adjacent to each other word gives a symmetric co-occurrence matrix, e.g. `I` co-occurs with `enjoy` (1) and `like` (2); `like` with `I` (2), `NLP` (1), `deep` (1); etc. Run SVD on that matrix, keep the top 2 columns of `U`, and you can plot each word in 2-D — `like`/`enjoy` land near each other because they share the neighbor `I`. That's a semantic space from pure counts.

**Training-pair generation (Skip-gram).** Sentence `"I like deep learning"`, window `m=1`, center = `deep`:
```
(deep, like)      (deep, learning)     ← positive pairs
(deep, "cat"), (deep, "the")           ← k negative samples (random noise words)
```
The model nudges `deep`'s vector toward `like`/`learning` and away from the noise words. Repeat across the corpus and `deep`/`learning`/`machine` drift together.

**Analogy check** (what you'd run after training): `king − man + woman` returns a vector whose nearest neighbor (by cosine) is `queen` — the payoff of the linear structure in §1.

---

## 5. Code / Implementation

**Train your own with gensim** (the domain-corpus case — e.g. the COVID-19 research-paper search engine from the lecture):
```python
from gensim.models import Word2Vec
# corpus = list of tokenized sentences: [["machine","learning",...], ...]
model = Word2Vec(corpus, vector_size=100, window=5, min_count=3,
                 sg=1, negative=5, workers=4)   # sg=1 → Skip-gram; negative=5 → NS
# NOTE: gensim ≥4.0 renamed size→vector_size and vectors live under .wv
vec        = model.wv["virus"]                  # the learned 100-d embedding
neighbors  = model.wv.most_similar("virus", topn=5)
analogy    = model.wv.most_similar(positive=["king","woman"], negative=["man"])  # → queen
```

**Use pretrained vectors** (the general-domain case — don't train from scratch on small data):
```python
import gensim.downloader as api
wv = api.load("glove-wiki-gigaword-100")        # or "word2vec-google-news-300", "fasttext-wiki-news-subwords-300"
wv.similarity("coffee", "tea")                  # ~0.7
```

**Document/query vectors via averaging** (the lecture's search engine — simple, and note its weakness in §6):
```python
import numpy as np
def doc_vector(tokens, wv, dim=100):
    vecs = [wv[w] for w in tokens if w in wv]   # skip OOV words
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)   # centroid of word vectors

def rank(query, docs_vecs, wv):                 # rank docs by cosine to the query centroid
    q = doc_vector(query.split(), wv)
    cos = lambda a,b: a@b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
    return sorted(docs_vecs, key=lambda d: cos(q, d["vec"]), reverse=True)
```

---

## 6. When It Breaks

- **Static — one vector per word (polysemy collapse).** `bank` (river vs money), `apple` (fruit vs company), `python` (snake vs language) all get a **single** averaged vector. Word2Vec/GloVe/fastText cannot disambiguate by sentence — this is the headline limitation and the reason **contextual embeddings** (ELMo → [BERT](BERT.md)) exist: they give a *different* vector per occurrence. 🎯
- **OOV — unseen words have no vector.** `model.wv["covid19"]` throws `KeyError` if it wasn't in training (see the `try/except` in the search-engine code). **fastText** fixes this by composing vectors from character n-grams (§7).
- **Averaging word vectors is a weak sentence representation.** The centroid throws away word order and lets stopwords/frequent words dominate; `"dog bites man"` and `"man bites dog"` get the *same* vector. For real sentence similarity, use TF-IDF-weighted averaging, SIF, or proper sentence encoders → [Embeddings](../../AI%20Engineering/Embeddings.md).
- **Distributional similarity ≠ synonymy.** Antonyms share contexts (`good`/`bad` both near *is very ___*), so they often come out **close** — embeddings capture "same slot," not "same meaning."
- **Window size changes what "similar" means.** Small window → syntactic/functional neighbors (`run`/`ran`/`walk`); large window → topical neighbors (`run`/`marathon`/`race`). Choose deliberately.
- **Social bias is baked in.** Trained on human text, embeddings reproduce it: `doctor − man + woman ≈ nurse`. A real fairness/production concern, not a curiosity.
- **Small corpora → noisy vectors.** Below tens of millions of tokens, prefer pretrained embeddings (or fine-tune them) over training from scratch.

---

## 7. The Embedding Zoo — GloVe, fastText, Contextual

- **GloVe (Global Vectors).** A **hybrid** of count- and prediction-based: it fits vectors so that `wᵢ · wⱼ ≈ log(co-occurrence count of i,j)`, training on the **global** co-occurrence matrix rather than local windows. Ratios of co-occurrence probabilities encode meaning. Often ties Word2Vec; pretrained GloVe (Wikipedia/Common Crawl) is a very common default.
- **fastText.** Represents each word as a **bag of character n-grams** (`where → <wh, whe, her, ere, re>`) and sums them. Two big wins: **no OOV** (compose any word from its n-grams) and **morphology awareness** (`run`/`running`/`runner` share subwords) — great for rare words and rich-morphology languages. This is the same subword idea that later powers transformer tokenizers.
- **The count↔prediction bridge.** Levy & Goldberg showed **Skip-gram with negative sampling implicitly factorizes a shifted PMI co-occurrence matrix** — so the "two families" are two views of the same thing. `(likely)` A crisp interview flex.
- **Static → contextual.** Word2Vec/GloVe/fastText are **static** (a lookup table). **ELMo** (biLSTM) and **[BERT](BERT.md)** (transformer) produce **contextual** vectors — the representation of `bank` depends on the whole sentence — which is why they dominate today. Word embeddings are the conceptual and historical foundation; [Embeddings](../../AI%20Engineering/Embeddings.md) covers the sentence-level, contextual successors.

---

## 8. Production & MLOps Notes

- **Train-your-own vs pretrained.** Domain-specific corpus with its own jargon (biomedical, legal, the COVID example) → train Word2Vec/fastText on it. General English → **load pretrained** GloVe/fastText/GoogleNews and skip training. Rule of thumb: pretrained unless your domain vocabulary is genuinely out-of-distribution.
- **As features in a downstream model.** Initialize a neural net's **embedding layer** with pretrained vectors, then **freeze** (small data — pure transfer) or **fine-tune** (enough data to specialize). This is transfer learning for words — see [Text Classification with Neural Networks](../Neural%20Networks/Text%20Classification%20with%20Neural%20Networks.md).
- **Storage & serving.** The matrix is `|V| × d` floats (a 400k×300 GloVe ≈ 480 MB). Persist with gensim `KeyedVectors` (`.kv`/`.bin`), memory-map for fast load, and ship only the `wv` (drop `W'` and the training state).
- **Document & query vectors.** Centroid averaging is the cheap baseline (and its weakness is real, §6). Better: **TF-IDF-weighted** average or **SIF** (smooth inverse frequency + removing the top principal component). Best: a real sentence encoder (SBERT). For search at scale, index doc vectors in an **ANN** store (FAISS) and retrieve by cosine — the classical ancestor of vector RAG ([Embeddings](../../AI%20Engineering/Embeddings.md)).
- **Evaluation.** *Intrinsic* — word-similarity correlation (WordSim-353, SimLex) and analogy accuracy (Google analogy set). *Extrinsic* — does swapping embeddings improve the actual downstream task? Extrinsic wins ties.
- **Drift & retraining.** New vocabulary (products, slang, `covid`) → rising OOV; retrain periodically or switch to a subword/contextual model. Pin the corpus + seed for reproducibility (Word2Vec is stochastic).
- **Visualization.** Project to 2-D with **[PCA & t-SNE](../Unsupervised%20ML/PCA%20&%20t-SNE.md)** (or UMAP) to sanity-check clusters — but remember t-SNE distances between clusters aren't meaningful.

---

## 9. Interview Lens

The question is really testing whether you understand **why dense beats sparse, and where static embeddings stop working**.

🎯 **Kill-shots:**
- *"One-hot vectors are orthogonal — every word equidistant, so meaning can't be shared. Embeddings learn a dense space from context (distributional hypothesis) where distance encodes similarity, which is why `king − man + woman ≈ queen` works."*
- *"Word2Vec is a linear log-bilinear model, not a deep net — that's why the input weight matrix rows *are* the embeddings."*
- *"Static embeddings give one vector per word, so polysemy collapses; contextual models (BERT) give a vector per occurrence."*

**Likely follow-ups:**
- *CBoW vs Skip-gram?* CBoW predicts center from context (fast, good for frequent words); Skip-gram predicts context from center (better for rare words, larger data). `(certain)`
- *Why negative sampling / hierarchical softmax?* Full softmax normalizes over all `|V|` words every step — too expensive. NS turns it into `k` binary classifications; hier-softmax uses a tree for `O(log|V|)`. `(certain)`
- *Count-based vs prediction-based?* SVD on a co-occurrence matrix vs iteratively learning to predict; SGNS provably ≈ factorizing a shifted-PMI matrix, so they're related. `(likely)`
- *Cosine vs Euclidean for embeddings?* Cosine — direction carries meaning, magnitude tracks frequency. `(certain)`
- *How do you handle OOV?* fastText subword composition, or an `<UNK>`/skip; ultimately subword/contextual models. `(certain)`
- *Why not average word vectors for a sentence?* Loses order, stopwords dominate; use weighted averaging or a sentence encoder. `(certain)`

---

## 10. Alternatives & How to Choose

| Representation | Type | Strength | Reach for it when |
|---|---|---|---|
| **One-hot / BoW / TF-IDF** | sparse, count | interpretable, no training, strong baseline | small data, linear models ([Naive Bayes](../Supervised%20ML/Naive%20Bayes.md), SVM), you need explainable features |
| **Word2Vec / GloVe** | static dense | semantic space, transferable, cheap | general word-level features, similarity, initializing embedding layers |
| **fastText** | static dense + subword | OOV-proof, morphology, rare words | noisy text, rich-morphology or low-resource languages |
| **Contextual (ELMo, [BERT](BERT.md))** | dynamic dense | per-sentence meaning, polysemy resolved, SOTA | you can afford a transformer and accuracy matters most |

**Decision rule:** need a baseline or interpretability → TF-IDF. Need cheap semantic word features / lots of OOV → GloVe or **fastText**. Need to resolve word sense and top accuracy → **contextual embeddings** ([Embeddings](../../AI%20Engineering/Embeddings.md), [BERT](BERT.md)). Historically the ladder runs sparse → static → contextual, each fixing the last one's blind spot.

---

## 🧠 Self-Test

1. Why can't one-hot vectors express that "cat" and "dog" are similar, and how do embeddings fix it?
   <details><summary>answer</summary>One-hot vectors are orthogonal basis vectors — any two distinct words have dot product 0 and cosine 0, so there's no notion of similarity and no parameter sharing. Embeddings learn a low-dimensional dense space (from the distributional hypothesis) where semantically related words sit close, enabling similarity and analogies.</details>

2. Contrast CBoW and Skip-gram, including which handles rare words better and why.
   <details><summary>answer</summary>CBoW predicts the center word from averaged context (fast, smooths, can overfit frequent words). Skip-gram predicts each context word from the center word, producing a separate training signal per context word — so it learns **rare** words better, at higher compute. Skip-gram + negative sampling is the usual default.</details>

3. What problem does negative sampling solve, and what is the objective?
   <details><summary>answer</summary>The full softmax normalizes over the entire vocabulary each step — prohibitively expensive. Negative sampling replaces it with `k` binary logistic problems: maximize `log σ(u_o·v_c)` for the true pair and `Σ log σ(−u_neg·v_c)` for `k` noise words sampled from `freq^{0.75}`. Hierarchical softmax is the tree-based alternative (`O(log|V|)`).</details>

4. A colleague says Word2Vec is a deep neural network with a ReLU hidden layer. Correct them.
   <details><summary>answer</summary>It's a **shallow, log-bilinear** model: an embedding lookup + a **linear** projection + softmax, with *no* hidden nonlinearity. That linearity is precisely why the input weight matrix's rows can be used directly as the word vectors.</details>

5. Why do "bank" (river) and "bank" (money) get the same vector, and what fixes it?
   <details><summary>answer</summary>Word2Vec/GloVe/fastText are **static** — one vector per word type, averaged over all senses (polysemy collapse). **Contextual** embeddings (ELMo, BERT) produce a different vector per occurrence based on the surrounding sentence, resolving word sense.</details>

6. Name two ways to turn word vectors into a document vector and a weakness of the simplest one.
   <details><summary>answer</summary>(1) **Centroid** — average the word vectors: simple but discards word order and lets stopwords/frequent words dominate (`"dog bites man"` == `"man bites dog"`). (2) **TF-IDF-weighted average / SIF**, or better a **sentence encoder (SBERT)**. Index with FAISS for retrieval at scale.</details>
