# Text Classification with Neural Networks — From Raw Text to a Trained Classifier

> **TL;DR.** A neural net eats numbers, not words, so text needs a pipeline: **tokenize** (word → integer id) → **pad/truncate** to a fixed length → an **Embedding layer** that maps each id to a *learned* dense vector (semantic meaning, not one-hot). That gives a `(seq_len × embed_dim)` matrix per document; you then **collapse it to one vector** — `Flatten` (keeps everything but explodes parameters) or **GlobalAveragePooling/MaxPooling** (tiny, regularizing, the modern default) — and finish with `Dense → Dropout → sigmoid`. Everything downstream (Dense, ReLU, backprop, Adam) is ordinary [NN Fundamentals](Neural%20Network%20Fundamentals.md); the *only* text-specific pieces are the **vectorization pipeline, the Embedding layer, and the pooling choice**. Worked example: an **AI-vs-human text detector**.

**Where it fits:** the applied bridge from [neural networks](Neural%20Network%20Fundamentals.md) into **NLP** — how you actually get text *into* a network. Deeper theory lives in the sequence-model note and the future NLP deep-dives.
**Prereqs:** [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md) (Dense, ReLU, sigmoid, backprop), [Weight Initialization & Optimizers](Weight%20Initialization%20&%20Optimizers.md) (Adam), [Logistic Regression](../Supervised%20ML/Logistic%20Regression.md) (binary output). Deeper token/vector theory → [[Text Preprocessing]], [[Word Embeddings]]; sequence models → [RNN · LSTM · Transformers](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [From Text to Tensors](#2-from-text-to-tensors)
3. [The Embedding Layer](#3-the-embedding-layer)
4. [Collapsing the Sequence: Flatten vs Pooling](#4-collapsing-the-sequence-flatten-vs-pooling)
5. [Building & Training the Classifier](#5-building--training-the-classifier)
6. [Code / Implementation](#6-code--implementation)
7. [When It Breaks](#7-when-it-breaks)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Three problems stand between raw text and a network, each solved by one pipeline stage:
```
"AI writes fluently"   ── tokenize ──▶  [42, 8, 315]        (words → ids; but ids are meaningless magnitudes)
     variable length   ── pad/trunc ─▶  [42, 8, 315, 0, 0]  (every doc same length → fits a fixed tensor)
   ids carry no meaning ── embed ────▶  128-D vector per id  (learned meaning: "fluently"≈"eloquently")
                                        └─ (seq_len × 128) matrix ─┐
                        collapse to one vector (flatten / pool) ───┴─▶ Dense → Dropout → sigmoid → P(AI)
```
- 🎯 **The whole trick of "NN for text" is turning a variable-length string into a fixed-size tensor of *meaningful* numbers.** Tokenization+padding fixes the shape; the Embedding layer supplies the meaning; pooling reduces it to something a Dense head can classify. `(certain)`
- Word **id 315 is not "bigger" than id 8** — ids are arbitrary labels. Feeding ids straight to a Dense layer would (wrongly) treat them as ordered magnitudes; the Embedding layer is what replaces that nonsense with a learned vector. `(certain)`

---

## 2. From Text to Tensors

Three deterministic preprocessing steps (Keras `Tokenizer` + `pad_sequences`): `(certain)`

1. **Tokenization** — build a vocabulary mapping each word to an integer id, ranked by frequency. `Tokenizer(num_words=10000)` keeps only the **10k most frequent** words (the long tail is noise and blows up the embedding matrix).
2. **OOV (out-of-vocabulary)** — any word not in the vocab (rare, or unseen at inference) maps to a reserved `<OOV>` id via `oov_token='<OOV>'`. Without it, unknown words are silently dropped. `(certain)`
3. **Sequences → padding/truncating** — `texts_to_sequences` turns each doc into a list of ids (order preserved); `pad_sequences(..., maxlen=L, padding='post', truncating='post')` forces every doc to length `L` by **padding short docs with 0** and **cutting long ones**. A fixed `L` is what lets you stack docs into one `(batch × L)` tensor.

```
"AI writes fluently"        → [42, 8, 315]           → pad L=5 → [42, 8, 315, 0, 0]
"humans write with nuance"  → [5, 22, 91, 7, 640]    → (already 5) [5, 22, 91, 7, 640]
```
- **Choosing `L`:** too short → you truncate signal; too long → wasted compute and mostly-padding rows. Pick around the **95th-percentile document length**, not the max. `(likely)`
- ⚠️ **Fit the tokenizer on the *training set only*.** Fitting it on all data (incl. test) leaks vocabulary/frequency info — a subtle but real train/test leak. `(certain)`

---

## 3. The Embedding Layer

An **Embedding layer is a trainable lookup table**: a matrix `E` of shape `(vocab_size × embed_dim)`. Token id `i` simply selects **row `i`** — `embedding = E[i]`. `(certain)`

```
layers.Embedding(input_dim=10000,   # vocab size → number of ROWS
                 output_dim=128,     # embedding dim → number of COLUMNS
                 input_length=L)      # (optional) sequence length
# Parameters = 10000 × 128 = 1,280,000  — all LEARNED by backprop
# Input:  (batch, L) integer ids   →   Output: (batch, L, 128) float vectors
```

- 🎯 **It's not a fixed encoding — the embedding matrix is *weights*, trained end-to-end with the rest of the network, so words that help the task drift into useful positions** (e.g. "verbose", "furthermore", "moreover" cluster if they signal AI text). `(certain)`
- **Why not one-hot?** One-hot gives each of 10k words a 10,000-D sparse vector: enormous, and *equidistant* — it encodes zero similarity ("love" and "like" are as far apart as "love" and "banana"). Embeddings are dense (128-D), and similar words end up close (cosine-similar). `(certain)`
- **Pretrained embeddings** (GloVe/word2vec/fastText) can initialize `E` and be frozen or fine-tuned — a big win on **small datasets** where you can't learn good vectors from scratch. This is transfer learning for words. Deeper treatment: [[Word Embeddings]]. `(likely)`
- These learned vectors are the same kind of object you get from an [autoencoder bottleneck](Autoencoders.md) or a recommender, and you can visualize them with PCA / t-SNE.

---

## 4. Collapsing the Sequence: Flatten vs Pooling

The Embedding output is `(L × embed_dim)` per doc, but a `Dense` head needs a **single vector**. How you collapse the sequence axis is the model's most consequential architecture choice. `(certain)`

| Option | `(L=200, dim=128)` → | Params into next `Dense(64)` | Trade-off |
|---|---|---|---|
| **`Flatten`** | `25,600` | `25,600×64 ≈ 1.64M` | keeps *all* info + positions, but **parameter explosion** → overfits, slow |
| **`GlobalAveragePooling1D`** | `128` (mean over words) | `128×64 ≈ 8.2K` | ~**198× fewer params**, strong regularizer; loses word order |
| **`GlobalMaxPooling1D`** | `128` (max over words) | `128×64 ≈ 8.2K` | captures the *strongest* signal per feature; loses order & averages |

```
GlobalAveragePooling1D:  out[j] = mean over the L words of embedding[:, j]   # "average meaning of the doc"
GlobalMaxPooling1D:       out[j] = max  over the L words of embedding[:, j]   # "most salient feature"
```
- 🎯 **Flatten preserves everything but multiplies parameters by `L`; global pooling throws away word order to get a tiny, regularized, position-invariant vector — for a bag-of-words-ish task like AI-vs-human detection, pooling usually generalizes better.** `(likely)`
- Pooling layers have **zero parameters** — they're pure reductions. `Flatten` also has zero params itself; the cost is in the *next* Dense layer it feeds. `(certain)`
- **If word order matters** (negation, syntax), don't pool a plain embedding — use a **sequence model** (LSTM/GRU/Transformer, see [RNN · LSTM · Transformers](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)) or **`Conv1D`** to capture local n-gram patterns before pooling. `(certain)`

---

## 5. Building & Training the Classifier

The AI-vs-human dataset: a text column `text_content` and a binary `label` (`0=Human, 1=AI`), split **stratified 80/20** (stratify to preserve class balance). Binary target ⇒ **1 sigmoid output + `binary_crossentropy`**. `(certain)`

**Model A — the naive baseline (Flatten):**
```
Embedding(10000, 128) → Flatten → Dense(64, relu) → Dense(1, sigmoid)
```
Works, but the `Flatten` produces 25,600 features → a 1.6M-param Dense layer that overfits fast.

**Model B — the better default (pooling + dropout):**
```
Embedding(10000,128) → GlobalAveragePooling1D
   → Dense(128, relu) → Dropout(0.5)
   → Dense(64,  relu) → Dropout(0.3)
   → Dense(1, sigmoid)
```
- **[Dropout](Batch%20Normalization%20&%20Dropout.md)** randomly zeros a fraction of activations *each training step* (0.5 = half), forcing redundancy so no single unit dominates → less overfitting. It's **on during training, off at inference** (Keras handles this automatically). Higher rate nearer the input where overfitting risk is largest. `(certain)`
- **[Batch Normalization](Batch%20Normalization%20&%20Dropout.md)** is the complementary knob some text models add (`Dense → BatchNorm → Activation → Dropout`): it normalizes activations to stabilize and speed up training, rather than regularize. Both come from the same tutorial — see the dedicated note for the mechanics, ordering, and train/inference switch. `(certain)`
- Compile with **`adam`** ([why](Weight%20Initialization%20&%20Optimizers.md)) + `binary_crossentropy`, monitor **validation loss** for early stopping. Everything here — the Dense math, ReLU, the Adam updates — is the ordinary machinery from the fundamentals notes; text only changed the *front end*.

---

## 6. Code / Implementation

**End-to-end Keras pipeline:**
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers

MAX_WORDS, MAX_LEN, EMB = 10000, 200, 128

# --- text → tensor (FIT ON TRAIN ONLY) ---
tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tok.fit_on_texts(X_train_text)                      # builds the vocab from training text only
Xtr = pad_sequences(tok.texts_to_sequences(X_train_text), maxlen=MAX_LEN,
                    padding="post", truncating="post")
Xte = pad_sequences(tok.texts_to_sequences(X_test_text),  maxlen=MAX_LEN,
                    padding="post", truncating="post")   # same tokenizer, no re-fit

# --- model: pooling + dropout head ---
model = models.Sequential([
    layers.Embedding(MAX_WORDS, EMB, input_length=MAX_LEN),   # 1.28M learned params
    layers.GlobalAveragePooling1D(),                          # (L,128) → (128,)  ~198× fewer params than Flatten
    layers.Dense(128, activation="relu"), layers.Dropout(0.5),
    layers.Dense(64,  activation="relu"), layers.Dropout(0.3),
    layers.Dense(1,   activation="sigmoid"),                  # P(AI)
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(Xtr, y_train, validation_split=0.1, epochs=15, batch_size=32)

# --- inference: SAME tokenizer + SAME maxlen ---
def predict_text(text):
    seq = pad_sequences(tok.texts_to_sequences([text]), maxlen=MAX_LEN,
                        padding="post", truncating="post")
    p = float(model.predict(seq)[0, 0])
    return ("AI" if p >= 0.5 else "Human"), p        # threshold is tunable (§8)
```

---

## 7. When It Breaks

- **Tokenizer not saved with the model** → 🎯 *the single most common text-serving bug.* The model's integer ids are meaningless without the exact vocab that produced them; ship the `tok` object (pickle) alongside the weights and reuse it at inference. `(certain)`
- **Fitting the tokenizer on all data** → vocabulary/frequency leakage from test into train. Fit on train only.
- **Wrong/mismatched `maxlen` at inference** → shapes differ from training; always pad inference text to the *same* `MAX_LEN`.
- **Heavy OOV / vocabulary drift** — new slang, new domain, or a too-small `num_words` sends many words to `<OOV>` and accuracy silently rots. Monitor OOV rate. `(certain)`
- **`Flatten` on long sequences** → millions of params → overfitting + slow training + memory blowup. Prefer pooling or a sequence model.
- **Class imbalance** (e.g. mostly-human corpus) — accuracy misleads; watch precision/recall/F1/AUC and consider class weights (see [Classification Metrics](../Supervised%20ML/Classification%20Metrics.md)).
- **Padding treated as signal** — with mean-pooling, lots of `0` padding dilutes the average; use `mask_zero=True` on the Embedding (or a masking-aware layer) so pads are ignored. `(likely)`
- **Losing word order** — pooling a plain embedding is bag-of-words: it can't tell "not good" from "good not". If order matters, add Conv1D/LSTM/attention.

---

## 8. Production & MLOps Notes

- **Version the tokenizer *with* the model as one artifact.** Vocab is part of the model; a weights-only deploy is broken. Track `num_words`, `MAX_LEN`, and OOV token as config.
- **Threshold, don't just argmax.** Sigmoid outputs a probability; the 0.5 cut is a business choice. For a detector, pick the threshold from the precision/recall trade-off you need (e.g. high precision to avoid falsely flagging humans), and **calibrate** if you report probabilities. `(likely)`
- **Monitoring & drift:** track **OOV rate**, input length distribution, and score distribution over time — language drifts (new AI models write differently), so an AI-vs-human detector decays and needs periodic retraining. `(certain)`
- **Cost:** the Embedding matrix dominates parameters (vocab×dim); shrink `num_words`/`embed_dim`, or use pretrained + frozen embeddings to cut trainable params.
- **The real upgrade path** is not a fancier Dense head — it's a **better text representation**: pretrained embeddings → a sequence model (LSTM/GRU) → a **fine-tuned transformer (BERT)**, which sets the SOTA on text classification by contextualizing each token. That's the top of the ladder in §10. See [[BERT]]. `(likely)`
- **Baseline first:** a **TF-IDF + [Logistic Regression](../Supervised%20ML/Logistic%20Regression.md)** model trains in seconds and is often within a few points of an embedding-NN on medium data — always ship it as the yardstick before reaching for deep models.

---

## 9. Interview Lens

**"How do you feed text into a neural network?"** → 🎯 *"Tokenize words to integer ids, pad/truncate to a fixed length, then an Embedding layer maps each id to a learned dense vector. You collapse the resulting (length × dim) matrix to one vector with Flatten or global pooling and finish with Dense→sigmoid. The only text-specific parts are tokenization, the embedding, and the pooling."*

**"Why an Embedding layer instead of one-hot?"** → 🎯 *"One-hot is huge (vocab-sized) and equidistant — it encodes no similarity. An embedding is a small dense trainable lookup table, so similar words learn similar vectors and the representation is far more efficient and meaningful."*

**Likely follow-ups:**
- *Flatten vs GlobalAveragePooling for text?* → Flatten keeps all info + order but multiplies params by sequence length (overfits); pooling gives a tiny, position-invariant, regularized vector — usually better for bag-of-words tasks.
- *What does `num_words` / OOV do?* → Caps the vocab to the top-k frequent words; everything else (incl. unseen words at inference) maps to a reserved `<OOV>` id so it isn't dropped.
- *Are embeddings trained or fixed?* → Trained end-to-end by default; you can also initialize from pretrained GloVe/word2vec and freeze or fine-tune.
- *How is padding handled so it doesn't corrupt pooling?* → `mask_zero=True` / masking so padded positions are ignored.
- *Biggest deployment gotcha?* → Forgetting to save and reuse the exact tokenizer; ids are meaningless without it.
- *Order-sensitive text?* → Plain embedding+pooling is bag-of-words; use Conv1D/LSTM/Transformer to model order.

---

## 10. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Baseline / small data / need speed | **TF-IDF + [Logistic Regression](../Supervised%20ML/Logistic%20Regression.md)** | seconds to train, strong yardstick, interpretable |
| Medium data, order not critical | **Embedding + Global Pooling + Dense** (this note) | cheap, regularized, learns task-specific word vectors |
| Local phrase/n-gram patterns matter | **Embedding + `Conv1D` + pooling** | detects local patterns cheaply; some order sensitivity |
| Word order / long-range dependencies | **LSTM / GRU** ([sequence models](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)) | models sequence, not bag-of-words |
| Best accuracy, enough compute | **Fine-tuned Transformer (BERT)** | contextual embeddings → SOTA text classification |

**Decision rule:** always start with **TF-IDF + LogReg** as the baseline. Move to an **embedding + pooling** net when you have enough data to learn vectors and want a task-specific representation. Escalate to **Conv1D/LSTM** only if word **order** carries the signal, and to a **fine-tuned transformer** when accuracy justifies the cost. Don't add depth to the Dense head to fix accuracy — upgrade the *representation*.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Walk through the pipeline that turns a raw sentence into something a Dense layer can classify.
   <details><summary>answer</summary>Tokenize (word→id) → pad/truncate to fixed length `L` → Embedding layer (id→learned dense vector) giving `(L×dim)` → collapse to one vector via Flatten or global pooling → Dense→Dropout→sigmoid.</details>
2. Why is an Embedding layer better than one-hot encoding?
   <details><summary>answer</summary>One-hot vectors are vocab-sized (e.g. 10,000-D), sparse, and equidistant (no similarity). Embeddings are small dense (e.g. 128-D) trainable vectors where similar words end up close, so they're efficient and carry meaning.</details>
3. What's the parameter trade-off between `Flatten` and `GlobalAveragePooling1D`?
   <details><summary>answer</summary>Flatten yields `L×dim` features (e.g. 200×128=25,600) → the next Dense layer has ~L× more params (overfits, slow) but keeps all info+order. Global pooling averages the sequence to `dim` (128) features → ~198× fewer params, a regularizer, but loses word order.</details>
4. What is `oov_token` for, and why cap `num_words`?
   <details><summary>answer</summary>`oov_token` gives unseen/rare words a reserved id so they aren't silently dropped at inference. `num_words` keeps only the top-k frequent words to bound the embedding matrix size and drop noisy rare words.</details>
5. Name two ways this pipeline leaks or breaks in production.
   <details><summary>answer</summary>(1) Fitting the tokenizer on all data (test vocab leaks into train). (2) Not saving/reusing the exact tokenizer at inference — the model's ids are meaningless without it. Also: mismatched `maxlen`, rising OOV rate.</details>
6. When should you NOT use embedding + global pooling, and what do you use instead?
   <details><summary>answer</summary>When word **order** matters (negation, syntax) — pooling is bag-of-words. Use Conv1D (local n-grams), an LSTM/GRU, or a fine-tuned Transformer (BERT) that models sequence/context.</details>

---

*Covers: the text→tensor pipeline (tokenization, `num_words`, `<OOV>`, sequences, padding/truncating, choosing `L`, train-only tokenizer fit); the Embedding layer as a trainable lookup table (`vocab×dim` params, learned end-to-end, vs one-hot, pretrained/transfer); collapsing the sequence (Flatten parameter-explosion vs GlobalAverage/MaxPooling, masking); building the AI-vs-human classifier (sigmoid + BCE, Dropout, Adam); production (save the tokenizer, threshold/calibration, OOV & score drift, TF-IDF baseline, upgrade path to Conv1D/LSTM/BERT). Duplicated fundamentals (Dense, ReLU, gradient descent, optimizers) are cross-referenced, not repeated. Sourced from the "AI vs Human text detection" tutorial notebook, distilled to the text-specific concepts. Foundations: [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md), [Weight Initialization & Optimizers](Weight%20Initialization%20&%20Optimizers.md).*
