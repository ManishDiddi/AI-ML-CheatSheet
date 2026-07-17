# Autoencoders — Learning Compressed Representations by Reconstructing the Input

> **TL;DR.** An autoencoder is an hourglass-shaped network trained to **output its own input** (`x̂ ≈ x`). Because the signal must squeeze through a narrow **bottleneck** layer, the network is forced to learn a compact **latent representation (embedding)** that keeps only the information needed to rebuild `x`. The **encoder** compresses `x → z`, the **decoder** rebuilds `z → x̂`; you train end-to-end on reconstruction loss, then **throw the decoder away and keep the encoder** as a feature extractor. It's *self-supervised* (the label is the input), so it needs **no labels**. Uses: non-linear dimensionality reduction (a non-linear [PCA](../Unsupervised%20ML/PCA%20%26%20t-SNE.md)), **denoising**, feature extraction / transfer learning, embeddings for recommenders/search/clustering, and [anomaly detection](../Unsupervised%20ML/Anomaly%20Detection.md) via reconstruction error.

**Where it fits:** unsupervised **representation learning** — sits next to [PCA & t-SNE](../Unsupervised%20ML/PCA & t-SNE) but learns *non-linear* compressions with a neural net.
**Prereqs:** [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md) (dense layers, backprop, reconstruction via BCE/MSE), [PCA & t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md) (the linear baseline it generalizes).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [Autoencoders vs PCA](#3-autoencoders-vs-pca)
4. [How It Works](#4-how-it-works)
5. [Denoising Autoencoders](#5-denoising-autoencoders)
6. [Applications: Features, Transfer Learning, Recommenders](#6-applications-features-transfer-learning-recommenders)
7. [Worked Example](#7-worked-example)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks](#9-when-it-breaks)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose](#12-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Force a network to copy its input to its output — but make it pass through a layer far smaller than the input. To rebuild `x` from just a few numbers, those numbers **must** encode the input's essential structure. `(certain)`

```
   x (6-D)      ENCODER          bottleneck        DECODER        x̂ (6-D)
  ┌───────┐    ↓ compress          z (2-D)        ↑ rebuild     ┌───────┐
  │ ▓▓▓▓▓ │──▶ 6 → 4 → ──────────▶  ● ●  ──────────▶ → 4 → 6 ──▶│ ▓▓▓▓▓ │
  └───────┘                      "latent code"                  └───────┘
             the SQUEEZE is the point ──────▶  loss = how different x̂ is from x
```

- 🎯 **The reconstruction is a *pretext* — you don't care about `x̂`. You care that a good reconstruction proves the tiny bottleneck `z` captured all the information, so `z` is a learned compressed feature vector (embedding).** `(certain)`
- If you allowed **linear** activations and equal-width layers, the net could trivially learn the **identity** (copy in = out) and learn nothing. The **bottleneck** (fewer neurons than inputs) is what makes copying *impossible without compression*. `(certain)`
- It's **self-supervised**: no labels — the target *is* the input, so autoencoders learn from unlabeled data.

---

## 2. The Formal Core

An autoencoder is two functions trained jointly: `(certain)`
```
z  = f(x)     # ENCODER:  x ∈ ℝ^d  →  z ∈ ℝ^{d'}   (d' < d = "undercomplete")
x̂  = g(z)     # DECODER:  z ∈ ℝ^{d'} →  x̂ ∈ ℝ^d
minimize  L(x, x̂)   over encoder+decoder weights     # reconstruction loss
```
- **`z` = latent code / embedding / bottleneck / code layer.** Its dimension `d'` is the compression target.
- **Loss depends on the data type** (same choices as any output layer): `(certain)`
  - continuous values → **MSE / RMSE** (linear output)
  - values in [0,1] (e.g. normalized pixels) → **binary cross-entropy** (sigmoid output)
  - one-hot categorical → **categorical cross-entropy** (softmax output)
- **Undercomplete** (`d' < d`) forces compression — the standard case. **Overcomplete** (`d' ≥ d`) has room to cheat by learning the identity, so it *needs regularization* (denoising §5, or a sparsity penalty). `(certain)`
- **Encoder/decoder need not be symmetric.** Historically they mirrored each other and **tied weights** (`W_dec = W_encᵀ`) to halve parameters; modern practice uses independent weights and whatever shapes work. `(certain)`

---

## 3. Autoencoders vs PCA

The cleanest way to understand an autoencoder is as a **generalization of [PCA](../Unsupervised%20ML/PCA%20%26%20t-SNE.md)**: `(certain)`

- 🎯 **A single-hidden-layer autoencoder with linear activations and MSE loss learns the same subspace as PCA** (it spans the top-`d'` principal components). The non-linearity is what makes it more powerful. `(likely)`
- Add **non-linear activations and depth**, and the encoder learns a **curved (non-linear) manifold** that PCA — restricted to a linear projection — cannot represent. On data lying on a non-linear manifold (e.g. MNIST digits), a deep AE gives a much better low-D encoding than PCA. `(certain)`

| | PCA | Autoencoder |
|---|---|---|
| Mapping | linear projection | arbitrary non-linear (with non-linear activations) |
| Fit | closed-form (SVD/eigdecomposition) | gradient descent, iterative |
| Invertibility | exact linear inverse | learned decoder (approximate) |
| Cost / data | cheap, deterministic | needs more data + compute, non-convex |
| When it wins | linear structure, few samples | non-linear structure, lots of data |

---

## 4. How It Works

1. **Build** encoder layers stepping *down* to the bottleneck, then decoder layers stepping *back up* to the input size (last decoder activation matches the data: sigmoid for [0,1], linear for continuous). `(certain)`
2. **Train** end-to-end with `fit(X, X)` — inputs and targets are both `X` — minimizing reconstruction loss with [Adam](Weight%20Initialization%20%26%20Optimizers.md). No labels involved.
3. **Extract the encoder.** Once trained, build a new model whose output is the **bottleneck layer's activation** — that's your embedding producer. In Keras: `Model(ae.input, ae.layers[k].output)` where layer `k` is the bottleneck. `(certain)`
4. **Use `z`** for whatever you need — feed embeddings to a classifier, cluster them, compute cosine similarity, or visualize with [t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md). The decoder is usually discarded (kept only if you actually need reconstructions, e.g. denoising). `(certain)`

---

## 5. Denoising Autoencoders

The failure mode of a too-powerful AE: it learns the **identity function** (memorizes copy-through) instead of useful structure — especially when overcomplete. The fix is elegant: `(certain)`

```
train:   x̃ = x + noise   ──▶ encoder ─▶ z ─▶ decoder ──▶ x̂        loss = L(x, x̂)
                (corrupt the INPUT)                     (compare to CLEAN x)
```
- 🎯 **Feed a *corrupted* input but ask it to reconstruct the *clean* original.** The network can't win by copying (its input now contains random noise with no pattern), so it's forced to learn the underlying signal and, as a side effect, **removes noise** — the model learns to clean data. `(certain)`
- Acts as a **regularizer**: because noise is unpredictable, the AE learns robust features rather than a brittle identity map. Common noise: additive Gaussian (`x + 0.5·N(0,1)`, then clip to valid range), or masking (randomly zero inputs — "dropout on the input"). `(certain)`
- **Uses:** image/audio denoising, robust feature learning, and pre-training. (Related regularized variants: **sparse AEs** add an activation-sparsity penalty on `z`; **contractive AEs** penalize the encoder's sensitivity to input. Same goal — stop the identity shortcut.) `(likely)`

---

## 6. Applications: Features, Transfer Learning, Recommenders

**Feature extraction.** The bottleneck `z` is a dense feature vector. Raw pixels are useless to a [decision tree](../Supervised%20ML/Decision%20Trees.md) or logistic regression; AE embeddings often aren't. So: run the encoder to get `z`, then do classic ML/DL (KNN, logistic regression, trees) **on the embeddings**. `(certain)`

**Transfer learning.** Reuse a network trained for one task to help another: take a model trained on a big labeled dataset (e.g. cancer-detection images), **use an intermediate layer's output as embeddings**, and train a small model for *your* task (e.g. COVID detection) on those embeddings — you inherit the representation someone paid weeks of GPU to learn. Autoencoders are one way to produce such reusable embeddings from **unlabeled** data. (Deeper, with CNNs: [[Transfer Learning]].) `(certain)`

**Recommender systems.** Take the sparse user–item rating matrix; treat each item's column of ratings as its input vector, run it through an AE, and use the **bottleneck embedding** as a dense item vector. Then **cosine similarity** between item embeddings gives "movies similar to X" — a content-free, purely interaction-based recommender. See [Recommendation Systems](../Recommendation%20Systems/Recommendation%20Systems.md) for the matrix-factorization view of the same idea. `(certain)`

**Anomaly detection.** Train the AE on **normal** data only; at inference, **high reconstruction error** flags anomalies (the AE never learned to rebuild them). Strong for images/sequences — see [Anomaly Detection](../Unsupervised%20ML/Anomaly%20Detection.md). `(certain)`

---

## 7. Worked Example

**MNIST compression (784-D images).** Encoder `784 → 128 → 64 → 32`, decoder `32 → 64 → 128 → 784`, sigmoid output + `binary_crossentropy` (pixels normalized to [0,1]). Train `fit(x_train, x_train)`. The **32-D bottleneck** reconstructs recognizable digits — a 24× compression that kept the essence. Push the bottleneck to **2-D** and you can plot every digit on a plane; the classes separate into clusters, comparably to [t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md) but via a *learned, reusable* encoder (t-SNE can't embed new points; the AE encoder can). `(certain)`

**Movie recommender (MovieLens).** Ratings matrix pivoted to **movies × users**, ~**1.5% filled** (very sparse), 668 user-columns. AE `668 → 512 → 256 → 128 → 256 → 512 → 668`, **linear output + MSE** (ratings are continuous). After training, slice out the **128-D bottleneck** as each movie's embedding, build the `cosine_similarity` matrix, and for *Liar Liar* the top-cosine neighbors are its recommendations. `(certain)`

```
movie 1485 (Liar Liar) embedding  ─cos─▶  sort other movies by similarity  ─▶  top-10 = recommendations
```

---

## 8. Code / Implementation

**Keras functional AE + pulling out the encoder:**
```python
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input(shape=(784,))
x = layers.Dense(128, activation="relu")(inp)
x = layers.Dense(64,  activation="relu")(x)
z = layers.Dense(32,  activation="relu")(x)          # ← bottleneck (the embedding)
x = layers.Dense(64,  activation="relu")(z)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(784, activation="sigmoid")(x)     # match data range → BCE loss

ae = keras.Model(inp, out)
ae.compile(optimizer="adam", loss="binary_crossentropy")
ae.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))  # X→X

encoder = keras.Model(ae.input, z)                   # keep ONLY the encoder
embeddings = encoder.predict(x_test)                 # (n, 32) reusable features
```

**Denoising AE — the only change is noisy input, clean target:**
```python
import numpy as np
noise = 0.5
x_noisy = np.clip(x_train + noise * np.random.normal(size=x_train.shape), 0., 1.)
ae.fit(x_noisy, x_train, epochs=100, batch_size=256)   # input NOISY, target CLEAN → learns to denoise
```

---

## 9. When It Breaks

- **Learns the identity function** — too wide/overcomplete or under-regularized → it copies input to output and the embedding is useless. Fix: a real bottleneck, or denoising/sparsity regularization (§5). `(certain)`
- **Bottleneck too small** → underfits, blurry/lossy reconstructions, discarded information. **Too large** → poor compression, identity risk. `d'` is the key hyperparameter to tune. `(certain)`
- **MSE → blurry images.** Pixel-wise MSE averages plausible reconstructions, producing blur; perceptual/adversarial losses help but add complexity. `(likely)`
- **Not a generator.** A vanilla AE's latent space is **not continuous or regularized** — sampling a random `z` and decoding gives garbage, because training never organized the space for interpolation. The **[[Variational Autoencoders|VAE]]** fixes this by encoding a *distribution* (mean+variance) and regularizing the latent space (KL term), enabling generation. Don't claim a plain AE is generative. `(certain)`
- **Reconstruction ≠ semantics.** Low reconstruction error means "rebuilt the pixels," not "understood the content"; embeddings can capture texture/lighting rather than the label you cared about.
- **Linear AE ≈ PCA** — if you use linear activations you've just re-implemented PCA more slowly and less reliably; the non-linearity is the entire reason to reach for an AE. `(certain)`
- **Distribution shift for anomaly detection** — reconstruction-error thresholds drift as "normal" changes; recalibrate.

---

## 10. Production & MLOps Notes

- **Ship the encoder, not the whole AE.** Serving needs only `encoder` (and the decoder *only* for denoising/reconstruction use-cases) — smaller, faster.
- **The embedding is a versioned artifact.** Retraining the AE produces a *different* latent space — downstream indexes (cosine-similarity tables, vector DBs, clusterings) must be **rebuilt together**, or old and new embeddings become incomparable. `(certain)`
- **Anomaly-detection deployment:** set the reconstruction-error threshold from a validation set of known-normal data (e.g. a high percentile), and **monitor for drift** — a rising baseline error means "normal" moved and the threshold needs recalibration.
- **Latent-space monitoring:** track embedding statistics over time; drift signals the encoder is stale. Retrain when input distribution shifts.
- **Modern context:** for images, convolutional AEs beat dense ones; for generation you want a **VAE** or diffusion model; for pretrained *text/image* embeddings, large self-supervised models (BERT, CLIP) have largely superseded task-specific AEs — but AEs remain the go-to for **cheap, unsupervised, domain-specific** compression, denoising, and reconstruction-based anomaly detection. `(likely)`

---

## 11. Interview Lens

**"What is an autoencoder and why does the bottleneck matter?"** → 🎯 *"A network trained to reconstruct its own input through a narrow bottleneck. Reconstructing well from a few latent numbers forces those numbers to encode the input's essential structure, so the bottleneck becomes a learned compressed embedding. Without the bottleneck it could just learn the identity and learn nothing."*

**"How is an autoencoder related to PCA?"** → 🎯 *"A linear autoencoder with MSE recovers the same subspace as PCA. Add non-linear activations and depth and it learns a non-linear manifold PCA can't represent — it's a non-linear generalization of PCA."*

**Likely follow-ups:**
- *Is it supervised?* → No — self-supervised; the target is the input, so no labels are needed.
- *Loss function?* → Matches the data: MSE for continuous, binary cross-entropy for [0,1] pixels, categorical CE for one-hot.
- *What's a denoising AE and why?* → Corrupt the input but reconstruct the clean original; it can't copy noise, so it's forced to learn structure and denoise — a regularizer that prevents the identity function.
- *Undercomplete vs overcomplete?* → Undercomplete (`d'<d`) forces compression; overcomplete needs regularization (denoising/sparsity) or it learns identity.
- *Can an AE generate new data?* → Not a vanilla one — its latent space isn't regularized; use a VAE (encodes a distribution + KL regularizer) for generation.
- *How do you use it for anomaly detection?* → Train on normal data; flag high reconstruction error at inference.
- *Do encoder and decoder have to be symmetric / share weights?* → No; symmetry and weight-tying were historical parameter-saving tricks, not requirements.

---

## 12. Alternatives & How to Choose

| Goal | Reach for | Why |
|---|---|---|
| Linear dim-reduction, few samples | **[PCA](../Unsupervised%20ML/PCA%20%26%20t-SNE.md)** | closed-form, cheap, deterministic; AE overkill |
| Non-linear dim-reduction, lots of data | **Autoencoder** | learns curved manifolds PCA can't |
| 2-D visualization only (no reuse) | **[t-SNE / UMAP](../Unsupervised%20ML/PCA%20%26%20t-SNE.md)** | better cluster separation, but can't embed new points |
| Denoising / robust features | **Denoising AE** | learns to reconstruct clean signal from corrupted input |
| **Generating** new samples | **[[Variational Autoencoders|VAE]]** / diffusion | regularized, samplable latent space |
| Pretrained general text/image features | **BERT / CLIP** | huge self-supervised models beat task-specific AEs |
| Reconstruction-based anomaly detection | **Autoencoder** | high reconstruction error = anomaly ([note](../Unsupervised%20ML/Anomaly%20Detection.md)) |

**Decision rule:** if the structure is **linear or data is scarce**, use **PCA**. Reach for an **autoencoder** when you need a **non-linear** compression, **denoising**, or **unsupervised embeddings** from domain-specific unlabeled data — and you have the data/compute to fit it. If you need to **generate** samples, a plain AE won't do; use a **VAE** or diffusion model. For off-the-shelf text/image embeddings, prefer large pretrained models over training an AE from scratch.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why does training a network to reconstruct its input teach it anything useful?
   <details><summary>answer</summary>Because the signal must pass through a narrow bottleneck: reconstructing `x` from few latent numbers forces those numbers to encode the input's essential structure. The reconstruction is a pretext; the bottleneck embedding is the payoff.</details>
2. What breaks if there's no bottleneck (or the AE is overcomplete and unregularized)?
   <details><summary>answer</summary>It can learn the identity function — copy input to output — and learn no useful representation. Fix with an undercomplete bottleneck or regularization (denoising, sparsity).</details>
3. How does an autoencoder relate to PCA?
   <details><summary>answer</summary>A linear-activation AE with MSE recovers PCA's subspace (top principal components). Non-linear activations + depth let it learn non-linear manifolds PCA can't — it's a non-linear generalization of PCA.</details>
4. Describe a denoising autoencoder and why it helps.
   <details><summary>answer</summary>Feed a noise-corrupted input but reconstruct the clean original. The net can't copy (its input has random noise), so it must learn the underlying structure — a regularizer that prevents identity-learning and yields a model that removes noise.</details>
5. Which reconstruction loss for [0,1]-normalized pixels vs continuous values?
   <details><summary>answer</summary>Binary cross-entropy (sigmoid output) for [0,1] pixels; MSE/RMSE (linear output) for continuous values; categorical cross-entropy for one-hot.</details>
6. Why can't a vanilla autoencoder generate new data, and what fixes it?
   <details><summary>answer</summary>Its latent space isn't continuous/regularized, so sampling a random `z` decodes to garbage. A VAE encodes a distribution (mean+variance) with a KL regularizer, organizing the latent space so you can sample and interpolate.</details>
7. Give three concrete uses of the bottleneck embedding.
   <details><summary>answer</summary>Feature extraction (feed embeddings to a classifier), recommender similarity (cosine between item embeddings), anomaly detection (high reconstruction error), transfer learning, clustering, or low-D visualization.</details>

---

*Covers: the encoder–bottleneck–decoder architecture and self-supervised reconstruction; the formal core (`z=f(x)`, `x̂=g(z)`, reconstruction loss by data type, undercomplete vs overcomplete, weight-tying/symmetry); AE as non-linear PCA; extracting the encoder; denoising AEs (and sparse/contractive variants) as regularizers against the identity function; applications (feature extraction, transfer learning, recommender embeddings via cosine, anomaly detection via reconstruction error); MNIST compression + MovieLens recommender worked examples; failure modes (identity, bottleneck sizing, MSE blur, not generative → VAE, reconstruction≠semantics); production (ship the encoder, versioned embeddings, drift). Sourced from the Scaler Autoencoders lecture. Foundations: [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md), [PCA & t-SNE](../Unsupervised%20ML/PCA%20%26%20t-SNE.md).*
