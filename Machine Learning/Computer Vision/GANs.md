# GANs — Generative Adversarial Networks: two nets in a forgery game that learn to generate data

> **TL;DR.** A **generator** maps random noise `z` to a fake image; a **discriminator** learns to tell real from fake. They train **adversarially** — a minimax game — until the generator's fakes are indistinguishable from real data (`D ≈ 0.5` everywhere). Immensely powerful for image synthesis, but *notoriously* unstable to train (mode collapse, vanishing gradients, non-convergence). **DCGAN** gives the standard conv recipe. As of the 2020s, **diffusion models have overtaken GANs** for most image generation (quality + stability), but the adversarial idea still powers super-resolution, style transfer, and unpaired translation (SRGAN, StyleGAN, CycleGAN).

**Where it fits:** the flagship **generative** model in classic deep learning — *unsupervised* (no labels), learning the data *distribution* to sample new examples, as opposed to the *discriminative* models (classifiers/detectors) in the rest of this folder.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (strided conv, BatchNorm), **transposed convolution** (built up in [Image Segmentation §2](Image%20Segmentation.md#2-the-formal-core--tasks-upsampling-skips-losses-metrics)), [Autoencoders](../Neural%20Networks/Autoencoders.md) (the other encoder-decoder generative model), and binary cross-entropy + Adam ([Weight Initialization & Optimizers](../Neural%20Networks/Weight%20Initialization%20&%20Optimizers.md)).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — the minimax game](#2-the-formal-core--the-minimax-game)
3. [How It Works — generator, discriminator, DCGAN recipe](#3-how-it-works--generator-discriminator-dcgan-recipe)
4. [Training Dynamics (in place of a static worked example)](#4-training-dynamics-in-place-of-a-static-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks — the failure modes GANs are infamous for](#6-when-it-breaks--the-failure-modes-gans-are-infamous-for)
7. [The GAN Zoo, Evaluation & Production](#7-the-gan-zoo-evaluation--production)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**Discriminative vs generative.** A discriminative model learns `P(y|x)` — a *boundary* between classes ("is this a cat?"). A generative model learns `P(x)` — how the data itself is *distributed* — so it can **sample new `x`** that plausibly came from the training set. GANs are generative: feed anime faces, get *new* anime faces that were never in the data.

**Why not just an autoencoder?** An [autoencoder](../Neural%20Networks/Autoencoders.md) reconstructs its *input* — its job is to copy, so its outputs cluster tightly around training examples (low variety) and tend to come out **blurry**. GANs instead optimize for *fooling a critic*, which pushes toward sharp, novel, realistic samples.

**The forgery game (the whole idea):**

```
   Generator (counterfeiter)                Discriminator (detective)
   noise z ──▶ [G] ──▶ fake image ──┐
                                     ├──▶ [D] ──▶ P(real) ∈ [0,1]
              real image ───────────┘
   G's goal: make fakes D calls "real"   D's goal: correctly label real=1, fake=0
```

Two networks locked in competition, each forcing the other to improve. The counterfeiter starts terrible; the detective easily catches it; the counterfeiter learns from being caught; the detective adapts to smarter fakes; round after round both become experts — until the counterfeits are indistinguishable from real currency. At that equilibrium the generator has learned the true data distribution.

🎯 *"A GAN pits a generator that turns noise into fakes against a discriminator that classifies real vs fake; training is a minimax game, and at the ideal equilibrium the generator's distribution matches the data so the discriminator can't do better than a coin flip."*

---

## 2. The Formal Core — the minimax game

**The value function** (`x` real data, `z` latent noise ~ `N(0,I)` or uniform):
```
min_G  max_D   V(D,G) = E_{x~p_data}[ log D(x) ]  +  E_{z~p_z}[ log(1 − D(G(z))) ]
        ▲                └── D wants this ≈ 1 (real→1) ──┘   └── D wants D(G(z))≈0 (fake→0) ──┘
        │
   D maximizes V (be a good critic);  G minimizes V (fool the critic on the 2nd term only —
   G can't touch the first term, which has no G in it).
```

**Discriminator** = a binary classifier trained with **binary cross-entropy**: label real=1, fake=0, maximize `log D(x) + log(1 − D(G(z)))`.

**Generator**, naively, minimizes `log(1 − D(G(z)))` — but early on `D` rejects confidently (`D(G(z))≈0`), where that term is **flat → vanishing gradient**, so `G` can't learn. Fix (**non-saturating loss**, what everyone actually uses): have `G` *maximize* `log D(G(z))` instead — same optimum, strong gradients when they're most needed. `(certain)`

**What the game converges to.** For a fixed `G`, the optimal discriminator is `D*(x) = p_data(x) / (p_data(x) + p_g(x))`. Substituting it back, `G` ends up minimizing the **Jensen–Shannon divergence** between the real and generated distributions — globally minimized when `p_g = p_data`, at which point `D* = ½` everywhere (the critic is reduced to guessing). That's the "indistinguishable fakes" endpoint, made precise.

**Alternating optimization.** You do **not** update `G` and `D` together — you freeze one while stepping the other: (1) train `D` on a real+fake batch, (2) freeze `D`, train `G` through `D` with fake images labeled "real." Repeat.

---

## 3. How It Works — generator, discriminator, DCGAN recipe

**Generator: noise → image (upsampling).** Take a latent vector (e.g. `128`-D `z`), then **transposed convolutions** ([mechanics in the segmentation note](Image%20Segmentation.md#2-the-formal-core--tasks-upsampling-skips-losses-metrics)) progressively upsample `1×1 → 4×4 → 8×8 → … → 64×64×3`.
- **ReLU / LeakyReLU** in hidden layers; **`tanh` at the output** → pixels in `[−1,1]`.
- ⚠️ Therefore **scale real images to `[−1,1]`** too (`(x−127.5)/127.5`) so real and fake share a range — a classic silent bug if forgotten.

**Discriminator: image → P(real) (downsampling).** A CNN with **strided convolutions** (stride 2) shrinking the map, ending in a single `sigmoid` unit.
- **LeakyReLU** (not plain ReLU): plain ReLU zeros all negatives, killing gradient there; LeakyReLU passes a small negative slope so **gradient always flows back into the generator** — critical, since G only learns *through* D.
- No global-average-pooling head — GAP stabilizes classifiers but empirically **hurts GAN convergence** (DCGAN finding).

**DCGAN — the recipe that made conv GANs train** (memorize; it's the interview default):
```
• Replace pooling with STRIDED conv (D) and TRANSPOSED conv (G) — learn the res/downsampling.
• BatchNorm in G and D (but NOT on G's output layer or D's input layer).
• No fully-connected hidden layers — fully convolutional.
• ReLU in G (tanh output); LeakyReLU in D.
• Adam, lr ≈ 2e-4, β1 = 0.5  (not the usual 0.9 — lower β1 stabilizes GAN training).
• Real images scaled to [−1,1] to match tanh.
```

**One training step** (alternating):
```
1. z ~ N(0,I) → G(z) = fake batch;  take a real batch.
2. Train D: BCE on {real→1, fake→0}, update D only.
3. z ~ N(0,I) again → G(z);  label them "real" (1).
4. Train G: BCE through the (frozen) D, update G only → G is rewarded for fooling D.
```

---

## 4. Training Dynamics (in place of a static worked example)

*(A numeric single-pass example doesn't capture GANs — the phenomenon **is** the two-player trajectory, so here's what to watch.)*

**Healthy training** is a *balanced tug-of-war*, not a loss that marches to zero:
```
loss
 │   d_loss  ~~~~~~~~~~  hovers, drifting UP a bit as fakes get harder to catch
 │   g_loss  ‾‾\__ ~~~   drifts DOWN / oscillates as samples improve
 │
 └──────────────────────── epochs      (samples get sharper each epoch — the real signal)
```
Counter-intuitively, **rising discriminator loss is often good**: it means the generator is producing harder fakes, so the same-quality critic is fooled more. There's no single "loss = quality" number — the losses are *relative* to an opponent that's also moving. **Judge by the samples** (and FID), not the loss curves.

**The equilibrium** you're chasing: `p_g → p_data`, `D(x) → ½` for everything. In practice you rarely reach it cleanly — you stop when samples look good and diverse.

**Pathologies show up as characteristic curves:** `d_loss → 0` while `g_loss` explodes = discriminator won, generator starved (vanishing gradient). `g_loss` low but samples all look the same = mode collapse. Wild oscillation with no sample improvement = non-convergence. (Fixes in §6.)

---

## 5. Code / Implementation

**Generator & discriminator (DCGAN-style, Keras):**
```python
from tensorflow.keras import layers, Input, Model

def generator(latent_dim=128):                       # noise (1,1,128) → 64×64×3 in [-1,1]
    z = Input((1, 1, latent_dim))
    x = z
    for f, s, p in [(512,1,"valid"), (256,2,"same"), (128,2,"same"), (64,2,"same")]:
        x = layers.Conv2DTranspose(f, 4, strides=s, padding=p)(x)   # upsample
        x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)  # → [-1,1]
    return Model(z, out)

def discriminator(shape=(64,64,3)):                  # image → P(real)
    img = Input(shape); x = img
    for f in [64, 128, 256, 512]:
        x = layers.Conv2D(f, 4, strides=2, padding="same")(x)       # strided downsample
        if f > 64: x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.LeakyReLU(0.2)(x)                                # LeakyReLU → gradient to G
    out = layers.Dense(1, activation="sigmoid")(layers.Flatten()(x))
    return Model(img, out)
```

**The adversarial train step (alternating D then G):**
```python
class GAN(keras.Model):
    def train_step(self, real):
        bs = tf.shape(real)[0]
        z  = tf.random.normal((bs, 1, 1, self.latent_dim))
        # 1) Train D on real (label 1) + fake (label 0)
        combined = tf.concat([self.generator(z), real], 0)
        labels   = tf.concat([tf.zeros((bs,1)), tf.ones((bs,1))], 0)   # fake=0, real=1
        with tf.GradientTape() as t:
            d_loss = self.loss_fn(labels, self.discriminator(combined))
        self.d_opt.apply_gradients(zip(t.gradient(d_loss, self.discriminator.trainable_weights),
                                       self.discriminator.trainable_weights))
        # 2) Train G: fresh noise, label fakes as "real" (1) → G rewarded for fooling frozen D
        z = tf.random.normal((bs, 1, 1, self.latent_dim))
        with tf.GradientTape() as t:
            g_loss = self.loss_fn(tf.ones((bs,1)), self.discriminator(self.generator(z)))
        self.g_opt.apply_gradients(zip(t.gradient(g_loss, self.generator.trainable_weights),
                                       self.generator.trainable_weights))    # only G updates
        return {"d_loss": d_loss, "g_loss": g_loss}

gan.compile(d_opt=Adam(2e-4, beta_1=0.5), g_opt=Adam(1.5e-4, beta_1=0.5),   # β1=0.5; slightly
            loss_fn=keras.losses.BinaryCrossentropy())                       # different LRs = TTUR
```
Monitor by generating from a **fixed** noise batch every epoch (so you see the *same* seeds improve). Note the two different learning rates — a mild **two-time-scale (TTUR)** trick that helps stability.

---

## 6. When It Breaks — the failure modes GANs are infamous for

```
❌ MODE COLLAPSE — the signature GAN failure. G finds a few outputs that reliably fool D and emits only
   those → all samples look alike, ignoring most of the data's modes. It's a degenerate equilibrium.
   FIXES: minibatch discrimination / feature matching (let D see batch diversity), unrolled GANs,
   Wasserstein loss (WGAN/WGAN-GP), lower/tuned LRs, more diverse minibatches.

❌ VANISHING GRADIENT — discriminator too strong. If D becomes near-perfect, log(1−D(G(z)))≈0 gradient
   → G stops learning. FIXES: the non-saturating G loss (max log D(G(z))); don't over-train D per G step;
   Wasserstein/WGAN gives useful gradients even when D is confident.

❌ NON-CONVERGENCE / OSCILLATION. The minimax dynamics need not settle — G and D can cycle forever,
   samples never stabilizing. FIXES: TTUR (different LRs), spectral normalization, one-sided label
   smoothing (real target 0.9 not 1.0), historical averaging.

❌ HYPERPARAMETER FRAGILITY. GANs are "extremely sensitive to hyperparameters, activations, and
   regularization." A wrong LR ratio, β1, or init can wreck a run. Start from a KNOWN recipe (DCGAN) and
   change one thing at a time.

❌ EVALUATION IS HARD. Loss ≠ quality (it's relative to a moving opponent). You cannot early-stop on loss.
   → Use FID / Inception Score (§7) and eyeball samples.

❌ D–G IMBALANCE. If one overpowers the other, quality collapses. Watch generated samples every epoch;
   rebalance LRs / update ratios.
```

---

## 7. The GAN Zoo, Evaluation & Production

**The variants worth naming** (interviewers ask "which GAN for X?"):
- **Conditional GAN (cGAN)** — condition G and D on a label → control *what* is generated (class-conditional).
- **pix2pix** — **paired** image-to-image translation (edges→photo, map→satellite); cGAN + L1 loss, needs aligned pairs.
- **CycleGAN** — **unpaired** translation (horse↔zebra, summer↔winter) via **cycle-consistency loss**: two generators `G:X→Y`, `F:Y→X` with `F(G(x)) ≈ x` — no paired data needed. `(certain)`
- **SRGAN** — super-resolution: upsamples low-res → high-res with a perceptual+adversarial loss for sharp detail.
- **StyleGAN (1/2/3)** — style-based generator + progressive growing → photorealistic high-res faces with a disentangled, controllable latent (`w`) space.
- **WGAN / WGAN-GP** — replace JS-divergence BCE with the **Wasserstein (earth-mover) distance** and a "critic"; far more stable, meaningful loss, resists mode collapse (gradient penalty ≻ weight clipping).
- **BigGAN** — large-scale class-conditional, high-fidelity ImageNet synthesis.

**Evaluation (there's no accuracy here):**
- **FID (Fréchet Inception Distance)** — distance between real and generated feature distributions (Inception activations); **lower = better**. The de-facto standard.
- **Inception Score (IS)** — rewards samples that are both confidently classified and diverse; **higher = better** (but weaker than FID).

**Production reality.** Training is expensive and brittle: checkpoint often, fix seeds for reproducible sample tracking, log FID over training, and expect to babysit runs. For **deployment** you ship only the **generator** (the discriminator is scaffolding). GANs are fast at *inference* (one forward pass) — an advantage over diffusion's iterative sampling.

**The 2020s reality check (the lecture predates this).** For most *image generation*, **diffusion models** (DDPM → Stable Diffusion, DALL·E, Imagen) have **overtaken GANs**: comparable-or-better quality, far better mode coverage/diversity, and vastly more stable training — at the cost of slow multi-step sampling. GANs remain competitive where **fast one-shot sampling** or a specific adversarial objective matters (super-resolution, real-time style transfer, some unpaired translation). See [[Diffusion Models]]. `(likely)`

---

## 8. Interview Lens

> ⚡ The core story every time: generator vs discriminator, minimax objective, alternating training, equilibrium at `p_g=p_data`/`D=½`, and the instability failure modes (mode collapse, vanishing gradient) with their fixes.

**"Explain a GAN and how it's trained."** → 🎯 *"Two nets: a generator turns noise into fakes, a discriminator classifies real vs fake. They play a minimax game — D maximizes `E[log D(x)] + E[log(1−D(G(z)))]`, G minimizes it — trained by alternating (freeze one, update the other). In practice G uses the non-saturating loss `max log D(G(z))` for gradients. At the optimum G matches the data distribution and D is stuck at 0.5."*

**Likely follow-ups:**
- *Generator vs discriminator loss/activation?* → G: transposed-conv upsampler, ReLU + `tanh` output ([−1,1]). D: strided-conv classifier, LeakyReLU + `sigmoid`. Scale real images to [−1,1]. `(certain)`
- *Why LeakyReLU in D?* → Passes negative-side gradient so signal always flows back to G (which only learns through D).
- *Why the non-saturating generator loss?* → `log(1−D(G(z)))` saturates (flat, no gradient) when D confidently rejects early on; `max log D(G(z))` gives strong gradients. `(certain)`
- *What is mode collapse and how do you fix it?* → G emits a few outputs that fool D, losing diversity; fix with minibatch discrimination, feature matching, WGAN-GP, LR tuning.
- *Why is GAN training unstable / how judge progress?* → Two moving objectives → oscillation, D–G imbalance, vanishing gradients; loss ≠ quality, so track samples + **FID**.
- *DCGAN key choices?* → strided/transposed conv (no pooling), BatchNorm, LeakyReLU(D)/ReLU(G), tanh out, no FC, Adam β1=0.5.
- *Paired vs unpaired translation?* → pix2pix (paired, L1+cGAN) vs CycleGAN (unpaired, cycle-consistency).
- *GAN vs VAE vs diffusion?* → GAN: sharp, fast sampling, unstable. VAE: stable, blurry. Diffusion: best quality+diversity, stable, slow sampling — current SOTA.

---

## 9. Alternatives & How to Choose

| Goal | Reach for | Why |
|---|---|---|
| Photorealistic image synthesis (today) | **Diffusion models** | SOTA quality + diversity + stable training (slow sampling) |
| Sharp samples with **fast** one-shot sampling | **GAN (StyleGAN / BigGAN)** | single forward pass; great fidelity |
| Stable training, latent space, OK with blur | **VAE** ([Autoencoders](../Neural%20Networks/Autoencoders.md)) | probabilistic, easy to train, smooth latent |
| Class-controlled generation | **Conditional GAN / diffusion** | condition on labels/text |
| Paired image-to-image | **pix2pix** | aligned pairs → cGAN + L1 |
| Unpaired image-to-image | **CycleGAN** | cycle-consistency, no pairs needed |
| Super-resolution | **SRGAN / ESRGAN** | adversarial + perceptual loss → sharp upsampling |
| Stabilize a flaky GAN | **WGAN-GP + spectral norm** | Wasserstein loss resists mode collapse, gives useful gradients |
| Exact likelihood / sequential | **Autoregressive (PixelCNN, image-GPT)** | tractable likelihood, strong but slow |

**Decision rule:** *for new image-generation work, start with diffusion; reach for a GAN when you need fast single-pass sampling or a specific adversarial objective (super-res, real-time translation); use a VAE when you want a stable, structured latent and can tolerate softness.* Whatever you pick, if it's a GAN, start from the DCGAN recipe and evaluate with FID.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. What do the generator and discriminator each optimize, and what's the combined objective?
   <details><summary>answer</summary>D maximizes `E[log D(x)] + E[log(1−D(G(z)))]` (label real 1, fake 0); G minimizes it (only the 2nd term). Combined: `min_G max_D V(D,G)`. In practice G uses the non-saturating `max log D(G(z))`.</details>
2. Why scale real images to [−1,1], and what activations sit at G's and D's outputs?
   <details><summary>answer</summary>G's output is `tanh` → range [−1,1], so real images must match that range. D's output is `sigmoid` → P(real).</details>
3. Why LeakyReLU in the discriminator rather than ReLU?
   <details><summary>answer</summary>ReLU zeros all negative activations (and their gradient); LeakyReLU keeps a small negative slope so gradient always flows back into the generator, which only learns through D.</details>
4. What does the game converge to at the global optimum?
   <details><summary>answer</summary>`p_g = p_data` (generator matches the data distribution), where the optimal discriminator `D*(x)=p_data/(p_data+p_g) = ½` everywhere — it can only guess. Equivalent to minimizing JS divergence.</details>
5. Define mode collapse and give two fixes.
   <details><summary>answer</summary>G produces only a few outputs that fool D → samples lack diversity (ignores most data modes). Fixes: minibatch discrimination / feature matching, WGAN-GP, unrolled GANs, LR tuning.</details>
6. Why can't you judge GAN progress from the loss, and what do you use instead?
   <details><summary>answer</summary>Losses are relative to a moving opponent (rising D loss can mean better fakes); there's no "loss=quality." Use FID (lower better) / Inception Score (higher better) and inspect fixed-seed samples.</details>
7. List four DCGAN design rules.
   <details><summary>answer</summary>Strided conv (D) + transposed conv (G) instead of pooling; BatchNorm (not on G-output/D-input); LeakyReLU in D, ReLU + tanh in G; no FC layers; Adam lr≈2e-4, β1=0.5.</details>
8. GAN vs VAE vs diffusion in one line each.
   <details><summary>answer</summary>GAN: sharp samples, fast one-pass sampling, unstable/mode-collapse-prone. VAE: stable, structured latent, blurry. Diffusion: best quality + diversity, stable training, slow iterative sampling — current SOTA for image gen.</details>

---

*Covers: generative vs discriminative · why not autoencoders · generator (transposed conv, ReLU/tanh) vs discriminator (strided conv, LeakyReLU/sigmoid) · minimax value function & non-saturating loss · optimal D & JS-divergence equilibrium · alternating training · DCGAN recipe (BatchNorm, no pooling/FC, Adam β1=0.5, [−1,1]) · training dynamics & reading the curves · mode collapse / vanishing gradient / non-convergence + fixes (WGAN-GP, spectral norm, TTUR, label smoothing) · cGAN/pix2pix/CycleGAN/SRGAN/StyleGAN/BigGAN · FID & Inception Score · GAN vs VAE vs diffusion.*
