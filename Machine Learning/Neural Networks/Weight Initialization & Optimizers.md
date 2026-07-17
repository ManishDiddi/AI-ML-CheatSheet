# Weight Initialization & Optimizers — Where You Start and How You Descend

> **TL;DR.** Two knobs that decide whether a deep net trains at all and how fast. **Initialization** sets *where the ball starts*: you must break symmetry (never zeros/constants) and scale the random weights so signal + gradients keep their variance across layers — **Xavier/Glorot** for tanh/sigmoid (`Var(w)≈1/n_in`, precisely `2/(n_in+n_out)`), **He** for ReLU (`Var(w)=2/n_in`, doubled because ReLU kills half the signal). **Optimizers** decide *how the ball rolls*: plain mini-batch GD takes noisy, wasteful steps, so we smooth them. The one idea underneath all modern optimizers is the **exponential moving average (EMA)** of gradients: **Momentum** EMAs the gradient (build velocity, damp oscillation), **RMSprop** EMAs the *squared* gradient (per-parameter adaptive step), and **Adam** = both + bias-correction. Cap it with a **learning-rate schedule** (decay/warmup/cosine). Default today: **He init + AdamW + warmup→cosine**.

**Where it fits:** the training-dynamics layer of [deep learning](Neural%20Network%20Fundamentals.md) — same forward/backward math, but *how you initialize and step* is what separates a net that diverges or stalls from one that converges in a fraction of the epochs.
**Prereqs:** [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md) (forward/backprop, `dW`, vanishing/exploding gradients), [gradient descent](../Supervised%20ML/Linear%20Regression.md), basic variance algebra.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [Weight Initialization](#2-weight-initialization)
3. [The Optimizer Problem: Why Plain GD Is Slow](#3-the-optimizer-problem-why-plain-gd-is-slow)
4. [Exponential Moving Average — The Core Primitive](#4-exponential-moving-average--the-core-primitive)
5. [Momentum, RMSprop & Adam](#5-momentum-rmsprop--adam)
6. [Learning Rate Schedules](#6-learning-rate-schedules)
7. [Worked Example](#7-worked-example)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks](#9-when-it-breaks)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose](#12-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Training is a ball rolling down a loss surface. **Initialization is where you drop the ball; the optimizer is the physics of how it rolls.** Both can ruin you: drop it in a flat/exploded region and the gradient is dead or infinite before the optimizer gets a chance; roll it with dumb physics and it zig-zags for thousands of epochs.

```
        loss
         │   plain mini-batch GD: noisy, zig-zags across the valley
         │        \  /\    /\   /\
         │         \/  \  /  \ /  \___ ...many wasted steps
         │              \/    v
         │   momentum/Adam: averages past steps → rolls straight down
         │         \___
         │             \____
         └────────────────────▶ weights
   ▲ where INIT drops you        ▲ where the OPTIMIZER takes you
```

- 🎯 **The unifying trick for every modern optimizer is the exponential moving average of the gradient.** Momentum averages the gradient itself (direction); RMSprop averages its square (scale); Adam does both. Everything else is bias-correction and scheduling. `(certain)`
- **Init and optimizer solve *different* failures.** Init fixes *signal/gradient scale at step 0* (symmetry, vanishing/exploding). The optimizer fixes *the path over many steps* (noise, oscillation, slow valleys). A perfect optimizer can't rescue an exploded init, and perfect init still needs a good optimizer to converge fast.

---

## 2. Weight Initialization

**Two things can go wrong before the first gradient step: symmetry and scale.**

### 2a. Symmetry — never initialize to zeros or a constant
If every weight in a layer is identical, every neuron computes the **same output** and receives the **same gradient** in backprop → they update identically and stay clones **forever**. The layer has the representational power of a *single* neuron, permanently. `(certain)`

- 🎯 **Zero-init (or any constant) is fatal because it never breaks symmetry — random init is what lets neurons specialize.** `(certain)`
- Biases *can* safely start at 0 (the weights already break symmetry). ReLU layers sometimes use a small positive bias (e.g. `0.01`) to keep units active early, but `0` is the standard.

### 2b. Scale — keep the variance stable across layers
Random isn't enough; the *magnitude* matters. For a pre-activation `z = Σᵢ wᵢxᵢ` over `n_in` zero-mean independent inputs:
```
Var(z) = n_in · Var(w) · Var(x)
```
- If `n_in·Var(w) > 1`, variance **grows** every layer → activations & gradients **explode**.
- If `n_in·Var(w) < 1`, variance **shrinks** every layer → activations & gradients **vanish** (§7 of [Fundamentals](Neural%20Network%20Fundamentals.md)).
- 🎯 **The goal: pick `Var(w)` so signal variance stays ≈1 layer-to-layer, in both the forward pass and the backward (gradient) pass.** That single requirement is what Xavier and He solve. `(certain)`

### 2c. The two initializers you must know
| Init | Use with | `Var(w)` | Draw (normal form) | Why |
|---|---|---|---|---|
| **Xavier / Glorot** | tanh, sigmoid, linear | `2/(n_in+n_out)` | `w ~ N(0, 2/(n_in+n_out))` | balances forward **and** backward variance for a symmetric, ~linear-at-0 activation |
| **He / Kaiming** | ReLU, Leaky ReLU | `2/n_in` | `w ~ N(0, 2/n_in)` | ReLU zeros ~half its inputs → halves variance, so **double** it to compensate |
| **LeCun** | SELU | `1/n_in` | `w ~ N(0, 1/n_in)` | preserves variance for the self-normalizing SELU |

- 🎯 **He vs Xavier is the classic question: the only difference is the factor of 2 that He adds because ReLU throws away the negative half of the distribution.** Use **He with ReLU-family**, **Xavier with tanh/sigmoid**. `(certain)`
- **Uniform vs normal** is a minor variant of the same variance: Glorot-uniform draws `w ~ U[−√(6/(n_in+n_out)), +√(6/(n_in+n_out))]` — same variance, bounded support. Keras `Dense` defaults to **`glorot_uniform`** (that's the `kernel_initializer='glorot_uniform'` you see in the lecture); with ReLU you should usually switch it to **`he_normal`/`he_uniform`**. `(certain)`
- **`n_in` = fan-in** (inputs to the layer), **`n_out` = fan-out** (neurons in the layer). "Fan" = number of connections.
- **Where this later got automated:** Batch/Layer Norm and residual connections make deep nets far less sensitive to init by renormalizing activations mid-network — but good init still matters for the first steps and for norm-free architectures. See [Batch Normalization & Dropout](Batch%20Normalization%20%26%20Dropout.md).

---

## 3. The Optimizer Problem: Why Plain GD Is Slow

**Mini-batch GD** updates weights once per batch: batch of `1` = **Stochastic GD (SGD)** (frequent but noisy updates), batch = whole set = **Batch GD** (smooth but one slow update per epoch), in-between = **mini-batch** (the practical default). `(certain)`

The catch the lecture demonstrates: on the 24Seven data, plain mini-batch SGD needed **~200+ epochs** to converge.
```
Why so slow? Each batch has a different loss, so the gradient direction
jumps batch-to-batch:  → ↗ → ↘ → ↗  (noisy). Many steps point AWAY
from the minimum and cancel out. The net progress per epoch is tiny.
```
- 🎯 **Mini-batch gradients are noisy because different batches disagree on the gradient; the optimizer wastes steps oscillating instead of descending.** The fix is to *average the recent gradients* so the noise cancels and the consistent direction survives — that average is the EMA. `(certain)`

---

## 4. Exponential Moving Average — The Core Primitive

An EMA is a running, exponentially-decaying average of a stream of values `gₜ`:
```
Vₜ = β·Vₜ₋₁ + (1−β)·gₜ            # β ∈ [0,1): how much history to keep
```
Unrolling shows *why* it's "exponential" — older gradients decay geometrically:
```
V₃ = (1−β)g₃ + β(1−β)g₂ + β²(1−β)g₁ + β³V₀
       now         1 ago        2 ago     (fades as βᵏ)
```
- **`β` = memory length.** It averages roughly the last **`1/(1−β)`** values: `β=0.9`→~10 steps, `β=0.99`→~100. Higher `β` = smoother but laggier. `(certain)`
- **Bias problem (matters for §5's Adam).** Starting at `V₀=0` makes early estimates too small — `V₁=(1−β)g₁` is only 10% of `g₁` at `β=0.9`. **Bias correction** rescales: `V̂ₜ = Vₜ/(1−βᵗ)`. As `t` grows `βᵗ→0` so the correction fades; it only matters for the first ~`1/(1−β)` steps. `(certain)`

---

## 5. Momentum, RMSprop & Adam

All three apply the EMA to the gradient — they differ in *what* they average and *how* they use it.

### 5a. Gradient Descent with Momentum — EMA of the gradient
```
V_dw = β·V_dw + (1−β)·dw          # EMA of the gradient (velocity)
w    = w − α·V_dw                 # step along the smoothed direction
```
Instead of stepping along the *current* (noisy) gradient, step along the *running average*. Oscillating components cancel; the consistent downhill component accumulates. `(certain)`

> **Ball-down-a-hill analogy** (the lecture's mental model): the gradient `dw` is **acceleration**, `V_dw` is **velocity**, and `β` is **friction** that stops the ball speeding up without limit. The ball builds *momentum* down the consistent slope and coasts through small bumps. On the 24Seven net, momentum cut ~200 epochs down to ~25.

- **Nesterov (NAG)** is a sharper variant: evaluate the gradient at the *look-ahead* position `w − β·V_dw` (where momentum is about to carry you), so it can "brake" before overshooting. Usually a small, free win. `(likely)`
- ⚠️ **Framework nuance:** the lecture's EMA form has the `(1−β)`; Keras/PyTorch `SGD(momentum=β)` use `v = β·v − α·dw; w = w + v` (no `(1−β)`), folding that factor into the learning rate. Same behavior, different bookkeeping — don't get thrown when the code doesn't match the formula. `(certain)`

### 5b. RMSprop — EMA of the *squared* gradient
Momentum smooths *direction*; RMSprop adapts the *step size per parameter*. It EMAs the squared gradient, then divides the step by its root:
```
S_dw = β·S_dw + (1−β)·(dw)²                 # EMA of squared gradient
w    = w − α · dw / (√(S_dw) + ε)           # per-parameter adaptive step
```
- 🎯 **Intuition: parameters with large, jumpy gradients get a *big* denominator → small, damped steps; parameters with small, flat gradients get a *small* denominator → larger steps.** It equalizes progress across directions, killing the oscillation that plagues steep-in-one-axis, flat-in-another valleys. `(certain)`
- **`ε ≈ 1e-8`** guards against divide-by-zero when `S_dw≈0`.
- **AdaGrad** is the ancestor: it uses a *cumulative sum* `S = S + (dw)²` instead of an EMA. Because the sum only grows, the effective learning rate decays monotonically to ~0 and training **stalls**. RMSprop's EMA (which forgets old gradients) is exactly the fix. `(certain)`

### 5c. Adam — momentum + RMSprop + bias correction
Adam runs **both** EMAs and bias-corrects them: `(certain)`
```
V = β₁·V + (1−β₁)·dw          # 1st moment  (momentum, direction)
S = β₂·S + (1−β₂)·(dw)²       # 2nd moment  (RMSprop, scale)
V̂ = V / (1−β₁ᵗ)               # bias-correct (both start at 0)
Ŝ = S / (1−β₂ᵗ)
w = w − α · V̂ / (√(Ŝ) + ε)
```
- **Defaults you should have memorized:** `β₁=0.9, β₂=0.999, ε=1e-8`. `β₂` is high because squared gradients are noisier and need a longer memory. `(certain)`
- 🎯 **Adam = "momentum for direction + RMSprop for per-parameter scale + bias-correction for the cold start." It's the robust default because it needs little LR tuning to work.** `(certain)`
- **AdamW** (what transformers actually use) fixes a subtle bug: in plain Adam, L2 weight decay gets divided by the adaptive `√Ŝ` term, so heavily-updated weights are decayed *less* — not what you want. **AdamW decouples** decay from the gradient step: `w = w − α·V̂/(√Ŝ+ε) − α·λ·w`. Use AdamW whenever you use weight decay. `(likely)`

```
       averages DIRECTION   averages SCALE     bias-corrects
SGD          –                    –                 –
Momentum     ✓ (EMA of dw)        –                 –
RMSprop      –                    ✓ (EMA of dw²)    –
Adam         ✓                    ✓                 ✓
```

---

## 6. Learning Rate Schedules

Even Adam plateaus if `α` is fixed: too high and it **bounces around** the minimum (loss stuck at, say, 0.4); too low and it **crawls**. The fix is a high LR early (fast approach) that **decays** as you near the minimum (fine settling). `(certain)`

- **Time-based decay** (the lecture's, via a Keras `LearningRateScheduler` callback applied *after* each epoch):
  ```
  α = α₀ / (1 + r·epoch)          # r = decay rate; α₀ = initial LR
  ```
- **Other schedules you'll meet in production:** `(likely)`
  - **Step decay** — cut LR ×0.1 every N epochs (classic for CNNs).
  - **Exponential** — `α = α₀·e^(−k·epoch)`.
  - **Cosine annealing** — smoothly anneal to ~0 along a cosine; the modern default for vision + transformers.
  - **Warmup** — ramp LR *up* linearly for the first few hundred/thousand steps **before** decaying. 🎯 **Essential for Adam on transformers: early second-moment estimates `Ŝ` are unreliable, so a full-size step can destabilize training — warmup lets the statistics settle first.** `(likely)`
- **ReduceLROnPlateau** — a reactive alternative: drop LR when validation loss stops improving. Good when you don't know the schedule up front.

---

## 7. Worked Example

Tiny arithmetic so the formulas become muscle memory (all from the lecture's quizzes).

**EMA step.** `V₀=0, β=0.9, g₁=0.1`:
```
V₁ = 0.9·0 + (1−0.9)·0.1 = 0.1·0.1 = 0.01
```
**Recover a gradient from an EMA.** Previous `V=0.04`, `β=0.9`, new `V=0.06`. Solve for `dw`:
```
0.06 = 0.9·0.04 + 0.1·dw  →  dw = (0.06 − 0.036)/0.1 = 0.24
```
**Momentum weight update.** `α=0.01, V_dw=0.1, w=1.2`:
```
w = 1.2 − 0.01·0.1 = 1.2 − 0.001 = 1.199
```
**Learning-rate decay.** `α₀=0.1, r=0.1, epoch=10`:
```
α = 0.1 / (1 + 0.1·10) = 0.1 / 2 = 0.05
```
**One Adam step (t=1).** `dw=0.1, V=S=0, β₁=0.9, β₂=0.999, α=0.001, ε=1e-8`:
```
V = 0.1·0.1 = 0.01      V̂ = 0.01/(1−0.9)   = 0.1
S = 0.001·0.01 = 1e-5   Ŝ  = 1e-5/(1−0.999) = 0.01
step = α·V̂/(√Ŝ+ε) = 0.001·0.1/(0.1) = 0.001     # bias-correction makes step 1 sane, not tiny
```

---

## 8. Code / Implementation

**Keras — swap the initializer and the optimizer (the lecture's model):**
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation="relu", kernel_initializer="he_normal"),   # ReLU → He, not the glorot default
    Dense(32, activation="relu", kernel_initializer="he_normal"),
    Dense(3,  activation="softmax", kernel_initializer="glorot_uniform"),  # softmax layer ~ linear → Glorot
])

# The optimizer progression, each cutting epochs vs the last:
# SGD()                         → plain mini-batch GD  (needs ~200 epochs)
# SGD(momentum=0.9)             → GD + momentum        (~25)
# RMSprop(rho=0.9)              → adaptive per-param LR (fast but oscillates)
# Adam(beta_1=0.9, beta_2=0.999)→ momentum + RMSprop    (smooth + fast)
model.compile(optimizer="adamw", loss="categorical_crossentropy")   # AdamW = Adam + decoupled weight decay

# Learning-rate decay as an after-epoch callback:
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):            # α = α₀ / (1 + r·epoch)
    return lr / (1 + 0.01 * epoch)
model.fit(X_train, y_train, epochs=50, batch_size=128,
          callbacks=[LearningRateScheduler(scheduler)])
```

**From scratch — Adam is just the EMAs in ~6 lines:**
```python
def adam_step(w, dw, state, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    state["V"] = b1*state["V"] + (1-b1)*dw          # 1st moment (momentum)
    state["S"] = b2*state["S"] + (1-b2)*dw**2       # 2nd moment (RMSprop)
    Vh = state["V"] / (1 - b1**t)                    # bias correction
    Sh = state["S"] / (1 - b2**t)
    return w - lr * Vh / (np.sqrt(Sh) + eps)         # adaptive, direction-smoothed step
# state = {"V": 0, "S": 0}; call with t = 1, 2, 3, … (t must start at 1, not 0)
```

---

## 9. When It Breaks

- **Bad init → dead or diverged before step 1.** Too-small init (or zeros) → vanishing activations/gradients; too-large → exploding NaNs. Symptom: loss `NaN` on the first batches, or loss flat from epoch 0. Fix: He/Xavier per activation. `(certain)`
- **Symmetry from constant init** — network learns as if it had one neuron per layer; accuracy caps far below capacity.
- **Exploding gradients** (deep/recurrent nets) — even with good init, gradients can blow up. Fix: **gradient clipping** (`clipnorm`/`clipvalue`) — rescale the gradient if its norm exceeds a threshold. Standard for RNNs/LSTMs.
- **LR too high** — loss oscillates, spikes, or goes NaN; **too low** — loss barely moves. The single most impactful hyperparameter; tune it first (LR range test / warmup).
- **Adam's generalization gap** — Adam often reaches low *training* loss fastest but can generalize slightly *worse* than SGD+momentum on vision; AdamW narrows this. Some pipelines "Adam early, SGD late." `(likely)`
- **`ε` in mixed precision (fp16)** — a too-small `ε` can underflow to 0 and cause NaNs; frameworks bump it (e.g. `1e-4`) under fp16. `(likely)`
- **Forgetting to reset optimizer state** — Momentum/Adam carry `V`/`S` buffers; reloading weights but not clearing/loading these state buffers corrupts the first steps after a resume. `(certain)`

---

## 10. Production & MLOps Notes

- 🎯 **Default recipe by domain** `(likely)`:
  - **Transformers / NLP / LLMs** → **AdamW + linear warmup → cosine decay**. Non-negotiable at scale.
  - **CNNs / vision** → **SGD + momentum(0.9) + step or cosine decay** — often generalizes best; Adam(W) is the fast-prototyping alternative.
  - **Tabular / small nets** → **Adam** as a robust, low-tuning default.
- **LR is the hyperparameter that matters most.** Use an **LR range test** (ramp LR, watch loss) to find the max stable LR; then warmup + decay around it. Batch size and LR scale together — the **linear scaling rule**: double the batch → roughly double the LR (up to a limit).
- **Weight decay ≠ L2 with adaptive optimizers.** Use **AdamW** so decay is applied directly to weights, not warped by the `√Ŝ` denominator. Typical `λ` ~ `0.01–0.1` for transformers.
- **Gradient clipping** (`clipnorm≈1.0`) is cheap insurance against loss spikes, especially for RNNs/transformers.
- **Memory cost:** Adam stores **2 extra buffers** (`V`, `S`) per parameter → ~3× the parameter memory of SGD. Relevant when a model barely fits; SGD or 8-bit Adam saves it.
- **Reproducibility:** seed NumPy **and** the framework (`tf.random.set_seed`/`torch.manual_seed`); init is random, so an unseeded run isn't reproducible. Full determinism also needs deterministic GPU kernels.
- **Monitoring:** watch **gradient norms** and **per-layer activation stats** in early training — a norm trending to 0 (vanishing) or ∞ (exploding) tells you init/LR is wrong *before* the loss curve does.

---

## 11. Interview Lens

**"Why can't you initialize all weights to zero?"** → 🎯 *"It never breaks symmetry: every neuron in a layer computes the same output and gets the same gradient, so they stay identical forever — the layer collapses to a single neuron. You need random init so neurons specialize."*

**"Xavier vs He — when and why?"** → 🎯 *"Both keep activation variance stable across layers by scaling `Var(w)` to the fan-in. Xavier (`1/n_in`, or `2/(n_in+n_out)`) is for symmetric activations like tanh/sigmoid; He (`2/n_in`) is for ReLU — the factor of 2 compensates for ReLU zeroing half the inputs and halving the variance."*

**"Explain Adam."** → 🎯 *"Adam combines momentum — an EMA of the gradient for a smoothed direction — with RMSprop — an EMA of the squared gradient for a per-parameter adaptive step — plus bias correction to fix the cold start when both EMAs are initialized at zero."*

**Likely follow-ups:**
- *Momentum vs RMSprop?* → Momentum EMAs the gradient to smooth **direction**; RMSprop EMAs the squared gradient to adapt **step size** per parameter (damps oscillating directions).
- *Why bias correction?* → EMAs start at 0, so early estimates are biased low; dividing by `1−βᵗ` corrects it, and the correction fades as `t` grows.
- *What is `β`?* → The memory of the EMA; it averages ~`1/(1−β)` recent steps (0.9→~10). Friction in the ball analogy.
- *Why does mini-batch GD need so many epochs?* → Batch-to-batch gradient noise makes it oscillate; averaging (momentum/Adam) cancels the noise and cuts the step count.
- *AdaGrad vs RMSprop?* → AdaGrad **sums** squared gradients so the LR decays to 0 and stalls; RMSprop uses an **EMA** that forgets, so it keeps learning.
- *Why warmup for transformers?* → Adam's second-moment estimate is unreliable in the first steps; a large step then can diverge, so ramp the LR up first.
- *AdamW vs Adam?* → AdamW decouples weight decay from the adaptive update, so decay isn't scaled by `√Ŝ`; it's the correct default with weight decay.
- *Does init still matter with BatchNorm?* → Less — normalization re-centers activations mid-network — but yes for the first steps and norm-free models.

---

## 12. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| ReLU-family hidden layers | **He (`he_normal`)** | compensates for ReLU halving the variance |
| tanh / sigmoid layers | **Xavier / Glorot** | balances forward+backward variance for symmetric activations |
| SELU (self-normalizing) | **LeCun** | matches SELU's variance-preserving assumption |
| Fast prototype, minimal tuning | **Adam / AdamW** | robust out-of-the-box, adaptive per-parameter LR |
| Transformers / LLMs | **AdamW + warmup + cosine** | stable at scale; correct weight decay |
| Vision CNNs, best generalization | **SGD + momentum(0.9)** | often generalizes better than Adam; pairs with step/cosine decay |
| Very sparse features / embeddings | **AdaGrad** | large early steps for rarely-seen parameters |
| Exploding gradients (RNN/deep) | **any + gradient clipping** | caps the step regardless of optimizer |

**Decision rule:** choose the **initializer by activation** (He for ReLU, Xavier for tanh/sigmoid — this is a lookup, not a tuning knob), then choose the **optimizer by domain** (AdamW for transformers/NLP and quick iteration; SGD+momentum for vision when you want the last bit of generalization), and always pair it with a **schedule** (warmup→cosine for transformers, step/cosine for CNNs). Tune the **learning rate first** — it dominates everything else.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why is zero/constant weight initialization fatal, and what does random init fix?
   <details><summary>answer</summary>Symmetry: identical weights → every neuron in a layer produces the same output and gets the same gradient → they never differentiate, so the layer acts as one neuron. Random init breaks symmetry so neurons specialize. (Biases can still start at 0.)</details>
2. State Xavier and He initialization and when to use each.
   <details><summary>answer</summary>Scale `Var(w)` to fan-in to keep variance stable. **Xavier/Glorot** `Var(w)=2/(n_in+n_out)` (≈`1/n_in`) for tanh/sigmoid; **He** `Var(w)=2/n_in` for ReLU — the ×2 compensates for ReLU zeroing half the signal.</details>
3. Write the EMA update and explain `β`.
   <details><summary>answer</summary>`Vₜ = β·Vₜ₋₁ + (1−β)·gₜ`. `β` sets the memory: it averages ~`1/(1−β)` recent values (0.9→~10 steps). Higher β = smoother but laggier.</details>
4. How do Momentum, RMSprop, and Adam differ?
   <details><summary>answer</summary>Momentum = EMA of the gradient (smooths direction, builds velocity). RMSprop = EMA of the *squared* gradient, divide step by its root (per-parameter adaptive LR, damps oscillation). Adam = both + bias correction (`V̂=V/(1−β₁ᵗ)`, `Ŝ=S/(1−β₂ᵗ)`).</details>
5. Why does Adam need bias correction?
   <details><summary>answer</summary>`V` and `S` start at 0, so early EMA estimates are biased toward 0 (e.g. `V₁` is only `(1−β)g₁`). Dividing by `1−βᵗ` rescales them; the correction fades as `t` grows and `βᵗ→0`.</details>
6. Compute a learning-rate-decay step: `α₀=0.1, r=0.1, epoch=10`, schedule `α=α₀/(1+r·epoch)`.
   <details><summary>answer</summary>`α = 0.1/(1+0.1·10) = 0.1/2 = 0.05`.</details>
7. What's AdaGrad's failure mode and how does RMSprop fix it?
   <details><summary>answer</summary>AdaGrad **sums** squared gradients, so the denominator grows without bound and the effective LR decays to ~0 → training stalls. RMSprop uses an **EMA** (forgets old gradients), keeping the step size alive.</details>
8. Why is warmup needed for Adam on transformers?
   <details><summary>answer</summary>Early second-moment estimates `Ŝ` are unreliable, so a full-size adaptive step can destabilize/diverge training. Warmup ramps the LR up over the first steps to let the statistics settle before taking large steps.</details>

---

*Covers: symmetry-breaking and zero-init; variance-preserving init (`Var(z)=n_in·Var(w)·Var(x)`) → Xavier/Glorot (tanh/sigmoid) vs He/Kaiming (ReLU, ×2) vs LeCun (SELU), plus Keras defaults; why mini-batch GD is noisy; the EMA primitive + bias correction; GD-with-momentum (ball analogy, Nesterov, framework `(1−β)` nuance); RMSprop (+ AdaGrad ancestor); Adam (β₁/β₂ defaults, bias correction) and AdamW (decoupled weight decay); learning-rate schedules (time-based, step, exponential, cosine, warmup, ReduceLROnPlateau); gradient clipping, LR range test, batch↔LR scaling, optimizer-state memory, and reproducibility. Sourced from the Scaler "Weight Initialization & Optimizers" lecture (24Seven business case) + production practice. Foundations: [Neural Network Fundamentals](Neural%20Network%20Fundamentals.md).*
