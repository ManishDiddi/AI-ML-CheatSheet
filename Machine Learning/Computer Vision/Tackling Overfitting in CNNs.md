# Tackling Overfitting in CNNs — the systematic ladder from a memorizing model to one that generalizes

> **TL;DR.** A CNN overfits when it has far more parameters than training images — the classic symptom is a widening gap between rising train accuracy and stalling validation accuracy. You fix it with two orthogonal levers: **shrink effective capacity** (fewer/pooled params via GAP, BatchNorm, Dropout, L1/L2 weight penalties, early stopping) and **grow effective data** (augmentation). The senior move is to apply them *in order as a diagnostic ladder* — kill the parameter explosion first (Flatten→GAP), then regularize, then augment — reading the train/val curves after each step, not throwing all of them on at once.

**Where it fits:** the training-discipline layer that sits on top of any [CNN](Convolutional%20Neural%20Networks%20for%20Vision.md) (or any deep net). Vision overfits *hard* because image models are huge and labeled images are scarce, so this is where most of a vision project's accuracy is actually won or lost.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (conv/pool/GAP), [Batch Normalization & Dropout](../Neural%20Networks/Batch%20Normalization%20&%20Dropout.md), [Weight Initialization & Optimizers](../Neural%20Networks/Weight%20Initialization%20&%20Optimizers.md) (LR schedules), and L1/L2 penalties (see [Linear Regression](../Supervised%20ML/Linear%20Regression.md)). Bias–variance framing: [Ensemble Methods that Trade Off Bias vs Variance](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works — the regularization ladder](#3-how-it-works--the-regularization-ladder)
4. [Worked Example — the clothing-dataset journey](#4-worked-example--the-clothing-dataset-journey)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Production & MLOps Notes](#7-production--mlops-notes)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**The symptom.** Overfitting is a *gap*, not a single number. Read the two curves together:

```
Accuracy                              Diagnosis
  │           ┌─────── train ~99%     train HIGH, val LOW & flat  → OVERFIT (high variance)
  │          /                        train LOW,  val LOW         → UNDERFIT (high bias)
  │         /  ┌──── val ~55% (flat)  train ≈ val, both rising    → still learning; train longer
  │        /  /                       train ≈ val, both high      → 🎯 the goal
  │_______/__/___________ epoch
```

**The root cause (say this cold):** overfitting = *too many trainable parameters relative to the number of training samples*. A 2-conv-block CNN on a 128×128 clothing dataset has **~17M parameters but only ~3000 training images** — the model has enough capacity to *memorize* the training set (and its noise) rather than learn generalizable patterns.

**The two levers** — every technique in this note is one or the other:

```
        OVERFITTING = big model  ÷  small data
                          │             │
      ┌───────────────────┘             └──────────────────┐
      ▼                                                     ▼
1. MODEL PIPELINE                                  2. DATA PIPELINE
   shrink *effective* capacity                        grow *effective* data
   • GAP instead of Flatten (kill param explosion)    • augmentation
   • Dropout, BatchNorm                                  (flip/crop/rotate/jitter…)
   • L1 / L2 weight penalty
   • early stopping (implicit capacity limit)
   • simpler/shallower architecture
```

The discipline is to treat these as a **ladder you climb one rung at a time**, re-plotting train/val after each change so you know *which* rung moved the needle. Dumping BN + Dropout + L2 + augmentation on at once and hoping is how you end up unable to explain your own model.

---

## 2. The Formal Core

**Regularization taxonomy** — four places to inject the bias "prefer simpler models":

| Lever | Where it acts | Effect |
|---|---|---|
| **Architecture** (GAP, fewer/smaller layers) | model capacity | fewer params → less to memorize |
| **Weight penalty** (L1/L2) | loss function | pull weights toward 0 → smoother function |
| **Stochastic noise** (Dropout, BN, augmentation) | activations / data | forces redundancy → can't rely on any one feature/sample |
| **Training control** (early stopping, LR schedule) | optimization | stop before weights specialize to noise |

**L2 (weight decay / ridge):** `L = L_data + λ Σ wᵢ²`. Gradient adds `2λw` → every step shrinks weights toward (but not to) 0. Non-sparse; distributes influence across correlated features. Smaller weights → smaller Lipschitz constant → smoother, lower-variance function.

**L1 (lasso):** `L = L_data + λ Σ |wᵢ|`. The constant-magnitude gradient `λ·sign(w)` drives many weights **exactly to 0** → sparsity → implicit feature selection. In a CNN, rarely used on conv weights; more a general-ML tool.

**Dropout** (keep-prob `p`): at train, each unit is kept with probability `p`, else zeroed; this samples a random sub-network each step, so the full net behaves like an **ensemble of exponentially many thinned networks** sharing weights, which breaks *co-adaptation* (no neuron can depend on a specific partner always being present). At test, all units are active — with **inverted dropout** the kept activations are scaled by `1/p` at train time so train/test expectations match and inference needs no change. Dropout has **no learnable parameters**.

**Batch Norm** (per channel, over `batch×H×W`): `x̂ = (x−μ_B)/√(σ²_B+ε)`, then `y = γx̂ + β`. It regularizes as a *side effect* — each sample's normalization depends on the random minibatch it landed in, injecting noise like a mild dropout. It **does** add learnable params (`γ, β` per channel) plus **non-trainable** running `μ, σ²` used at inference. (Full mechanism: [Batch Normalization & Dropout](../Neural%20Networks/Batch%20Normalization%20&%20Dropout.md).)

**Data augmentation as an empirical-risk expander:** instead of minimizing loss over `N` samples, you minimize expected loss over a distribution of *label-preserving transforms* `T`: `E_{x,y} E_{T} [L(f(T(x)), y)]`. You're telling the model "these transforms don't change the class" — encoding an invariance prior for free.

**Early stopping ≈ implicit L2.** Halting gradient descent early keeps weights near their small initialization; for a quadratic loss this is provably equivalent to an L2 penalty whose strength falls as training proceeds. That's why it "just works" as a regularizer with zero tuning.

---

## 3. How It Works — the regularization ladder

Climb these **in order**; re-read the curves after each rung.

**Rung 0 — Locate the parameters.** Run `model.summary()` and find where the params live. In a Flatten→Dense CNN, **~99% of params sit in the first Dense layer**, not the convolutions — flattening a `64×64×16` map into 65,536 units and connecting to a 256-wide Dense costs `65,536×256 ≈ 16.8M` weights (no weight sharing in an MLP head). The convolutions themselves are tiny (~thousands). *You cannot regularize your way out of a 16M-param head — you delete it.*

**Rung 1 — Kill the param explosion: Flatten → Global Average Pooling.** Add a few more conv blocks to shrink the spatial map, then replace `Flatten` with `GlobalAveragePooling2D`: each `H×W×C` map collapses to one number per channel → a `C`-vector straight into the head. The 16.8M-param head becomes `256×256 ≈ 65K`. Overfitting often *vanishes* here — but watch for the opposite failure: the curves are still rising when training ends → **underfitting**, fix by training more epochs (with early stopping so you don't have to guess how many).

**Rung 2 — Add BatchNorm + Dropout.**
- **BatchNorm** after every Conv (and Dense): stabilizes activations, lets you use a higher LR, mildly regularizes.
- **Dropout** in the **head** (`p≈0.5` between Dense layers; `0.1–0.25` if used after pooling). In modern conv stacks, BN largely *replaces* dropout inside the conv body — dropout mostly lives in the FC head.
- **Placement caveat:** the BN paper put BN *before* the non-linearity, but empirically **Conv → ReLU → BN** (BN after activation) often trains better. Reason: if you normalize right before ReLU, ReLU then clamps all negatives to 0, re-skewing the distribution you just centered — defeating the normalization. Both orders appear in the wild; be able to argue it.
- **Symptom you'll now see:** the val curve gets *jittery*. Every regularizer injects noise (dropout flips connections each step), so some instability is expected — but **heartbeat/fluctuating curves = learning rate too high.** That's the cue for Rung 3.

**Rung 3 — Training control: the callback trio.** Don't hand-tune epochs or LR; let callbacks do it.
- `ReduceLROnPlateau(monitor=val_loss, factor=0.3, patience=5, min_lr=1e-5)` — when val loss stalls, cut LR (×0.3) so the optimizer stops bouncing around the minimum and settles into it. Monitor **val** loss (train loss always falls — useless as a plateau signal).
- `EarlyStopping(monitor=val_loss, patience=10, min_delta=0.001)` — stop when val loss stops improving by a meaningful amount; `min_delta` ignores noise-level "improvements," `patience` gives it room to escape plateaus. Pair with `restore_best_weights=True`.
- `ModelCheckpoint(monitor=val_accuracy, mode=max, save_best_only=True)` — persist the *best* weights so that even if the model overfits after the peak, you keep the peak. (Note the asymmetry: checkpoint on **val_accuracy/max**, early-stop on **val_loss/min** — loss is the smoother, earlier warning of overfitting; accuracy is what you ship on.)

**Rung 4 — Weight penalty: L2 on Conv + Dense.** Add `kernel_regularizer=l2(1e-3)` to conv and dense layers to shrink weights and squeeze out a few more points. (In Keras/SGD this equals weight decay; with Adam, L2-in-the-loss and true weight decay differ — prefer **AdamW** for decoupled decay. See §7.)

**Rung 5 — Grow the data: augmentation.** Apply random label-preserving transforms to the **training set only** (resize→random-crop, flip, small rotation/translation, brightness/contrast jitter). This is usually the single biggest lever when data is scarce — it's what finally breaks the 80% barrier on the clothing set. Crucial detail: augmentation layers fire **only in training mode**; val/test see the clean `resize+rescale` pipeline. (Exception: deliberate Test-Time Augmentation, §7.)

```
Baseline (17M params, Flatten)  ──▶  ~50% test, badly overfit
 + GAP + deeper convs           ──▶  overfit gone (now underfit → train longer)
 + BN + Dropout                 ──▶  better, but unstable (LR too high)
 + LR scheduler + early stop    ──▶  stable, higher
 + L2                           ──▶  a few more points
 + data augmentation            ──▶  🎯 >80% val — biggest single jump
```

---

## 4. Worked Example — the clothing-dataset journey

10-class clothing classifier, 128×128 RGB, ~3000 train images.

| Step | Change | Params | Behaviour |
|---|---|---|---|
| Baseline | `Conv(16)→MaxPool→Flatten→Dense(256)→Dense(10)` | **~16.8M** (99% in Dense₁) | ~50% test, train≫val (overfit) |
| Mod 1 | conv stack 16→32→64→128→256, **GAP** replaces Flatten | **~0.4M** | overfit gone; underfits at 10 epochs → 30 epochs |
| Mod 2 | + BatchNorm (every conv+dense) + Dropout(0.5) in head | ~0.4M | higher but jittery val curve |
| Mod 3 | + ReduceLROnPlateau + EarlyStopping + Checkpoint | — | stable, smooth convergence |
| Mod 4 | + L2(1e-3) on conv+dense | ~0.4M | +a few % test |
| Mod 5 | + augmentation (resize 156→RandomCrop 128, jitter) | ~0.4M | **>80% val**, best test |

**The param arithmetic that matters:** the entire fix in Mod 1 is `Flatten(64×64×16=65,536)→Dense(256)` costing **16.8M** vs `GAP(256)→Dense(256)` costing **65K** — a **~250× reduction in the head**, with the conv body doing the real feature work. Memorize this: *the parameter explosion in a naive CNN is in the Flatten→Dense join, and GAP is the structural fix.*

---

## 5. Code / Implementation

**The final architecture (Keras) — GAP + BN + Dropout + L2, the whole ladder baked in:**
```python
from tensorflow.keras import layers, regularizers, Sequential

def cnn(height=128, width=128, n_classes=10, wd=1e-3):
    l2 = regularizers.l2(wd)
    def block(f):                                   # Conv → ReLU → BN → Pool
        return [layers.Conv2D(f, 3, padding="same", kernel_regularizer=l2),
                layers.Activation("relu"),
                layers.BatchNormalization(),        # after activation (empirically better here)
                layers.MaxPooling2D()]
    return Sequential(
        [layers.Input((height, width, 3))]
        + block(16) + block(32) + block(64) + block(128)
        + [layers.Conv2D(256, 3, padding="same", kernel_regularizer=l2),
           layers.Activation("relu"), layers.BatchNormalization(),
           layers.GlobalAveragePooling2D(),         # ← kills the Flatten→Dense param explosion
           layers.Dense(256, kernel_regularizer=l2),
           layers.Activation("relu"), layers.BatchNormalization(),
           layers.Dropout(0.5),                     # dropout lives in the HEAD, after BN
           layers.Dense(n_classes, activation="softmax")])
```

**The callback trio — let training tune itself instead of guessing epochs/LR:**
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-5),  # settle into minima
    EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-3,
                  restore_best_weights=True),                                     # stop + rewind to best
    ModelCheckpoint("best.weights.h5", monitor="val_accuracy", mode="max",
                    save_best_only=True, save_weights_only=True),                 # keep the peak
]
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

**Augmentation as a training-only preprocessing stack (train sees random crops; val/test see clean):**
```python
augment = Sequential([                       # applied to TRAIN only
    layers.Resizing(156, 156),               # upsize, then...
    layers.RandomCrop(128, 128),             # ...random 128 crop = translation invariance for free
    layers.RandomFlip("horizontal"),         # clothes are L/R symmetric → label-preserving
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2), layers.RandomContrast(0.2),
    layers.Rescaling(1/255.),
])
clean = Sequential([layers.Resizing(128, 128), layers.Rescaling(1/255.)])  # VAL/TEST

train_ds = train_data.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_data.map(  lambda x, y: (clean(x),   y))   # NEVER augment val/test (except TTA)
```
Keras augmentation layers are inert at inference automatically, but keeping augment/clean as separate pipelines makes the train-only intent explicit and leak-proof. (PyTorch: the equivalents are `torchvision.transforms` in the *train* `Dataset` only, with `nn.Dropout`, `nn.BatchNorm2d`, `weight_decay=` in the optimizer, and a `ReduceLROnPlateau`/`CosineAnnealing` scheduler.)

---

## 6. When It Breaks

```
❌ Augmentation that CHANGES the label — the #1 trap.
   • vertical-flip a "6" → "9"; horizontal-flip text/street-signs → garbage labels
   • hue-shift when color IS the class (ripe vs unripe fruit, blood vs rust)
   • aggressive rotation on digits/characters
   → Rule: an augmentation is valid only if a human would give the transformed image the SAME label.

❌ Over-regularization → underfitting. Too much dropout + heavy L2 + brutal augmentation can push
   train accuracy DOWN to meet val — both mediocre. Regularize until the gap closes, then STOP.

❌ Data leakage via augmentation ordering. Augment AFTER the train/val/test split, never before —
   augmenting then splitting puts transformed copies of the same image on both sides → inflated val.

❌ BatchNorm–Dropout disharmony ("variance shift", Li et al. 2019). Dropout BEFORE a BN layer changes
   the activation variance between train (dropout on) and test (dropout off), so BN's running stats are
   wrong at inference → accuracy drops. Fix: put Dropout AFTER all BN, or don't interleave them —
   which is exactly why modern nets keep BN in the conv body and Dropout only in the FC head.

❌ "Val accuracy > train accuracy?!" Not a bug. Train accuracy is measured WITH dropout on and on
   AUGMENTED (harder) images; val is measured clean with dropout off. The handicap is on the train side.

❌ BatchNorm with tiny batches. BN's statistics get noisy for batch size ≲ 8 (common with big images) →
   unstable training. Use GroupNorm / LayerNorm instead, or accumulate/sync BN across devices.

❌ Fluctuating ("heartbeat") loss curves = LR too high, not "needs more regularization." Lower the LR /
   add a scheduler first before reaching for more dropout.
```

---

## 7. Production & MLOps Notes

**Modern augmentation beats hand-picked transforms.** The lecture's geometric/photometric ops are the floor; in production reach for:
- **Cutout / Random Erasing** — mask a random rectangle → forces the net to use the whole object, not one discriminative patch.
- **Mixup** — train on convex blends `λ·xᵢ+(1−λ)·xⱼ` with correspondingly blended labels → smoother decision boundaries, better calibration.
- **CutMix** — paste a patch of image B into image A and mix labels *by area* → localization-friendly regularizer, usually > Mixup for classification.
- **RandAugment / AutoAugment / TrivialAugment** — apply random ops at random magnitudes from a policy; RandAugment has just 2 knobs (N ops, magnitude M) and is the practical default.
- **Conv-specific structured dropout:** `SpatialDropout2D` (drop whole channels — plain dropout is weak in convs because neighboring pixels are correlated and "fill in" for a dropped one), **DropBlock** (drop contiguous spatial regions), **Stochastic Depth** (randomly skip whole residual blocks — standard in EfficientNet/ResNet training).
- **Label smoothing** (`ε≈0.1`) — soften one-hot targets → curbs overconfidence and helps calibration; cheap and almost always on.

**Weight decay ≠ L2 under Adam.** L2-in-the-loss and decoupled weight decay are identical for SGD but *not* for Adam (the adaptive denominator rescales the L2 gradient). Use **AdamW** so decay is applied directly to weights — meaningfully better generalization for the same nominal λ. `(certain)`

**Test-Time Augmentation (TTA).** At inference, run several augmented views (flips, crops) and average the softmaxes → a small, reliable accuracy bump for a linear latency cost. The one time you *do* augment val/test.

**Where augmentation actually helps.** Biggest wins on **small datasets**; returns diminish as data grows (a 10M-image set already contains the variations you'd synthesize). Always cheaper than collecting/labeling more data, so it's the first thing to try — but it's not free accuracy forever.

**Reproducibility & monitoring.** Seed the augmentation RNG for repeatable runs; log the exact augmentation policy as a versioned artifact (it's a hyperparameter). Track the **train–val gap** as a first-class metric in your dashboards — a widening gap over retrains is your earliest overfitting/drift alarm. Do augmentation **on-the-fly on CPU workers** (prefetch) so the GPU never stalls; reserve offline/materialized augmentation for expensive transforms only.

---

## 8. Interview Lens

> ⚡ Diagnose before prescribing: name the symptom (train↑/val-flat gap), then the *root cause* (params ≫ samples), then the two-lever fix. Interviewers want the ordered ladder, not a bag of tricks.

**"Your CNN overfits — walk me through fixing it."** → 🎯 *"First I read the train/val curves to confirm it's variance not bias, then I check `model.summary()` — in a naive CNN ~99% of params are in the Flatten→Dense head, so I replace Flatten with Global Average Pooling to kill that explosion. Then I add BatchNorm + Dropout, let a LR scheduler + early stopping control training, add L2/weight decay, and finally data augmentation — which is usually the biggest single jump when data is limited. I add them one at a time and re-check the curves so I know which lever paid off."*

**Likely follow-ups:**
- *Flatten vs GAP?* → Flatten→Dense has no weight sharing → millions of params + overfitting; GAP → one value per channel → tiny head, structural regularization. `(certain)`
- *Do BN and Dropout add trainable params?* → BN yes (`γ,β` per channel, plus non-trainable running stats); Dropout no; MaxPool no. Say all three unprompted.
- *Where do you place Dropout in a CNN?* → In the FC head (`p≈0.5`), not the conv body; BN covers the conv body. Interleaving Dropout before BN causes variance shift.
- *L1 vs L2 for a CNN?* → L2/weight-decay by default (smooth shrinkage); L1 only if you want sparsity/feature selection, rare on conv weights.
- *An augmentation that would hurt?* → any that changes the label: vertical-flip a digit, h-flip text, hue-shift when color is the class.
- *Early stopping monitors which metric, and why?* → val **loss** (smoother, earlier overfitting signal); checkpoint on val **accuracy** (what you ship).
- *Fluctuating val curve — cause?* → LR too high → add ReduceLROnPlateau / lower LR, not more dropout.
- *Adam + L2 gotcha?* → they're not equivalent under Adam; use AdamW for correct decoupled weight decay.

---

## 9. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Param explosion in the head | **GAP** (drop Flatten) | structural fix, biggest single reduction |
| Deep conv net, unstable training | **BatchNorm** | stabilizes + mild regularization + higher LR |
| Small labeled dataset | **Augmentation** (then Mixup/CutMix/RandAugment) | grows effective data — the top lever when data-limited |
| Overconfident / miscalibrated | **Label smoothing**, Mixup | softer targets, better calibration |
| Tiny batches (big images) | **GroupNorm/LayerNorm** over BN | BN stats too noisy at batch ≲ 8 |
| Very deep residual net | **Stochastic Depth**, DropBlock | structured regularization designed for depth/convs |
| Don't know how long to train | **EarlyStopping + Checkpoint** | zero-tuning capacity control, keeps best weights |
| Still overfitting after all of the above | **more data / transfer learning** | pretrained features need far fewer labels → [Transfer Learning](Transfer%20Learning.md) |

**Decision rule:** *shrink the model until it stops memorizing, then grow the data until it stops underfitting.* When you've exhausted both and still fall short, the real answer is usually **transfer learning** — start from ImageNet features instead of fighting overfitting from scratch.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. State the one-line root cause of overfitting and the two orthogonal levers to fix it.
   <details><summary>answer</summary>Too many trainable params relative to training samples. Levers: (1) shrink *effective capacity* (GAP, BN, Dropout, L1/L2, early stopping, smaller net); (2) grow *effective data* (augmentation).</details>
2. In a naive `Conv→MaxPool→Flatten→Dense(256)` CNN on 128×128 images, where do ~99% of the parameters live, and what's the fix?
   <details><summary>answer</summary>The `Flatten→Dense` join (e.g. `65,536×256 ≈ 16.8M`) — no weight sharing in the MLP head. Fix: replace Flatten with **Global Average Pooling** → `256→Dense(256) ≈ 65K`, a ~250× cut.</details>
3. Which of BatchNorm, Dropout, MaxPool have trainable parameters?
   <details><summary>answer</summary>Only **BatchNorm** (`γ, β` per channel + non-trainable running mean/var). Dropout and MaxPool have none.</details>
4. You add Dropout + BN and the val curve starts fluctuating wildly. Cause and fix?
   <details><summary>answer</summary>Learning rate too high (regularizers add noise, high LR amplifies it). Fix: `ReduceLROnPlateau` / lower LR — not more regularization.</details>
5. Give two augmentations that would *hurt*, and why.
   <details><summary>answer</summary>Vertical-flip a digit (6↔9) and horizontal-flip text/signs — both change the correct label. Also hue-shift when color defines the class. Valid aug ⇔ a human gives the transformed image the same label.</details>
6. Why can validation accuracy exceed training accuracy, and why is early stopping monitored on val *loss* but checkpointing on val *accuracy*?
   <details><summary>answer</summary>Train metric is measured with dropout on and on augmented (harder) images; val is clean with dropout off. Early-stop on val **loss** (smoother, earlier overfit signal); checkpoint on val **accuracy** (the deployment metric).</details>
7. Under Adam, why prefer AdamW over `kernel_regularizer=l2(...)`?
   <details><summary>answer</summary>L2-in-the-loss and weight decay coincide for SGD but not Adam — Adam's adaptive denominator rescales the L2 gradient. AdamW applies decay directly to weights (decoupled) → better generalization.</details>

---

*Covers: overfitting symptom/diagnosis · overparametrization root cause · model-vs-data levers · Flatten→GAP param explosion · BatchNorm & Dropout (placement, params, disharmony) · L1/L2 & weight decay vs AdamW · LR scheduler + EarlyStopping + ModelCheckpoint trio · data augmentation (basic → Mixup/CutMix/RandAugment/Cutout/SpatialDropout/DropBlock/Stochastic-Depth) · label smoothing · TTA · leakage & reproducibility · the ordered regularization ladder.*
