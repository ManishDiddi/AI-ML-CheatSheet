# Transfer Learning — reuse a pretrained backbone instead of training vision from scratch

> **TL;DR.** A CNN trained on ImageNet has already learned *generic* visual features (edges → textures → parts) in its convolutional layers; only the final classifier is task-specific. Transfer learning **reuses that frozen conv backbone and swaps in a new head** — so you get high accuracy on a small dataset in minutes instead of failing to train a SOTA net from scratch. The single decision that governs everything is a 2×2 of **how much data you have × how similar your domain is to ImageNet**, which sets how much of the backbone you freeze vs fine-tune.

**Where it fits:** the default starting point for *almost every* real vision task — you rarely train from scratch. It's the escalation you reach for when [augmentation](Tackling%20Overfitting%20in%20CNNs.md) alone can't beat overfitting on a small dataset.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (conv/GAP, the LeNet→EfficientNet architecture evolution), [Tackling Overfitting in CNNs](Tackling%20Overfitting%20in%20CNNs.md), and BatchNorm behaviour at train vs inference ([Batch Normalization & Dropout](../Neural%20Networks/Batch%20Normalization%20&%20Dropout.md)) — critical for the frozen-backbone gotcha in §8.

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../../_STUDY%20LOOP.md))*


> ### ⚡ Fast Pass — 5 minutes
>
> **Model.** A conv trunk learns edges → textures → parts, and that hierarchy is **generic**; only the classifier head is task-specific. So: keep the pretrained trunk, delete its 1000-class head, bolt on your own.
>
> **Core.** Two regimes — **feature extraction** (backbone frozen, only the head trains, normal LR) and **fine-tuning** (top blocks unfrozen, LR ≤ `1e-4`). Which one you're in is set by a 2×2 of **dataset size × domain similarity**: *small + similar → freeze more; large + different → freeze less*. Always warm the head up frozen **first**, then unfreeze — a random head's gradients will otherwise overwrite good weights.
>
> **Backbones.** `ResNet50` = the default (ecosystem, not accuracy) · `EfficientNet-B0` = best accuracy per parameter, +2.2 pts on ResNet50 at 1/5 the size · `MobileNetV3` = edge/latency · `InceptionV3`/`Xception` = +3 pts at the same size, but **299×299** · `VGG16` = perceptual loss only · `AlexNet` = history.
>
> **Traps.** ① **Preprocessing mismatch** — resolution *and* normalisation are properties of the checkpoint (224 / 299 / 380; `[-1,1]` vs mean-std vs Keras-EfficientNet's raw `[0,255]`). ② `trainable=False` freezes the **weights but not the BatchNorm running statistics**. ③ Fewer FLOPs ≠ faster — depthwise convs are memory-bandwidth bound.
>
> 🎯 **Kill-shot.** *"Freeze the ImageNet trunk and retrain only the head — 700 images goes from 11% to 85%. And 'frozen' has to mean the BatchNorm running statistics too, not just the weights."*

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — feature extraction vs fine-tuning](#2-the-formal-core--feature-extraction-vs-fine-tuning)
3. [How It Works](#3-how-it-works)
4. [Worked Example — landmarks: 11% from scratch → 85% transferred](#4-worked-example--landmarks-11-from-scratch--85-transferred)
5. [The Backbone Zoo — what each architecture actually solved](#5-the-backbone-zoo--what-each-architecture-actually-solved)
   · [AlexNet](#51-alexnet-2012--proof-that-deep-cnns-work-at-all) · [VGG](#52-vgg-16--vgg-19-2014--depth-and-the-death-of-the-big-filter) · [Inception](#53-googlenet--inception-2014--stop-choosing-filter-sizes-and-kill-the-fc-head) · [ResNet](#54-resnet-2015--the-degradation-problem-and-the-shortcut-that-fixed-it) · [MobileNet](#55-mobilenet-v1--v2--v3-201719--the-phone-budget) · [EfficientNet](#56-efficientnet-2019--scaling-stops-being-guesswork) · [What came after](#57-what-came-after--the-three-you-should-be-able-to-name)
6. [Choosing & Fine-Tuning a Backbone](#6-choosing--fine-tuning-a-backbone)
7. [Code / Implementation](#7-code--implementation)
8. [When It Breaks](#8-when-it-breaks)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**Why training from scratch fails on small data.** VGG16 has ~138M parameters; the landmark dataset has ~700 training images. Training that net from random init lands at **~11% test accuracy on 10 classes — barely above the 10% random-guess floor.** There simply isn't enough signal to fit 138M weights. Collecting more labeled images is slow and expensive. Transfer learning sidesteps the whole problem.

**The key insight — features are generic near the input, specific near the output:**

```
     ImageNet-pretrained CNN
INPUT ─▶ [conv early] ─▶ [conv mid] ─▶ [conv late] ─▶ [FC head] ─▶ 1000 ImageNet classes
          edges,          textures,     object          "is this a
          colors          patterns      parts            golden retriever?"
        └──────── GENERIC (transfers to ANY vision task) ────┘   └─ TASK-SPECIFIC ─┘
                              REUSE THESE (freeze)                  THROW AWAY, replace
```

![The freeze-versus-fine-tune strategy on a pretrained backbone — early conv blocks holding generic edge and texture detectors stay frozen and reused as-is, the late blocks that encode task-specific object parts are optionally fine-tuned at a tiny learning rate, and only the new head trains on your classes; how much you unfreeze slides with dataset size and domain gap.](attachments/transfer-learning-freeze-finetune-strategy.png)

Edge and texture detectors are useful for *any* image task — landmarks, X-rays, satellite tiles. Only the last layers encode "these features → ImageNet's 1000 classes." So you **keep the conv backbone, delete the ImageNet head, and bolt on a fresh head for your classes.** The backbone stays frozen (the majority of the weights, already good); only the small new head trains → fast, data-efficient, high accuracy.

🎯 *"Transfer learning works because the convolutional backbone learns a generic edge→texture→part hierarchy that's reusable across vision tasks — I freeze that and only retrain the task-specific head, so 700 images is enough to hit 85% where from-scratch gives 11%."*

---

## 2. The Formal Core — feature extraction vs fine-tuning

Two regimes, distinguished by whether backbone weights update:

| | **Feature extraction** | **Fine-tuning** |
|---|---|---|
| Backbone | **frozen** (`trainable=False`) | **unfrozen** (some/all blocks train) |
| What trains | only the new head | head **+** upper backbone blocks |
| Learning rate | normal | **very low** (1e-4/1e-5) — don't wreck good weights |
| Data needed | little | more (else you overfit the unfrozen weights) |
| Cost | cheap (backbone is a fixed feature map) | expensive (full backprop) |
| Use when | small data, similar domain | more data and/or domain differs from ImageNet |

**The decision matrix — memorize this 2×2** (data size × domain similarity to the pretrained source):

```
                        SIMILAR domain              DIFFERENT domain
                 ┌──────────────────────────┬──────────────────────────┐
  SMALL dataset  │ Freeze backbone,         │ Freeze EARLY layers only, │
                 │ train new head only.     │ train head + a few late   │
                 │ (classic feature         │ blocks. Late features are │
                 │  extraction)             │ ImageNet-specific → adapt. │
                 ├──────────────────────────┼──────────────────────────┤
  LARGE dataset  │ Fine-tune the whole net  │ Fine-tune everything (or  │
                 │ at a low LR — you can    │ even train from scratch — │
                 │ afford to adapt.         │ enough data, ImageNet may │
                 │                          │ not help / can hurt).     │
                 └──────────────────────────┴──────────────────────────┘
```

**Rule of thumb:** *small + similar → freeze more; large + different → freeze less.* The two axes independently push toward freezing (little data can't safely update many weights) or fine-tuning (big domain gap means late features must adapt).

**Discriminative / layer-wise learning rates.** When you do fine-tune, use **lower LRs for earlier layers** (their generic features barely need to change) and higher for later layers/head. Early conv ≈ 1e-6, late conv ≈ 1e-5, head ≈ 1e-3. This protects the general-purpose features while letting task-specific ones move.

---

## 3. How It Works

**Feature-extraction pipeline (the 90% case):**

1. **Match the input contract.** Resize to the backbone's expected size (224×224 for VGG/ResNet) and apply the **exact preprocessing the backbone was trained with** — for torchvision that's the ImageNet channel mean/std; for `keras.applications` it's the model's own `preprocess_input`. Mismatch silently tanks accuracy (§8), and the exact contract differs per backbone (§5).
2. **Load the backbone without its head:** `include_top=False` (Keras) / `num_classes=0` or slice off `fc` (PyTorch). You get the conv feature extractor ending in a feature map (e.g. `7×7×512` for VGG).
3. **Freeze it:** `backbone.trainable = False`. Its weights are now constants; forward passes just compute features.
4. **Attach a new head** sized to *your* classes: `GAP → Dense(n_classes, softmax)` (prefer GAP over Flatten — see §5.2's VGG FC bottleneck). 
5. **Train only the head** with a normal optimizer/LR. Converges in a few epochs because you're fitting a tiny number of params on top of fixed features.

**Optional fine-tuning stage (squeeze out more, if you have the data):**
6. **Unfreeze the top blocks** of the backbone (leave early layers frozen).
7. **Recompile with a very low LR** (1e-4 or less) and continue training. Recompiling is mandatory in Keras after changing `trainable` flags. Consider **gradual unfreezing** — unfreeze one block at a time from the top — to avoid destabilizing the pretrained weights.

```
Stage 1 (feature extraction):   [FROZEN backbone] → [new head]      train head, normal LR
Stage 2 (fine-tuning):          [frozen early | UNFROZEN late] → [head]   tiny LR, adapt late features
```

**Why freeze first, then fine-tune?** A randomly-initialized head produces large, noisy gradients at the start. If the backbone is unfrozen during those first steps, those garbage gradients flow back and **corrupt the good pretrained weights** ("catastrophic forgetting"). Warming up the head while frozen, *then* unfreezing at a low LR, avoids this.

---

## 4. Worked Example — landmarks: 11% from scratch → 85% transferred

10-class world-landmark classifier, ~70 train images/class (tiny).

| Approach | Setup | Result |
|---|---|---|
| **VGG16 from scratch** | random init, SGD, 5 epochs | **~11.6% test** (≈ random for 10 classes) |
| **VGG16 transfer** | ImageNet weights, `include_top=False`, freeze conv base, `GAP/Flatten → Dense(10)`, train head 5 epochs | **~84.5% val** |

Same architecture, same data, same epochs — the *only* difference is initialization from ImageNet features. That 11% → 85% jump is the entire argument for transfer learning.

![Transfer-learning training curves on the 10-class landmark set — validation accuracy climbs past 0.84 in just five epochs while train and validation loss fall together and stay close, the healthy well-fit signature of a frozen ImageNet backbone, versus the roughly 11 percent a from-scratch VGG reaches on the same data.](attachments/transfer-learning-training-curves.png)

**Same experiment, different backbone, and the ranking changes.** Which backbone you start from is its own decision with its own trade-offs — that is all of [§5](#5-the-backbone-zoo--what-each-architecture-actually-solved) and [§6](#6-choosing--fine-tuning-a-backbone). VGG16 is used here because it makes the from-scratch failure vivid, not because it's what you'd ship.

**Metric aside — top-k accuracy.** For fine-grained classes (pickup-truck vs minivan) where even human labelers disagree, report **top-k accuracy** (correct if the true label is among the model's top *k* guesses). ImageNet is benchmarked on top-5 precisely because top-1 penalizes near-ties that aren't real errors; it's also the natural metric when you'll surface *k* candidates (search, recommendations).

---

## 5. The Backbone Zoo — what each architecture actually solved

You will be asked *"which pretrained model would you use, and why?"* The wrong answer is a name. The right answer names a **constraint** — accuracy, latency, memory, domain gap — and then the architecture that was designed for it. Every backbone below exists because the previous one hit a specific wall.

![The backbone family tree read as a chain of problems and fixes — AlexNet proves deep CNNs work but guesses its filter sizes, VGG standardises on 3x3 and goes deep but buries 120M parameters in its FC head, Inception runs multiple filter sizes in parallel and replaces the FC head with global average pooling, ResNet adds the identity shortcut that finally lets depth past 20 layers, MobileNet splits convolution into depthwise plus pointwise to fit a phone, and EfficientNet scales depth width and resolution together with one coefficient.](attachments/backbone-lineage-problem-solved.png)

**All numbers in this section come from one benchmark — [Keras Applications](https://keras.io/api/applications/) — so they're comparable to each other.** Published top-1/top-5 figures for the same model vary by a few points across sources (single-crop vs 10-crop, 224 vs 299 input, different training recipes). Treat them as **ordinal, not absolute**: don't be thrown when a lecture slide lists VGG16 at 74.4% and this table says 71.3%. `(certain)`

---

### 5.1 AlexNet (2012) — proof that deep CNNs work at all

**The problem it solved.** Before 2012, vision was hand-crafted features (SIFT, HOG) fed to an SVM. Deep nets were widely considered untrainable on real images. AlexNet won ILSVRC 2012 with **15.3% top-5 error against 26.2% for the runner-up** — a ~10-point gap that ended the hand-crafted-feature era overnight.

**What was actually new** (none of it is the architecture — that's the point):
- **ReLU instead of tanh/sigmoid** → no saturation, ~6× faster convergence. This is the single change that made depth trainable.
- **Dropout** in the FC layers → the first workable regulariser for a 60M-param net.
- **Two-GPU training** (GTX 580, 3 GB each) with the network literally split across them.
- **Heavy augmentation** (random crops, flips, PCA colour jitter).

**Shape:** 5 conv + 3 FC = 8 layers, **60M params**, `11×11` stride-4 first conv (a huge, coarse filter chosen by hand), then `5×5`, then `3×3`.

> **Transfer verdict: ❌ never.** It survives in `torchvision` for history only. **56.5% top-1 for 233 MB** — MobileNetV2 beats it by ~15 accuracy points at 1/16 the download. Know the *story* (ReLU + dropout + GPUs), not the checkpoint.

---

### 5.2 VGG-16 / VGG-19 (2014) — depth, and the death of the big filter

**The problem it solved.** AlexNet's filter sizes were guesswork. VGG asked: *hold everything else fixed, use one filter size everywhere, and just add depth — what happens?*

**The answer: use only `3×3`, and depth is what pays.** Two facts you should be able to derive on a whiteboard:

```
Two stacked 3×3 convs  vs  one 5×5 conv        Three stacked 3×3  vs  one 7×7
  same 5×5 receptive field                       same 7×7 receptive field
  2·(3·3·C·C) = 18C²  vs  25C²  params           3·(3·3·C·C) = 27C²  vs  49C²
  TWO non-linearities vs one                     THREE non-linearities vs one
  → fewer weights AND more expressive            → 45% fewer weights
```

**But VGG's own numbers indict it.** ~**120M of its 138M params live in the three FC layers** — the first one alone is `(7·7·512 + 1) · 4096 ≈ 102M`. The conv trunk you actually want to transfer is only ~20M. That's why the checkpoint is **528 MB**.

**And depth alone had already topped out:** VGG-19 adds three more conv layers over VGG-16 and scores **71.3% top-1 — identical**. Three extra layers bought literally nothing. That plateau is the observation ResNet was built to explain.

> **Transfer verdict: ⚠️ rarely — but not never.** For classification it's dominated on every axis (see the Pareto plot in §6). It stays alive for one real reason: **perceptual / style loss.** Style transfer, super-resolution and LPIPS all measure distance in VGG feature space, because those features are smooth, well-studied, and every paper baselines against them — reproducibility beats efficiency there. It's also the teaching default, which is why §4's worked example uses it.
>
> 🚨 **Gotcha:** the original VGG weights are Caffe-style — **BGR channel order, mean-subtracted, not scaled to [0,1]**. `keras.applications.vgg16.preprocess_input` does this for you; hand-rolled normalisation silently costs you several points.

---

### 5.3 GoogLeNet / Inception (2014) — stop choosing filter sizes, and kill the FC head

**Two problems solved at once.**

*Problem 1 — which filter size?* `5×5` catches big objects, `3×3` catches small distributed detail, `1×1` catches per-pixel channel structure. You can't know in advance which a layer needs. **Inception's answer: don't choose. Run all of them in parallel and concatenate along the channel axis** — the next layer learns which branch to weight. One layer becomes a **multi-scale feature extractor**.

The intuition the lecture uses: a photo of *wheat, people, sky*. The sky fills the frame and wants a large kernel; a person's face is small and wants a small one. A single filter size has to compromise; an Inception module doesn't.

*Problem 2 — VGG's parameter bloat.* Two fixes:
- **`1×1` convolutions as a cheap bottleneck** *before* the expensive `3×3`/`5×5` branches. A `1×1` conv is a learned weighted mix across channels at one pixel — it costs almost nothing and lets you squeeze 192 channels down to 16 before the `5×5` ever runs.
- **Global Average Pooling instead of `Flatten → FC`** for the head: `7×7×1024 → 1024`, deleting the ~120M-param FC block entirely.

![The Inception module with and without 1x1 bottlenecks — in the naive version every branch reads all 192 input channels so the 5x5 branch alone costs 153,600 weights and about 120M multiply-adds, while inserting a 1x1 convolution that first reduces 192 channels to 16 drops the same branch to 15,872 weights and about 12.4M multiply-adds for the same output shape.](attachments/inception-module-1x1-bottleneck.png)

Result: **GoogLeNet ships ≈7M parameters — the paper's headline is 12× fewer than AlexNet — while being more accurate than both AlexNet and VGG.**

**One more trick worth naming:** GoogLeNet bolts **auxiliary classifiers** onto two intermediate layers during training (their losses added at weight `0.3`, discarded at inference). This was a pre-ResNet hack to inject gradient into early layers when backprop through 22 layers was still fragile. Later analysis concluded they act mostly as regularisers. `(likely)`

**The family line:** v1 (GoogLeNet) → **v2/v3** (factorise `5×5` into two `3×3`, then `n×n` into `1×n` + `n×1`; add BatchNorm and label smoothing; input moves to `299×299`) → v4 / Inception-ResNet (bolt residuals on) → **Xception**, which pushes the Inception hypothesis to its limit: *fully* decouple spatial from channel mixing. That limit **is** the depthwise-separable convolution — Xception is the bridge from Inception to MobileNet.

> **Transfer verdict: ✅ a genuinely good workhorse.** InceptionV3 is **23.9M params / 92 MB / 77.9% top-1** — it beats ResNet50 by **3.0 points at slightly fewer parameters**. Reach for it (or Xception, 79.0%) when you want more accuracy than ResNet50 without changing your cost envelope.
>
> 🚨 **Gotcha:** InceptionV3 and Xception expect **`299×299`**, not `224×224`, and their `preprocess_input` scales to **`[-1, 1]`** — *not* ImageNet mean/std. Feed one a 224px image normalised the ResNet way and you lose several points with no error message.
>
> Side note: InceptionV3 is also the network behind **FID and Inception Score** — see [GANs](GANs.md).

---

### 5.4 ResNet (2015) — the degradation problem, and the shortcut that fixed it

**The problem it solved — and almost everyone states it wrong.** ResNet did *not* fix overfitting. He et al. observed that a **56-layer plain CNN had higher *training* error than a 20-layer one.** Higher *training* error. If it were overfitting, training error would have gone *down*. 🎯 **Deep plain nets were failing to optimise, not failing to generalise — that's the "degradation problem," and naming it correctly is what separates a real answer from a memorised one.**

**The fix.** Make each block learn a **residual** instead of a whole mapping:

```
y = F(x) + x          F = the conv stack,  x = identity shortcut

Optimisation view:  if the best thing a block can do is nothing, a plain stack has to
                    LEARN the identity through two convs (hard). A residual block just
                    drives F(x) → 0 (easy). Useless depth becomes free.

Gradient view:      ∂y/∂x = ∂F/∂x + 1
                    That "+1" is a highway. Backprop multiplies many such terms together;
                    with the +1 present the product cannot collapse to zero.
```

![The three ResNet block types side by side — the basic block used in ResNet-18 and 34 stacks two 3x3 convolutions at full width, the bottleneck block used in ResNet-50 and deeper squeezes 256 channels to 64 with a 1x1 convolution before the 3x3 and expands back afterwards, and the projection block puts a 1x1 stride-2 convolution on the shortcut itself so the input can still be added when the spatial size and channel count change.](attachments/resnet-basic-bottleneck-projection-blocks.png)

**Two block types, and why the depth ladder splits where it does:**

| | **Basic block** | **Bottleneck block** |
|---|---|---|
| Shape | `3×3 → 3×3` at full width | `1×1 squeeze → 3×3 → 1×1 expand` |
| Used by | ResNet-**18 / 34** | ResNet-**50 / 101 / 152** |
| Why | simple, fine at 64–512 channels | runs the costly `3×3` in a thin space → same cost, **3× the layers** |

**Two shortcut types.** When input and output shapes match, the shortcut is a pure **identity** — zero parameters, gradient exactly 1. When a stage downsamples (stride 2, channels change), `x` can't be added as-is, so a **`1×1` stride-2 projection conv sits on the shortcut**. Every ResNet stage begins with one.

**The descendants worth naming:** **ResNetV2** (pre-activation — reorder to `BN → ReLU → conv`, giving a completely clean identity path, ~1 point better); **ResNeXt** (grouped convs); **Wide ResNet** (fewer, fatter blocks). And an important humility point: **ResNet-50 reaches ~80% top-1 with a modern training recipe** (longer schedules, RandAugment, label smoothing, EMA) versus the 74.9% of the original recipe — *most of the gap between 2015 and 2020 architectures was training recipe, not architecture.* `(likely)`

> **Transfer verdict: ✅✅ the default, and it isn't close.** ResNet-50 is the most-transferred backbone in existence — not because it wins any accuracy-per-FLOP contest (it doesn't), but because of **ecosystem**: every detection and segmentation framework (Faster R-CNN, Mask R-CNN, DeepLab, FPN) ships ResNet backbones, every paper baselines on it, and every serving stack has hand-tuned kernels for it. Pick it when you want the answer nobody argues with.
>
> **Depth choice:** **R-18** when data is tiny or you're iterating fast · **R-50** as the default · **R-101/152** only when you have the data and the ~+1.5 points actually matter.

---

### 5.5 MobileNet V1 / V2 / V3 (2017–19) — the phone budget

**The problem it solved.** ResNet-50 is **4.1 GFLOPs per image**. No phone runs that at 30 fps. MobileNet asks: what's the cheapest thing that still behaves like a convolution?

**V1's answer — factor the convolution into two cheap steps.** A standard conv does spatial filtering *and* channel mixing simultaneously, and pays `k²·C·N` for the privilege. Split them:

![Standard 3x3 convolution versus its depthwise-separable factorisation — a standard conv over 32 input channels producing 64 outputs costs 18,432 weights, while a depthwise 3x3 that filters each channel independently costs 288 and a pointwise 1x1 that mixes channels costs 2,048, for 2,336 total or 7.9 times fewer weights and operations at the same output shape.](attachments/depthwise-separable-convolution-cost.png)

```
Reduction ratio  =  1/N + 1/k²   =  1/64 + 1/9  ≈  0.127     → ~8–9× cheaper
Accuracy cost on ImageNet: ~1–2 points. That's the whole trade.
```

Plus two dials you can turn without retraining your intuition: the **width multiplier α** (scale every channel count; params and ops scale by `α²`) and the **resolution multiplier ρ** (scale the input; ops scale by `ρ²`, params unchanged). Together they hit any point on a speed/accuracy curve.

**V2's answer — two more ideas.** V1's weakness: the depthwise conv operates on *narrow* feature maps, so it can only detect as many spatial patterns as it has channels.

![ResNet's bottleneck compared with MobileNet V2's inverted residual — ResNet goes wide to narrow to wide, squeezing 256 channels to 64 so the expensive 3x3 runs cheaply, while V2 goes narrow to wide to narrow, expanding 32 channels to 192 so the already-cheap depthwise 3x3 filters in a much richer space, with the skip connection joining the narrow ends and no ReLU after the final compression.](attachments/mobilenetv2-inverted-residual-linear-bottleneck.png)

- **Inverted residual** — ResNet's bottleneck is `WIDE → narrow → WIDE` (compress so the costly `3×3` is affordable). But a *depthwise* `3×3` is already cheap, so V2 inverts it: `narrow → WIDE (×6) → narrow`. The depthwise conv now filters in a 192-channel space instead of 32 → far richer features at similar cost. The skip connection joins the **narrow** ends.
- **Linear bottleneck** — **no ReLU after the final compression.** ReLU zeros negatives; on 192 channels that's survivable, on the 32-channel compressed output it destroys most of what you just computed. So the last `1×1` gets no activation at all. 🎯 *"ReLU is cheap regularisation in high dimensions and information destruction in low ones — that's why V2's bottleneck is linear."*

Net effect: V2 is **smaller (3.4M vs 4.2M), faster (300M vs 569M MAdds), and more accurate (72.0% vs 70.6%)** than V1 — all three at once.

**V3** adds NAS-searched block layout, **squeeze-and-excitation** channel attention, and **hard-swish** activation: V3-Large = 5.4M / 219M MAdds / 75.2% top-1; V3-Small = 2.5M / 66M / 67.4%.

> **Transfer verdict: ✅ when latency, memory, battery, or cost-per-inference is a real constraint** — mobile, embedded, browser, or high-QPS serving where you're paying per inference. MobileNetV2 is **14 MB at 71.3% top-1**.
>
> 🚨 **The gotcha interviewers love: FLOPs ≠ latency.** Depthwise convolutions have terrible **arithmetic intensity** — very little compute per byte moved — so they're **memory-bandwidth bound**, and a GPU's FLOPs go unused. Look at the same benchmark: MobileNetV2 has ~8× fewer FLOPs than ResNet50 but runs **3.8 ms vs 4.6 ms** on an A100. Barely faster. The 8× win is real on mobile CPUs and NPUs; on a server GPU it largely evaporates. 🎯 *"I'd benchmark on the target hardware rather than assume a smaller FLOP count means faster — depthwise convs are bandwidth-bound, so the FLOP saving doesn't translate on a GPU."*
>
> **Fine-tuning nuance:** MobileNets are already thin, so they carry far less redundant capacity than a ResNet. Pure feature-extraction (everything frozen) often underperforms here — plan to **unfreeze more of the net** than you would for ResNet50.

---

### 5.6 EfficientNet (2019) — scaling stops being guesswork

**The problem it solved.** You have a good small network and a bigger compute budget. Do you make it **deeper**? **wider**? feed it **bigger images**? Every prior paper picked one axis, by hand, arbitrarily.

**The finding: the three axes are coupled.** A higher-resolution input needs *more depth* (to grow the receptive field enough to cover it) and *more width* (to capture the finer patterns it exposes). Scale one axis alone and accuracy saturates fast, because the other two become the bottleneck.

**Compound scaling** — one coefficient φ moves all three, with the exponents fixed by a small grid search on the baseline:

```
depth      d = α^φ        α = 1.20
width      w = β^φ        β = 1.10        subject to   α · β² · γ²  ≈  2
resolution r = γ^φ        γ = 1.15        → FLOPs roughly double per +1 φ

B0 … B7  is literally just  φ = 0 … 7.
```

![EfficientNet compound scaling across five panels — the B0 baseline, then width-only scaling with more channels, depth-only scaling with more layers, resolution-only scaling with a bigger input, and finally compound scaling that moves all three axes together by one coefficient, which is what takes B0's 5.3M parameters and 77.1 percent top-1 to B7's 66.7M and 84.3 percent.](attachments/efficientnet-compound-scaling.png)

The B0 baseline itself was found by neural architecture search, and its block is **MBConv — MobileNetV2's inverted residual — plus squeeze-and-excitation.** EfficientNet is, structurally, a well-scaled MobileNet.

**Numbers that matter:** B0 = **5.3M params, 77.1% top-1** — that's **+2.2 points over ResNet50 at 1/5 the parameters**. B7 = 66.7M, 84.3%.

> **Transfer verdict: ✅ the best accuracy-per-parameter of the CNN era.** B0–B2 when you want to beat ResNet50 cheaply; **EfficientNetV2-S** (21.6M / 83.9%) when you want near-SOTA CNN accuracy. It is, however, **fussier to fine-tune** than a ResNet.
>
> 🚨 **Gotcha 1 — every variant has its own input resolution.** B0 `224` · B1 `240` · B2 `260` · B3 `300` · B4 `380` · B5 `456` · B6 `528` · B7 `600`. Feeding B4 a 224px image throws away most of what the compound scaling bought you.
>
> 🚨 **Gotcha 2 — in Keras, `efficientnet.preprocess_input` is a NO-OP.** Normalisation is baked into the model as a `Rescaling`/`Normalization` layer, and it expects **raw `[0, 255]` pixels**. Helpfully dividing by 255 first — the thing every other backbone wants — silently wrecks it. This is the one exception to §3's "always call `preprocess_input`" rule. `(certain)`
>
> 🚨 **Gotcha 3 — low FLOPs ≠ fast training.** High-resolution inputs mean enormous activation tensors and heavy depthwise memory traffic, so EfficientNet trains slowly and forces small batches. **EfficientNetV2** fixes exactly this: swap `Fused-MBConv` (a plain `3×3` conv) into the early stages where depthwise doesn't pay, plus progressive resizing → ~4× faster training at higher accuracy.

---

### 5.7 What came after — the three you should be able to name

- **ConvNeXt (2022)** — a ResNet modernised with transformer-era design choices: `7×7` depthwise convs, LayerNorm instead of BN, GELU, an inverted bottleneck, fewer activations. It matches Swin Transformers using only convolutions, which settled an argument: **the CNN-vs-transformer gap was mostly training recipe and design details, not attention itself.** ConvNeXt-Tiny = 28.6M / 81.3%. `(certain)`
- **Vision Transformers (ViT)** — split the image into patches, run self-attention, no convolution anywhere. Beats CNNs *given enough pretraining data*, and underperforms them on small data because it has none of the CNN's built-in locality and translation-equivariance priors — it has to learn them. Fine-tune ViT when your checkpoint came from something large (ImageNet-21k, JFT, or CLIP). See [RNN · LSTM · Transformers](../NLP/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) for the attention mechanics.
- **Foundation backbones — CLIP, DINOv2, SAM.** CLIP gives you text-aligned embeddings and **zero-shot** classification with no training at all; **DINOv2** is self-supervised and its features often transfer *better than ImageNet supervision*, especially when your domain is far from natural photos. Increasingly these are the things you actually fine-tune. See [Multimodal RAG](../../AI%20Engineering/Multimodal%20RAG.md) and [Siamese Networks & Image Similarity](Siamese%20Networks%20&%20Image%20Similarity.md).

🎯 *"My default in 2026 isn't ResNet50 out of habit — it's ResNet50 or ConvNeXt-Tiny for a standard supervised task, EfficientNet or MobileNet when there's a latency budget, and a DINOv2 or CLIP backbone when my domain is far from ImageNet or my labels are scarce."*

---

## 6. Choosing & Fine-Tuning a Backbone

### 6.1 The Pareto view — most of the zoo is dominated

![Two scatter plots of ImageNet top-1 accuracy against parameter count and against download size for twelve backbones — the efficient frontier runs from MobileNetV2 through EfficientNet-B0, B4, V2S and B7, while AlexNet and VGG16 sit far below and to the right of every other model, dominated on both axes, and EfficientNet-B0 beats VGG16 by 5.8 accuracy points at one eighteenth the download size.](attachments/backbone-accuracy-vs-cost-pareto.png)

Read the frontier, not the individual dots. **AlexNet and VGG are strictly dominated** — for any accuracy they reach, something far smaller reaches it too. ResNet50 sits *below* the frontier and is still the right default, which is the useful lesson: **you are not optimising accuracy-per-parameter, you are optimising accuracy-per-parameter subject to ecosystem, tooling, and the risk of being the only person using a weird backbone.**

| Model | Params | Size | Top-1 | GPU ms | Input | One-line reason to pick it |
|---|---|---|---|---|---|---|
| AlexNet | 61.1M | 233 MB | 56.5% | — | 224 | history only |
| VGG16 | 138.4M | 528 MB | 71.3% | 4.2 | 224 | perceptual/style loss; teaching |
| InceptionV3 | 23.9M | 92 MB | 77.9% | 6.9 | **299** | +3 pts on ResNet50, same size |
| Xception | 22.9M | 88 MB | 79.0% | 8.1 | **299** | Inception taken to its depthwise limit |
| **ResNet50** | 25.6M | 98 MB | 74.9% | 4.6 | 224 | **the default — ecosystem wins** |
| ResNet101 | 44.7M | 171 MB | 76.4% | 5.2 | 224 | +1.5 pts if you have the data |
| **MobileNetV2** | 3.5M | 14 MB | 71.3% | 3.8 | 224 | **edge / mobile / cost-per-inference** |
| MobileNetV3-L | 5.4M | — | 75.2% | — | 224 | better mobile net than V2 |
| **EfficientNetB0** | 5.3M | 29 MB | 77.1% | 4.9 | 224 | **best accuracy per parameter** |
| EfficientNetB4 | 19.5M | 75 MB | 82.9% | 15.1 | **380** | strong accuracy, still moderate |
| EfficientNetV2-S | 21.6M | 88 MB | 83.9% | — | 384 | near-SOTA CNN, trains fast |
| ConvNeXt-Tiny | 28.6M | 109 MB | 81.3% | — | 224 | modern ResNet, drop-in replacement |

*Keras Applications benchmark; GPU ms is batch-32 on an A100. MobileNetV3 isn't in that table — its numbers are from the paper.*

### 6.2 Constraint → backbone

```
What is actually binding you?
│
├─ "Nothing — I need a solid baseline"            → ResNet50   (R18 if data is tiny)
│                                                    everyone recognises it, everything supports it
├─ "Accuracy per parameter / per FLOP"            → EfficientNet-B0…B3, EfficientNetV2-S
├─ "Phone, browser, embedded, or QPS × $"         → MobileNetV3 / EfficientNet-Lite → then distil
├─ "More accuracy, same budget as ResNet50"       → InceptionV3, Xception, ConvNeXt-Tiny
├─ "Detection or segmentation downstream"         → ResNet50-FPN — the whole toolchain assumes it
├─ "Perceptual loss, style transfer, or repro"    → VGG16 (the one job it still owns)
├─ "My domain is nothing like ImageNet"           → DINOv2 / in-domain self-supervised pretraining
├─ "Labels are scarce or classes keep changing"   → CLIP (zero-shot, open-vocabulary)
└─ "I need to explain the history in an interview"→ AlexNet → VGG → Inception → ResNet → MobileNet → EfficientNet
```

**How to actually decide, in about 30 minutes.** Don't argue about it — measure it. Because a frozen backbone is a *fixed function*, you can **extract features once and cache them**, then fit a cheap head:

```python
# 3 candidates, ~1 backbone forward pass each over your data, then instant head training
for name, backbone, prep in candidates:            # e.g. ResNet50, EfficientNetB0, MobileNetV2
    Xtr = backbone.predict(prep(imgs_tr))          # ONE forward pass — cache it to disk
    Xva = backbone.predict(prep(imgs_va))          # (frozen backbone = a fixed function)
    head  = LogisticRegression(max_iter=1000).fit(Xtr, y_tr)
    print(name, head.score(Xva, y_va))
```
That ranking almost always survives to the fine-tuned model, and it costs one forward pass per candidate instead of three training runs. `(likely)`

### 6.3 Fine-tuning each family — what to unfreeze

The 2×2 in §2 tells you *how much* to unfreeze. This tells you *where the seams are*:

| Family | Unfreeze from | BatchNorm? | Notes |
|---|---|---|---|
| **VGG16** | `block5_conv1` (last conv block) | **no BN at all** | the only family immune to the frozen-BN trap — simplest to fine-tune |
| **ResNet50** | `layer4` (PyTorch) / `conv5_block*` (Keras), then `layer3` | BN everywhere | the classic; keep BN in inference mode |
| **InceptionV3** | `mixed8` / `mixed9` — the Keras tutorial's "freeze the first 249 layers" | BN everywhere | remember `299×299` |
| **MobileNetV2/V3** | last few inverted-residual blocks (`block_13_*` onward) | BN everywhere | thin net → usually needs **more** unfrozen than ResNet |
| **EfficientNet** | `block6a_*` onward + head | BN everywhere | high res + big activations → small batches; go gentle on LR |
| **ViT / ConvNeXt** | last transformer blocks / stage 4 | LayerNorm, not BN | **LN has no running stats → no frozen-BN trap**; still use a low LR |

🎯 **The tie-back to §8:** every family in that table except VGG (and the LayerNorm-based ones) is BatchNorm-heavy — so **"freeze the weights *and* the BN running statistics"** applies to essentially every backbone you'll actually use. It's the #1 transfer-learning bug precisely because it's the default failure mode of the default backbone.

**Two rules that hold across families:**
- **Match the input contract per-model, not per-project.** Resolution and preprocessing are properties of the *checkpoint* (224 vs 299 vs 380; `[-1,1]` vs mean/std vs raw `[0,255]`). Use the bundled transform — `weights.transforms()` in torchvision, the model's own `preprocess_input` in Keras — and remember EfficientNet's no-op exception.
- **Swap backbones behind one interface.** Because every one of these is "conv trunk → pooled feature vector → your head," changing backbone should be a one-line config change in your code. If it isn't, you've hard-coded a feature dimension somewhere.

---

## 7. Code / Implementation

**Keras — feature extraction, then fine-tuning:**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# 1. Load ImageNet backbone WITHOUT its 1000-class head
base = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                                   input_shape=(224, 224, 3))
base.trainable = False                         # 2. FREEZE the conv backbone

# 3. New task head — GAP (not Flatten) keeps params tiny
inputs  = layers.Input((224, 224, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)   # ← the EXACT preprocessing VGG expects
x = base(x, training=False)                    # training=False keeps frozen BN in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)      # only the head trains

# --- Optional Stage 2: fine-tune the top block at a TINY LR ---
base.trainable = True
for layer in base.layers[:-4]:                 # keep early layers frozen
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy",
              metrics=["accuracy"])            # MUST recompile after changing trainable flags
model.fit(train_ds, validation_data=val_ds, epochs=5)
```

**PyTorch / torchvision — the same idea:**
```python
import torch, torch.nn as nn, torchvision.models as models
from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
net = models.resnet50(weights=weights)
preprocess = weights.transforms()             # ships the EXACT resize+normalize the weights expect

for p in net.parameters():                    # freeze everything...
    p.requires_grad = False
net.fc = nn.Linear(net.fc.in_features, 10)    # ...then replace head (new layer is trainable by default)

opt = torch.optim.Adam(net.fc.parameters(), lr=1e-3)   # optimizer sees ONLY head params
# Fine-tune later: unfreeze net.layer4, add its params to a param group at lr=1e-5.
```
For a broader model zoo (ResNet/ViT/ConvNeXt/EfficientNet + thousands more) use **`timm`** or **Hugging Face**; the pattern is identical — load pretrained, replace head, freeze/fine-tune.

---

## 8. When It Breaks

```
❌ Preprocessing mismatch — the silent killer. If you don't normalize with the SAME mean/std / 
   preprocess_input the backbone was trained on, its features are garbage and accuracy craters with 
   no error message. Always use the weights' bundled transform (weights.transforms() / preprocess_input).

❌ BatchNorm in a "frozen" backbone. Setting trainable=False freezes the WEIGHTS but, unless you also 
   run the layer in inference mode (Keras: call base(x, training=False); PyTorch: net.eval() for those 
   modules), BN keeps UPDATING its running mean/var from your small batches → the backbone's statistics 
   drift and accuracy silently degrades. This is the #1 transfer-learning bug. 🎯 Freeze BOTH the weights
   AND the BN running-stat updates.

❌ Fine-tuning at too high an LR → catastrophic forgetting. Big gradients from a fresh, high-loss head 
   overwrite the pretrained weights. Fix: train the head frozen first, THEN unfreeze at LR ≤ 1e-4, 
   optionally gradual-unfreezing.

❌ Negative transfer / large domain gap. ImageNet (natural photos) → medical/satellite/microscopy/depth 
   or a different input modality (grayscale, spectral, RGB-D): late features may not transfer and can 
   even hurt vs training from scratch. Freeze fewer layers, or pretrain on in-domain / self-supervised data.

❌ Forgetting to recompile (Keras) after changing trainable flags → your unfreeze does nothing.

❌ Data leakage. Split BEFORE any augmentation; and if the pretrained model already saw images overlapping
   your test set, your reported accuracy is inflated.

❌ Overfitting the fine-tuned weights. Unfreezing many layers on a tiny dataset re-introduces the exact 
   overparametrization problem transfer learning was solving — keep more frozen when data is scarce.
```

---

## 9. Production & MLOps Notes

**Model zoos are your library.** `torchvision`, **`timm`** (best-in-class, thousands of pretrained CNNs + ViTs), and Hugging Face give you weights in one line. Prefer modern backbones (**ResNet/ConvNeXt/EfficientNet/ViT**) over VGG in production — VGG is a great teaching example but heavy and dated.

**The backbone as an embedding service.** Feature extraction has a second life beyond classification: run the frozen backbone and take the **penultimate vector as an image embedding**, then do k-NN / metric search / clustering / few-shot on top — no head training at all. This is the bridge to [Siamese Networks & Image Similarity](Siamese%20Networks%20&%20Image%20Similarity.md) and image retrieval.

**Beyond ImageNet — self-supervised & foundation models.** When labels are scarce even for pretraining, **self-supervised pretraining** (SimCLR, MoCo, BYOL, DINO, **MAE**) learns transferable features from *unlabeled* images and often transfers better than supervised ImageNet, especially cross-domain. Vision **foundation models** (CLIP for image-text, DINOv2 as a general backbone, SAM for segmentation) are increasingly the thing you fine-tune. `(likely)`

**Cost, latency, licensing.**
- **Compute:** feature extraction is cheap (backbone forward pass, cache features if the dataset is fixed); full fine-tuning is a full training run — budget accordingly.
- **Serving:** you ship the *whole* net (backbone + head), so backbone size drives latency/memory — pick EfficientNet/MobileNet for edge; compress with quantization/pruning/**distillation** (train a small student from the fine-tuned teacher).
- **Reproducibility:** pin the exact weights version (`IMAGENET1K_V1` vs `V2` differ) and its preprocessing — an upgraded weight enum can change results.
- **Licensing:** check pretrained-weight and dataset licenses before commercial use; not all ImageNet-derived or foundation weights are freely usable.

**Monitoring.** Same drift concerns as any CNN, plus: watch for **domain shift away from the pretraining distribution** — the further your production images drift from ImageNet-like photos, the more the transferred features underperform, signalling it's time to fine-tune deeper or re-pretrain in-domain.

---

## 10. Interview Lens

> ⚡ The question is almost always "small dataset, how do you get good accuracy?" — answer *transfer learning*, then show you know the freeze-vs-fine-tune decision and the two classic gotchas (preprocessing match, frozen-BN).

**"You have 700 images and 10 classes — how do you build a good classifier?"** → 🎯 *"I wouldn't train from scratch — 700 images can't fit a deep net (VGG from scratch gets ~11% here). I take an ImageNet-pretrained backbone, drop its 1000-class head, freeze the conv base, add a GAP → Dense(10) head, and train just the head — that's ~85% in a few epochs. If I had more data or a domain gap, I'd then unfreeze the top blocks and fine-tune at a low LR."*

**"Which pretrained model would you pick, and why?"** → 🎯 *"Depends what's binding. **ResNet50 by default** — not because it's the best accuracy-per-parameter (it isn't) but because every detection and segmentation framework and every serving stack is built around it. If accuracy-per-parameter is what matters, **EfficientNet-B0** beats it by 2.2 points at a fifth of the parameters. If it ships on a phone, **MobileNetV3**. And if my domain is nothing like ImageNet, I'd stop reaching for ImageNet supervision at all and fine-tune **DINOv2**."*

**Likely follow-ups:**
- *What problem did ResNet actually solve?* → **Degradation, not overfitting.** A 56-layer plain net had higher **training** error than a 20-layer one — an *optimisation* failure. `y = F(x) + x` makes "do nothing" trivially learnable and gives the gradient a `+1` highway. Saying "vanishing gradients" alone is the half-answer. `(certain)`
- *Why does the Inception module have 1×1 convs?* → Cheap channel **dimensionality reduction before** the expensive `3×3`/`5×5` branches — the `5×5` branch drops from ~120M to ~12.4M multiply-adds. Separately, **GAP replacing `Flatten→FC`** is what deleted VGG's ~120M-param head.
- *Why didn't VGG-19 beat VGG-16?* → It doesn't (71.3% both). Depth alone had already saturated — precisely the observation that motivated ResNet's shortcut. `(certain)`
- *What makes MobileNet cheap?* → Depthwise-separable conv: reduction `1/N + 1/k² ≈ 0.127`, so ~8× fewer params and ops for ~1–2 accuracy points. V2 adds the inverted residual and the linear bottleneck.
- *Why is there no ReLU at the end of a MobileNetV2 block?* → **Linear bottleneck.** ReLU zeroing negatives is survivable across 192 channels but destroys most of the information in the 32-channel compressed output.
- *What is EfficientNet's contribution?* → **Compound scaling.** Depth, width and resolution are coupled, so scale all three with one coefficient φ under `α·β²·γ² ≈ 2` instead of picking one axis by hand.
- *Trap — is a model with 8× fewer FLOPs 8× faster?* → **No.** Depthwise convs are memory-bandwidth bound; MobileNetV2 has ~8× fewer FLOPs than ResNet50 but runs 3.8 ms vs 4.6 ms on an A100. Benchmark on the target hardware. 🎯
- *Feature extraction vs fine-tuning — when each?* → Freeze + train head when data is small / domain similar; unfreeze + low-LR fine-tune when you have more data or a domain gap. The 2×2 (data × similarity). `(certain)`
- *Which layers transfer best?* → Early conv (edges/textures) are generic and transfer everywhere; late layers are ImageNet-specific and are the ones you replace or adapt.
- *Why freeze the head-warmup before unfreezing?* → A random head's large early gradients would corrupt the pretrained weights (catastrophic forgetting).
- *Biggest silent bug?* → Preprocessing mismatch (wrong normalization) and BatchNorm still updating in a "frozen" backbone. Say both.
- *When does transfer learning NOT help?* → Huge domain gap (medical/satellite), different modality, or you already have millions of in-domain labels — then fine-tune deep or train from scratch (negative transfer).
- *Why prefer a GAP head over Flatten when transferring VGG?* → VGG's `Flatten→FC` head is ~120M of its ~138M params; GAP (`7×7×512→512`) removes that bottleneck.
- *How to fine-tune without forgetting?* → Low + discriminative LRs (earlier layers lower), gradual unfreezing.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Small data, ImageNet-like images | **Feature extraction** (freeze backbone) | fastest, most data-efficient, hard to overfit |
| Moderate data or mild domain gap | **Fine-tune top blocks**, low LR | adapt task-specific features without losing generic ones |
| Lots of data, different domain | **Fine-tune all / train from scratch** | enough signal; ImageNet features may not help |
| Labels scarce even for pretraining | **Self-supervised pretrain** (DINO/MAE/SimCLR) | learns transferable features from unlabeled images |
| Need embeddings / retrieval / few-shot | **Frozen backbone as feature extractor** | penultimate vector = image embedding → k-NN / metric learning |
| Edge/latency constraints | Transfer onto **MobileNetV3 / EfficientNet-Lite**, then distill | small backbone drives serving cost |
| Unsure *which backbone* to start from | **Cache frozen features, fit a logistic head, compare** ([§6.2](#62-constraint--backbone)) | one forward pass per candidate beats three training runs |
| Multimodal / open-vocabulary | **CLIP / foundation model** fine-tune | text-image features generalize beyond fixed label sets |

**Decision rule:** default to transfer learning for *any* vision task with < ~10k images; freeze by default, fine-tune only when data/domain justify it. Training from scratch is the exception, reserved for large in-domain datasets or when no relevant pretrained backbone exists.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why does VGG16 get ~11% on the 10-class landmark set from scratch but ~85% via transfer learning?
   <details><summary>answer</summary>138M params can't be fit on ~700 images (11% ≈ random). Transfer learning reuses ImageNet's generic edge→texture→part features (frozen backbone) and trains only a small head → data-efficient, high accuracy.</details>
2. Which layers transfer, and which do you replace?
   <details><summary>answer</summary>Early/mid conv layers (generic edges, textures, parts) transfer to any vision task → keep/freeze. Late layers + the classifier head are ImageNet-specific → replace with a head for your classes.</details>
3. State the freeze-vs-fine-tune decision as a 2×2.
   <details><summary>answer</summary>Axes = dataset size × domain similarity. Small+similar → freeze all, train head. Small+different → freeze early, adapt late. Large+similar → fine-tune all (low LR). Large+different → fine-tune all or train from scratch. *Small+similar → freeze more; large+different → freeze less.*</details>
4. What are the two classic silent bugs in transfer learning?
   <details><summary>answer</summary>(1) **Preprocessing mismatch** — not using the backbone's exact normalization → garbage features. (2) **Frozen-BN drift** — `trainable=False` freezes weights but BN keeps updating running stats from your batches unless you run it in inference mode (`training=False` / `.eval()`).</details>
5. Why warm up the new head with the backbone frozen before fine-tuning it?
   <details><summary>answer</summary>A randomly-initialized head has high loss → large gradients that, if backpropagated into an unfrozen backbone, overwrite the good pretrained weights (catastrophic forgetting). Freeze → warm up head → unfreeze at low LR.</details>
6. Two 3×3 convs vs one 5×5 — which does VGG use and why?
   <details><summary>answer</summary>Two 3×3 (18 params + two non-linearities) beat one 5×5 (25 params, one non-linearity) for the same receptive field → fewer params, more expressive. Why VGG dropped AlexNet's big filters.</details>
7. When would you NOT use ImageNet transfer learning?
   <details><summary>answer</summary>Large domain gap (medical/satellite/microscopy), different input modality, or you already have millions of in-domain labels — features may not transfer (negative transfer). Use self-supervised/in-domain pretraining or train from scratch.</details>

8. ResNet solved *which* problem — and why is "vanishing gradients" only half the answer?
   <details><summary>answer</summary>The **degradation problem**: a 56-layer *plain* net had higher **training** error than a 20-layer one. Higher *training* error means it's an **optimisation** failure, not overfitting. `y = F(x)+x` fixes it two ways: driving `F(x)→0` makes an identity block trivially learnable, and `∂y/∂x = ∂F/∂x + 1` gives backprop a highway that can't multiply down to zero.</details>
9. What two problems does the Inception module solve, and how?
   <details><summary>answer</summary>(1) *Which filter size?* — don't choose: run `1×1`, `3×3`, `5×5` and pool **in parallel**, concatenate along channels, let the next layer weight them. (2) *Parameter bloat* — `1×1` convs squeeze channels **before** the expensive branches (`5×5` branch: 153,600 → 15,872 weights), and **GAP replaces the FC head**, deleting VGG's ~120M params. GoogLeNet ≈ 7M params, more accurate than VGG's 138M.</details>
10. What is a depthwise-separable convolution, and what does it save?
    <details><summary>answer</summary>Split a standard conv into **depthwise** (one `k×k` filter per channel — spatial only, no channel mixing) then **pointwise** (`1×1` — channel mixing only). For `C=32, N=64, k=3`: 18,432 → 288 + 2,048 = 2,336 weights. Reduction `1/N + 1/k² = 1/64 + 1/9 ≈ 0.127` → ~8× cheaper for ~1–2 accuracy points.</details>
11. Why does a MobileNetV2 block end *without* a ReLU, and why is its residual called "inverted"?
    <details><summary>answer</summary>**Linear bottleneck:** ReLU zeros negatives — fine across 192 channels, but it destroys most of the information in the 32-channel compressed output, so the final `1×1` gets no activation. **Inverted:** ResNet goes `WIDE→narrow→WIDE` (compress so the costly `3×3` is affordable); a depthwise `3×3` is already cheap, so V2 goes `narrow→WIDE(×6)→narrow` and the skip joins the **narrow** ends.</details>
12. What is compound scaling, and name two EfficientNet gotchas that bite in transfer learning.
    <details><summary>answer</summary>Depth/width/resolution are **coupled**, so scale all three with one coefficient: `d=1.20^φ, w=1.10^φ, r=1.15^φ` with `α·β²·γ²≈2`; B0→B7 is just φ=0→7. Gotchas: (1) **each variant has its own input resolution** (B0 224 … B7 600) — feeding B4 a 224px image throws the scaling away; (2) in Keras `efficientnet.preprocess_input` is a **no-op** — the model expects raw `[0,255]`, so dividing by 255 first silently wrecks it.</details>
13. A 5k-image classifier must run in ~50 ms on a phone. Which backbone, and what do you unfreeze?
    <details><summary>answer</summary>**MobileNetV3-Large** (or EfficientNet-Lite) — 5.4M params, built for the constraint. 5k images with a similar domain is "small-to-moderate + similar", so: freeze, train the head, then **unfreeze the last few inverted-residual blocks** at `1e-4`. Unfreeze *more* than you would on a ResNet — MobileNets are thin and carry less redundant capacity. Then **distil** if you still need latency. Keep BN in inference mode throughout.</details>
14. Is a model with 8× fewer FLOPs 8× faster?
    <details><summary>answer</summary>No. Depthwise convolutions have low **arithmetic intensity** (little compute per byte moved) so they're **memory-bandwidth bound** — on an A100, MobileNetV2 runs 3.8 ms vs ResNet50's 4.6 ms despite ~8× fewer FLOPs. The saving is real on mobile CPUs/NPUs, largely absent on server GPUs. Always benchmark on the target hardware.</details>

---

*Covers: why scratch fails on small data · generic→specific feature gradient · feature extraction vs fine-tuning · the data×similarity 2×2 · discriminative/layer-wise LRs · gradual unfreezing & catastrophic forgetting · preprocessing-match & frozen-BN gotchas · **the backbone zoo — AlexNet (ReLU/dropout/GPU), VGG (3×3 depth + the FC bottleneck), Inception (parallel filters, 1×1 reduction, GAP, aux heads, Xception), ResNet (degradation, residual, basic vs bottleneck vs projection, V2/ResNeXt), MobileNet (depthwise-separable, α/ρ, inverted residual, linear bottleneck, V3), EfficientNet (compound scaling, MBConv, V2), ConvNeXt/ViT/CLIP/DINOv2** · **choosing a backbone — the Pareto frontier, constraint→model table, cached-feature bake-off, per-family unfreeze points** · FLOPs≠latency · per-checkpoint input contracts (224/299/380, Keras EfficientNet no-op) · top-k accuracy · negative transfer & domain gap · self-supervised/foundation-model pretraining · model zoos, embeddings, distillation, licensing.*
