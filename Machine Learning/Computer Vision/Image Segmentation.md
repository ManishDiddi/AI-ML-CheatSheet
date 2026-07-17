# Image Segmentation — per-pixel labels: encoder–decoder, U-Net, Mask R-CNN

> **TL;DR.** Segmentation classifies *every pixel*, not the whole image (classification) or a rectangle (detection). **Semantic** segmentation labels each pixel by class (all cars → "car"); **instance** segmentation gives each object its own mask (car₁ ≠ car₂). The universal recipe is an **encoder–decoder**: the encoder downsamples for *semantic* meaning ("what"), the decoder upsamples back to full resolution ("where"), and **skip connections** feed the decoder the fine spatial detail the encoder threw away. FCN pioneered it, **U-Net** perfected it (concatenation skips at every scale — the go-to for medical/small data), and **Mask R-CNN** does instance segmentation by bolting a mask head onto Faster R-CNN. Because foreground is tiny, you train with **Dice/IoU loss**, not plain cross-entropy.

**Where it fits:** the most *spatially precise* vision task — used where a bounding box isn't enough: medical tumor/organ delineation, portrait-mode background blur, self-driving road/lane/free-space, satellite land cover.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (conv, 1×1 conv, dilated/atrous conv, GAP), [Object Detection](Object%20Detection.md) (Faster R-CNN, RPN, RoI Align, IoU/NMS/mAP — Mask R-CNN builds directly on these), and encoder–decoder intuition ([Autoencoders](../Neural%20Networks/Autoencoders.md)).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — tasks, upsampling, skips, losses, metrics](#2-the-formal-core--tasks-upsampling-skips-losses-metrics)
3. [How It Works — FCN → U-Net → Mask R-CNN → DeepLab](#3-how-it-works--fcn--u-net--mask-r-cnn--deeplab)
4. [Worked Examples — transposed conv, U-Net shapes, Dice vs IoU](#4-worked-examples--transposed-conv-u-net-shapes-dice-vs-iou)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Production & MLOps Notes](#7-production--mlops-notes)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**The task ladder — precision increases at every rung:**

```
Classification   → 1 label for the whole image           "there is a cat"
Detection        → a box per object                       "cat in this rectangle"
Semantic seg.    → a class per PIXEL                       "these exact pixels are cat"
Instance seg.    → a mask per OBJECT                       "these pixels are cat #1, those are cat #2"
```
A bounding box around a person still includes background around the arms/hair — useless for portrait blur or tumor boundaries. Segmentation traces the *exact* shape.

**The core tension: "What" vs "Where".** A classification CNN downsamples hard (`224→112→…→7`) to build **deep, coarse, semantic** features — great at "*what* is this" but it has **destroyed the spatial detail** needed for "*where* exactly is the boundary." Segmentation needs *both*:

```
Deep layers  (7×7, 512ch)   →  WHAT it is   (semantic, coarse)   ← lost the "where"
Shallow layers (112×112, 64ch) →  WHERE it is (edges, fine detail) ← lacks the "what"

Encoder–decoder + skip connections fuse them:
  Encoder (downsample) → bottleneck → Decoder (upsample back to H×W)
        │  skip: copy fine encoder features across  ▲
        └────────────────────────────────────────────┘
```

🎯 *"Segmentation is an encoder–decoder: the encoder gives semantic 'what' by downsampling, the decoder recovers spatial 'where' by upsampling, and skip connections hand the decoder the fine detail the encoder discarded — otherwise the mask comes out blurry."*

This is the same encoder–decoder shape as an [autoencoder](../Neural%20Networks/Autoencoders.md), but supervised to output a *label map* instead of reconstructing the input.

---

## 2. The Formal Core — tasks, upsampling, skips, losses, metrics

**Task definitions & output shapes:**

| Task | Labels every pixel? | Distinguishes instances? | Output | Models |
|---|---|---|---|---|
| **Semantic** | yes | no (all cars = "car") | `H×W×C` (per-pixel class map) | FCN, U-Net, DeepLab |
| **Instance** | no (foreground "things" only) | yes | `N` binary masks `H×W` | Mask R-CNN |

Semantic output is per-pixel softmax over `C` classes → argmax gives the label map.

**Upsampling — how the decoder grows `7×7` back to `H×W`.** Two ways:
- **Transposed convolution** (`Conv2DTranspose`, aka "deconvolution"): *learned* upsampling. Each input value stamps the whole (scaled) kernel onto the output; overlapping stamps are **summed**. It reverses conv *dimensions*, not values. Output size:
  ```
  out = stride·(in − 1) + kernel − 2·padding      (stride here SPREADS input cells, not shrinks)
  stride=2, kernel=2, in=2 → 2·(2−1)+2 = 4  (doubles)
  ```
  ⚠️ **Checkerboard artifacts** when `kernel % stride ≠ 0` (uneven overlap) → use kernel divisible by stride (e.g. 4×4/stride-2).
- **Bilinear upsampling / `UpSampling2D`**: *fixed* geometric interpolation (weighted avg of 4 nearest pixels). **No learnable params → cheaper, less memory, mobile-friendly, no artifacts**, but can't learn task-specific patterns. Modern best practice: **bilinear upsample → then a regular conv** (separate the geometry from the learning). U-Net uses `UpSampling2D`; FCN uses transposed conv.

**Skip-connection fusion — two flavors:**
```
Addition (FCN):        encoder + decoder  → element-wise sum (channels must match; merges, loses identity)
Concatenation (U-Net): [encoder | decoder] → stack channels (doubles depth; next conv learns the mix)
```
Concatenation keeps coarse and fine features *separate* so the following conv picks the optimal blend → **sharper boundaries** than addition. That single choice is most of why U-Net beats FCN.

**Losses — why plain cross-entropy fails.** Pixel-wise CE averages a per-pixel loss with **no notion of the whole shape**, and foreground is often <1% of pixels (a 50-pixel tumor in a 512×512 image). CE then happily predicts "all background" → 99.98% pixel accuracy, **0% tumor detected**. Fixes:
```
Dice coefficient = 2·|A∩B| / (|A|+|B|) = 2·TP / (2·TP + FP + FN)          (double-counts TP)
Dice loss        = 1 − Dice           → optimizes overlap directly, imbalance-robust
Focal loss       = −(1−p_t)^γ · log p_t → down-weights easy background pixels
Best practice    = CrossEntropy + λ·(1 − Dice)  → CE's stable gradients + Dice's imbalance handling
```
Dice ↔ IoU: `Dice = 2·IoU/(1+IoU)`. IoU has a "squaring" effect that punishes worst-case errors harder (≈ worst-case performance); Dice tracks *average* performance.

**Metrics:**
- **IoU / mIoU** (`= TP/(TP+FP+FN)`, meaned over classes) — the semantic-segmentation standard.
- **Dice coefficient** — the medical-imaging standard.
- **mask mAP** — instance segmentation, same as detection [mAP](Object%20Detection.md#2-the-formal-core--boxes-anchors-iou-losses-map) but IoU computed on the **mask**, not the box (`mAP@[.5:.95]`).

---

## 3. How It Works — FCN → U-Net → Mask R-CNN → DeepLab

### 3a. FCN (Fully Convolutional Network) — the first end-to-end semantic segmenter
Key move: **replace the FC classifier head with `1×1` convolutions** → the network becomes *fully convolutional*, accepts any input size, and outputs a *spatial* class map instead of a vector. Encoder = a pretrained backbone (VGG/ResNet) minus its FC layers. Variants differ only in how aggressively the decoder upsamples and how many skips it fuses (by **addition**):
```
FCN-32s: bottleneck 7×7 → ONE 32× transposed-conv → 224×224   → blurry (all detail lost)
FCN-16s: 2×up + ADD Pool4 skip → 16×up                        → better
FCN-8s : 2×up + ADD Pool4 → 2×up + ADD Pool3 → 8×up           → sharpest FCN
```
More skips + smaller final upsample jump = sharper boundaries. Even FCN-8s stays a bit coarse at edges → motivates U-Net.

### 3b. U-Net — the segmentation workhorse
A **symmetric** encoder–decoder ("U" shape): the decoder *mirrors* the encoder step-for-step, with a **concatenation skip at every level** and **convolutions in the decoder** (to refine while upsampling). One decoder block:
```
upsample (2×)  →  concatenate the matching encoder feature map  →  Conv3×3 → Conv3×3
   (channels ÷2)          (channels ×2 from concat)                 (channels ÷2 back)
```
Dense per-scale skips mean the decoder *never* has to hallucinate fine detail — it's handed the encoder's. Consequences: precise boundaries, and famously it trains on **as few as ~30 annotated images** (with **elastic-deformation** augmentation), which is why it dominates **medical imaging**.

### 3c. Mask R-CNN — instance segmentation = Faster R-CNN + a mask head
Reuses the entire two-stage detector ([backbone+FPN → RPN → RoI Align](Object%20Detection.md#3-how-it-works--the-two-families)) and adds a **third parallel head**:
```
RoI Align (e.g. 14×14×256 per proposal)
   ├── FC  → class label      (existing)
   ├── FC  → box offsets       (existing)
   └── small FCN → 28×28×C binary masks   (NEW: 4×Conv3×3 → transposed-conv 2× → 1×1 conv + sigmoid)
```
Three things make it work:
- **RoI Align, not RoI Pooling.** Pooling *rounds* proposal coords to feature cells (≤0.5-cell error) — fine for boxes, **catastrophic for masks** (1px off = wrong outline). RoI Align uses exact float coords + bilinear interpolation → sub-pixel accuracy. This is *the* reason masks are crisp.
- **Mask loss is per-pixel BCE on the ground-truth class only** — the mask head just learns "what *shape* is here," decoupled from the class head's "what *is* it." Simpler, more stable.
- **Masks are RoI-relative**: the `28×28` mask is predicted in the RoI's local frame, then **resized to the box's size and placed top-left-aligned at (x1,y1)** in image space — *not* centered, *not* stride-scaled.
Loss = `L_cls + L_box + L_mask` (multi-task).

### 3d. DeepLab — semantic segmentation without a decoder, via atrous conv
Instead of an encoder–decoder, keep resolution high using **atrous (dilated) convolution** ([mechanics in the CNN note](Convolutional%20Neural%20Networks%20for%20Vision.md#3-how-it-works)): insert gaps between filter taps → the receptive field grows **without pooling**, so spatial detail is never lost (`effective_RF = k + (k−1)(rate−1)`). **ASPP (Atrous Spatial Pyramid Pooling)** runs several dilated convs at *different rates in parallel* (rate 6/12/18 + a 1×1 and image-level pooling) → captures small *and* large objects at once, then concatenates. This multi-scale-without-downsampling design is DeepLab's signature.

---

## 4. Worked Examples — transposed conv, U-Net shapes, Dice vs IoU

**Transposed convolution (2×2 input, 2×2 filter, stride 1 → 3×3 output).** Each input value stamps the filter; overlaps sum:
```
input  [1 2 / 3 4]   filter [1 2 / 3 4]
stamp 1·filter at (0,0), 2·filter at (0,1), 3·filter at (1,0), 4·filter at (1,1), sum overlaps →
output = [ 1   4   4 /  6  18  16 /  9  24  16 ]      out = stride·(in−1)+k = 1·1+2 = 3 ✓
```

**U-Net decoder shape arithmetic** — the rule "*concat doubles channels, then Conv halves them back*":
```
bottleneck 32×32×1024
 → transposed-conv 2×            → 64×64×512
 → concat encoder E4 (64×64×512) → 64×64×1024      (channels DOUBLE)
 → Conv3×3 ×2                    → 64×64×512        (channels HALVE)
 … repeat up to full resolution → 1×1 conv → H×W×C
```

**Dice vs IoU on an imbalanced mask.** Predict 40 of 50 tumor pixels correctly, 5 false positives (TP=40, FP=5, FN=10):
```
IoU  = 40 / (40+5+10)      = 40/55  ≈ 0.73
Dice = 2·40 / (2·40+5+10)  = 80/95  ≈ 0.84     (Dice ≥ IoU always; Dice = 2·IoU/(1+IoU) = 2·0.73/1.73 ≈ 0.84 ✓)
```
Cross-entropy on this image would be dominated by the ~262k background pixels; Dice/IoU focus on the 50 that matter.

---

## 5. Code / Implementation

**U-Net (Keras) — symmetric encoder–decoder with concatenation skips:**
```python
import tensorflow as tf
from tensorflow.keras import layers as L

def conv_block(x, f):                                  # Conv → Conv (the U-Net unit)
    x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
    return L.Conv2D(f, 3, padding="same", activation="relu")(x)

def unet(n_classes=2):
    inp = tf.keras.Input((128,128,3))
    c1 = conv_block(inp, 64);  p1 = L.MaxPooling2D()(c1)   # encoder: save c1..c4 for skips
    c2 = conv_block(p1, 128);  p2 = L.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256);  p3 = L.MaxPooling2D()(c3)
    c4 = conv_block(p3, 512);  p4 = L.MaxPooling2D()(c4)
    bn = conv_block(p4, 1024)                              # bottleneck
    def up(x, skip, f):                                    # decoder: upsample → CONCAT skip → conv
        x = L.Conv2D(f, 3, padding="same", activation="relu")(L.UpSampling2D()(x))
        x = L.Concatenate()([x, skip])                    # ← the U-Net skip (concat, not add)
        return conv_block(x, f)
    u = up(bn, c4, 512); u = up(u, c3, 256); u = up(u, c2, 128); u = up(u, c1, 64)
    out = L.Conv2D(n_classes, 3, padding="same", activation="softmax")(u)  # per-pixel class map
    return tf.keras.Model(inp, out)
```

**Dice coefficient / loss (the metric you actually optimize on imbalanced masks):**
```python
import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred, smooth=1.):
    yt, yp = K.flatten(y_true), K.flatten(y_pred)
    inter = K.sum(yt * yp)
    return (2.*inter + smooth) / (K.sum(yt) + K.sum(yp) + smooth)   # smooth avoids /0 on empty masks
def dice_loss(y_true, y_pred): return 1 - dice_coef(y_true, y_pred)

model = unet()
model.compile("adam", loss="categorical_crossentropy",           # or  CE + dice_loss  (best practice)
              metrics=["accuracy", dice_coef])
```

**Applying the mask (portrait-mode background blur — the lecture's business case):**
```python
import cv2, numpy as np
pred = np.argmax(model.predict(img[None]), -1)[0]          # per-pixel class (1=person, 0=bg)
blur = cv2.GaussianBlur(img, (21,21), 0)
portrait = np.where(pred[...,None] == 1, img, blur)         # keep person sharp, blur background
```
For production, don't hand-roll: **`segmentation_models`** (U-Net/FPN/DeepLab with pretrained encoders), **`torchvision.models.segmentation`** (`deeplabv3_resnet50`, `fcn_resnet50`), and **`torchvision...maskrcnn_resnet50_fpn`** ship ready to fine-tune.

---

## 6. When It Breaks

```
❌ Class imbalance → plain cross-entropy collapses. Foreground is often <1% of pixels; CE predicts
   "all background" for ~99% pixel accuracy and 0% object recall. FIX: Dice / focal / CE+Dice loss.

❌ Augmentation applied to the image but NOT the mask (or applied differently). Every SPATIAL transform
   (flip/rotate/crop/scale) MUST be applied IDENTICALLY to image and mask, or labels desync. Photometric
   transforms (brightness/color jitter) go on the IMAGE ONLY — the mask holds class IDs, not colors.
   Use paired transforms (albumentations, tf paired ops), never two independent random calls.

❌ Blurry boundaries from too-aggressive downsampling / too-few skips (FCN-32s upsamples 32× in one jump).
   FIX: more skip levels (FCN-8s → U-Net), or atrous conv (DeepLab) to avoid downsampling at all.

❌ Checkerboard artifacts from transposed conv when kernel % stride ≠ 0. FIX: kernel divisible by stride,
   or "bilinear upsample + conv" instead.

❌ RoI Pooling for instance masks. Its ≤0.5-cell rounding wrecks mask outlines. FIX: RoI Align (Mask R-CNN).

❌ Coarse 28×28 Mask R-CNN masks on large/thin objects. The fixed low mask resolution loses fine structure
   (wispy hair, thin poles) → PointRend / higher-res mask heads / boundary-aware losses.

❌ Wrong mask placement in Mask R-CNN post-processing: the 28×28 mask is RoI-relative → resize to the box
   and align top-left; don't center it or stride-scale it.

❌ Evaluating with pixel accuracy. On imbalanced masks it's meaningless — always report mIoU / Dice.
```

---

## 7. Production & MLOps Notes

**Don't train from scratch — use pretrained encoders.** U-Net/DeepLab with an ImageNet-pretrained ResNet/EfficientNet encoder ([transfer learning](Transfer%20Learning.md)) converges faster and needs far fewer masks. Libraries: `segmentation_models` (Keras/PyTorch), `torchvision.models.segmentation`, MMSegmentation.

**Foundation models changed the game.** **SAM (Segment Anything)** produces high-quality class-agnostic masks from a point/box prompt with zero task training — use it to *bootstrap annotation* (prompt → mask → correct) or as a zero-shot segmenter; it's often faster than training a bespoke model. `(likely)` For open-vocabulary, text-prompted variants exist.

**Annotation is the bottleneck.** Pixel masks are ~10× costlier to label than boxes. Mitigate with SAM-assisted labeling, weak/scribble supervision, polygon tools, and heavy augmentation. **Elastic deformation** is the classic medical-imaging augmentation (mimics tissue variation); **mosaic/scale-jitter** help general scenes.

**Metrics & monitoring.** Track **mIoU** (or Dice for medical) per class, plus **boundary IoU** if edges matter (portrait/medical). Watch for domain drift (new scanner, camera, lighting) degrading masks; segmentation is sensitive to it. Report at multiple IoU thresholds for instance tasks.

**Latency & edge.** Full-resolution decoders are expensive. For mobile/real-time: **bilinear `UpSampling2D`** (fewer params/memory than transposed conv), lightweight backbones (MobileNet), lower input resolution, and model compression. Test-time augmentation (flip/scale, averaged) buys accuracy at a latency cost. **DeepLabv3+** and **BiSeNet/Fast-SCNN** are common accuracy/speed picks; **SegFormer** (transformer-based, [[Vision Transformers]]) is current SOTA when compute allows.

**Data format discipline.** Masks are `H×W` single-channel images where pixel value = class id (0=background). Keep the class-id↔color map versioned; a silent remap corrupts every label.

---

## 8. Interview Lens

> ⚡ Through-line: segmentation = per-pixel classification; encoder gives "what," decoder + skips give "where," and imbalance forces Dice/IoU losses over cross-entropy.

**"Semantic vs instance segmentation?"** → 🎯 *"Semantic labels every pixel by class but can't separate two cars; instance gives each object its own mask but ignores background. Semantic → U-Net/DeepLab; instance → Mask R-CNN."*

**Likely follow-ups:**
- *Why an encoder–decoder, and what do skip connections add?* → Encoder downsamples for semantic "what" but loses spatial "where"; decoder upsamples back; skips hand the decoder the encoder's fine detail so masks aren't blurry. `(certain)`
- *U-Net vs FCN?* → U-Net uses concatenation skips at *every* level + decoder convs (symmetric) → sharper boundaries; FCN uses addition skips at 2–3 levels. U-Net wins on small data (medical).
- *Transposed conv vs bilinear upsample?* → Transposed conv is learned (can checkerboard); bilinear is fixed/cheap/smooth. Best practice: bilinear + conv.
- *Why Dice loss over cross-entropy?* → CE averages per-pixel and drowns in the majority background class; Dice optimizes overlap so missing a tiny tumor gives Dice≈0 regardless of background. `(certain)`
- *Mask R-CNN vs Faster R-CNN?* → Same detector + a parallel FCN mask head + **RoI Align** (vs RoI Pooling) for sub-pixel accuracy; mask loss on the GT class only.
- *Why RoI Align matters for masks specifically?* → 0.5-cell rounding in RoI Pooling shifts the whole mask outline; masks need sub-pixel alignment, boxes tolerate it.
- *How is instance segmentation evaluated?* → mask mAP (IoU on masks, @[.5:.95]); semantic → mIoU/Dice.
- *DeepLab's trick?* → Atrous conv (bigger RF, no resolution loss) + ASPP (parallel dilated convs for multi-scale) — no decoder needed.
- *Augmentation gotcha?* → spatial transforms must be applied identically to image and mask; photometric to image only.

---

## 9. Alternatives & How to Choose

| Need | Reach for | Why |
|---|---|---|
| General semantic seg | **U-Net** / **DeepLabv3+** | dense skips / atrous+ASPP → strong boundaries |
| Medical / tiny dataset | **U-Net** (+ elastic deformation, Dice loss) | dense skips + trains on ~30 images |
| Per-object masks (counting, tracking) | **Mask R-CNN** | instance masks via RoI Align + mask head |
| Real-time / mobile | **BiSeNet / Fast-SCNN / DeepLabv3+ (MobileNet)** | lightweight, bilinear upsampling |
| Best accuracy, compute available | **SegFormer / Mask2Former** | transformer segmenter, strong across semantic + instance |
| Zero-shot / annotation bootstrap | **SAM (Segment Anything)** | promptable masks, no task training |
| Severe class imbalance (any) | **Dice / focal / CE+Dice loss** | overlap-based, imbalance-robust |

**Decision rule:** *semantic → U-Net or DeepLab (medical/small-data → U-Net); instance → Mask R-CNN; real-time → a lightweight decoder with bilinear upsampling; unsure or few labels → prompt SAM.* Start from a pretrained encoder and a Dice-inclusive loss every time.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Distinguish semantic and instance segmentation by output.
   <details><summary>answer</summary>Semantic = per-pixel class map `H×W×C`, no instance IDs (all cars alike). Instance = `N` per-object binary masks, foreground only (car₁ ≠ car₂).</details>
2. Why does a classification CNN need a *decoder* for segmentation, and what do skip connections fix?
   <details><summary>answer</summary>The encoder downsamples to gain semantic "what" but destroys spatial "where"; the decoder upsamples back to `H×W`. Upsampling restores resolution but not lost detail — skip connections copy fine encoder features to the decoder so boundaries are sharp.</details>
3. Transposed conv vs bilinear upsampling — one advantage each, and one failure mode.
   <details><summary>answer</summary>Transposed conv: learned (task-specific) but can produce checkerboard artifacts when kernel % stride ≠ 0. Bilinear: no params/cheap/smooth but can't learn. Best practice: bilinear upsample + conv.</details>
4. Why does U-Net beat FCN on boundaries?
   <details><summary>answer</summary>U-Net concatenates skips at *every* scale (keeps coarse + fine features separate for the next conv to blend) and has convs in a symmetric decoder; FCN adds skips at only 2–3 levels. Concatenation → sharper edges.</details>
5. Why is plain cross-entropy bad for a 50-pixel tumor, and what replaces it?
   <details><summary>answer</summary>CE averages per-pixel loss; predicting "all background" scores ~99.98% pixel accuracy with 0% tumor recall. Dice loss (`1 − 2TP/(2TP+FP+FN)`) optimizes overlap → missing the tumor gives Dice≈0. Often CE + Dice.</details>
6. What three things does Mask R-CNN add to Faster R-CNN, and why RoI Align?
   <details><summary>answer</summary>A parallel FCN mask head (28×28×C), **RoI Align** instead of RoI Pooling, and a per-pixel mask loss on the GT class only. RoI Align avoids the ≤0.5-cell rounding that would misalign a whole mask outline.</details>
7. What is atrous convolution and why does DeepLab use it (with ASPP)?
   <details><summary>answer</summary>Dilated conv inserts gaps between taps → larger receptive field with *no* pooling, so spatial resolution (and boundary detail) is preserved. ASPP runs several dilation rates in parallel for multi-scale context — no decoder needed.</details>
8. State the Dice↔IoU relationship and what each emphasizes.
   <details><summary>answer</summary>`Dice = 2·IoU/(1+IoU)` (Dice ≥ IoU). IoU penalizes worst-case errors harder (≈ worst-case perf); Dice tracks average performance.</details>

---

*Covers: classification→detection→semantic→instance ladder · "what vs where" & encoder–decoder · transposed conv (formula, checkerboard) vs bilinear/UpSampling2D · addition vs concatenation skips · FCN-32s/16s/8s & 1×1-conv trick · U-Net (symmetric, concat skips, medical/30-images, elastic deformation) · Mask R-CNN (mask head, RoI Align, GT-class mask loss, RoI-relative placement) · DeepLab (atrous conv, ASPP) · pixel-CE vs Dice vs focal vs CE+Dice · IoU/mIoU/Dice/mask-mAP · paired image+mask augmentation · SAM & transformer segmenters · pretrained encoders, edge deployment, annotation cost.*
