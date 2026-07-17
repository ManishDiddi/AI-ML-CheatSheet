# Object Detection — locate *and* classify every object: sliding windows → Faster R-CNN → YOLO

> **TL;DR.** Detection = **classification + localization for *multiple* objects**: the model outputs a *set* of `(box, class, confidence)` triples. Two families split on the speed/accuracy trade-off: **two-stage** (R-CNN → Faster R-CNN — first *propose* regions, then *classify+refine* them; most accurate) and **single-stage** (YOLO, SSD — one forward pass over a dense grid of predictions; real-time fast). Both share the same plumbing: **anchors** (reference boxes), **IoU** (overlap metric), **NMS** (dedupe overlapping boxes), and **mAP** (the evaluation metric). Pick single-stage for real-time/edge, two-stage when accuracy on small/dense objects matters most.

**Where it fits:** the vision task one rung above classification — the backbone is still a [CNN](Convolutional%20Neural%20Networks%20for%20Vision.md), but with detection *heads* and a matching loss/metric stack. It's what powers self-driving perception, surveillance, retail analytics, medical screening.
**Prereqs:** [CNN fundamentals](Convolutional%20Neural%20Networks%20for%20Vision.md) (conv/1×1/GAP, depthwise-separable, architecture evolution), [Transfer Learning](Transfer%20Learning.md) (every detector rides a pretrained backbone), and IoU/NMS (built up in §2–3 here).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — boxes, anchors, IoU, losses, mAP](#2-the-formal-core--boxes-anchors-iou-losses-map)
3. [How It Works — the two families](#3-how-it-works--the-two-families)
4. [Worked Examples — box regression, YOLO grid, IoU](#4-worked-examples--box-regression-yolo-grid-iou)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Efficient Backbones & Production Notes](#7-efficient-backbones--production-notes)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

**The task ladder — each rung adds output:**

```
Classification        Localization             Object Detection
"what is it?"         "what + WHERE (1 box)"   "what + where for EVERY object"
   [cat]                 [cat]┌───┐               ┌───┐cat   ┌──┐dog
                              │   │               │   │      │  │
                              └───┘               └───┘  ┌───┴─┐person
   1 label               1 label + 4 coords       └─────┘
                                              a SET of (box, class, score)
```

- **Classification** → one label for the whole image (CNN + softmax).
- **Localization** → *one* object: add a **regression head** predicting 4 box coordinates alongside the class head. Output is *fixed-size* (4 numbers) → **exactly one box** → can't handle multiple objects.
- **Detection** → *variable* number of objects → you need machinery to output an unknown-length set of boxes. That's the whole problem.

**Two ways to produce that set — the family split:**

```
TWO-STAGE  (propose → classify)          SINGLE-STAGE (one shot)
Image → backbone → "where might objects   Image → backbone → dense grid of predictions
        be?" (region proposals)                   (every cell/anchor predicts box+class
      → classify + refine each proposal            in ONE forward pass)
   ✅ accurate (esp. small/dense)          ✅ FAST / real-time
   ❌ slower (two passes)                   ❌ weaker on tiny/overlapping objects
   R-CNN → Fast R-CNN → Faster R-CNN        YOLO, SSD, RetinaNet
```

🎯 *"Two-stage detectors trade speed for accuracy by first proposing regions then classifying them; single-stage detectors like YOLO regress boxes and classes directly from a grid in one pass — real-time, at some cost on small/dense objects. Both dedupe with NMS and are scored by mAP."*

---

## 2. The Formal Core — boxes, anchors, IoU, losses, mAP

**Box representation.** A box is 4 numbers, in one of two conventions — know both and the conversion (a constant source of bugs):
```
center form:  (x_c, y_c, w, h)          ← YOLO / annotations
corner form:  (x1, y1, x2, y2)          ← drawing / IoU
convert:  x1 = x_c − w/2,  y1 = y_c − h/2,  x2 = x_c + w/2,  y2 = y_c + h/2
```
Coordinates are usually **normalized to [0,1]** by image width/height so they're resolution-independent.

**IoU (Intersection over Union)** — the overlap metric that underpins *everything* (evaluation, anchor labeling, NMS):
```
IoU = area(intersection) / area(union)          ∈ [0, 1]
      0 = no overlap, 1 = perfect match
```

**Anchors / default boxes** — the trick that lets a *fixed*-output CNN emit variable detections. At every feature-map location you pre-place `k` reference boxes of assorted **scales × aspect ratios**; the network predicts, *per anchor*, an offset + objectness + class. Faster R-CNN: scales `[128,256,512]` × ratios `[0.5,1,2]` → **9 anchors/location**. SSD: ~6 default boxes/cell. YOLO: anchors per grid cell (from v2 on).

**Box regression is predicted as offsets to an anchor, not absolute coords** (easier to learn, scale-normalized):
```
t_x = (x − x_a) / w_a       t_w = log(w / w_a)
t_y = (y − y_a) / h_a       t_h = log(h / h_a)          (a = anchor)
```
Log for w,h keeps them positive and makes the target symmetric around 0. Smooth-L1 is the loss on these (robust to outliers vs L2).

**Anchor labeling (the RPN/SSD ground truth), by IoU with GT boxes:**
```
IoU ≥ 0.7  → positive (object)        IoU ≤ 0.3  → negative (background)
0.3 < IoU < 0.7 → ignore (don't train on ambiguous anchors)
```

**Output tensor — YOLO's grid.** Split the image into an `S×S` grid; a cell is responsible for an object **iff the object's center falls in it** (this rule prevents double-counting). Per cell you predict box coords + an **objectness/confidence** (= P(object)·IoU) + `C` class scores:
```
YOLO output = S × S × (B·5 + C)     B = boxes/anchors per cell, 5 = (x,y,w,h,conf)
lecture's simplest case (B=1):  S × S × (C + 5)  →  e.g. 4×4×(3+5) for 3 classes
```

**Losses** — detection loss is always a *sum* of a classification term and a localization term:
```
Two-stage (Faster R-CNN) = 4 losses:
   RPN objectness (obj/no-obj)  +  RPN box regression
 + detector classification      +  detector box regression
SSD "MultiBox" loss  =  confidence_loss (softmax CE)  +  α · localization_loss (Smooth-L1)
```

**Metrics — precision/recall at an IoU threshold, summarized as mAP:**
- A prediction is a **True Positive** iff `IoU(pred, GT) ≥ threshold` (e.g. 0.5) *and* the class matches; else False Positive. Missed GTs are False Negatives.
- **AP** (Average Precision) = area under the precision–recall curve for one class (sweeping the confidence threshold).
- **mAP** = mean AP over all classes. `mAP@0.5` (PASCAL VOC) uses IoU≥0.5; **`mAP@[.5:.95]`** (COCO) averages mAP over IoU thresholds 0.5→0.95 — the strict modern standard.
- **top-k accuracy** (from the classification side) is still useful for fine-grained classes where near-ties aren't real errors.

---

## 3. How It Works — the two families

### 3a. Single-object localization (the starting point)
Take a pretrained backbone, remove the classifier, attach **two heads** off the shared features:
- **classification head** → `Dense(n_classes, softmax/sigmoid)`
- **regression head** → `Dense(4, sigmoid)` for the box (normalized coords)

Train with a **combined loss** (e.g. `BCE` for class + `MSE`/Smooth-L1 for box), optionally weighting the box loss higher. Simple and effective — but the fixed 4-coord output means **exactly one box**, so it can't detect multiple objects. That limitation motivates everything below.

### 3b. Multi-object, the naïve way: sliding window (why it's bad)
Slide a fixed box across the image at many positions **and scales**, classifying each crop ("car / not car"). Turns detection into classification — but:
```
❌ O(N²) windows × several scales → thousands of CNN passes per image → far too slow
❌ redundant computation on overlapping crops
❌ one object fires in many windows → duplicate detections (→ need NMS)
```
Everything after this is "sliding window, but smart."

### 3c. Non-Maximum Suppression (NMS) — dedupe overlapping boxes
Every detector produces multiple boxes for the same object; NMS keeps the best:
```
1. Pick the box with the highest confidence → move it to the KEEP list.
2. Compute IoU of that box with all remaining boxes.
3. Drop every box with IoU > threshold (they're duplicates of the same object).
4. Repeat with the next-highest remaining box until none are left.
```
Run per class. Threshold ~0.45. (Failure mode: two genuinely-overlapping objects → NMS may delete a true box; **Soft-NMS** decays scores instead of hard-dropping to mitigate.)

### 3d. Two-stage family: R-CNN → Fast R-CNN → Faster R-CNN
```
R-CNN (2014)         Selective Search → ~2000 region proposals
                     → run the CNN on EACH cropped region (2000 forward passes!)
                     → SVM classify + separate box regressor
                     ❌ agonizingly slow, multi-phase training, not end-to-end

Fast R-CNN (2015)    Run the CNN ONCE on the whole image → feature map
                     → Selective Search proposals mapped onto the feature map
                     → RoI Pooling crops each proposal to a fixed 7×7 → FC heads
                     → softmax class + box regression, trained end-to-end
                     ❌ still bottlenecked by Selective Search (slow, offline, fixed)

Faster R-CNN (2015)  Replace Selective Search with a learned Region Proposal Network (RPN)
                     that SHARES the backbone features → proposals in-network
                     🔥 near-real-time, fully learned, one unified model
```

- **Selective Search** = a classical region proposer: over-segment the image, then greedily **merge adjacent similar segments** into ~2000 candidate boxes. Cheaper than exhaustive sliding windows, but hand-crafted and fixed (doesn't learn).
- **RoI Pooling** = max-pool each variable-size proposal down to a fixed `7×7` grid so a fixed FC head can consume it (FC layers need constant input size). **RoI Align** (from Mask R-CNN) is the upgrade: RoI Pooling *rounds* proposal coordinates to feature-map cells → sub-pixel **misalignment**; RoI Align uses **bilinear interpolation, no rounding** → crucial for precise localization/segmentation.
- **RPN** = a small **fully-convolutional** net on the backbone feature map: a `3×3` conv then two `1×1` heads — an **objectness** head (`H×W×k`: object vs background per anchor) and a **box-offset** head (`H×W×4k`). It says "*where* to look"; the second-stage head says "*what* it is." Proposal pipeline: generate anchors → RPN scores+offsets → apply offsets → clip to image → drop tiny boxes → sort by score → NMS → keep top-N (~300).
- **Feature-map ↔ image coordinates (a classic gotcha):** the backbone downsamples by a **stride** (e.g. 800→50 ⇒ stride 16). Anchors are placed on the feature map but **boxes always live in *image* space** — multiply feature coords by stride to get image coords, divide to go back. Mix these up and your boxes land in the wrong place.

### 3e. Single-stage family: YOLO & SSD
- **YOLO ("You Only Look Once"):** one CNN pass predicts the entire `S×S×(B·5+C)` grid — all boxes, objectness, and classes simultaneously. A cell owns an object iff the object's **center** lands in it. Backbone = **Darknet-53** (53 conv layers, residual blocks; chosen for best accuracy-per-FLOP and GPU efficiency vs ResNet-101/152). This single-pass design is why YOLO hits real-time frame rates. Version arc: **v1 (2015)** grid regression → **v2/YOLO9000** anchors + higher mAP → **v3** multi-scale predictions, Darknet-53 → **v4** (2020) → **v5** (Ultralytics, PyTorch, the applied-community default) → v6/v7/v8+.
- **SSD (Single Shot Detector):** VGG base + extra conv layers; crucially, **multiple feature maps at different depths each feed the detector**, so *different scales are detected at different layers* (early/large maps → small objects, deep/small maps → big objects). ~6 **default boxes** per cell. Loss = **MultiBox** = confidence (softmax CE) + α·localization (Smooth-L1).
- **YOLO vs SSD:** both are single-pass; YOLO(v1) ended in FC layers on a coarse grid, SSD is fully-convolutional and multi-scale (better across object sizes). Modern YOLO adopted multi-scale (FPN-like) too.

---

## 4. Worked Examples — box regression, YOLO grid, IoU

**Box-regression targets.** Anchor `(x_a,y_a,w_a,h_a) = (100,100,50,50)`, ground-truth box `(120,110,60,40)`:
```
t_x = (120−100)/50 = 0.40      t_w = log(60/50) = log(1.2) ≈  0.18
t_y = (110−100)/50 = 0.20      t_h = log(40/50) = log(0.8) ≈ −0.22
→ the RPN/regressor learns to output (0.40, 0.20, 0.18, −0.22) for this anchor.
```

**YOLO output shape.** 3 classes (Car, Light, Pedestrian), `4×4` grid, 1 box/cell:
```
output = 4 × 4 × (C + 5) = 4 × 4 × (3 + 5) = 4 × 4 × 8
per cell = [x, y, w, h, objectness, p(Car), p(Light), p(Pedestrian)]
```
Only the cell containing an object's *center* is trained to fire for it → no double counting.

**IoU.** Two 50×50 boxes offset by (20,10): intersection = `(50−20)×(50−10) = 30×40 = 1200`; union = `2·2500 − 1200 = 3800`; `IoU = 1200/3800 ≈ 0.32` → below a 0.5 threshold, so this would count as a **False Positive** at mAP@0.5.

---

## 5. Code / Implementation

**Single-object localization — one backbone, two heads (Keras):**
```python
import tensorflow as tf
base = tf.keras.applications.ResNet101(weights="imagenet", include_top=False,
                                       input_tensor=tf.keras.Input((416,416,3)))
base.trainable = False                                   # transfer learning: freeze backbone
flat = tf.keras.layers.Flatten()(base.output)

def head(x, units=(128,64,32)):
    for u in units: x = tf.keras.layers.Dense(u, activation="relu")(x)
    return x

cls = tf.keras.layers.Dense(1, activation="sigmoid", name="class_output")(head(flat))  # gun / no-gun
box = tf.keras.layers.Dense(4, activation="sigmoid", name="box_output")(head(flat))     # (x,y,w,h) norm
model = tf.keras.Model(base.input, [box, cls])

model.compile(tf.keras.optimizers.Adam(1e-4),
    loss   ={"class_output": "binary_crossentropy", "box_output": "mse"},
    loss_weights={"class_output": 1.0, "box_output": 4.0},   # weight box loss higher
    metrics={"class_output": "accuracy", "box_output": "mse"})
```

**NMS (the dedupe every detector needs):**
```python
import numpy as np
def nms(boxes, scores, iou_thr=0.45):           # boxes as (x1,y1,x2,y2)
    keep, idxs = [], scores.argsort()[::-1]      # process high-confidence first
    while len(idxs):
        i = idxs[0]; keep.append(i)
        ious = np.array([iou(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][ious <= iou_thr]         # drop boxes overlapping the kept one
    return keep
```

**Practical detection: pretrained YOLOv5 (ONNX via OpenCV DNN — CPU real-time):**
```python
import cv2, numpy as np
net = cv2.dnn.readNet("yolov5s.onnx")            # or use `ultralytics` YOLO for the modern API
blob = cv2.dnn.blobFromImage(frame, 1/255., (640,640), swapRB=True, crop=False)
net.setInput(blob); outputs = net.forward(net.getUnconnectedOutLayersNames())
# post-process: filter by objectness ≥ 0.5 and class-confidence ≥ 0.45, then cv2.dnn.NMSBoxes(...)
```
For training/fine-tuning use **`ultralytics`** (`YOLO("yolov5s.pt").train(data=...)`). In PyTorch, `torchvision.models.detection` ships **`fasterrcnn_resnet50_fpn`**, **`retinanet_resnet50_fpn`**, and **`ssd300_vgg16`** pretrained on COCO — load, swap the head's `num_classes`, fine-tune. Export to **ONNX** for cross-framework deployment.

---

## 6. When It Breaks

```
❌ Foreground–background class imbalance — the defining single-stage problem. A one-stage detector
   scores ~10⁴–10⁵ candidate boxes/image; almost all are background → the loss is swamped by easy
   negatives and the model under-learns objects. FIX: RetinaNet's FOCAL LOSS, which down-weights
   easy examples: FL = −(1−p_t)^γ · log(p_t). Two-stage sidesteps this because the RPN pre-filters
   to ~300 proposals. 🎯 This imbalance is THE reason early one-stage detectors trailed two-stage on accuracy.

❌ Small / dense / occluded objects. YOLO-style grids assign one object per cell → crowds and tiny
   objects (satellite imagery, far-away pedestrians) get missed. FIX: multi-scale features (FPN),
   higher input resolution, more anchors, or a two-stage detector.

❌ NMS deletes true overlapping objects. Two people hugging → high mutual IoU → NMS suppresses one.
   FIX: Soft-NMS (decay scores instead of hard-drop) or a higher NMS threshold.

❌ Anchor sensitivity. Wrong anchor scales/ratios for your domain (e.g. long thin objects) tanks recall.
   FIX: k-means the anchor sizes on YOUR data, or use anchor-free detectors (FCOS, CenterNet).

❌ RoI Pooling misalignment. Rounding proposal coords to feature cells shifts boxes by up to a stride
   → hurts precise localization. FIX: RoI Align (bilinear, no rounding).

❌ Coordinate-space bugs. Predicting/clipping boxes in feature-map space instead of image space; mixing
   center vs corner form; forgetting to un-normalize. These are the most common implementation errors.

❌ Threshold coupling. Objectness threshold, class-confidence threshold, and NMS IoU threshold jointly
   control precision/recall — they must be tuned together on a val set, not guessed.
```

---

## 7. Efficient Backbones & Production Notes

**MobileNet — the efficient backbone that makes on-device detection (MobileNet-SSD) possible.** Standard conv does spatial filtering *and* channel mixing at once (cost `k²·C·N`). **Depthwise-separable convolution** ([mechanics in the CNN note](Convolutional%20Neural%20Networks%20for%20Vision.md#7-architecture-evolution)) splits it into **depthwise** (one `k×k` filter per channel — spatial only) + **pointwise** (`1×1` — channel mixing only), cutting cost to `≈ 1/N + 1/k²` (**~8–9× cheaper**) for ~1–2% accuracy loss.
- **MobileNet V1:** depthwise-separable everywhere + two knobs — **width multiplier α** (scales channel counts; params/ops ∝ α²) and **resolution multiplier ρ** (scales input size; ops ∝ ρ²) — to dial any speed/accuracy point. (4.2M params, 569M ops, 70.6% ImageNet top-1.)
- **MobileNet V2** adds two ideas: the **inverted residual block** — *expand* channels (×6) → depthwise-filter in that *rich, wide* space → *compress* back (narrow→wide→narrow, opposite of ResNet's bottleneck) — and the **linear bottleneck** — **no ReLU after the final compression**, because ReLU zeroing on a *low*-dimensional (narrow) output destroys too much information. Plus ResNet-style skip connections when shapes match. Result: *smaller AND faster AND more accurate* than V1 (3.4M params, 300M ops, 72.0%). `(certain)`

**Beyond anchors and NMS (know these exist):**
- **FPN (Feature Pyramid Network)** — fuse features top-down across scales → detect small and large objects well; now standard in Faster R-CNN, RetinaNet, YOLO.
- **Anchor-free detectors** — **FCOS, CenterNet** predict boxes per-pixel/as keypoints, dropping anchor tuning entirely.
- **DETR** — transformer-based *set prediction*: no anchors, no NMS (bipartite matching gives one box per object). Simpler pipeline, needs more training. See [[Vision Transformers]].
- **Mask R-CNN** — Faster R-CNN + a mask head + RoI Align → instance segmentation; the bridge to [[Image Segmentation]].

**Deployment & ops.**
- **Real-time is the constraint.** Autonomous/surveillance need CPU-real-time inference; a pretrained YOLOv5-nano processes a frame in a few hundred ms on CPU. For **video**, sample frames (you rarely need all 24–30 fps) to cut cost proportionally.
- **Runtimes/frameworks:** export to **ONNX**; serve via OpenCV DNN / TensorRT / TFOD API / `ultralytics`. Quantize (INT8) and pick a MobileNet/EfficientDet backbone for edge.
- **Tune the three thresholds** (objectness, class-confidence, NMS-IoU) to your precision/recall needs; a safety system (gun/pedestrian detection) wants high recall, a retail counter wants high precision.
- **Data & annotation:** boxes are expensive to label; use pretrained COCO weights and fine-tune. **Mosaic/mixup augmentation** (stitch 4 images) is a big win for detectors. Watch **train/inference resolution** consistency.
- **Monitoring:** track **mAP** (per-class, @[.5:.95]) on a labeled feedback set, plus latency percentiles; detection quality drifts as scene/camera/lighting changes.

---

## 8. Interview Lens

> ⚡ The through-line: detection = classification + localization for a *variable* number of objects; anchors + IoU + NMS + mAP are the shared machinery; the two families trade speed vs accuracy.

**"Two-stage vs single-stage — when each?"** → 🎯 *"Two-stage (Faster R-CNN) proposes regions with an RPN then classifies/refines them — most accurate, best on small/dense objects, but slower. Single-stage (YOLO/SSD) regresses boxes and classes directly from a grid in one forward pass — real-time, at some accuracy cost on tiny/overlapping objects, largely because of foreground-background class imbalance (fixed by focal loss). I'd use YOLO for real-time/edge and Faster R-CNN when accuracy on hard objects dominates."*

**Likely follow-ups:**
- *What is an anchor and why?* → Pre-placed reference boxes (scales × ratios) at each location; the net predicts *offsets* to them → turns a fixed-output CNN into a variable-box detector and makes regression easier. `(certain)`
- *What does the RPN do?* → A fully-conv net that predicts objectness + box offsets per anchor → proposes ~300 regions; "tells the network where to look." Faster R-CNN's key innovation over Fast R-CNN (learned proposals vs Selective Search).
- *RoI Pooling vs RoI Align?* → Pooling rounds coords → misalignment; Align uses bilinear interpolation, no rounding → precise (from Mask R-CNN).
- *What is NMS and when does it fail?* → Greedy dedupe by IoU; fails on genuinely overlapping objects → Soft-NMS.
- *Why do single-stage detectors underperform, and the fix?* → Extreme fg/bg imbalance (10⁴–10⁵ boxes, few objects) → focal loss (RetinaNet) down-weights easy negatives.
- *How is detection evaluated?* → mAP: TP needs IoU≥threshold + right class; AP = area under PR curve; mAP averages over classes; COCO mAP@[.5:.95] averages over IoU thresholds.
- *YOLO output shape?* → `S×S×(B·5+C)`; a cell owns an object iff its center is inside → no double counting.
- *Box regression parametrization?* → offsets to anchor: `t_x=(x−x_a)/w_a`, `t_w=log(w/w_a)`; Smooth-L1 loss (robust).
- *Common myths to reject?* → anchors are **not** learned; boxes live in **image** not feature space; objectness ≠ IoU (it's a predicted probability); the RPN gives **proposals**, not final detections.

---

## 9. Alternatives & How to Choose

| Need | Reach for | Why |
|---|---|---|
| Real-time / video / edge | **YOLO (v5/v8), SSD, MobileNet-SSD** | single pass, small backbone → high FPS |
| Max accuracy, small/dense objects | **Faster R-CNN + FPN** | region proposals + refinement + multi-scale |
| One-stage but accurate | **RetinaNet (focal loss)** | fixes fg/bg imbalance → one-stage speed, two-stage-ish accuracy |
| Avoid anchor tuning | **FCOS / CenterNet (anchor-free)** | per-pixel/keypoint boxes, fewer hyperparameters |
| Simplest pipeline, no NMS | **DETR / transformer detectors** | set prediction via bipartite matching |
| Also need per-pixel masks | **Mask R-CNN** | Faster R-CNN + mask head → [[Image Segmentation]] |
| Single object only | **2-head localization CNN** | classification + 4-coord regression, trivial to train |
| Tiny labeled dataset | **Pretrained COCO detector + fine-tune** | boxes are expensive; transfer learning is the default |

**Decision rule:** *default to a pretrained YOLO for real-time and Faster R-CNN+FPN when accuracy on hard objects wins; escalate to RetinaNet/anchor-free/DETR to remove a specific pain point (imbalance, anchor tuning, NMS).* Almost never train a detector from scratch — start from COCO weights.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why can a single-object localization CNN not do object detection?
   <details><summary>answer</summary>Its regression head outputs a *fixed* 4 coordinates → exactly one box. Detection needs a *variable* number of boxes, which requires anchors/grids + NMS.</details>
2. Trace R-CNN → Fast R-CNN → Faster R-CNN, naming what each fixed.
   <details><summary>answer</summary>R-CNN: Selective Search + a CNN pass per region (2000×) + SVM — slow. Fast R-CNN: one CNN on the whole image + **RoI Pooling** + FC heads, end-to-end — but still Selective Search. Faster R-CNN: replace Selective Search with a learned **RPN** sharing the backbone → near-real-time, fully learned.</details>
3. What are anchors, and what does the box head actually predict?
   <details><summary>answer</summary>Pre-placed reference boxes (scales × ratios) per location. The head predicts *offsets* to an anchor: `t_x=(x−x_a)/w_a`, `t_y=(y−y_a)/h_a`, `t_w=log(w/w_a)`, `t_h=log(h/h_a)` — not absolute coordinates.</details>
4. Give the NMS algorithm and a failure case.
   <details><summary>answer</summary>Pick highest-confidence box → drop all remaining boxes with IoU > threshold → repeat. Fails on genuinely overlapping objects (deletes a true box) → use Soft-NMS.</details>
5. Why do single-stage detectors historically lag on accuracy, and the fix?
   <details><summary>answer</summary>Extreme foreground/background imbalance (10⁴–10⁵ boxes, few objects) drowns the loss in easy negatives → **focal loss** (RetinaNet) down-weights easy examples: `−(1−p_t)^γ log p_t`.</details>
6. RoI Pooling vs RoI Align?
   <details><summary>answer</summary>RoI Pooling rounds proposal coords to feature-map cells → sub-pixel misalignment. RoI Align uses bilinear interpolation with no rounding → accurate localization (introduced by Mask R-CNN).</details>
7. Define mAP@[.5:.95] and what makes a detection a True Positive.
   <details><summary>answer</summary>TP = correct class **and** IoU(pred, GT) ≥ the IoU threshold. AP = area under the precision–recall curve per class; mAP = mean over classes; COCO's mAP@[.5:.95] averages that over IoU thresholds 0.5→0.95 (step 0.05).</details>
8. In one line each: depthwise-separable conv, inverted residual, linear bottleneck.
   <details><summary>answer</summary>Depthwise-separable = per-channel spatial conv + 1×1 channel mix (~8–9× cheaper). Inverted residual (MobileNetV2) = expand→depthwise→compress (narrow→wide→narrow). Linear bottleneck = no ReLU after compression, since ReLU destroys info in low-dim space.</details>

---

*Covers: classification→localization→detection ladder · box formats & IoU · anchors/default boxes · box-regression offsets & Smooth-L1 · YOLO grid `S×S×(B·5+C)` & SSD multi-scale · sliding window → NMS (+Soft-NMS) · two-stage R-CNN/Fast/Faster, Selective Search, RPN, RoI Pooling vs RoI Align · feature-map↔image stride · focal loss & class imbalance · mAP/AP/mAP@[.5:.95] · MobileNet V1/V2 (depthwise-separable, inverted residual, linear bottleneck, α/ρ) · FPN, anchor-free, DETR, Mask R-CNN · real-time deployment, thresholds, mosaic aug, monitoring.*
