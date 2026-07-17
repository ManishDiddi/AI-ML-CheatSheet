# CNN — Convolutional Neural Networks for Vision

> **TL;DR.** A CNN learns a *hierarchy* of spatial feature detectors: early layers find edges/colours, deeper layers combine them into textures → parts → objects. Three ideas make it work on images where dense nets fail — **local filters** (small learnable kernels), **weight sharing** (same filter everywhere → translation invariance + few parameters), and **spatial downsampling** (pooling/stride → bigger receptive field, less compute). Default backbone for vision tasks; ResNet-style skip connections are what let them go deep.

**Where it fits:** Supervised (and self-supervised) learning on grid-structured data — images first, but also audio spectrograms, video, and 1-D signals. Now sharing the SOTA crown with [[vision-transformers]] (see §11).
**Prereqs:** [[backpropagation]], [[gradient-descent]], [[activation-functions]] (ReLU), [[batch-normalization]].

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example--shapes--params-through-a-net)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Architecture Evolution](#7-architecture-evolution)
8. [Interpretability — Grad-CAM & Saliency](#8-interpretability--grad-cam--saliency)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

```
Raw pixels → low-level features → mid-level features → high-level features → prediction
            (edges, blobs)      (corners, textures)   (eyes, wheels, faces)

Each layer learns to detect patterns ON TOP of the previous layer's detections.
This hierarchical feature learning is WHY CNNs work for vision.
```

- **Filter = a learnable pattern detector** slid across the image. Layer-1 filters are interpretable (edge/colour detectors); deep-layer filters detect abstract combinations that don't map to human-describable concepts.
- **The big win over a dense net:** a fully-connected layer on a 224×224×3 image needs **150,528 weights per neuron**; a 3×3×3 conv filter covers the same job with **27 weights**, reused everywhere → fewer parameters *and* translation invariance (a cat top-left is detected the same as bottom-right).

**The three image properties CNNs exploit** (name them cold — this is the "why CNN over MLP" answer):
- **Locality** — neighbouring pixels are correlated; a feature only needs a small patch → *local filters* (sparse connections).
- **Stationarity** — the same pattern recurs at many positions → *weight sharing* (one filter applied everywhere).
- **Compositionality** — high-level features are pooled from lower-level ones → *stacked conv + pooling*.

These are the CNN's built-in **inductive bias**; they map one-to-one onto local filters / weight sharing / downsampling, and are *why* a CNN needs far less data than an MLP on images — it doesn't have to learn these facts, the architecture assumes them. The flip side (MLP on images): flattening destroys spatial structure, gives no translation invariance, and needs a separate weight per pixel → parameter explosion + relearning every pattern at every location.

---

## 2. The Formal Core

**Convolution at one position:** element-wise multiply the filter by the input patch, sum to one number = "how strongly this pattern is present here." One filter → one **feature map**; `K` filters → `K` feature maps stacked as output depth.

```
Input:  (H × W × C_in)        Filter bank: (f × f × C_in × K)        Output: (H' × W' × K)
```

**Output spatial size** (memorize cold — classic interview question):
```
H' = floor( (H − f + 2P) / S ) + 1        f=filter, P=padding, S=stride
```

**Parameter counts:**
```
Conv layer:  params = (f × f × C_in + 1) × C_out      (+1 = bias per filter)
Pooling:     params = 0                                ← trick question; say it unprompted
FC layer:    params = (n_in + 1) × n_out               ← where param counts explode
```

**Receptive field** (region of the *input* one neuron sees) grows `+(f−1)` per stacked layer: three 3×3 layers → 7×7 RF with `3×(3·3)=27` params vs one 7×7 layer's `49` params — same RF, **45% fewer params** (this is why VGG uses only 3×3). `(certain)` Pooling accelerates this: a 2×2/stride-2 pool roughly *doubles* the RF-growth rate, so RF grows ~exponentially with depth+pooling vs linearly with conv alone — that's a second reason pooling matters beyond compute. The **effective** RF is smaller than this theoretical box: central input pixels reach a neuron through many more paths than border pixels, so influence tapers ~Gaussian from the centre. `(certain)`

**Batch Norm (per channel):** `x̂ = (x − μ)/√(σ²+ε)`, then `y = γx̂ + β` with learnable `γ, β`. `μ, σ²` over `(batch, H, W)`.

---

## 3. How It Works

### Filters & weight sharing
Example 3×3 vertical-edge detector `[-1,0,1] / [-1,0,1] / [-1,0,1]` → strong response where left is bright and right is dark, zero where no edge. Many filters = many patterns: 32 filters on layer 1 → 32 feature maps, each answering *"where is pattern k present?"* The **same** weights are applied at every position (weight sharing) → translation invariance + parameter efficiency.

### Why kernels are odd-sized (3×3, 5×5 — never 2×2 or 4×4)
An odd kernel has a **single central pixel**, so its output maps back to a well-defined integer centre with the previous-layer pixels sitting symmetrically around it. An even kernel's centre falls *between* pixels (at x=0.5) → asymmetric padding and a half-pixel shift that accumulates distortion across stacked layers. 🎯 *"Odd sizes give a symmetric receptive field with a real centre pixel — that's why 3×3 is the workhorse and you never see 2×2 conv kernels."*

### Stride & padding
```
Stride S = pixels the filter hops. S=1 → dense/large output; S=2 → halves H,W.

VALID (no pad):  output SMALLER; border pixels used fewer times; edge info lost.
SAME  (zero-pad): output SAME size at S=1; borders contribute equally.
  → SAME for most hidden layers & segmentation (U-Net); VALID for explicit shrink / final layers.
```

### Pooling — fixed (parameter-free) downsampling
Purpose: shrink spatial dims, grow receptive field, add small-translation invariance, cut overfitting.

| | Max Pooling | Avg Pooling | Min Pooling |
|---|---|---|---|
| **Keeps** | Strongest signal | Overall activity | Weakest signal |
| **Best for** | Classification/detection | Texture, GAP | Absence detection |
| **Sparse features (post-ReLU)** | ✅ Excellent | ❌ Dilutes | ❌ |
| **Default choice** | ✅ Yes | Only for GAP | ❌ Rarely |

**Global Average Pooling (GAP):** window = whole feature map → each `(H×W)` map becomes one number → `K` maps → `K`-vector straight to softmax. Replaces the giant FC head (ResNet/Inception/MobileNet): e.g. `7×7×512 → 512`, ~96% fewer params, acts as a regularizer.

### Receptive field — small filters, deep stacks
Each layer adds `(f−1)`. To detect a 200-px face a neuron needs RF ≥ 200 px; deep nets build that from stacked 3×3s. **Dilated/atrous conv** inserts gaps between filter taps → RF grows (3×3 at dilation 2 → 5×5 RF) with the *same* params, no pooling — used in DeepLab segmentation and WaveNet audio.

### The standard block: `Conv → BN → ReLU`
```
Conv : linear feature extraction
BN   : normalize activations (stable training) — BEFORE the non-linearity → better gradient flow
ReLU : non-linearity
```
**Batch Norm** lets you use higher learning rates, reduces init sensitivity, mildly regularizes (often replacing dropout in conv stacks), and smooths the loss landscape. At **inference** it uses the running mean/variance saved during training (a batch may be size 1).

---

## 4. Worked Example — shapes & params through a net

Input `224×224×3`, all convs 3×3 / stride 1 / SAME pad:

```
Conv(32)  → 224×224×32   params = (3·3·3 + 1)·32   = 896
MaxPool2  → 112×112×32   params = 0
Conv(64)  → 112×112×64   params = (3·3·32 + 1)·64  = 18,496
MaxPool2  → 56×56×64     params = 0
GAP       → (64,)        params = 0
FC 64→10  → (10,)        params = (64 + 1)·10       = 650
                         ─────────────────────────────────────
                         TOTAL learnable params ≈ 20,042
```
**Why GAP matters:** flattening `56×56×64 = 200,704` into an FC→10 head instead would cost `(200,704+1)·10 ≈ 2.0M` params — **100× more** than the entire GAP network above. That single design choice is most of why modern CNNs are small.

---

## 5. Code / Implementation

**Convolution from scratch (NumPy) — makes the §2 output-size formula concrete:**
```python
import numpy as np

def conv2d(x, kernel, stride=1, padding=0):
    """2D convolution as done in deep learning — technically CROSS-CORRELATION
    (no kernel flip; true math convolution flips the kernel first).
      x:      (H, W)  single-channel input feature map
      kernel: (f, f)  filter of weights (a pattern detector)
      returns (H_out, W_out), where  H_out = floor((H - f + 2P)/S) + 1   ← the §2 formula
    """
    if padding > 0:
        x = np.pad(x, padding, mode="constant")          # zero-pad borders (SAME-style)
    H, W = x.shape
    f = kernel.shape[0]
    H_out = (H - f) // stride + 1                         # apply the output-size formula
    W_out = (W - f) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):                                # slide the window down…
        for j in range(W_out):                           # …and across
            r, c = i * stride, j * stride                # top-left of this receptive field
            patch = x[r:r + f, c:c + f]                   # the input region the filter sees
            out[i, j] = np.sum(patch * kernel)           # element-wise multiply, then SUM → one number
    return out

# Sanity check with the vertical-edge filter from §3:
img   = np.tile(np.array([10, 10, 10, 0, 0], float), (5, 1))   # bright left half, dark right half
vedge = np.array([[-1, 0, 1]] * 3, float)                      # detects a light→dark vertical edge
print(conv2d(img, vedge, stride=1))     # ~0 in flat regions, large |response| at the 10→0 boundary
                                        # 5×5 in, 3×3 filter, stride 1 → (5-3)//1+1 = 3×3 output
```
*Extension to real conv layers:* for an `(H, W, C_in)` input and `(f, f, C_in)` filter you also **sum over the input channels**; `K` filters produce `K` stacked output maps (`H_out × W_out × K`). Production code never loops like this — it uses **im2col** (unfold patches into a matrix, then one big matmul) or FFT, which is what makes GPU convolution fast.

**A small CNN (the canonical block, in PyTorch):**
```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # Conv→BN→ReLU
            nn.MaxPool2d(2),                                                  # 224→112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                                  # 112→56
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                           # GAP → (B,64)
            nn.Linear(64, n_classes),                                        # tiny head
        )
    def forward(self, x): return self.head(self.features(x))
```

**Transfer learning (the workhorse — match the strategy to your data):**
```python
import torchvision.models as models, torch.nn as nn
model = models.resnet50(weights="IMAGENET1K_V2")

# Strategy 1 — freeze all, train only the new head (tiny dataset, similar task)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)        # only this trains

# Strategy 2 — freeze early, fine-tune late blocks + head (medium data, somewhat different task)
for name, p in model.named_parameters():
    p.requires_grad = ("layer4" in name) or ("fc" in name)
# Strategy 3 — fine-tune ALL at a tiny LR (1e-4) for large/different datasets (don't wreck pretrained weights)
```
Rule of thumb: **small + similar → freeze more; large + different → freeze less.**

---

## 6. When It Breaks

```
❌ Max-pool info loss hurts when location/detail matter:
   • fine-grained recognition (bird species) → discards distinguishing detail
   • dense prediction (segmentation, depth) → need full resolution → U-Net / dilated conv
   • medical imaging → tiny critical features lost → smaller stride / attention
   • feature maps already < 8×8 → further pooling destroys too much
❌ Vanishing gradients in plain deep nets (>50 layers) → early layers stop learning → use ResNet skips.
❌ Dying ReLU — neurons stuck at 0; mitigate with LeakyReLU/ELU or careful init.
❌ BN ↔ Dropout conflict — dropout's random zeroing makes BN's batch stats noisy; modern CNNs use BN in
   conv stacks and reserve dropout (if any) for FC layers.
❌ Transfer learning fails when domains differ hard (ImageNet → X-ray/satellite/microscopy) or input
   modality differs (RGB → depth/spectral) — pretrained features may hurt.
❌ NOT translation-equivariant the way people assume: aggressive striding/pooling + border padding break it
   (the "CNNs can't tell where things are" aliasing issue → anti-aliased downsampling helps).
```

---

## 7. Architecture Evolution

Know the **one key innovation** of each (interviewers test depth here, not full details):

```
LeNet (1998)      First CNN (digits). Conv + AvgPool + FC. Proved CNNs work for vision.
AlexNet (2012)    Won ImageNet by ~10% → started the DL era. ReLU, Dropout, GPU training.
VGG (2014)        ALL 3×3 filters, very deep (16–19). Depth matters; but 138M params, slow.
GoogLeNet/Incep.  Inception module (parallel 1×1/3×3/5×5) + 1×1 dim-reduction + GAP. Fewer params than AlexNet.
ResNet (2015)     SKIP CONNECTIONS → solved vanishing gradients → 100+ layers. Still the default backbone.
MobileNet (2017)  Depthwise-separable convs → 8–9× fewer ops. Built for mobile/edge.
EfficientNet(2019) Compound scaling — scale depth/width/resolution together by one coefficient → SOTA efficiency.
```

**ResNet skip connection (explain crisply):**
```
y = F(x) + x         F = two conv layers, x = identity shortcut

Gradient flows through BOTH paths; the identity path has gradient = 1 (a "highway"),
so even if F(x)'s gradient vanishes, signal still reaches early layers.
Bonus: if F(x)=0 the block becomes identity → the net can "skip" a useless layer (self-regularizing).
```

**1×1 convolution** = a learned weighted mix across channels at one pixel ("network-in-network"). Uses: (1) **channel dim-reduction** cheaply (`28×28×256 → 28×28×64`), (2) add non-linearity without spatial mixing, (3) cross-channel mixing — the backbone of Inception and ResNet **bottleneck** blocks.

**Depthwise-separable conv** (MobileNet): split a standard `3×3×C` conv into **depthwise** (one 3×3 per channel = spatial mixing) + **pointwise** (1×1 across channels = channel mixing) → ~`C/9` the compute.

---

## 8. Interpretability — Grad-CAM & Saliency

You almost always must answer *"why did the model predict this?"* — to debug, build trust, or catch **shortcut learning** (e.g. classifying "horse" off a watermark, or pneumonia off the scanner type). For CNNs the standard tools:

- **Grad-CAM** — `P0`, the SHAP-equivalent for vision. Take the gradient of the class score w.r.t. the **last conv layer's** feature maps, global-average those gradients into per-channel weights, combine → a coarse **heatmap of where the network looked** for that class. Class-discriminative, needs no retraining, works on any CNN.
- **Saliency / Integrated Gradients** — gradient of the output w.r.t. **input pixels** → fine-grained but noisy pixel attributions.
- **Occlusion sensitivity** — slide a gray patch over the image and watch the prediction drop; wherever it drops most is what the model relied on. Slow but model-agnostic and very convincing to stakeholders.

```python
# Grad-CAM in ~5 lines of intent (pytorch-grad-cam):
from pytorch_grad_cam import GradCAM
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])   # last conv block
heatmap = cam(input_tensor=img)[0]      # overlay on img → "where it looked"
```
🎯 **Interview line:** *"For a CNN I reach for Grad-CAM — gradient-weighted activation maps of the last conv layer give a class-discriminative heatmap, which is how I catch the model latching onto background/watermark shortcuts instead of the object."*

---

## 9. Production & MLOps Notes

### Data & training (where most accuracy actually comes from)
- **Data augmentation** is the cheapest, biggest regularizer for vision — random flips, crops, rotation, colour jitter; stronger: **RandAugment**, **Mixup/CutMix** (blend images+labels). Often worth more than architecture tweaks. `(certain)`
- **Input preprocessing must match the backbone:** resize, then normalize with the *pretrained model's* channel mean/std (ImageNet `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`). Mismatch silently tanks transfer-learning accuracy.
- **Training tricks:** mixed-precision (AMP) for ~2× speed/half memory; LR **warmup + cosine decay**; class imbalance → weighted loss / focal loss / oversampling; **label smoothing** to curb overconfidence.

### Serving, compression & edge
- **Model compression** for latency/size: **quantization** (FP32→INT8, ~4× smaller/faster), **pruning** (drop low-magnitude weights/channels), **knowledge distillation** (train a small "student" to mimic a big "teacher"). Pick **MobileNet/EfficientNet** backbones for edge from the start.
- **Inference runtimes:** export to **ONNX**, accelerate with **TensorRT / OpenVINO / Core ML**; batch requests; pin a fixed input size (use `AdaptiveAvgPool` to tolerate variable sizes).

### Monitoring & reliability
- **CNNs are overconfident** — softmax probabilities aren't calibrated; apply **temperature scaling** on a held-out set if you threshold on confidence. `(certain)`
- **Watch image drift** (brightness/contrast/resolution/source-camera shifts) and **out-of-distribution** inputs (a confident prediction on garbage is the classic failure). Track per-class precision/recall on labeled feedback, not just accuracy.
- Different vision tasks reuse the CNN backbone with different heads — **detection** ([YOLO, Faster R-CNN](Object%20Detection.md)), **segmentation** ([U-Net, Mask R-CNN](Image%20Segmentation.md)); know that the backbone transfers but the head/loss/metrics (mAP, IoU) change.

---

## 10. Interview Lens

> ⚡ **Golden rule:** restate what's asked, diagnose the *why* before the *what*, then answer in 6–8 sentences.

**"Purpose of filters?"** → 🎯 *"Small learnable kernels slid across the input that detect a specific pattern at every location; stacking layers builds edges→textures→parts→objects, and weight sharing gives translation invariance with far fewer params than a dense net."*

**"Max vs average pooling?"** → 🎯 *"Max keeps the strongest activation ('is the feature present?') — best for classification on sparse post-ReLU maps; average keeps overall activity — best for texture and as Global Average Pooling replacing the FC head. Max's info loss bites in fine-grained, segmentation, and medical tasks → use stride/dilated convs there."*

**Likely follow-ups:**
- *ReLU vs sigmoid?* → Sigmoid saturates → vanishing gradients; ReLU gradient is 1 for positives, sparse, cheap. Dying-ReLU → Leaky/ELU. `(certain)`
- *Vanishing gradient & ResNet?* → Products of small gradients → ~0 for early layers; skip `y=F(x)+x` adds an identity path (grad 1) → trains 152 layers.
- *Depthwise-separable conv?* → depthwise (per-channel spatial) + pointwise (1×1 channel mix) → ~`C/9` ops (MobileNet).
- *Stride vs pooling for downsampling?* → both shrink; stride conv **learns** what to keep (ResNet/EfficientNet prefer it), pooling is a fixed parameter-free rule.
- *Params in `Conv(3×3, 64→128)`?* → `(3·3·64+1)·128 = 73,856`; pooling = 0.
- *When NOT to use transfer learning?* → big domain gap, huge task-specific data, or different input modality.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Standard image task, limited data | **CNN (ResNet/EfficientNet)** | Strong inductive bias (locality, translation) → data-efficient |
| Lots of data + compute, want SOTA | **Vision Transformer (ViT)** | Global self-attention, scales better at very large data |
| Edge / mobile latency | **MobileNet / EfficientNet-Lite** | Depthwise-separable, compound-scaled → small & fast |
| Best of both | **ConvNeXt / hybrid** | CNN design modernized to match ViT accuracy |
| Tiny dataset, simple task | Classical CV (HOG/SIFT) + classifier | No data to train conv filters |
| Dense prediction | U-Net / DeepLab | Encoder-decoder / dilated convs preserve resolution |

**ViT in one line:** split the image into patches, linearly embed them, run a Transformer with self-attention. **Less built-in inductive bias than CNNs → needs more data (or heavy augmentation/pretraining), but global context from layer 1 and better scaling.** In practice: CNNs still win on small/medium data and latency; ViTs/hybrids win at large scale. `(likely)`

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Output size of a 28×28 input, 3×3 filter, stride 2, padding 1?
   <details><summary>answer</summary>`floor((28−3+2)/2)+1 = 14`. Use `floor((H−f+2P)/S)+1`.</details>
2. Why do three stacked 3×3 convs beat one 7×7?
   <details><summary>answer</summary>Same 7×7 receptive field but `27` vs `49` params per channel (~45% fewer) and two extra non-linearities → more expressive. VGG's core insight.</details>
3. Why did GAP replace big FC heads?
   <details><summary>answer</summary>FC params explode (`(n_in+1)·n_out`; VGG's FCs were 119M of 138M). GAP turns each feature map into one number → tiny head, regularizes, near-free.</details>
4. How does a ResNet skip connection fix vanishing gradients?
   <details><summary>answer</summary>`y=F(x)+x`; the identity path has gradient 1, a highway that carries signal to early layers even when F's gradient vanishes. Also lets a block become identity if useless.</details>
5. Params in a pooling layer, and why is this a trick question?
   <details><summary>answer</summary>Zero — pooling is a fixed (max/avg) op with no learnable weights. Say it unprompted.</details>
6. How do you tell *why* a CNN made a prediction, and what bug does it catch?
   <details><summary>answer</summary>**Grad-CAM** (gradient-weighted last-conv activation heatmap), saliency, or occlusion. Catches **shortcut learning** — model keying on watermark/background/scanner instead of the object.</details>
7. You ship a fine-tuned ResNet and accuracy is far below validation. Two prime suspects?
   <details><summary>answer</summary>(1) **Preprocessing mismatch** — not normalizing with ImageNet mean/std the backbone expects; (2) **train/serve drift** or OOD inputs. Also check **overconfidence** → temperature-scale before thresholding.</details>
8. Edge device, model too big/slow. Three compression levers?
   <details><summary>answer</summary>**Quantization** (INT8), **pruning** (drop small weights/channels), **knowledge distillation** (small student mimics big teacher) — and pick a MobileNet/EfficientNet backbone.</details>

---

*Covers: filters · convolution · stride/padding · pooling · GAP · receptive field · dilated conv · BN · the Conv-BN-ReLU block · architecture evolution (LeNet→EfficientNet) · ResNet skips · 1×1 & depthwise-separable conv · parameter counting · Grad-CAM interpretability · augmentation · compression/deployment · ViT comparison.*
