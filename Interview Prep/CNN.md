# CNN — Interview Revision Notes
### Dense, no filler. Every concept an interviewer will probe.

---

## The One Mental Model for All of CNN

```
Raw pixels → low-level features → mid-level features → high-level features → prediction
            (edges, blobs)      (corners, textures)   (eyes, wheels, faces)

Each layer learns to detect patterns ON TOP of the previous layer's detections.
This hierarchical feature learning is WHY CNNs work for vision.
```

---

## Table of Contents
1. [Filters / Convolution](#1-filters--convolution)
2. [Stride and Padding](#2-stride-and-padding)
3. [Pooling Layers](#3-pooling-layers)
4. [Receptive Field](#4-receptive-field)
5. [CNN Architecture — Layer by Layer](#5-cnn-architecture--layer-by-layer)
6. [Batch Normalisation in CNNs](#6-batch-normalisation-in-cnns)
7. [Transfer Learning](#7-transfer-learning)
8. [Architecture Evolution](#8-architecture-evolution)
9. [Parameter Count — The Interview Trap](#9-parameter-count--the-interview-trap)
10. [Interview Answer Templates](#10-interview-answer-templates)
11. [Likely Follow-Up Questions & Answers](#11-likely-follow-up-questions--answers)
12. [Pre-Interview Checklist](#12-pre-interview-checklist)

---

# 1. Filters / Convolution

## What a Filter Is

```
A filter (kernel) is a small matrix of LEARNABLE WEIGHTS.
Typical sizes: 3×3, 5×5, 7×7 (almost always odd — to have a centre pixel)

Example 3×3 filter that detects vertical edges:
  [-1  0  1]
  [-1  0  1]
  [-1  0  1]

Strong response where left pixels are bright and right pixels are dark.
Zero response where no edge exists.
```

## What Convolution Actually Does

```
Filter slides across the input image (or feature map).
At each position:
  1. Element-wise multiply filter weights × patch of input
  2. Sum all products → single number
  3. That number = how strongly this filter pattern is present
     at this spatial location

Result = feature map (also called activation map)
         One feature map per filter.
```

## What Filters Learn at Each Layer

```
Layer 1:  simple patterns
  → vertical edges, horizontal edges, diagonal edges, colour blobs

Layer 2:  combinations of layer 1 patterns
  → corners, curves, textures, junctions

Layer 3+: combinations of layer 2 patterns
  → object parts (eyes, wheels, windows, fins)

Final layers: whole objects and their configurations
  → "this is a face", "this is a car"

Key interview point:
  Layer 1 filters are interpretable (edge detectors).
  Deep layer filters are not — they detect abstract combinations
  that don't map to human-describable concepts.
```

## Why Multiple Filters?

```
Each filter detects ONE type of pattern.
To detect many patterns, we use many filters.

32 filters on layer 1:
  → 32 feature maps
  → each map answers: "where in the image is pattern k present?"

Input:  (H × W × C)          e.g., 224 × 224 × 3
Filter: (f × f × C × K)      e.g., 3 × 3 × 3 × 32
Output: (H' × W' × K)        e.g., 224 × 224 × 32

K = number of filters = depth of output feature map
```

## Weight Sharing — Why CNNs Are Efficient

```
The SAME filter weights are used at every spatial position.
One vertical edge detector works everywhere in the image.

Without weight sharing (fully connected):
  224×224×3 input → 1 neuron = 150,528 weights

With weight sharing (conv filter 3×3×3):
  Same coverage = 27 weights TOTAL

This is why CNNs can process images with far fewer parameters
than fully connected networks.
It also gives CNNs TRANSLATION INVARIANCE — they detect
a cat in the top-left the same way as in the bottom-right.
```

---

# 2. Stride and Padding

## Stride

```
Stride = how many pixels the filter moves at each step.

Stride 1: filter moves 1 pixel at a time → dense, large output
Stride 2: filter moves 2 pixels at a time → halves spatial dimensions

Output size formula:
  Output = floor((Input - Filter + 2×Padding) / Stride) + 1

Example: Input=28, Filter=3, Stride=1, Padding=0
  Output = floor((28 - 3 + 0) / 1) + 1 = 26

Example: Input=28, Filter=3, Stride=2, Padding=1
  Output = floor((28 - 3 + 2) / 2) + 1 = 14
```

## Padding

```
VALID padding (no padding):
  → Output is SMALLER than input
  → Edge pixels are used fewer times than centre pixels
  → Spatial info at borders is lost

SAME padding (zero-pad input):
  → Output is SAME size as input (when stride=1)
  → Border pixels contribute equally
  → Preferred in deep networks to preserve spatial resolution

When to use SAME:
  → Most hidden conv layers (preserve resolution)
  → When you need output same size as input (U-Net, segmentation)

When to use VALID:
  → When you want explicit spatial reduction without pooling
  → Final layers before classification
```

---

# 3. Pooling Layers

## Purpose of Pooling

```
1. Reduce spatial dimensions → fewer parameters in later layers
2. Increase receptive field → each neuron "sees" more of the image
3. Achieve spatial invariance → small translations don't change output
4. Control overfitting → fewer parameters = less overfit

Pooling has NO LEARNABLE PARAMETERS.
It is a fixed downsampling operation.
```

## Max Pooling

```
Takes the MAXIMUM value in each pooling window.

Example (2×2 window, stride 2):
  Input patch:    Output:
  [1   3]
  [2   4]   →   4   (maximum value)

What it keeps:   the STRONGEST activation (strongest feature presence)
What it throws:  weaker activations and exact spatial location

When to use:
  ✅ Object detection / classification (presence matters, not location)
  ✅ When features are sparse (ReLU activations — many zeros)
  ✅ When you want the strongest signal from each region
  ✅ Most common default choice — works well in most cases

Limitation:
  ❌ If ALL values in window are important (not just the max)
     → information loss can hurt
  ❌ Small objects or fine-grained textures may be lost
```

## Average Pooling

```
Takes the AVERAGE value in each pooling window.

Example (2×2 window, stride 2):
  Input patch:    Output:
  [1   3]
  [2   4]   →   2.5   (average)

What it keeps:   overall activity level across the region
What it throws:  spatial structure within the window

When to use:
  ✅ Final pooling before classification head (Global Average Pooling)
  ✅ When background context matters (not just peak response)
  ✅ Texture classification (average texture intensity matters)
  ✅ When feature maps are dense (many non-zero values)

Limitation:
  ❌ A single strong feature gets diluted by surrounding weak values
     → weak detector if signal is sparse
```

## Global Average Pooling (GAP)

```
Special case: pooling window = entire feature map.
One feature map (H × W) → single number (the mean of all values)

K feature maps → K numbers → pass directly to softmax

Purpose:
  → Replaces the large fully-connected layers at the end
  → Drastically reduces parameters
  → Reduces overfitting
  → Used in: ResNet, MobileNet, GoogLeNet, modern architectures

Example:
  Feature maps: (7 × 7 × 512)
  After GAP:    (512,)         → 96% fewer parameters than FC layer!
```

## Min Pooling

```
Takes the MINIMUM value in each pooling window.

When to use:
  ✅ Detecting the ABSENCE of features (dark regions, gaps)
  ✅ Shadow detection, background modelling
  ✅ Rarely used in standard classification CNNs

Limitation:
  ❌ Rarely the right choice for standard vision tasks
  ❌ Throws away all positive activations (strong features)
```

## Side-by-Side Comparison

| | Max Pooling | Avg Pooling | Min Pooling |
|---|---|---|---|
| **Keeps** | Strongest signal | Overall activity | Weakest signal |
| **Throws** | Weaker signals | Spatial peaks | Strong signals |
| **Best for** | Classification, detection | Texture, GAP | Absence detection |
| **Sparse features** | ✅ Excellent | ❌ Dilutes signal | ❌ |
| **Dense features** | ⚠️ Loses context | ✅ Captures average | ❌ |
| **Default choice** | ✅ Yes | Only for GAP | ❌ Rarely |

## "When is max pooling's information loss NOT worth it?"

```
Interviewer will probe this. Answer:

1. Fine-grained recognition (bird species, car models)
   → Small distinguishing details can be discarded by max pool
   → Alternative: stride convolutions instead of pooling

2. Dense prediction tasks (semantic segmentation, depth estimation)
   → Need to recover full spatial resolution
   → Pooling throws away location info you need
   → Use: encoder-decoder (U-Net), dilated convolutions

3. Medical imaging (tumour detection)
   → Small but critical features must not be lost
   → Use: smaller stride, no pooling or use attention mechanisms

4. When feature maps are already small (< 8×8)
   → Further pooling destroys too much spatial information
```

---

# 4. Receptive Field

## What It Is

```
Receptive field = the region of the ORIGINAL INPUT IMAGE
                  that a single neuron in layer L can "see"

Layer 1 neuron with 3×3 filter:  receptive field = 3×3
Layer 2 neuron with 3×3 filter:  receptive field = 5×5
Layer 3 neuron with 3×3 filter:  receptive field = 7×7

Each layer adds (filter_size - 1) to the receptive field.

Why it matters:
  To detect a face (200px wide), a neuron must have
  receptive field ≥ 200px.
  Deep networks build large receptive fields from small filters.
```

## Growing Receptive Field Without Big Filters

```
One 7×7 filter:
  Parameters: 7×7 = 49 per channel
  Receptive field: 7×7

Three stacked 3×3 filters:
  Parameters: 3×(3×3) = 27 per channel   ← 45% fewer parameters!
  Receptive field: 7×7                   ← same receptive field!

This is WHY VGG uses all 3×3 filters.
Small filters, deep network = large receptive field, few parameters.
```

## Dilated / Atrous Convolutions

```
Dilated conv inserts gaps (zeros) between filter elements.
Dilation rate d=2 on a 3×3 filter:

Standard 3×3:      Dilated 3×3 (d=2):
  x x x              x . x . x
  x x x              . . . . .
  x x x              x . x . x
                      . . . . .
                      x . x . x

Receptive field: 3×3    →    5×5   (same parameters!)

Used in:
  → Semantic segmentation (DeepLab)
  → When you need large receptive field without pooling
  → Audio processing (WaveNet)
```

---

# 5. CNN Architecture — Layer by Layer

## Standard CNN Structure

```
INPUT IMAGE
    ↓
[CONV → BN → ReLU] × N     ← feature extraction block
    ↓
[POOLING]                   ← spatial reduction
    ↓
[CONV → BN → ReLU] × N     ← deeper feature extraction
    ↓
[POOLING]
    ↓
    ... repeat ...
    ↓
[GLOBAL AVERAGE POOLING]    ← spatial → vector
    ↓
[FULLY CONNECTED]           ← classification head
    ↓
[SOFTMAX]                   ← class probabilities
```

## Why This Order: Conv → BN → ReLU

```
Conv:   linear transformation (learns features)
BN:     normalise activations (stabilise training)
ReLU:   introduce non-linearity (enable complex functions)

Conv → BN → ReLU is the standard block in modern CNNs.
BN before ReLU: normalise BEFORE non-linearity → better gradient flow.
```

---

# 6. Batch Normalisation in CNNs

## What It Does

```
After conv, activations can have wildly different scales.
BN normalises activations across the batch:

For each feature map channel:
  μ    = mean activation across (batch, H, W)
  σ²   = variance of activation
  x̂   = (x - μ) / √(σ² + ε)      ← normalised
  y    = γ × x̂ + β                 ← learnable scale and shift

γ, β are LEARNED — BN can undo normalisation if useful.
```

## Why It Helps CNNs

```
✅ Allows higher learning rates → faster training
✅ Reduces sensitivity to weight initialisation
✅ Acts as mild regulariser → reduces need for dropout
✅ Smoother loss landscape → more stable training
✅ Reduces internal covariate shift

At inference: uses running mean/variance from training
              (not batch statistics — batch may be size 1)
```

---

# 7. Transfer Learning

## Why It Works for CNNs

```
Layer 1-3 of any CNN trained on images learns:
  → edge detectors, colour blobs, texture patterns
  → UNIVERSAL low-level features (same for cats, cars, X-rays)

Layer 4+ learns task-specific features.

Transfer learning: reuse the universal layers,
                   retrain only the task-specific ones.
```

## Three Strategies

```
Strategy 1: FREEZE ALL, train only classifier head
  → Use when: very small dataset (<1K images)
  → Use when: new task is similar to original task
  → Risk: if tasks are different, pretrained features may not apply

Strategy 2: FREEZE early layers, fine-tune later layers + head
  → Use when: medium dataset (1K–10K images)
  → Use when: task is somewhat different from original
  → Most common approach in practice

Strategy 3: FINE-TUNE ALL layers (very low learning rate)
  → Use when: large dataset (>10K images)
  → Use when: task is significantly different
  → Use tiny learning rate (0.0001) to not destroy pretrained weights

Rule of thumb:
  Small + similar dataset  → freeze more
  Large + different dataset → freeze less
```

## In Practice (PyTorch)

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Strategy 1: Freeze all, replace head
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for new task (e.g., 10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)
# Only model.fc parameters will be trained

# Strategy 2: Freeze early, fine-tune later
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True    # train these
    else:
        param.requires_grad = False   # freeze these
```

---

# 8. Architecture Evolution

## Why This Matters in Interviews

Interviewers ask "how has CNN architecture evolved?" to test depth.
Know the KEY INNOVATION of each architecture — not the full details.

```
LeNet (1998)
  → First CNN for digit recognition
  → Conv + AvgPool + FC
  → Key: proved CNNs work for vision

AlexNet (2012)
  → Won ImageNet by 10% margin — started deep learning era
  → Key innovations: ReLU (not tanh), Dropout, GPU training
  → 5 conv layers, 3 FC layers

VGGNet (2014)
  → Key innovation: ALL 3×3 filters, very deep (16-19 layers)
  → Proved: depth matters, small filters are better
  → Limitation: 138M parameters, very slow

GoogLeNet / Inception (2014)
  → Key innovation: Inception module (parallel 1×1, 3×3, 5×5 convs)
  → 1×1 convolutions for dimensionality reduction
  → Global Average Pooling replaced FC → drastically fewer params
  → 22 layers but FEWER parameters than AlexNet

ResNet (2015)
  → Key innovation: SKIP CONNECTIONS (residual connections)
  → Solved vanishing gradient → enabled 100+ layer networks
  → F(x) + x → network learns the RESIDUAL, not the full mapping
  → Still the default backbone for most vision tasks

MobileNet (2017)
  → Key innovation: Depthwise Separable Convolutions
  → Standard conv (3×3×C): expensive
  → Depthwise (3×3 per channel) + Pointwise (1×1): 8-9× fewer ops
  → Built for mobile / edge deployment
```

## ResNet Skip Connection — Explain This Clearly

```
Without skip connections (plain deep network):
  Gradients vanish over 50+ layers → early layers learn nothing

With skip connections:
  y = F(x) + x    (F = two conv layers, x = identity shortcut)

  Gradient flows through BOTH paths:
    Path 1: through F(x) → may vanish
    Path 2: through x directly → gradient = 1 (highway!)

  Even if path 1 gradient vanishes, path 2 carries gradient back.
  → Enables training of 152-layer networks.

Also: if F(x) = 0, layer becomes identity → network can choose
      to "skip" a layer if it's not useful. Self-regularisation.
```

---

# 9. Parameter Count — The Interview Trap

Interviewers love asking this. Know the formula cold.

## Conv Layer Parameters

```
Parameters = (F × F × C_in + 1) × C_out

F      = filter size (e.g., 3)
C_in   = input channels (e.g., 3 for RGB)
C_out  = number of filters (output channels)
+1     = bias term per filter

Example: Conv(3×3, input=3 channels, 32 filters)
  Parameters = (3 × 3 × 3 + 1) × 32 = 28 × 32 = 896

Example: Conv(3×3, input=64 channels, 128 filters)
  Parameters = (3 × 3 × 64 + 1) × 128 = 577 × 128 = 73,856
```

## Pooling Layer Parameters

```
Parameters = 0

Pooling has NO learnable parameters.
It is a fixed operation (max, average, or min).
This is a common trick question in interviews.
```

## Fully Connected Layer Parameters

```
Parameters = (input_neurons + 1) × output_neurons

Example: FC(4096 → 1000)
  Parameters = (4096 + 1) × 1000 = 4,097,000 ≈ 4M

This is why FC layers dominate parameter counts.
VGG's 3 FC layers = 119M of its 138M total parameters.
GAP replaced FC in modern CNNs to fix this.
```

## 1×1 Convolution — What Is It?

```
A 1×1 conv filter looks at ONE spatial location across ALL channels.
It is a learned weighted combination across channels.

Purpose 1: Dimensionality reduction (channel-wise)
  Input: 28×28×256  →  1×1 conv with 64 filters  →  28×28×64
  Parameters: (1×1×256+1)×64 = 16,448  (much cheaper than 3×3)

Purpose 2: Adding non-linearity without spatial mixing
  → Used heavily in Inception, ResNet bottleneck blocks

Purpose 3: Cross-channel information mixing
  → Each output combines information from all input channels

"Network in Network" — acts like a mini neural network at each pixel.
```

---

# 10. Interview Answer Templates

## "What is the purpose of filters in a CNN?"

> *"A filter is a small matrix of learnable weights that slides across the input and detects a specific pattern at every spatial location. The element-wise multiply-and-sum at each position produces a single number measuring how strongly that pattern is present there. The collection of these numbers forms a feature map. In layer 1, filters learn simple patterns like vertical edges, horizontal edges, and colour gradients. Deeper layers combine these to detect corners and textures, and deeper still to detect object parts and eventually whole objects. Different filters learn different patterns — 32 filters produce 32 different feature maps representing 32 different detected patterns. The key property is weight sharing: the same filter is applied everywhere in the image, giving CNNs translation invariance and making them far more parameter-efficient than fully connected networks."*

## "Explain pooling and when to use max vs average"

> *"Pooling reduces spatial dimensions by summarising each local region with a single value. It reduces parameters in later layers, grows the receptive field, and adds spatial invariance. Max pooling keeps the strongest activation in each window — it answers 'is this feature present anywhere in this region?' — making it ideal for classification and detection where feature presence matters more than exact location, especially when activations are sparse after ReLU. Average pooling computes the mean across the window — it captures overall activity level — making it better for texture recognition where context across the whole region matters. Global average pooling, where the window covers the entire feature map, is the modern replacement for large fully connected layers, reducing parameters dramatically. The assumption that max pooling has minimal information loss breaks down for fine-grained recognition, segmentation tasks, and medical imaging where small but critical spatial details must be preserved — in those cases, stride convolutions or dilated convolutions are better alternatives."*

## "What is a receptive field?"

> *"The receptive field of a neuron is the region of the original input image that can influence its activation. A neuron in layer 1 with a 3×3 filter has a 3×3 receptive field. A neuron in layer 2 also using 3×3 filters has a 5×5 receptive field — it sees the original image through the intermediate feature map. Receptive field grows with depth. This is why deep CNNs can detect large objects — they build up large receptive fields through stacked small filters. Three stacked 3×3 filters give the same 7×7 receptive field as one 7×7 filter but use 45% fewer parameters. That's the core insight behind VGG's design."*

---

# 11. Likely Follow-Up Questions & Answers

**Q: Why use ReLU instead of sigmoid in CNNs?**
> *"Sigmoid saturates at 0 and 1 — gradients become near-zero for large activations, causing vanishing gradients in deep networks. ReLU gradient is 1 for positive inputs — gradient flows unchanged. ReLU also makes activations sparse (negatives become zero), which is computationally efficient and acts as implicit regularisation. The dying ReLU problem (neurons stuck at zero) is handled by Leaky ReLU or ELU when it occurs."*

**Q: What is the vanishing gradient problem and how does ResNet solve it?**
> *"In deep networks, gradients are multiplied through each layer during backpropagation. With many layers, small gradients compound — the product approaches zero, and early layers receive essentially no learning signal. ResNet's skip connections add a direct path: output = F(x) + x. The gradient of x is 1, so even if F(x)'s gradient vanishes, the gradient flows back through the identity shortcut at full strength. This is what enabled training of 152-layer networks."*

**Q: What is depthwise separable convolution?**
> *"Standard conv applies a filter across all channels simultaneously — one 3×3×C filter. Depthwise separable splits this into two steps: depthwise conv applies one 3×3 filter per channel independently (spatial mixing), then pointwise conv applies 1×1 filters across channels (channel mixing). Total operations reduce by roughly C/9 times — about 8-9× fewer computations. MobileNet uses this to run on mobile devices with minimal accuracy loss."*

**Q: Dropout vs Batch Normalisation — can you use both?**
> *"They can conflict. BN uses batch statistics which are disrupted by dropout's random zeroing — the batch mean and variance become noisy. Modern CNNs typically use BN without dropout in conv layers, and dropout only in FC layers if at all. In practice, BN provides enough regularisation in the conv stack that dropout becomes unnecessary there."*

**Q: When would you NOT use transfer learning?**
> *"When the source and target domains are fundamentally different — pretrained weights on natural images may hurt rather than help on medical X-rays, satellite imagery, or microscopy. When you have massive task-specific data (millions of samples) — training from scratch may outperform fine-tuning. When the input modality is different — a model pretrained on RGB images doesn't directly transfer to depth maps or spectral data without adaptation."*

**Q: What is the difference between stride and pooling for downsampling?**
> *"Both reduce spatial dimensions. Stride convolution learns how to downsample — the filter weights are learned, so the model decides what to preserve. Pooling downsamples with a fixed rule (max or average) — no learning involved. Stride conv has more parameters but is more flexible. Pooling is parameter-free and computationally cheap. Recent architectures (ResNet, EfficientNet) prefer stride convolutions over pooling because learnable downsampling preserves more task-relevant information."*

---

# 12. Pre-Interview Checklist

```
□ Can I explain what a filter does and what it learns at each layer?
□ Can I explain weight sharing and why it matters?
□ Can I state the output size formula for a conv layer?
□ Do I know SAME vs VALID padding and when to use each?
□ Can I explain max, average, min pooling and when to use each?
□ Do I know when max pooling's information loss is NOT acceptable?
□ Can I explain Global Average Pooling and why it replaced FC?
□ Can I define receptive field and how it grows with depth?
□ Do I know why 3×3 filters are preferred over larger filters?
□ Can I explain skip connections in ResNet and why they work?
□ Can I calculate parameters in a conv layer? (and say pooling = 0)
□ Can I explain what a 1×1 convolution does?
□ Do I know the key innovation of LeNet, AlexNet, VGG, ResNet, MobileNet?
□ Can I explain the three transfer learning strategies?
□ Can I explain why BN and Dropout can conflict in CNNs?
□ Do I know stride conv vs pooling difference for downsampling?
```

---

*Interview revision notes — Filters · Convolution · Stride · Padding · Pooling · Receptive Field · BN · Transfer Learning · Architecture Evolution · Parameter Counting*