# Neural Network Fundamentals — Neurons, Forward Pass & Backpropagation

> **TL;DR.** A neuron is just [logistic regression](../Supervised%20ML/Logistic%20Regression.md): `z = w·x + b`, then a non-linear **activation** `a = f(z)`. Stack layers of them and the **non-linearities** let the network carve *non-linear* decision boundaries that a single linear model can't (spirals, XOR). Training is a loop: **forward propagation** (`Z = XW + b → A = f(Z)`, layer by layer, to a prediction + loss) → **backpropagation** (the chain rule pushes the error backward to get `∂Loss/∂W`) → **gradient descent** (`W ← W − η·∂Loss/∂W`). The gotcha that shaped modern DL: **sigmoid/tanh saturate** (derivative ≤ 0.25) so gradients **vanish** through deep stacks — **ReLU** (`max(0,z)`) fixed that and unlocked depth (at the cost of **dying ReLU**, patched by **Leaky ReLU**).

**Where it fits:** The foundation of **deep learning** — every CNN, RNN, and Transformer is this plus structure. It's [logistic/softmax regression](../Supervised%20ML/Logistic%20Regression.md) stacked and made non-linear. Optimizers/initialization get their own note: [Weight Initialization & Optimizers](Weight%20Initialization%20%26%20Optimizers.md).
**Prereqs:** [Logistic Regression](../Supervised%20ML/Logistic%20Regression.md) (sigmoid, log-loss), softmax & [cross-entropy](../Supervised%20ML/Classification%20Metrics.md), [gradient descent & the chain rule](../Supervised%20ML/Linear%20Regression.md), matrix multiplication.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Single Neuron = Logistic Regression](#2-the-single-neuron--logistic-regression)
3. [From Neurons to a Network — Architecture](#3-from-neurons-to-a-network--architecture)
4. [Forward Propagation](#4-forward-propagation)
5. [Loss Functions](#5-loss-functions)
6. [Backpropagation](#6-backpropagation)
7. [Activation Functions](#7-activation-functions)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks](#9-when-it-breaks)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Classical models struggle on data that isn't linearly separable — [logistic regression](../Supervised%20ML/Logistic%20Regression.md) draws a straight boundary, so it fails on a spiral; [trees](../Supervised%20ML/Decision%20Trees.md) make boxy cuts; [KNN](../Supervised%20ML/KNN.md) doesn't scale. A neural network **learns its own features** by stacking simple units.

```
INPUT ──▶ HIDDEN LAYER(s) ──▶ OUTPUT
 x        each neuron: z = w·x + b, then a = f(z)        ŷ
          (a linear cut, then bent by a non-linearity)

one neuron  = one linear boundary (logistic regression)
many layers = many bent boundaries combined → ANY shape (spiral, circle, XOR)
```

- 🎯 **Why it works:** a linear map then a non-linear activation, repeated, tiles the space into regions — the **universal approximation theorem** says a big enough network can approximate any function. Remove the activations and the whole stack **collapses into one linear layer** (matrix products of matrices are still a matrix). The non-linearity is the entire point. `(certain)`
- Each hidden neuron becomes a **feature detector**; deeper layers compose simpler features into complex ones.

---

## 2. The Single Neuron = Logistic Regression

A neuron takes inputs, weights them by *importance*, adds a *bias*, and squashes the result: `(certain)`

```
z = w₁x₁ + w₂x₂ + … + w_dx_d + b  =  w·x + b        # weighted sum ("linear combination")
a = f(z)                                             # activation (e.g. sigmoid → probability)
```

- **Weights** = how much each input matters (a big negative weight on "temperature" ⇒ hotter → less likely to touch). **Bias** = the neuron's default lean/threshold. **Activation** = turns the score into a usable output.
- 🎯 **A single neuron with a sigmoid is exactly [logistic regression](../Supervised%20ML/Logistic%20Regression.md)** (`σ(w·x+b)`). A neural network is just *many* of these, layered — which is why everything you know about log-loss and the sigmoid carries straight over. `(certain)`

---

## 3. From Neurons to a Network — Architecture

Stack neurons into **layers**: an **input layer**, one or more **hidden layers**, and an **output layer**. A **dense (fully-connected) layer** connects every input to every neuron. `(certain)`

```
For a layer with n inputs and h neurons:
   W ∈ ℝ^(n×h)   (each COLUMN = one neuron's weights)     b ∈ ℝ^(1×h)
   Z = X W + b        # one matrix multiply does ALL neurons for ALL samples at once
```

- The matrix form is a *speed* trick: `Z = XW + b` replaces looping over every neuron × sample. `(certain)`
- **Output layer** shape depends on the task: 1 sigmoid neuron (binary), `k` **softmax** neurons (multi-class), 1 linear neuron (regression).
- **Multi-class = multiple neurons + softmax** (vs logistic regression's one-vs-rest) — the softmax gives a joint probability distribution over classes.

---

## 4. Forward Propagation

"Forward prop" is the data flowing input → output to produce a prediction: `(certain)`

```
Layer 1:  Z⁽¹⁾ = X W₁ + b₁   →   A⁽¹⁾ = f(Z⁽¹⁾)      (e.g. ReLU)
Layer 2:  Z⁽²⁾ = A⁽¹⁾ W₂ + b₂ →  A⁽²⁾ = softmax/σ(Z⁽²⁾)   (the prediction ŷ)
Loss:     compare A⁽²⁾ to the true label
```

Each layer feeds the next; the final activation matches the task (softmax for classes). Use the **numerically stable softmax** (subtract the row max before `exp`) to avoid overflow. `(certain)`

---

## 5. Loss Functions

The loss scores how wrong the prediction is (the "report card"): `(certain)`
- **Binary cross-entropy** (1 sigmoid output): `L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]`.
- **Softmax + categorical cross-entropy** (k classes): softmax turns logits into probabilities (`P_i = e^{z_i}/Σe^{z_j}` — `exp` makes them positive and *amplifies* the top score), then `L = −log(P_correct)` — only the true class's probability matters. Perfect prediction → `log(1)=0` loss; a confident miss → huge loss. `(certain)`

(These are the same losses as in [Logistic Regression](../Supervised%20ML/Logistic%20Regression.md); the network just feeds them richer features.)

---

## 6. Backpropagation

Training needs `∂Loss/∂W` for every weight, but loss connects to `W` only *through* `Z → A → Loss`. **Backpropagation = the chain rule applied backward, layer by layer:** `(certain)`

```
∂L/∂W = ∂L/∂Z · ∂Z/∂W
```

🎯 **The elegant part:** for **softmax + cross-entropy** (and sigmoid + BCE), the output error collapses to a beautiful form: `(certain)`
```
dZ = A − Y              # prediction − truth  (Y = one-hot)   ← "how wrong, per class"
dW = Xᵀ · dZ            # blame each weight by its input
db = Σ dZ               # bias gradient
W ← W − η · dW          # gradient descent step
```
Intuition: if you predicted 24% for the true class, `dZ = 0.24 − 1 = −0.76` → negative → *push that weight up*; a wrong class at 0.32 → `dZ = +0.32` → *push it down*.

**Through hidden layers**, keep chaining (transpose to keep shapes right; `⊙` = elementwise): `(certain)`
```
dZ⁽²⁾ = A⁽²⁾ − Y
dW₂   = A⁽¹⁾ᵀ · dZ⁽²⁾
dA⁽¹⁾ = dZ⁽²⁾ · W₂ᵀ                       # each hidden neuron's blame = error × its outgoing weight
dZ⁽¹⁾ = dA⁽¹⁾ ⊙ f'(Z⁽¹⁾)                  # multiply by the activation's derivative
dW₁   = Xᵀ · dZ⁽¹⁾
```

**Worked weight update:** input `x=[1,2]`, weight `W[0,0]=0.1` (input-1 → Cat), predicted 24% Cat, truth = Cat.
```
dZ_Cat = 0.24 − 1 = −0.76
dW[0,0] = x[0]·dZ_Cat = 1·(−0.76) = −0.76
W_new = 0.1 − 0.1·(−0.76) = 0.1 + 0.076 = 0.176      # weight increased → higher Cat score next time ✅
```

---

## 7. Activation Functions

Without a non-linear activation, layers collapse to one linear layer (§1). But the *choice* of activation makes or breaks deep training. `(certain)`

**Ideal activation:** differentiable, non-linear, cheap to compute, and a **gradient near 1 over a wide range** of `z`.

| Activation | `f(z)` | Issue |
|---|---|---|
| **Sigmoid** | `1/(1+e⁻ᶻ)` → (0,1) | **saturates**: derivative ≤ **0.25**, ~0 in the tails → vanishing gradient |
| **tanh** | (−1,1), zero-centered | derivative ≤ 1 but still saturates in the tails |
| **ReLU** | `max(0, z)` | fast, non-saturating for `z>0` (derivative 1) → **enabled deep nets**; but **dying ReLU** |
| **Leaky ReLU** | `z` if `z>0` else `α·z` | small slope `α` for `z<0` keeps the gradient alive → fixes dying ReLU |

🎯 **The vanishing gradient problem:** a sigmoid's derivative is `σ(z)(1−σ(z)) ≤ 0.25`. Backprop *multiplies* these across layers, so through `n` sigmoid layers the gradient shrinks like `≤ 0.25ⁿ` → early layers barely update → deep sigmoid nets don't train. `(certain)`

**Why ReLU won:** its derivative is exactly **1** for positive inputs, so gradients don't shrink layer-to-layer — this is what made *deep* networks trainable. `(certain)`

**Dying ReLU:** for `z ≤ 0` the derivative is 0, so a neuron stuck in the negative region gets **zero gradient forever** and never updates (a "dead" neuron). **Leaky ReLU** gives negatives a small slope `α` (e.g. 0.01) so some gradient always flows. `(certain)`

---

## 8. Code / Implementation

**From scratch — the whole loop (one dense layer, softmax):**
```python
class Net:
    def __init__(self, d, k, lr=0.1):
        self.W = np.random.randn(d, k) * 0.01   # small random → breaks symmetry (see §9)
        self.b = np.zeros((1, k)); self.lr = lr
    def softmax(self, Z):
        e = np.exp(Z - Z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)  # stable
    def forward(self, X):
        self.X, self.A = X, self.softmax(X @ self.W + self.b); return self.A
    def step(self, y):                           # y = integer labels
        m = len(y)
        dZ = self.A.copy(); dZ[range(m), y] -= 1; dZ /= m   # dZ = A − Y
        self.W -= self.lr * (self.X.T @ dZ)                 # dW = Xᵀ dZ
        self.b -= self.lr * dZ.sum(0, keepdims=True)
```

**Keras — how you'd actually build it:**
```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(d,)),  # hidden layer
    keras.layers.Dense(k, activation="softmax"),                  # output layer
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
```

---

## 9. When It Breaks

- **Vanishing gradients** — sigmoid/tanh in deep stacks (§7). Fix: **ReLU/variants**, residual connections, batch norm, and careful **initialization** (see [Weight Initialization & Optimizers](Weight%20Initialization%20%26%20Optimizers.md)).
- **Exploding gradients** — gradients blow up (common in deep/recurrent nets). Fix: gradient clipping, normalization, better init.
- **Dying ReLU** — dead neurons stuck at 0 gradient → Leaky ReLU / smaller learning rate / better init.
- **Symmetry — why you must random-initialize.** If all weights start equal, every neuron computes the same thing and receives the same gradient → they stay identical forever and the layer is useless. Random init **breaks symmetry** so neurons specialize. (Init strategy is its own topic — [Weight Initialization & Optimizers](Weight%20Initialization%20%26%20Optimizers.md).) `(certain)`
- **Overfitting** — lots of parameters memorize noise; combat with **[dropout, batch norm](Batch%20Normalization%20%26%20Dropout.md), early stopping, weight decay (L2)**, and more data.
- **Data & compute hungry** — NNs need large datasets and (usually) GPUs; on small tabular data, [gradient-boosted trees](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md) often win.

---

## 10. Interview Lens

**"Why do neural networks need non-linear activations?"** → 🎯 *"Without them, stacking layers is just multiplying matrices — the whole network collapses to a single linear model. The non-linearity is what lets stacked layers carve non-linear boundaries and approximate arbitrary functions."*

**"Walk me through backpropagation."** → 🎯 *"Forward pass computes predictions and loss; backprop applies the chain rule backward to get ∂Loss/∂W. For softmax+cross-entropy the output error is simply `A − Y`; then `dW = Xᵀ·dZ`, pass `dZ·Wᵀ` to the previous layer, multiply by that activation's derivative, repeat. Gradient descent then updates `W ← W − η·dW`."*

**Likely follow-ups:**
- *How is a neuron related to logistic regression?* → A single sigmoid neuron *is* logistic regression; a network is many, layered.
- *What is the vanishing gradient problem?* → Sigmoid/tanh derivatives are ≤ 0.25/1 and backprop multiplies them, so gradients shrink exponentially with depth → early layers stop learning. ReLU (derivative 1) fixes it.
- *Dying ReLU and the fix?* → Neurons with `z≤0` get 0 gradient and never update; Leaky ReLU gives negatives a small slope.
- *Why random weight initialization?* → To break symmetry — equal weights make all neurons learn the same thing forever.
- *Why is the softmax+cross-entropy gradient so clean?* → It simplifies to `prediction − truth` (`A − Y`); the softmax and log derivatives cancel.
- *Why subtract the max in softmax?* → Numerical stability — prevents `exp` overflow without changing the result.
- *Sigmoid vs ReLU for hidden layers?* → ReLU — non-saturating, cheap, avoids vanishing gradients; sigmoid only really for a binary output.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Small/medium tabular data | **[Gradient-boosted trees](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)** | usually beat NNs on tabular; less data/compute |
| Linear-ish, need interpretability | **[Logistic Regression](../Supervised%20ML/Logistic%20Regression.md)** | a single neuron; transparent |
| Images | **[CNNs](../Convolutional%20Neural%20Networks%20for%20Vision.md)** | weight sharing / spatial structure |
| Sequences / text | **[RNNs / Transformers](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)** | model order and long-range dependencies |
| Large data, complex non-linear patterns | **Deep neural network (this note)** | learns features automatically, scales |

**Decision rule:** on tabular data start with trees/logistic regression; move to neural networks when the data is large and the structure is complex or perceptual (images, audio, text) — then add the right *structure* (CNN/RNN/Transformer) on top of these fundamentals.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Why does a network need non-linear activations?
   <details><summary>answer</summary>Without them, composed linear layers reduce to a single linear layer, so the network can only draw linear boundaries. Non-linearities let stacked layers approximate arbitrary (non-linear) functions.</details>
2. Write the softmax+cross-entropy output gradient and the weight gradient.
   <details><summary>answer</summary>`dZ = A − Y` (prediction − one-hot truth); `dW = Xᵀ·dZ`; `db = Σ dZ`; update `W ← W − η·dW`.</details>
3. Explain the vanishing gradient problem and why ReLU helps.
   <details><summary>answer</summary>Sigmoid's derivative ≤ 0.25; backprop multiplies derivatives across layers, so gradients shrink like `0.25ⁿ` → early layers stop learning. ReLU's derivative is 1 for positive inputs, so gradients don't shrink layer-to-layer.</details>
4. What is dying ReLU and how do you fix it?
   <details><summary>answer</summary>A neuron stuck at `z≤0` has derivative 0 → zero gradient → it never updates ("dead"). Leaky ReLU gives negatives a small slope `α` so gradient still flows.</details>
5. Why can't you initialize all weights to the same value?
   <details><summary>answer</summary>Symmetry: identical weights → identical outputs and identical gradients → neurons stay identical forever. Random init breaks symmetry so neurons specialize.</details>
6. How does a single neuron relate to logistic regression?
   <details><summary>answer</summary>A single neuron with a sigmoid activation *is* logistic regression (`σ(w·x+b)`); a neural network is many such units stacked in layers.</details>
7. In `Z = XW + b`, what are the shapes for n inputs, h neurons, m samples?
   <details><summary>answer</summary>`X`: (m×n), `W`: (n×h), `b`: (1×h), `Z`: (m×h). Each column of W is one neuron's weights.</details>

---

*Covers: why NN over linear models (non-linearity + universal approximation), the neuron as logistic regression, dense-layer architecture & the matrix form `Z=XW+b`, forward propagation, loss (BCE, softmax/cross-entropy), backpropagation (chain rule, `dZ=A−Y`, `dW=XᵀdZ`, layer-by-layer), activation functions (sigmoid/tanh saturation, vanishing gradients, ReLU, dying ReLU, Leaky ReLU), symmetry-breaking initialization, and overfitting. Sourced from Scaler NN lectures 1–2. Optimizers & initialization: [Weight Initialization & Optimizers](Weight%20Initialization%20%26%20Optimizers.md).*
