# RNN · LSTM · Transformers — Sequence Modeling

> **TL;DR.** Three answers to "how do I model a sequence." **RNN** reads step-by-step into one hidden vector → vanishing gradients + sequential bottleneck. **LSTM/GRU** adds a gated cell state (a gradient "highway") → learns long-range dependencies, but is still sequential and memory-bounded. **Transformer** drops recurrence for **self-attention** — every token attends directly to every other, in parallel → no bottleneck, long-range in one hop, but `O(n²)` cost. Transformers are today's default for almost all NLP (and beyond); RNNs survive in streaming/edge/tiny-data niches.

**Where it fits:** Any ordered data — text, speech, time series, code, biological sequences. The Transformer is the substrate of all modern LLMs.
**Prereqs:** [[backpropagation]], [[gradient-descent]], [[softmax]], [[word-embeddings]], [[layer-normalization]].

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works](#3-how-it-works)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Model Families — BERT vs GPT vs T5](#7-model-families--bert-vs-gpt-vs-t5)
8. [Interpretability — Attention & Its Caveat](#8-interpretability--attention--its-caveat)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

```
RNN         → reads sequence step by step into ONE fixed memory vector
              Problem: memory bottleneck + vanishing gradient
LSTM        → step by step, but separates short + long term memory (gated cell state)
              Problem: still sequential, still a fixed-size bottleneck
Transformer → reads the ENTIRE sequence at once, direct token-to-token attention
              Solves: parallelism + no bottleneck + long-range dependencies (at O(n²) cost)
```

---

## 2. The Formal Core

**RNN step:** `hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)`, optional output `ŷₜ = softmax(Wᵧ·hₜ)`. `hₜ` is the *only* memory.

**LSTM gates (know cold):**
```
fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf)     forget   iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi)   input
c̃ₜ = tanh(Wc·[hₜ₋₁,xₜ]+bc)  candidate  oₜ = σ(Wo·[hₜ₋₁,xₜ]+bo)   output
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ    cell update      hₜ = oₜ ⊙ tanh(cₜ)   hidden
```
σ squashes to (0,1) → acts as a gate; ⊙ = element-wise.

**Scaled dot-product attention (the heart of it all):**
```
Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
  Q=X·Wq "what I'm looking for"   K=X·Wk "what I contain"   V=X·Wv "what I contribute"
  /√d_k  scales dot products so softmax doesn't saturate (vanishing grads) as d_k grows
```
**Multi-head:** `Concat(head₁..headₕ)·Wₒ`, each `headᵢ = Attention(QWqᵢ, KWkᵢ, VWvᵢ)` with `d_k = d_model/h`.

**Positional encoding** (attention is order-blind, so inject position): original is sinusoidal `PE(pos,2i)=sin(pos/10000^{2i/d})`, `PE(pos,2i+1)=cos(...)`. Modern LLMs use **RoPE** (rotary, encodes *relative* position, extrapolates to longer contexts) or **ALiBi** (linear attention bias) instead. `(certain)`

**Complexity:** self-attention is `O(n²·d)` (every pair of tokens) vs RNN's `O(n·d²)` (linear in length) — the core Transformer trade-off.

---

## 3. How It Works

### RNN & the vanishing gradient (the problem that started it all)
Backprop through time multiplies a Jacobian per step: `∂L/∂h₁ = ∂L/∂hₙ · ∏ ∂hₜ/∂hₜ₋₁`, and each `∂hₜ/∂hₜ₋₁ = tanh'(·)·Wₕ`. Since `tanh' ∈ (0,1)`, over a length-500 sequence you multiply ~499 sub-1 terms → `∂L/∂h₁ ≈ 0` → early tokens get no gradient → long-range context is forgotten. The mirror case `Wₕ>1` gives **exploding gradients** (NaNs) → fixed by **gradient clipping** (`g ← g·thresh/‖g‖`).

### LSTM — the gated fix
Two memories: **cell state `cₜ`** (long-term "conveyor belt") and **hidden `hₜ`** (short-term output). Why it fixes vanishing gradients:
```
RNN path:   ∂cₜ/∂cₜ₋₁ = tanh'(·)·W   → always <1 → vanishes
LSTM path:  ∂cₜ/∂cₜ₋₁ = fₜ            → JUST the forget gate
            fₜ ≈ 1 → gradient flows unchanged → cell state is a gradient HIGHWAY
```
It's a *learned, adaptive* gate (set `fₜ≈1` to remember) vs RNN's fixed multiplication. **GRU** merges cell+hidden into 2 gates (reset, update) → fewer params, faster, similar accuracy; prefer it when compute is tight / sequences shorter.

### Transformer — read everything at once
High-level: `Embeddings + Positional Encoding → N×[Multi-Head Self-Attn → Add&Norm → FFN → Add&Norm]` (encoder); decoder adds **masked** self-attention + **cross-attention**, then `Linear+Softmax`.

- **Multi-head attention:** each head learns a different relation (head A: subject→verb, head B: coreference, head C: local context) → richer than one attention.
- **Add & LayerNorm:** `LayerNorm(x + SubLayer(x))`. The **residual** is a gradient highway (like ResNet) enabling deep stacks (BERT 12, GPT-3 96 layers). **LayerNorm not BatchNorm** because batch stats are unstable for variable-length, padded sequences — LayerNorm normalizes each token across features, stable at any batch size. *(Modern nets use **Pre-LN**: `x + SubLayer(LayerNorm(x))` — trains more stably than the original Post-LN.)*
- **FFN:** `max(0, xW₁+b₁)W₂+b₂`, expand 4× then compress (512→2048→512). Attention is a *linear* weighted average; the FFN adds the non-linearity. *"Attention decides WHERE to look; FFN decides WHAT to do with what it saw."* (Modern LLMs swap ReLU for **GELU/SwiGLU**.)
- **Masked self-attention (decoder):** set future-position scores to `−∞` before softmax → 0 attention to future → can't "cheat" by seeing the answer during generation.
- **Cross-attention:** `Q` from decoder, `K,V` from encoder output → decoder asks "which source tokens matter for the token I'm generating?"

**Why it beats LSTM on all fronts:** parallel (all positions at once) · no fixed bottleneck (direct token↔token) · long-range in one hop (token 1 ↔ token 500 directly) · residuals prevent vanishing gradients. The cost: `O(n²)` memory/compute.

---

## 4. Worked Example

**(a) Attention resolves coreference.** *"The animal didn't cross the street because **it** was tired."* — what is "it"?
```
scores for "it":  it→animal = 0.8   it→street = 0.1   it→The = 0.1   (after softmax)
output("it") = 0.8·V(animal) + 0.1·V(street) + 0.1·V(The)
            → "it"'s representation is dominated by "animal" → coreference resolved ✅
```

**(b) Why LSTM beats RNN on a long review.** *"Incredible visuals, stunning performances… **however**, the second half falls apart and the ending is a disappointment."* → sentiment = **negative**.
```
RNN : by "disappointment", the hidden state was overwritten ~60× and early gradients vanished;
      early positive words dominate → predicts POSITIVE ❌
LSTM: forget gate fires at "however", erasing the early positive sentiment from the cell state;
      preserves the negative turning point → predicts NEGATIVE ✅
```

---

## 5. Code / Implementation

**Scaled dot-product attention from scratch** (the whole idea in 6 lines):
```python
import torch, torch.nn.functional as F

def attention(Q, K, V, mask=None):           # Q,K,V: (batch, heads, seq, d_k)
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / d_k**0.5         # (.., seq, seq) pairwise scores
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # causal or padding mask
    weights = F.softmax(scores, dim=-1)                   # attention distribution per token
    return weights @ V, weights                           # blended values + the weights
```

**Don't hand-roll in practice — use the optimized building blocks:**
```python
import torch.nn as nn
enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
# Or, for real work, a pretrained model:
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")     # subword tokenizer (see §9)
model = AutoModel.from_pretrained("bert-base-uncased")
out = model(**tok("the cat sat", return_tensors="pt")).last_hidden_state  # contextual embeddings
```

---

## 6. When It Breaks

```
RNN
❌ Vanishing gradient → can't learn long-range dependencies
❌ Exploding gradient → instability (fix: clipping)
❌ Sequential → no parallelism → slow training
❌ Fixed hidden vector → information bottleneck

LSTM (after fixing gradients)
❌ Still sequential — 50-word × 1M sentences = 50M sequential steps; wastes parallel GPUs
❌ Fixed-size cell/hidden (e.g. 512-d) still bottlenecks very long documents
❌ Token 1 → token 1000 must survive 999 forget-gate multiplications (no direct path)

Transformer
❌ O(n²) memory/compute → long sequences (books, long context) are expensive
❌ Weak inductive bias (order injected artificially) → needs more data / pretraining than RNNs
❌ Hallucination & no grounding in plain LMs (see §9: RAG)
❌ Quadratic KV-cache memory growth at long context during generation
```

---

## 7. Model Families — BERT vs GPT vs T5

```
            Architecture     Objective            Best for
BERT        Encoder-only     Masked LM (+NSP)     UNDERSTANDING: classification, NER, QA, embeddings
                             bidirectional        sees full context; cannot generate
GPT         Decoder-only     Causal LM            GENERATION: completion, chat, agents
                             left-to-right        the modern LLM default
T5          Encoder-Decoder  Text-to-text         SEQ2SEQ: translation, summarization (in/out distinct)
```
- **BERT Masked LM:** input `"The [MASK] sat on the mat"` → predict `cat`; masking *forces* use of both-side context (a next-token objective would just be copied) → rich contextual embeddings.
- **GPT Causal LM:** `"The cat sat on"` → predict `the`; masked self-attention makes train and inference consistent (never see the future). Scales into today's LLMs.

---

## 8. Interpretability — Attention & Its Caveat

Attention weights are *seductively* readable — you can plot which tokens "it" attended to (§4a) and it often looks like a clean explanation. Use it, but know the trap:

- **"Attention is not explanation."** `(certain)` High attention weight ≠ causal importance: you can often permute/alter attention without changing the prediction, and multiple attention distributions yield the same output. Treat attention maps as a *hint*, not proof.
- **Better tools:** **Integrated Gradients / gradient×input** for token attribution, **probing classifiers** (can you recover POS/syntax from a layer's activations?), and **mechanistic interpretability** (identifying circuits like *induction heads* that do in-context copying). For LLM outputs specifically, attribution to retrieved context (in [[rag]]) is the practical "why."

🎯 **Interview line:** *"I'll show attention maps for intuition, but I won't claim them as explanations — 'attention is not explanation' is well established; for real attribution I use integrated gradients or probing."*

---

## 9. Production & MLOps Notes

### Tokenization — the step before everything (and a classic blind spot)
Models don't see words, they see **subword tokens**. **BPE** (GPT), **WordPiece** (BERT), **SentencePiece** (T5, multilingual) merge frequent character pairs into a ~30–50k vocab → handles out-of-vocabulary words by composition, balances vocab size vs sequence length. Token count drives both cost and the `O(n²)` budget. `(certain)`

### Generation / decoding strategies (decoder-only models)
```
Greedy        argmax each step           → fast, repetitive, can loop
Beam search   keep top-k partial seqs    → better for translation/summarization (closed-ended)
Temperature   sharpen/flatten softmax    → low=safe, high=creative
Top-k / Top-p random sample from top k / nucleus mass p  → the default for open-ended chat
```

### Inference efficiency
- **KV cache** — `P0`. During autoregressive generation, cache each token's K,V so you don't recompute the whole prefix every step → per-token cost drops from `O(n²)` to `O(n)`. The cache memory **grows linearly with context length** and dominates serving memory.
- **FlashAttention** (IO-aware exact attention), **paged attention / vLLM** (efficient KV-cache memory), **quantization** (8-/4-bit weights), **distillation** → the levers that make LLM serving affordable.

### Adapting models
- **Fine-tuning:** full FT (update all weights) vs **PEFT/LoRA** (train tiny low-rank adapters, ~0.1% of params, swap per task) — LoRA is the practical default. **Instruction tuning** + **RLHF/DPO** align base LMs to follow instructions/preferences.
- **Grounding:** plain LMs **hallucinate**; retrieve relevant documents and condition on them → **[[rag]]** is the standard production pattern for factual tasks.

### Evaluation & monitoring
No single metric: **perplexity** (LM fit), **BLEU/ROUGE** (translation/summarization, weak), task accuracy/F1, and increasingly **human or LLM-as-judge** for open-ended quality. Monitor **input drift** (new topics/languages), **latency/cost per token**, **toxicity/safety**, and refusal/hallucination rates.

---

## 10. Interview Lens

> ⚡ **Golden rule:** diagnose the *why* before the *what*; write the attention formula on the board when relevant.

**"RNN limitations?"** → 🎯 *"Vanishing gradient — BPTT multiplies `tanh'·W` (each <1) over the sequence, so early tokens get ~zero gradient and long-range context is lost; plus a fixed-vector bottleneck and no parallelism."*

**"How does LSTM fix it?"** → 🎯 *"A gated cell state whose gradient `∂cₜ/∂cₜ₋₁ = fₜ` — set the forget gate ≈1 and gradient flows unchanged. A learned highway, not a fixed shrinking product."*

**"Why Transformers over LSTM?"** → 🎯 *"Self-attention gives parallelism, no fixed bottleneck, and direct token-to-token connections at any distance, with residuals preventing vanishing gradients — at the cost of `O(n²)` in sequence length."*

**Likely follow-ups:**
- *Self- vs cross-attention?* → Self: Q,K,V from the same sequence. Cross: Q from decoder, K,V from encoder → bridges two sequences.
- *Why `√d_k`?* → Dot products grow with `d_k` → softmax saturates → tiny gradients; scaling keeps it healthy.
- *LayerNorm vs BatchNorm here?* → Variable-length, padded sequences make batch stats unstable; LayerNorm normalizes per-token across features, batch-size-independent.
- *Role of the FFN?* → The non-linearity; attention alone is a linear weighted average. Without it the stack collapses to linear.
- *Encoder-only vs decoder-only vs enc-dec?* → understanding/embeddings (BERT) · generation (GPT) · seq2seq (T5).
- *Transformer limitations?* → `O(n²)`, weak inductive bias (needs data), hallucination → sparse attention / FlashAttention / RAG.

---

## 11. Alternatives & How to Choose

| Need | Reach for | Why |
|---|---|---|
| Default NLP today | Transformer (decoder-only LLM) | Parallel, long-range, pretrained scale |
| Understanding / embeddings | Encoder (BERT-family) | Bidirectional context |
| Seq2seq (translate/summarize) | Encoder-decoder (T5) | Distinct in/out sequences + cross-attention |
| Very long context | **FlashAttention**, **Longformer/BigBird** (sparse), **Mamba/SSM** | Cut the `O(n²)` cost |
| Streaming / online / edge / tiny data | **RNN / GRU / LSTM** | Linear in length, low memory, strong inductive bias |
| Cheap per-task adaptation | **LoRA / PEFT** on a pretrained model | ~0.1% params, swappable adapters |

- **Efficient attention:** Longformer/BigBird (local+global+random sparse), Performer/linear attention (approximate), **FlashAttention** (exact but IO-aware — usually the first thing to try).
- **State Space Models / Mamba** — `P1`, the live frontier: linear-time sequence modeling with strong long-context, no quadratic attention. Worth knowing as "the post-Transformer contender."
- **When RNNs still win:** truly streaming inference (one token at a time, bounded memory), very small datasets (their locality/sequentiality inductive bias helps), and constrained edge devices.
- **Decoder-only LLMs** (GPT/Llama) are this note's decoder taken to scale + a training pipeline — see [[LLM]] for architecture (GPT-2 walk-through), decoding algorithms, and the pretrain→SFT→RLHF path to ChatGPT.

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Write the vanishing-gradient argument for RNNs in one line.
   <details><summary>answer</summary>BPTT multiplies `∂hₜ/∂hₜ₋₁ = tanh'(·)·Wₕ`, each factor <1; over n steps the product → 0, so early tokens get no gradient.</details>
2. Precisely why does the LSTM cell state avoid vanishing gradients?
   <details><summary>answer</summary>`∂cₜ/∂cₜ₋₁ = fₜ` (the forget gate). Learn `fₜ≈1` and gradient flows unchanged — an adaptive highway, not a fixed shrinking product.</details>
3. Write the attention formula and say why `√d_k` is there.
   <details><summary>answer</summary>`softmax(QKᵀ/√d_k)V`. Dot products grow with `d_k`; without scaling softmax saturates → vanishing gradients.</details>
4. Why LayerNorm rather than BatchNorm in Transformers?
   <details><summary>answer</summary>Variable-length, padded sequences make batch statistics unstable; LayerNorm normalizes each token across features, works at any batch size including 1.</details>
5. What does the decoder mask do, mechanically?
   <details><summary>answer</summary>Sets future-position attention scores to `−∞` before softmax → 0 weight on future tokens → autoregressive consistency between train and inference.</details>
6. What is a KV cache and why does it matter in production?
   <details><summary>answer</summary>Caches past tokens' K,V during generation so each new token costs `O(n)` not `O(n²)`. Its memory grows linearly with context and dominates serving memory.</details>
7. BPE/WordPiece solve what, and why should you care about token count?
   <details><summary>answer</summary>Subword tokenization handles OOV by composing frequent pieces with a small vocab. Token count drives cost and the `O(n²)` attention budget.</details>
8. Name one post-Transformer architecture for long sequences and its key property.
   <details><summary>answer</summary>**Mamba / SSMs** — linear-time sequence modeling with strong long-context, no quadratic attention. (Or FlashAttention/Longformer for cutting the `O(n²)` cost.)</details>

---

*Covers: RNN · BPTT & vanishing/exploding gradients · LSTM gates & cell-state highway · GRU · self/multi-head/cross attention · positional encoding (sinusoidal, RoPE, ALiBi) · residual+LayerNorm (Pre/Post-LN) · FFN · masking · BERT/GPT/T5 · tokenization · decoding strategies · KV cache · LoRA/PEFT · RAG · FlashAttention · Mamba/SSMs.*
