# RNN · LSTM · Transformers — Interview Revision Notes
### Dense, no filler. Every concept an interviewer will probe.

---

## The One Mental Model for All Three

```
RNN       → reads sequence step by step, one fixed memory vector
            Problem: memory bottleneck + vanishing gradient

LSTM      → reads step by step, separates short + long term memory
            Problem: still sequential, still fixed memory bottleneck

Transformer → reads entire sequence at once, direct token-to-token attention
              Solves: parallelism + no bottleneck + long-range dependencies
```

---

## Table of Contents
1. [Vanilla RNN — Architecture & Limitations](#1-vanilla-rnn--architecture--limitations)
2. [LSTM — Architecture & How It Solves RNN Problems](#2-lstm--architecture--how-it-solves-rnn-problems)
3. [Real-World Example: RNN vs LSTM](#3-real-world-example-rnn-vs-lstm)
4. [LSTM Limitations — Why Transformers Were Needed](#4-lstm-limitations--why-transformers-were-needed)
5. [Transformer Architecture — Full Detail](#5-transformer-architecture--full-detail)
6. [BERT vs GPT vs T5](#6-bert-vs-gpt-vs-t5)
7. [Interview Answer Templates](#7-interview-answer-templates)
8. [Likely Follow-Up Questions & Answers](#8-likely-follow-up-questions--answers)
9. [Pre-Interview Checklist](#9-pre-interview-checklist)

---

# 1. Vanilla RNN — Architecture & Limitations

## Architecture

```
At each time step t:

  hₜ = tanh(Wₕ · hₜ₋₁  +  Wₓ · xₜ  +  b)
              ↑                ↑
         previous          current
         hidden state       input

  ŷₜ = softmax(Wᵧ · hₜ)   ← output at step t (optional)

hₜ = the ONLY memory — carries all context from past steps
```

Visually:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → hₙ → ŷ
               ↑              ↑              ↑
             tanh           tanh           tanh
```

## The Vanishing Gradient Problem — Precise Explanation

```
During backpropagation through time (BPTT):

∂L/∂h₁ = ∂L/∂hₙ × ∏(t=2 to n) ∂hₜ/∂hₜ₋₁

Each term:
  ∂hₜ/∂hₜ₋₁ = tanh'(·) × Wₕ

  tanh'(x) ∈ (0, 1)  — always less than 1
  If Wₕ is also < 1  — product shrinks exponentially

For sequence length n=500:
  Product of 499 terms each < 1 → approaches 0

Result:
  ∂L/∂h₁ ≈ 0   → early timesteps receive no gradient
                → early words are NOT learned
                → the model effectively forgets long-term context
```

## The Exploding Gradient Problem

```
If Wₕ > 1 and tanh' is not small enough:
  Product of 499 terms each > 1 → approaches ∞

Result: NaN weights, training diverges

Fix: Gradient clipping
  if ||gradient|| > threshold:
      gradient = gradient × (threshold / ||gradient||)

Vanishing gradient: ignored early tokens
Exploding gradient: training crashes
Both are symptoms of the same sequential multiplication problem.
```

## Summary of RNN Limitations

```
❌ Vanishing gradient → can't learn long-range dependencies
❌ Exploding gradient → training instability (fixed by clipping)
❌ Sequential processing → can't parallelise → slow training
❌ Fixed-size hidden state → must compress entire history
   into one vector of fixed dimension → information bottleneck
❌ No direct path between distant tokens → must pass through
   every intermediate hidden state
```

---

# 2. LSTM — Architecture & How It Solves RNN Problems

## The Core Insight

```
RNN has ONE memory: hₜ (hidden state)
  → Must store everything in one vector
  → Short-term and long-term information compete

LSTM has TWO memories:
  cₜ = cell state   → long-term memory  (the "conveyor belt")
  hₜ = hidden state → short-term memory (output to next layer)

Three GATES control what flows through:
  fₜ = forget gate  → what to erase from long-term memory
  iₜ = input gate   → what new info to write to long-term memory
  oₜ = output gate  → what to read from long-term memory → hₜ
```

## Gate Equations — Know These Cold

```
Forget gate:   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
Input gate:    iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
Candidate:     c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
Cell update:   cₜ = fₜ ⊙ cₜ₋₁  +  iₜ ⊙ c̃ₜ
Output gate:   oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
Hidden state:  hₜ = oₜ ⊙ tanh(cₜ)

σ = sigmoid → squashes to (0,1) → acts as a "gate"
⊙ = element-wise multiplication
```

## Intuition for Each Gate

```
FORGET GATE (fₜ):
  "How much of the old cell state should I keep?"
  fₜ ≈ 1 → keep everything from cₜ₋₁
  fₜ ≈ 0 → erase everything from cₜ₋₁

  Example: Reading "The cat, which was hungry, sat on the mat."
  When we reach "sat", the subject "cat" is still relevant.
  Forget gate stays near 1 for subject information.

INPUT GATE (iₜ) + CANDIDATE (c̃ₜ):
  "What new information should I write to cell state?"
  iₜ = how much to write
  c̃ₜ = what to write (proposed new memory)

  Example: New subject appears → write it to cell state.

OUTPUT GATE (oₜ):
  "What part of cell state should I expose as output hₜ?"
  "What does the next word need to know about right now?"

  Example: Predicting verb → expose subject from cell state.
```

## How LSTM Solves Vanishing Gradient

```
RNN gradient path (through tanh):
  ∂cₜ/∂cₜ₋₁ = tanh'(·) × W    → always < 1 → vanishes

LSTM gradient path (through cell state):
  ∂cₜ/∂cₜ₋₁ = fₜ              → just the forget gate value!

If forget gate fₜ ≈ 1:
  ∂cₜ/∂cₜ₋₁ ≈ 1               → gradient flows unchanged

The cell state is a HIGHWAY for gradients.
When model wants to remember long-term info:
  → it sets fₜ ≈ 1
  → gradient flows back without vanishing

This is a learned, adaptive solution vs RNN's fixed multiplication.
```

## GRU — Simplified LSTM

```
GRU (Gated Recurrent Unit) merges cell and hidden state:
  Only 2 gates: reset gate, update gate
  No separate cell state

Fewer parameters than LSTM → faster training
Similar performance on most tasks
Use GRU when: compute budget is tight, sequences are shorter
Use LSTM when: need full expressivity, very long sequences
```

---

# 3. Real-World Example: RNN vs LSTM

## Sentiment Analysis on Long Reviews

```
Task: Classify movie review as positive or negative

Short review (RNN works fine):
  "The movie was fantastic!"   ← 5 words, no long-range dependency

Long review (RNN fails, LSTM works):
  "The movie started brilliantly with stunning visuals,
   an incredible soundtrack, and deeply moving performances.
   The first act drew me in completely. However, the second
   act lost the plot entirely. The ending was a complete
   disappointment, and I left the theatre feeling cheated."

Key signal: "disappointment", "cheated" → NEGATIVE
But context: "however", "lost the plot" 60 words earlier → NEGATIVE

RNN: by the time it reaches "disappointment",
     hidden state has been overwritten 60 times.
     Early positive words ("brilliant", "incredible")
     dominate hidden state → predicts POSITIVE ❌

LSTM: cell state maintains the "however" turning point
      as a long-term signal.
      Forget gate erases initial positive sentiment
      when "however" appears.
      Correctly predicts NEGATIVE ✅
```

## Machine Translation (Encoder-Decoder RNN)

```
Task: Translate "The cat sat on the mat." → French

RNN encoder must compress ENTIRE sentence into ONE vector hₙ.
For long sentences (20+ words):
  → hₙ bottleneck loses information about early words
  → Translation quality degrades sharply after ~10 words

LSTM encoder:
  → Cell state preserves subject/object from early tokens
  → Better translation quality for medium-length sentences
  → Still degrades for very long sentences (50+ words)

This degradation with length motivated the ATTENTION mechanism.
```

---

# 4. LSTM Limitations — Why Transformers Were Needed

```
After solving vanishing gradients, LSTM still has:

❌ Sequential processing — can't parallelise
   Each step depends on previous hₜ₋₁
   Must process word 1 before word 2 before word 3...
   Training 1M sentences × 50 words = 50M sequential steps
   → Extremely slow on modern GPU hardware
      (GPUs are built for parallel ops, not sequential)

❌ Fixed-size memory bottleneck
   hₜ and cₜ are fixed-dimension vectors (e.g., 512-dim)
   Must compress ENTIRE history into 512 numbers
   For long documents (1000+ words): critical information is lost
   No matter how well the gates work, 512 dims can only hold so much

❌ Limited long-range dependency
   Even with cell state, token at position 1 must
   "survive" 999 forget gate multiplications to reach token 1000
   If any forget gate is slightly < 1, signal attenuates
   Cannot directly connect token 1 to token 1000

❌ No direct token-to-token attention
   To relate word 1 to word 500:
     Must pass through 499 intermediate hidden states
   There is no direct path — information gets diluted

The fix for ALL of these: TRANSFORMERS
```

---

# 5. Transformer Architecture — Full Detail

## The Core Insight

```
What if instead of reading word by word,
we read the ENTIRE sequence at once and
let every word directly attend to every other word?

Attention(word_i, word_j) = "how relevant is word j
                              when processing word i?"

This gives:
  ✅ Direct token-to-token connection (no bottleneck)
  ✅ Full parallelism (all words processed simultaneously)
  ✅ Handles any sequence length
  ✅ Interpretable (attention weights are human-readable)
```

## High-Level Architecture

```
INPUT TOKENS
    ↓
[Token Embeddings] + [Positional Encoding]
    ↓
┌─────────────────────────────────┐
│         ENCODER STACK           │  ← N layers (e.g., 6 in original)
│                                 │
│  [Multi-Head Self-Attention]    │
│  [Add & LayerNorm]              │
│  [Feed Forward Network]         │
│  [Add & LayerNorm]              │
└─────────────────────────────────┘
    ↓ (encoder output = context vectors for every token)
┌─────────────────────────────────┐
│         DECODER STACK           │  ← N layers
│                                 │
│  [Masked Multi-Head Self-Attn]  │  ← can't see future tokens
│  [Add & LayerNorm]              │
│  [Cross-Attention]              │  ← attends to encoder output
│  [Add & LayerNorm]              │
│  [Feed Forward Network]         │
│  [Add & LayerNorm]              │
└─────────────────────────────────┘
    ↓
[Linear + Softmax]
    ↓
OUTPUT TOKEN PROBABILITIES
```

---

## Component 1: Positional Encoding

```
Problem: Attention has no inherent order.
  "Cat sat on mat" and "Mat on sat cat" look identical to attention.
  We must inject position information.

Solution: Add positional encoding to each token embedding.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why sinusoidal?
  → Different frequencies for different dimensions
  → Each position gets a unique fingerprint
  → Model can generalise to sequence lengths not seen in training
  → PE(pos+k) can be expressed as linear function of PE(pos)
    → relative positions are learnable

Input to each layer = Token Embedding + Positional Encoding
Both are d_model-dimensional vectors (e.g., 512-dim).
```

---

## Component 2: Scaled Dot-Product Attention

```
For each token, create 3 vectors:
  Query  (Q) = "what am I looking for?"
  Key    (K) = "what do I contain?"
  Value  (V) = "what do I contribute if selected?"

Q = X · Wq    (X = input, Wq = learned weight matrix)
K = X · Wk
V = X · Wv

Attention formula:
  Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

Step by step:
  1. Q · Kᵀ      → similarity score between every pair of tokens
                    shape: (seq_len × seq_len)
                    score[i,j] = how much token i attends to token j

  2. / √d_k       → scale by sqrt of key dimension
                    WHY: dot products grow large with high d_k
                         → softmax saturates → gradients vanish
                         → scaling keeps values in good range

  3. softmax(·)   → convert scores to probabilities (sum to 1)
                    attention_weights[i] = distribution over all tokens

  4. × V          → weighted sum of value vectors
                    output[i] = Σⱼ attention_weight[i,j] × V[j]
                    "blend of all values, weighted by relevance"
```

### Concrete Example

```
Sentence: "The animal didn't cross the street because it was tired."
Question: What does "it" refer to?

Q = query vector for "it"
K = key vectors for all words

Scores:
  it → The:      0.1
  it → animal:   0.8   ← HIGH — "it" attends strongly to "animal"
  it → street:   0.1

After softmax:
  weights = [0.1, 0.8, 0.1, ...]

Output for "it" = 0.1×V("The") + 0.8×V("animal") + 0.1×V("street")
                → representation of "it" heavily influenced by "animal"
                → coreference resolved! ✅
```

---

## Component 3: Multi-Head Attention

```
One attention head learns ONE type of relationship.
  Head 1: might learn subject-verb relationships
  Head 2: might learn coreference (it → animal)
  Head 3: might learn local context (adjacent words)
  Head 4: might learn syntactic dependencies

Multi-head runs h attention operations in parallel:
  head_i = Attention(Q·Wqᵢ, K·Wkᵢ, V·Wvᵢ)

All heads concatenated, then projected:
  MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · Wₒ

h=8 heads in original Transformer, d_k = d_model/h = 64

Why multiple heads?
  → Different heads can simultaneously capture different relationships
  → Single attention can only focus on one pattern at a time
  → Multiple heads = richer, multi-faceted representation
```

---

## Component 4: Add & Layer Norm (Residual Connection)

```
After each sub-layer (attention or FFN):

  output = LayerNorm(x + SubLayer(x))

Two components:

RESIDUAL CONNECTION (x + SubLayer(x)):
  Same as ResNet skip connections.
  → Gradient highway: ∂/∂x = 1 + ∂SubLayer/∂x
  → Prevents vanishing gradient in deep transformers
  → Allows very deep models (BERT=12 layers, GPT-3=96 layers)

LAYER NORMALISATION:
  Normalise across the feature dimension (not batch):
  For each token: subtract mean, divide by std across d_model dims
  Then scale and shift with learned γ, β

  Why LayerNorm not BatchNorm?
  → BatchNorm normalises across batch dimension
  → For variable-length sequences, batch stats are unstable
  → LayerNorm normalises each token independently → stable
```

---

## Component 5: Feed Forward Network (FFN)

```
Applied identically to each position independently:

  FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
          = ReLU(Linear → 4×d_model) → Linear → d_model

Example: d_model=512
  Linear: 512 → 2048   (expand 4×)
  ReLU
  Linear: 2048 → 512   (compress back)

Purpose:
  Attention is a weighted average — it's a linear operation.
  FFN introduces NON-LINEARITY after attention.
  "Attention decides WHERE to look.
   FFN decides WHAT to do with what it saw."

Each position's FFN is computed independently — fully parallelisable.
Weights are SHARED across positions (like conv weight sharing).
```

---

## Component 6: Masked Self-Attention (Decoder)

```
Decoder generates output ONE TOKEN AT A TIME.
When generating token t, it must NOT see tokens t+1, t+2, ...
(That would be cheating — looking at the answer)

Masking: set attention scores for future positions to -∞
  → After softmax: -∞ → 0 → zero attention to future tokens

Mask matrix (for seq_len=5):
  [0    -∞   -∞   -∞   -∞ ]   ← token 1 sees only itself
  [0     0   -∞   -∞   -∞ ]   ← token 2 sees tokens 1-2
  [0     0    0   -∞   -∞ ]   ← token 3 sees tokens 1-3
  [0     0    0    0   -∞ ]
  [0     0    0    0    0  ]   ← token 5 sees all
```

---

## Component 7: Cross-Attention (Encoder-Decoder Bridge)

```
In the decoder, after masked self-attention:

  Q = from decoder (what the decoder is currently generating)
  K = from encoder output (the full source sequence)
  V = from encoder output

"The decoder token asks: which encoder tokens are most
 relevant for generating my next output token?"

Example (translation):
  Generating French word "chat" (cat):
  Q = decoder query for current position
  K,V = encoder outputs for ["The", "cat", "sat", "on", "mat"]
  High attention weight on encoder's "cat" representation ✅

This is how the decoder uses source information while generating.
```

---

## Why Transformers Solve LSTM's Problems

```
LSTM problem          Transformer solution
─────────────────────────────────────────────────────────────
Sequential processing → All tokens processed in PARALLEL
                        Q, K, V computed simultaneously for all positions

Fixed memory bottleneck → No bottleneck!
                          Every token has direct attention to every other
                          No compression into fixed-size vector

Long-range dependencies → Direct attention: token 1 attends to token 500
                          in ONE step, not 499 sequential steps
                          Distance doesn't matter

Vanishing gradient     → Residual connections throughout
                          Gradient flows directly through skip connections
```

---

## Complexity Trade-Off

```
Self-attention complexity: O(n² × d)
  n = sequence length, d = model dimension

  For n=512: 512² = 262,144 attention scores → fine
  For n=10,000: 10,000² = 100M scores → memory explosion

RNN complexity: O(n × d²)
  Linear in sequence length — better for very long sequences

This is why Transformers struggle with very long sequences
(books, long documents) → addressed by:
  Longformer: sparse attention (local + global)
  BigBird:    random + local + global attention
  FlashAttention: memory-efficient exact attention
```

---

# 6. BERT vs GPT vs T5

```
Architecture    Training           Use case
──────────────────────────────────────────────────────────────
BERT            Encoder only       Understanding tasks:
                Masked LM          classification, NER, QA
                Next Sentence Pred Bidirectional: sees full context
                                   Cannot generate text

GPT             Decoder only       Generation tasks:
                Causal LM          text generation, completion
                (predict next tok) Left-to-right only
                                   Cannot see future tokens

T5              Encoder-Decoder    Both understanding + generation:
                Text-to-text       translation, summarisation, QA
                All tasks as seq2seq format
```

## BERT's Masked Language Model

```
Input:  "The [MASK] sat on the mat."
Target: predict "cat" for [MASK]

Why masking?
  BERT is bidirectional — sees left AND right context.
  If trained on next-token prediction, it would just copy next token.
  Masking forces it to use full context to predict hidden words.
  → Rich contextual representations.
```

## GPT's Causal Language Model

```
Input:  "The cat sat on"
Target: predict "the"

Then: "The cat sat on the"
Target: predict "mat"

Why causal (left-to-right)?
  Generation requires predicting next token without seeing future.
  Masked self-attention enforces this.
  Consistent at train and inference time.
```

---

# 7. Interview Answer Templates

## "Explain RNN and its limitations"

> *"RNN processes sequences step by step. At each step, it computes a hidden state from the current input and the previous hidden state using a tanh activation. The limitation is vanishing gradient: during backpropagation through time, gradients are multiplied by tanh' × W at each step. Since tanh' is always less than 1, this product shrinks exponentially over sequence length — a 500-word sequence multiplies 499 such terms, making the gradient effectively zero for early tokens. The model can't learn long-range dependencies. The opposite problem, exploding gradients, is fixed with gradient clipping. There's also the bottleneck problem: all context must be compressed into a single fixed-size hidden vector, losing information for long sequences. And training is inherently sequential — each step depends on the previous — preventing parallelisation."*

## "How does LSTM solve RNN's problems?"

> *"LSTM introduces a separate cell state as a long-term memory alongside the short-term hidden state. Three gates control information flow: the forget gate decides what to erase from cell state, the input gate decides what new information to write, and the output gate decides what to read from cell state for the current output. The key gradient fix is in the cell state update: dc_t/dc_{t-1} equals the forget gate value. When the model wants to preserve long-term information, it learns to set the forget gate close to 1 — gradient flows back unchanged. This is a learned, adaptive solution unlike RNN's fixed shrinking multiplication. The cell state acts as a gradient highway for long-term dependencies."*

## "Give a real example where LSTM outperforms RNN"

> *"Sentiment analysis on long reviews. Consider a 200-word review that starts with strongly positive language — 'incredible visuals, stunning performances' — then pivots with 'however, the second half completely falls apart, and the ending is a disappointment.' The sentiment is negative. A vanilla RNN will have its hidden state dominated by the early positive words by the time it reaches 'however' — the gradient from early tokens has vanished, and it predicts positive. LSTM's forget gate fires when it encounters 'however', erasing the early positive sentiment from the cell state and correctly preserving the negative turning point — predicting negative. The same failure occurs in machine translation: RNN encoder collapses an entire long sentence into one vector, losing early words; LSTM maintains them in cell state."*

## "What are LSTM's limitations and how do Transformers solve them?"

> *"Three fundamental limitations remain in LSTM. First, sequential processing: each step depends on the previous hidden state, so training cannot be parallelised — a 50-word sentence requires 50 sequential steps, which is catastrophically slow on GPUs built for parallel computation. Second, the fixed-size memory bottleneck: cell state and hidden state are fixed-dimension vectors — 512 dimensions must compress thousands of words, inevitably losing information for very long sequences. Third, no direct long-range connections: even with the cell state, relating token 1 to token 1000 requires surviving 999 forget gate multiplications. Transformers solve all three: self-attention processes all tokens simultaneously (parallelism), every token attends directly to every other token (no bottleneck, direct connections), and residual connections throughout prevent gradient issues. The trade-off is quadratic memory complexity in sequence length — O(n²) — which makes transformers expensive for very long sequences."*

## "Explain Transformer architecture in detail"

> *"The transformer has an encoder-decoder structure. Input tokens are converted to embeddings and positional encodings are added — sinusoidal functions of position — since attention has no inherent order. The encoder stack has N identical layers, each with two sub-layers: multi-head self-attention followed by a feed-forward network, each wrapped with a residual connection and layer normalisation. Self-attention computes Query, Key, Value matrices from the input. Attention scores are dot products of Q and K, scaled by root d_k to prevent saturation, then softmaxed to give attention weights, then multiplied with V to produce output. Running h attention heads in parallel captures different types of relationships simultaneously. Multi-head outputs are concatenated and projected. The FFN applies two linear transformations with ReLU — it introduces non-linearity since attention itself is linear. The decoder adds masked self-attention (prevents looking at future tokens during generation) and cross-attention where decoder queries attend to encoder keys and values. Residual connections throughout act as gradient highways enabling very deep models."*

---

# 8. Likely Follow-Up Questions & Answers

**Q: What is the difference between self-attention and cross-attention?**
> *"In self-attention, Q, K, V all come from the same sequence — each token attends to all other tokens in the same sequence. In cross-attention, Q comes from one sequence (decoder) and K, V come from another (encoder output). Self-attention builds representations within a sequence; cross-attention bridges two sequences — the decoder uses it to decide which source tokens are most relevant when generating each output token."*

**Q: Why scale by √d_k in attention?**
> *"Dot products of Q and K grow in magnitude as d_k increases — for d_k=64, expected magnitude is 8. Large values push softmax into regions of very small gradients — the softmax saturates, attention becomes a near-one-hot vector, and gradient flow degrades. Dividing by √d_k keeps dot products in a range where softmax has healthy gradients regardless of model dimension."*

**Q: Why Layer Norm instead of Batch Norm in Transformers?**
> *"Batch Norm normalises across the batch dimension — it requires a large, consistent batch size and becomes unstable for variable-length sequences where padding affects statistics. Layer Norm normalises across the feature dimension for each token independently — it works for any batch size including batch size 1, handles variable-length sequences naturally, and is stable at inference. Transformers process sequences of varying length, making Layer Norm the natural choice."*

**Q: What is the role of the FFN in Transformers? Isn't attention enough?**
> *"Attention is a weighted average — it's a linear operation over value vectors. It's excellent at deciding which tokens to combine, but cannot introduce non-linearity or perform complex transformations on the combined representation. The FFN applies two linear layers with ReLU between them, expanding to 4× dimension then compressing back — it processes each position's representation independently after attention has mixed information across positions. Attention decides where to look; FFN decides what to compute from what it saw. Without FFN, the model is a stack of linear operations — it can't learn complex functions."*

**Q: What is the difference between encoder-only, decoder-only, and encoder-decoder models?**
> *"Encoder-only (BERT): bidirectional attention — every token sees all others. Ideal for understanding tasks where full context is available: classification, NER, question answering. Cannot generate. Decoder-only (GPT): causal/masked attention — each token sees only past tokens. Ideal for generation: autoregressive text generation, completion. Encoder-decoder (T5, original Transformer): encoder processes input with full attention, decoder generates output attending to encoder via cross-attention. Ideal for seq2seq tasks: translation, summarisation, where input and output are distinct sequences."*

**Q: How does BERT's masked language modelling differ from GPT's causal LM?**
> *"BERT masks random tokens and trains the model to predict them using both left and right context — bidirectional. This produces rich contextual embeddings but cannot generate text sequentially since it requires full context at inference. GPT predicts the next token using only left context — autoregressive. This enables open-ended text generation but each token only sees its past. BERT is better for understanding; GPT is better for generation. T5 reformulates all tasks as text-to-text to get benefits of both."*

**Q: What are the limitations of Transformers?**
> *"Quadratic complexity in sequence length: O(n²) memory and compute for self-attention — for n=10,000 tokens, 100M attention scores. Makes transformers expensive for long documents. No inherent sequential bias — positional encoding is added artificially, and the model doesn't naturally capture local structure the way RNNs do implicitly. Transformers need more data than RNNs to train from scratch — they don't have the inductive biases (locality, sequentiality) that RNNs have built in. Solutions: sparse attention (Longformer, BigBird), linear attention approximations, FlashAttention for memory efficiency."*

---

# 9. Pre-Interview Checklist

```
RNN
□ Can I explain the vanishing gradient mathematically?
   (tanh' < 1, multiplied n times → 0)
□ Can I explain exploding gradient and its fix (clipping)?
□ Can I list all 4 RNN limitations?

LSTM
□ Can I name the two memory types and their purpose?
   (cell state = long-term, hidden state = short-term)
□ Can I name and explain all 3 gates?
   (forget, input, output)
□ Can I explain precisely why cell state solves vanishing gradient?
   (dc_t/dc_{t-1} = forget gate ≈ 1 → gradient flows)
□ Can I give the real-world RNN vs LSTM example?
□ Do I know GRU and when to prefer it over LSTM?

Transformers
□ Can I explain why positional encoding is needed?
□ Can I explain Q, K, V intuitively?
□ Can I write the attention formula from memory?
   Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V
□ Do I know why we scale by √d_k?
□ Can I explain multi-head attention purpose?
□ Can I explain masked self-attention in the decoder?
□ Can I explain cross-attention (Q from decoder, K/V from encoder)?
□ Can I explain the FFN's role (non-linearity after linear attention)?
□ Can I explain residual + LayerNorm (and why LayerNorm not BN)?
□ Do I know BERT vs GPT vs T5 differences?
□ Can I explain transformers' O(n²) limitation?
□ Can I explain all 4 ways transformers solve LSTM's problems?
```

---

*Interview revision notes — RNN · LSTM · GRU · Attention · Self-Attention · Multi-Head Attention · Positional Encoding · Transformer Architecture · BERT · GPT · T5*