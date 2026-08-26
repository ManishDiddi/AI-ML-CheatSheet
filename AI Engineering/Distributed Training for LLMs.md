# Distributed Training for LLMs — what to split across GPUs, and which problem each split actually solves

> **TL;DR.** Every strategy here falls out of one question: **what is sitting in GPU memory, and how can you cut it?** Three things live there. The **batch** — cut it and you get **data parallelism** (fixes *speed*, gives **zero** memory relief). The **model state** — `16 bytes × params` (2 bf16 weights + 2 bf16 grads + 4 fp32 master + 4 Adam `m` + 4 Adam `v`) — cuttable three structurally different ways: **by layer** = pipeline, **inside a layer** = tensor, **by ownership** = ZeRO/FSDP. And the **activations**, which no amount of weight sharding touches — cut them with **activation checkpointing** and **sequence parallelism**. 🎯 The line that wins the question: *"Pipeline and tensor parallelism change what each GPU **computes**; ZeRO changes only what each GPU **stores** — which is why ZeRO is a config change and the other two are code changes."* `(certain)` Every number below is computed on **GPT-2 small (124 M)** so the arithmetic stays holdable in your head.

**Where it fits:** The back half of the *Advanced Fine-Tuning* lecture, and the direct sequel to [Fine-Tuning LLMs](Fine-Tuning%20LLMs.md) — which ends at exactly the wall this note starts from: full fine-tuning a 7B needs ≈168 GB, a 70B needs **>1.1 TB**, and no single GPU exists at that size. [PEFT/LoRA](Fine-Tuning%20LLMs.md#5-peft-and-lora--the-mechanism) is how you *avoid* this problem; distributed training is how you *solve* it when avoiding isn't an option (pre-training, full fine-tuning, or a model too big to load at all). Its sequel is [Model Quantization](Model%20Quantization.md) — the **third** answer to the same wall: instead of splitting the bytes across GPUs, make every byte smaller.
**Prereqs:** [Fine-Tuning LLMs](Fine-Tuning%20LLMs.md) (the memory arithmetic, gradient accumulation), [Weight Initialization & Optimizers](../Machine%20Learning/Neural%20Networks/Weight%20Initialization%20&%20Optimizers.md) (what Adam stores), [RNN · LSTM · Transformers](../Machine%20Learning/NLP/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) (the layer stack you're about to cut up).

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../_STUDY%20LOOP.md))*

---

> ### ⚡ Fast Pass — 5 minutes
> **Model:** Three things live in GPU memory — the **batch**, the **model state**, the **activations**. Enumerate the ways to cut each and the entire field falls out. Batch → **data parallel**. State by layer → **pipeline**; inside a layer → **tensor**; by ownership → **ZeRO/FSDP**. Activations → **checkpointing / sequence parallel**.
> **Core:**
> - `state = 16 B × P` — **2** bf16 weights + **2** bf16 grads + **4** fp32 master + **4** `m` + **4** `v`. GPT-2 small (124 M) = **2 GB**; GPT-2 XL (1.5 B) = **24 GB**.
> - DDP: same weights everywhere, different batches, `ḡ = (1/N)Σgᵢ` via **all-reduce**. **Zero memory relief.**
> - `all-reduce = reduce-scatter + all-gather` → `2·(N−1)/N·S` per GPU, **flat in N** — why DDP scales to thousands.
> - ZeRO-1/2/3 shards optimizer states / +grads / +weights → `4P+12P/N`, `2P+14P/N`, `16P/N`. FSDP2 ≈ ZeRO-3.
> - Pipeline bubble `= (G−1)/(M+G−1)`; rule `M ≥ 4G`.
> - Tensor: **4 all-reduces per block per step**, on the critical path, unhideable → **never crosses a server**.
> **Traps:** ① thinking DDP helps a model that doesn't fit — it doesn't, at any N; ② still OOM after ZeRO-3 and not realising it's **activations**, which ZeRO never sharded; ③ scaling out without rescaling the LR (effective batch just went ×N); ④ offering **ZeRO** as the answer to "how would you serve a 175B model" — there is no optimizer state at inference.
> 🎯 **Kill-shot:** *"Data parallelism scales throughput; everything else scales capacity. And the ordering between the capacity strategies is set by communication cost — data parallelism talks once per step and hides it behind the backward pass, tensor parallelism talks four times per block on the critical path with nothing to hide behind."*

---

## Table of Contents
1. [The Reference Model — GPT-2 small](#1-the-reference-model--gpt-2-small)
2. [What's in GPU Memory — the six buffers](#2-whats-in-gpu-memory--the-six-buffers)
3. [The Taxonomy, Derived](#3-the-taxonomy-derived)
4. [The Collectives — identified by payload, not by name](#4-the-collectives--identified-by-payload-not-by-name)
5. [Data Parallelism and DDP](#5-data-parallelism-and-ddp)
6. [ZeRO and FSDP — sharding what you *store*](#6-zero-and-fsdp--sharding-what-you-store)
7. [Pipeline Parallelism — and the bubble](#7-pipeline-parallelism--and-the-bubble)
8. [Tensor Parallelism — cutting inside the block](#8-tensor-parallelism--cutting-inside-the-block)
9. [Sequence and Expert Parallelism — the two often missed](#9-sequence-and-expert-parallelism--the-two-often-missed)
10. [Composing Them — 3D parallelism and HSDP](#10-composing-them--3d-parallelism-and-hsdp)
11. [Worked Example — sizing a 70B run](#11-worked-example--sizing-a-70b-run)
12. [Code / Implementation](#12-code--implementation)
13. [When It Breaks](#13-when-it-breaks)
14. [Production & LLMOps Notes](#14-production--llmops-notes)
15. [Interview Lens](#15-interview-lens)
16. [Alternatives & How to Choose](#16-alternatives--how-to-choose)
17. [Formula Sheet](#17-formula-sheet)
- [🧠 Self-Test](#-self-test)

---

## 1. The Reference Model — GPT-2 small

Distributed training goes abstract fast. The fix is to anchor every number to **one model you know the shape of cold**, then scale. Use **GPT-2 small: 124 M parameters, 12 blocks, hidden size `d = 768`, 12 heads × 64, context 1024, vocab 50257.**

```
PER BLOCK (four weight matrices — this is all a transformer block is)
  W_qkv     768 × 2304    1.77 M      (2304 = 3 × 768: query/key/value fused)
  W_out     768 ×  768    0.59 M
  W_fc      768 × 3072    2.36 M      (3072 = 4 × 768: the FFN expansion)
  W_proj   3072 ×  768    2.36 M
                          ───────
                          7.08 M      ( = 12 · d² )

WHOLE MODEL
  12 blocks × 7.08 M            =  84.9 M
  token embedding 50257 × 768   =  38.6 M
  position embedding 1024 × 768 =   0.8 M
                                   ──────
                                   124.3 M
```

Two details worth carrying: **`params_per_block = 12·d²`** is the number to memorize (biases and LayerNorm add a few thousand more per block — ignorable at this precision), and **GPT-2 ties its output head to the token embedding**, so there is no separate `768 × 50257` weight matrix — which matters in §8 when we split the vocabulary.

**The scale ladder — the only four rows you need:**

| Model | Blocks | `d` | Params | Training state (16 B/param) |
|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 124 M | **1.99 GB** |
| GPT-2 XL | 48 | 1600 | 1.5 B | **24 GB** |
| GPT-3 | 96 | 12288 | 175 B | **2,800 GB** |
| *one GPT-3 block* | 1 | 12288 | 1.81 B | **29 GB** |

🎯 **A single GPT-3 block has more parameters than the whole of GPT-2 XL.** That one line is the entire intuition for why anyone would ever cut *inside* a block rather than between blocks. `(certain)`

---

## 2. What's in GPU Memory — the six buffers

**Nothing below makes sense until you can name what's being split.** For `P` parameters under mixed-precision Adam:

| # | Buffer | Precision | B/param | Why it exists |
|---|---|---|---|---|
| ① | Weights | bf16 | 2 | what the matmuls actually consume |
| ② | Gradients | bf16 | 2 | the output of backward |
| ③ | Master weights | fp32 | 4 | bf16 keeps fp32's *range* but only ~3 decimal digits of *precision* (7 stored mantissa bits), so `w − 1e-7·g` in bf16 is a literal **no-op**. This copy is what makes small updates survive. |
| ④ | Adam `m` | fp32 | 4 | first moment — momentum |
| ⑤ | Adam `v` | fp32 | 4 | second moment — per-parameter scaling |
| | **persistent total** | | **16** | ③+④+⑤ = the 12 bytes ZeRO calls *optimizer states* |
| ⑥ | **Activations** | bf16 | — | scales with `batch × seq × d × layers` — **not with `P`** |

```
GPT-2 small:  124 M × 16 B = 1.99 GB persistent state
GPT-2 XL:     1.5 B × 16 B = 24 GB     ← already OOMs a 24 GB card before activations
```

🚩 **Buffer ⑥ is the one that gets you.** Persistent state divides by `N` under sharding; **activations don't**. Once FSDP is running and the weights are sharded, activations are usually what actually OOMs — and no amount of extra GPUs fixes it, because activation memory grows *with* the batch you spread across them. The two levers are **activation checkpointing** (store only block boundaries, recompute the interiors — roughly a third more compute for a large memory cut) and **sequence parallelism** (§9). `(certain)`

🎯 **Always state the convention before quoting a number**, because different papers apportion the same 16 bytes differently: *"16 bytes per parameter — 2 for bf16 weights, 2 for bf16 gradients, 4 for the fp32 master copy, 4 for `m`, 4 for `v` — so 124 M is 2 GB of state, plus activations on top."* That sentence pre-empts the follow-up.

---

## 3. The Taxonomy, Derived

Don't memorize a list of four strategies. **Enumerate the ways to cut the three things in memory, and the whole field falls out:**

```
① DATA (the batch)
   └─ different samples per GPU ──────────────────► DATA PARALLEL
      does NOT touch model state → zero memory relief

② MODEL STATE — three structurally different cuts:
   ├─ by LAYER       (GPU0: blocks 0-2)  ─────────► PIPELINE
   ├─ INSIDE a layer (split W_fc)  ───────────────► TENSOR
   └─ by OWNERSHIP   (store 1/N, borrow the rest) ► ZeRO / FSDP

③ ACTIVATIONS
   ├─ split along SEQUENCE ───────────────────────► SEQUENCE PARALLEL
   └─ recompute instead of store ─────────────────► ACTIVATION CHECKPOINTING

(MoE only: split by EXPERT ─────────────────────► EXPERT PARALLEL)
```

![Four ways to split a model across GPUs — data parallelism replicates the full model on every GPU and gives each a different batch, model or pipeline parallelism gives each GPU a contiguous slice of layers, tensor parallelism splits a single weight matrix column-wise across GPUs, and ZeRO or FSDP shards optimizer states then gradients then parameters across stages one to three.](attachments/parallelism-four-kinds.png)

🎯 **The load-bearing distinction, and the one most people never articulate:** *"Pipeline and tensor parallelism change what each GPU **computes**. ZeRO changes only what each GPU **stores** — every GPU still runs the whole model. That's exactly why ZeRO is a config change and the other two are code changes."* `(certain)`

### Same matrix, three cuts — the picture that separates them

Take `W_fc` (768 × 3072) on 4 GPUs. Tensor parallelism and ZeRO both "split the matrix" — but they are not the same operation, and the difference is the point:

```
TENSOR — cut by COLUMN (dim 1)          ZeRO — cut by ROW (dim 0)
┌───────┬───────┬───────┬───────┐       ┌──────────────────────┐
│ 0-767 │768-...│  ...  │  ...  │       │ rows   0-191 → GPU0  │
│ GPU0  │ GPU1  │ GPU2  │ GPU3  │       │ rows 192-383 → GPU1  │
└───────┴───────┴───────┴───────┘       │ rows 384-575 → GPU2  │
   each 768 × 768                       │ rows 576-767 → GPU3  │
                                        └──────────────────────┘
GPU0 can compute NOW. Its columns          each 192 × 3072
are real, final output features.        GPU0 can compute NOTHING.
The cut is MATHEMATICALLY MEANINGFUL    A quarter of the rows is a
(chosen precisely because GeLU is       FRAGMENT. The cut is just a
 elementwise — see §8).                 FILING SYSTEM.
                                        → must be UNDONE (all-gather)
                                          before every single use.
```

**That's the whole difference.** Tensor parallelism's cut is chosen so the pieces are individually *useful*; ZeRO's cut is chosen so the pieces are individually *small*, and it pays an all-gather to undo it every time it needs them. `(certain)`

| Strategy | What it splits | Fixes | Communication cost | What you pay |
|---|---|---|---|---|
| **Data (DDP)** | the batch | speed | gradient all-reduce, once per step, overlapped | full model replica on every GPU |
| **ZeRO / FSDP** | *ownership* of state | fit + speed | all-gather weights per unit + reduce-scatter grads | ~1.5× DDP volume |
| **Pipeline** | contiguous layers | fit | activations at stage seams — tiny | idle GPUs (the bubble) |
| **Tensor** | one matrix, internally | fit + **inference latency** | 4 all-reduces **per block per step** | needs NVLink; dies across nodes |
| **Sequence** | the token dimension | long context | moderate | the only one that shards activations |

---

## 4. The Collectives — identified by payload, not by name

Everything in this note is assembled from a handful of **collective operations**, implemented on GPU by **NCCL**. The mistake is reading `[1,2,3,4]` in a diagram as "data." It's a **flattened parameter tensor** — each slot is one real weight, gradient or activation. **Name the payload and the right collective becomes obvious.**

```
                 GPU0   GPU1   GPU2   GPU3
  start          [1]    [2]    [3]    [4]

① Reduce         [10]    ·      ·      ·      combine → ONE destination rank
② All-Reduce     [10]   [10]   [10]   [10]    combine → EVERY rank gets the result
③ Broadcast      [w]    [w]    [w]    [w]     one rank's buffer copied to all
④ All-Gather     [1234][1234][1234][1234]     concatenate → every rank holds all pieces
⑤ Reduce-Scatter [10ₐ]  [10_b] [10_c] [10_d]  combine, but each rank keeps only ITS slice
⑥ All-to-All     each rank sends DIFFERENT data to every other rank
```

**The table that actually matters — sorted by payload:**

| Collective | Payload | Scales with | Fires when | Used by |
|---|---|---|---|---|
| **Broadcast** | weights | params | once at startup | data-parallel init (identical replicas) |
| **All-reduce** | **gradients** | params | during backward | **data parallel** |
| **All-reduce** | **activations** | `batch × seq` | inside fwd *and* bwd | **tensor parallel** |
| **All-gather** | weights | params | before each unit, fwd & bwd | **ZeRO-3 / FSDP** |
| **Reduce-scatter** | gradients | params | after each unit's bwd | **ZeRO / FSDP** |
| **All-to-all** | tokens | `batch × seq` | at each MoE layer | expert parallel |
| **Reduce** | one scalar | — | logging | `dist.reduce(loss, dst=0)` |
| **Send/Recv** *(point-to-point, not a collective)* | activations | `batch × seq` | at stage seams | **pipeline** |

Notice that the same collective (**all-reduce**) appears twice with completely different economics — because in DDP it carries *gradients* (sized by `P`, once per step) and in tensor parallelism it carries *activations* (sized by `batch × seq`, four times per block). Same primitive, opposite scaling behaviour. That distinction is §8's entire argument.

### All-reduce vs reduce-scatter — same sum, different endpoint

```
              rank0        rank1        rank2        rank3
  ∂L/∂W    [.1,.2,.3,.4] [.5,.1,.2,.6] [.3,.3,.1,.2] [.1,.4,.4,.8]
           ↑ each slot is ONE ACTUAL WEIGHT, e.g. W = [[w₁,w₂],[w₃,w₄]]

  ALL-REDUCE    [1,1,1,2] on every rank   ← DDP: every rank owns all 4
                                             weights, so it must update all 4
  REDUCE-SCATTER  [1]  [1]  [1]  [2]      ← FSDP: rank0 owns only w₁ and
                                             holds only w₁'s m, v, master copy.
                                             The other 3 summed gradients are
                                             USELESS to it — so don't ship them.
```

**The identity, its cost, and the payoff — derive this, don't memorize it:**

```
all-reduce = reduce-scatter + all-gather

  reduce-scatter:  (N−1)/N · S      ┐
  all-gather:      (N−1)/N · S      ┘→ all-reduce: 2·(N−1)/N · S

Derivation: ring topology, N−1 steps per phase, S/N bytes moved per step.
Payoff:     per-GPU traffic is FLAT IN N → data parallelism scales to 1000s.
```

That identity is not trivia. It is *why* ring all-reduce costs `2·(N−1)/N·S`, and it is the reason FSDP's gradient sync is **literally half the cost of DDP's** (§6) — reduce-scatter is one of the two phases, not both.

> 🚩 **The deadlock rule that follows:** collectives are **barriers**. Every rank must call every collective, **in the same order**. A single `if rank == 0: dist.all_reduce(x)` hangs the entire job until the NCCL timeout fires (~30 min of burnt GPU-hours). It hides in rank-0-only logging, per-rank early exits (`if not batch: continue`), and dataloaders of uneven length across ranks. This is the single most common distributed-training bug. `(certain)`

---

## 5. Data Parallelism and DDP

The most common strategy by a wide margin, and the one to reach for first.

**The mental model:** four students each do a different quarter of the homework, then average their answers and all write down the same average. **The averaging *is* the sync** — there is no separate weight-sync step, which is why the replicas never drift.

![Distributed Data Parallel — four GPUs each hold the same weights and process a different mini-batch of eight samples, producing local gradients that are averaged by an all-reduce so every GPU applies an identical update, giving an effective batch of thirty-two.](attachments/ddp-gradient-all-reduce.png)

### Traced on GPT-2 small, 4 GPUs, per-device batch 8

```
  startup:   BROADCAST weights rank0 → all       (bit-identical start)
  forward:   no collectives at all
  backward:  grads arrive lm_head → block11 → block10 → …
             bucketed (~25 MB); ALL-REDUCE fires per bucket as soon as it
             fills → OVERLAPPED with the still-running backward pass
  optimizer: no collectives. Identical ḡ → identical update → still identical ✓

  volume:    2 × (3/4) × 248 MB = 372 MB per GPU per step
             (248 MB = 124 M gradients × 2 B)
```

**Memory per GPU: still 1.99 GB. Completely unchanged.**

🚩 **DDP gives ZERO memory relief.** Every GPU holds a *complete* replica — weights, gradients, master copy, and both Adam moments. **If a model OOMs on one GPU it OOMs on a thousand.** This is *the* trap question, and it is precisely the gap ZeRO was invented to close. `(certain)`

### Effective batch — the bug that follows you

```
effective_batch = per_device × grad_accum × num_gpus = 8 × 1 × 4 = 32
```

Going from 1 GPU to 8 **multiplied your effective batch by 8 without you asking for it**. Fewer, less-noisy updates per epoch is a genuinely different optimization problem. The fix is **`√N` LR scaling for Adam-family optimizers** (the linear rule is the SGD-era result from the Goyal et al. ImageNet-in-1-hour work) **plus warmup**, since a large LR straight into an untrained state diverges. This is the entire answer to *"I added GPUs and my loss curve got worse."* `(certain)`

> **Gradient accumulation is data parallelism across *time* instead of *devices*** — same effective batch, one GPU, more wall-clock. See [Fine-Tuning LLMs §11](Fine-Tuning%20LLMs.md#11-production--llmops-notes). The two multiply, which is how you hit a target batch size on whatever hardware you actually have.

### `DataParallel` vs `DistributedDataParallel`

| | `DataParallel` (legacy) | `DistributedDataParallel` |
|---|---|---|
| Processes | **one**, multi-threaded — GIL-bound | **one per GPU** |
| Gradient sync | gather to GPU 0, then scatter | ring all-reduce, overlapped with backward |
| Verdict | legacy, **never use it** | the standard |

**Diagnostic worth memorizing: GPU 0 at 90% memory while the others sit at 40% → you are on `DataParallel`**, because rank 0 gathers every replica's outputs to compute the loss.

---

## 6. ZeRO and FSDP — sharding what you *store*

**The observation:** under DDP with `N = 4`, all four GPUs store **bit-identical** copies of `m`, `v` and the master weights. That is 4× redundancy on data that is provably identical. So store `1/4` each, and borrow the rest on demand.

![The four ZeRO stages — baseline data parallelism stores parameters, gradients and optimizer states in full on every GPU at 120 GB for a 7.5 B model, P-os shards optimizer states down to 31.4 GB, P-os+g adds gradient sharding at 16.6 GB, and P-os+g+p adds parameter sharding at 1.9 GB, with communication volume staying at 1x for the first three stages and rising to 1.5x for the last.](attachments/zero-stages-memory-breakdown.png)
*Source: Rajbhandari et al., ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054), via the Hugging Face parallelism docs. `Ψ` = parameters, `K = 12` = optimizer-state bytes per parameter — the same 2 + 2 + 12 accounting as §2.*

### The four-beat cycle — per unit, which means per transformer block

```
all-gather  →  compute  →  free  →  reduce-scatter
(rebuild)      (use)       (drop)   (backward only)
```

### Traced on GPT-2 small, 4 GPUs

```
AT REST     GPU0 holds rows 0-191 of every matrix, plus the matching quarter of
            grads / master / m / v = 496 MB.
            It CANNOT run the model in this state. Nothing it holds is usable.

FORWARD     block 0:  ALL-GATHER (the others send their row-slices; GPU0 now holds
                        block 0 whole — 7.08 M params, 14.2 MB in bf16)
                      COMPUTE
                      FREE the 3/4 it doesn't own   ← no comms, just free()
            blocks 1-11: identical
            (block 6's all-gather is prefetched during block 5's compute)

BACKWARD    block 11: ALL-GATHER the weights again (they were freed)
                      COMPUTE gradients for ALL of block 11
                      REDUCE-SCATTER → GPU0 keeps only rows 0-191; the other
                        three quarters are summed and DISCARDED — they aren't
                        its shard, and it has no m/v/master for them
                      FREE
            blocks 10..0: identical

OPTIMIZER   no communication whatsoever. GPU0 updates rows 0-191 from its own
            shard of master/m/v. It cannot update anything else. It doesn't need to.

peak memory = 496 MB + one full block (14.2 MB) + activations
```

### The stages

| Stage | Shards | B/param at `N = 4` | GPT-2 small | GPT-2 XL |
|---|---|---|---|---|
| DDP | nothing | 16 | 1.99 GB | 24 GB ✗ |
| **ZeRO-1** | optimizer states (the 12 B) | `2 + 2 + 12/N` = **7** | 868 MB | **10.5 GB** ✓ |
| **ZeRO-2** | + gradients | `2 + 2/N + 12/N` = **5.5** | 682 MB | 8.25 GB |
| **ZeRO-3** | + weights | `16/N` = **4** | 496 MB | 6 GB |
| **+ offload** | to CPU / NVMe | lower still | — | much slower |

🎯 **Stage 1 is free money** — **56% of your memory back for essentially no extra communication**, because no GPU ever needs another GPU's optimizer state at any point. It is badly underused; people leap straight to stage 3 and pay for weight all-gathers they didn't need. `(certain)`

### The cost, derived rather than quoted

```
grads in bf16 = 248 MB, N = 4 → the (N−1)/N factor is 3/4

DDP    : all-reduce                       2 × 3/4 × 248  =  372 MB
FSDP   : all-gather weights (forward)         3/4 × 248  =  186 MB
         all-gather weights (backward)        3/4 × 248  =  186 MB
         reduce-scatter gradients             3/4 × 248  =  186 MB
                                                            ───────
                                                            558 MB   = 1.5× ✓
```

**Two non-obvious things fall out of that arithmetic:**

- **FSDP's gradient sync is *cheaper* than DDP's** — 186 MB vs 372 MB, because reduce-scatter is literally half an all-reduce. The entire 1.5× premium is the **two weight all-gathers**, not the gradient handling.
- Setting `reshard_after_forward=False` keeps the forward's gathered weights alive into the backward, dropping the second all-gather: `186 + 186 = 372 MB`, **identical to DDP**, at higher memory. That flag stops being arbitrary the moment you've done this sum. `(certain)`

**The one-line contrast to keep:**

```
DDP :  same model everywhere  +  different data
FSDP:  sharded model (weights + grads + optimizer state)  +  different data
       +  temporary reconstruction of ONE UNIT at a time
```

> **Why this is the modern default.** ZeRO-3 is still *data* parallelism — same programming model, no model surgery, no rewriting your layers — but with per-GPU state falling as `16P/N`. You get capacity scaling with the simplicity of DP. Reach for pipeline or tensor parallelism only when ZeRO-3 genuinely isn't enough. `(certain)`

### FSDP2 — your old config is deprecated

PyTorch's `FullyShardedDataParallel` wrapper (**FSDP1**) is deprecated as of PyTorch 2.11; **FSDP2** is the current API and is not backwards-compatible. Know both, because half the tutorials online are still FSDP1. `(certain)`

| | FSDP1 (deprecated) | **FSDP2 (current)** |
|---|---|---|
| Shard representation | `FlatParameter` — flatten, concat, chunk | **`DTensor`**, per-parameter, chunked on dim 0 |
| API | `FullyShardedDataParallel(model)` wrapper | **`fully_shard(module)`**, applied in place |
| Parameter names | mangled | preserved (so LoRA / partial freezing work out of the box) |
| Stage dial | `sharding_strategy=FULL_SHARD` | **`reshard_after_forward=True/False`** |
| Mixed dtypes (fp8) | no | yes |
| Composition | awkward | `DeviceMesh` is first-class → composes with TP/HSDP |

```yaml
# accelerate config — FSDP2. Note accelerate still defaults to fsdp_version: 1,
# so you must set it explicitly.
distributed_type: FSDP
mixed_precision: bf16                              # prefer bf16: no loss-scaling drama
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: true                 # true = ZeRO-3; false ≈ ZeRO-2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer   # ← ONE BLOCK PER UNIT
  fsdp_offload_params: false                       # true → CPU offload; last resort
  fsdp_state_dict_type: SHARDED_STATE_DICT         # the FSDP2 default, and correct
```

🚩 **Wrap granularity is *the* knob.** Wrap the whole model as one unit and you get one giant all-gather — DDP's peak memory, having already paid FSDP's communication. Wrap every `nn.Linear` and you get hundreds of tiny latency-bound collectives. **One transformer block per unit** is the answer, and it's what lets unit `k+1` prefetch its all-gather during unit `k`'s compute. `(certain)`

**FSDP vs DeepSpeed in practice:** **FSDP2** if you're in pure PyTorch/HF and want the native, `DTensor`-based path that composes with tensor parallelism. **DeepSpeed** if you want the mature stage-by-stage config, CPU/NVMe offload, and its pipeline integration.

---

## 7. Pipeline Parallelism — and the bubble

Cut the model **between layers**, and the only thing that crosses a seam is an activation tensor.

```
GPT-2 small, 4 stages:
GPU0: blocks 0-2  ──▶  GPU1: blocks 3-5  ──▶  GPU2: 6-8  ──▶  GPU3: 9-11

boundary payload: activations only — (4, 1024, 768) in bf16 = 6.3 MB
point-to-point send/recv, NOT a collective.
→ the CHEAPEST dimension to stretch across a slow link.
```

**And it is catastrophic if you stop there** — one GPU works while `G−1` idle:

```
time →   1    2    3    4    5    6    7    8
GPU0   [F1]   ·    ·    ·    ·    ·    ·  [B1]
GPU1     ·  [F2]   ·    ·    ·    ·  [B2]   ·
GPU2     ·    ·  [F3]   ·    ·  [B3]  ·     ·
GPU3     ·    ·    ·  [F4][B4]  ·     ·     ·
                                    utilization = 1/4
```

![Two Gantt charts comparing naive model parallelism with pipeline parallelism — naive processes one batch and leaves seventy-five percent of GPU-time idle in the bubble, while splitting the same batch into four micro-batches drops idle time to forty-three percent and finishes in 3.5 versus 8 time units.](attachments/pipeline-bubble-micro-batches.png)

**The fix: split the batch into `M` micro-batches and stream them**, so GPU0 starts micro-batch 2 while GPU1 works on micro-batch 1.

```
ideal  = M slots (every stage busy every slot)
actual = M + G − 1 slots  (G−1 slots to fill the pipe, G−1 to drain it)

bubble = (M + G − 1 − M) / (M + G − 1) = (G−1)/(M + G − 1)
```

| `G = 4`, `M =` | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| bubble | **75%** | 43% | 27% | 16% | 8.6% |

**Design rule: `M ≥ 4G`.** The cost is memory — more micro-batches in flight means more concurrently-live activation sets. **1F1B scheduling** (one-forward-one-backward, as in PipeDream and Megatron) achieves the *same* bubble fraction while holding far fewer activations live, and it's what production stacks actually run. `(certain)`

⚠️ **The `16P/G` per-GPU figure quietly assumes equal stages, and real transformers aren't equal.** GPT-2's token embedding is 38.6 M parameters and the output head is the same size again — both land on the *end* stages, which then become stragglers while the middle stages wait. **Production configs balance stages by *cost*, not by layer count**, and often give the embedding stages fewer blocks to compensate. `(certain)`

> ⚠️ Any operation needing **global batch statistics** breaks when the batch is streamed as micro-batches — the classic clash is with `batch_norm`. Transformers use [LayerNorm/RMSNorm](../Machine%20Learning/Neural%20Networks/Batch%20Normalization%20&%20Dropout.md), which normalize *per-token, per-feature*, so this is a non-issue for LLMs — but it's why the technique reads differently in CNN literature. `(certain)`

---

## 8. Tensor Parallelism — cutting inside the block

Tensor parallelism (Megatron-LM style) cuts the four matrices **inside** a block.

🚩 **Hard constraint first: the parallelism degree must divide the head count.** GPT-2 small has 12 heads → **2, 3, 4, 6, 12 all work; 8 does not.** With modern **grouped-query attention** the binding constraint is the number of *KV* heads, not query heads — Llama-3-70B has 64 query heads but only 8 KV heads, so TP > 8 requires replicating KV heads across ranks rather than splitting them. `(certain)`

![Megatron-LM's tensor-parallel MLP block — the first weight matrix A is split column-wise into A1 and A2 so GeLU can be applied independently on each GPU with no communication, the second matrix B is split row-wise into B1 and B2 producing partial sums, and the g operator all-reduces them into the final output Z.](attachments/megatron-mlp-column-then-row.png)
*Source: Shoeybi et al., ["Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053), Figure 3a, via the Hugging Face parallelism docs. `f` is identity forward / all-reduce backward; `g` is all-reduce forward / identity backward.*

### Feed-forward: `Y = GeLU(X · W_fc) · W_proj`

```
W_fc  split by COLUMN          W_proj split by ROW
┌─────────┬─────────┐          ┌──────────────────┐
│ cols    │ cols    │          │rows 0-1535 →GPU0 │
│ 0-1535  │1536-3071│          ├──────────────────┤
│  GPU0   │  GPU1   │          │rows 1536-3071    │
└─────────┴─────────┘          │            →GPU1 │
                               └──────────────────┘
GPU0: H_left = X · W_fc[:, :1536]   GPU0: Y₀ = GeLU(H_left) · W_proj[:1536, :]
      shape (4, 1024, 1536)               shape (4, 1024, 768) — a PARTIAL SUM
      complete, but half-WIDTH            full-width, but INCOMPLETE

★ WHY COLUMN FIRST — the whole reason the order is what it is:
  GeLU is applied per-number, so
      GeLU([H_left | H_right]) == [GeLU(H_left) | GeLU(H_right)]
  → each GPU GeLUs its own half. ZERO communication.

  Cut W_fc by ROW instead and each GPU holds a PARTIAL SUM, and
      GeLU(a + b) ≠ GeLU(a) + GeLU(b)   — it is NONLINEAR
  forcing an all-reduce in the MIDDLE of the block, before the activation.
  The nonlinearity is the entire reason orientation matters.

Then: Y = Y₀ + Y₁  →  ONE ALL-REDUCE.
And the column-cut's output shape == the row-cut's input shape, so the two
chain together for free. That's why Megatron pairs them.
```

### Attention: heads are the free seam

Heads never talk to each other — head 7 computes its own Q/K/V, its own softmax, its own weighted sum, and only meets head 8 at `W_out`. So the head dimension is a seam that already exists in the math.

```
W_qkv split by COLUMN (= by head)  →  GPU0 gets heads 0-5, GPU1 heads 6-11
       the entire attention computation is local. No comms.
W_out split by ROW                 →  partial sums → ONE ALL-REDUCE
```

### What stays whole, and the count that matters

```
input X                     ← IDENTICAL on both GPUs
layer norm                  ← DUPLICATED (wasted memory → §9 fixes exactly this)
attention  → ALL-REDUCE ①
residual add                ← DUPLICATED
layer norm                  ← DUPLICATED
feed-forward → ALL-REDUCE ②
residual add                ← DUPLICATED
output                      ← IDENTICAL

2 forward + 2 backward = 4 ALL-REDUCES PER BLOCK PER STEP
  GPT-2 small (12 blocks) →   48 per step
  GPT-3       (96 blocks) →  384 per step
```

### Why it can't cross servers — do the GPT-3 arithmetic

```
payload per all-reduce = 1 × 2048 × 12288 × 2 B  =  50 MB
ring cost              = 2 × 7/8 × 50            =  88 MB per GPU

NVLink   @ 600 GB/s :  0.15 ms × 384  ≈   56 ms/step   ✓
Ethernet @  12 GB/s :  7.3  ms × 384  ≈  2.8 s/step    ✗ dead

Total TP volume = 384 × 88 MB                    =  33.8 GB/step
Total DP volume = 2 × 7/8 × (175e9 × 2 B grads)  =   613 GB/step   (N=8)
                                                    → TP moves 18× LESS data.
```

🎯 **So it was never about volume.** *"DDP moves its bytes in one stream, overlapped with the backward pass. Tensor parallelism moves 18× fewer bytes, but in 384 blocking stalls where the next matmul cannot start until the result lands — plus ~10 µs of fixed kernel-launch cost per collective, which is 4 ms per step on its own. Latency you can't hide beats bandwidth you can."* **TP stays inside one server.** `(certain)`

🚩 **A correction to the framing you'll see everywhere.** *"Tensor parallelism is for when a single layer won't fit"* is a **teaching case, not the production motive** — one GPT-3 block is 29 GB of state and fits comfortably on an 80 GB card. The real reasons production runs use TP: **(1)** it automatically shards the enormous `(batch, seq, 4d)` FFN activation, which nothing else does; **(2)** more blocks per GPU means fewer pipeline stages, which means a smaller bubble; **(3)** it is **the main latency lever at inference**, where data parallelism can do nothing for a single request. `(certain)`

### Vocabulary parallelism — the trick nobody mentions

GPT-2's output projection is `768 × 50257`. At batch 4, seq 1024 the **logit tensor alone** is `4 × 1024 × 50257 × 2 B` = **412 MB** — bigger than the entire model's weights. Megatron splits by **vocabulary** (GPU0 owns tokens 0–12564, GPU1 the next slice, …) and **fuses cross-entropy into the split**: the loss only needs the correct token's logit and the sum of exponentials over all tokens, and both are obtainable by all-reducing a couple of *scalars* per position. **The full logit tensor is never assembled anywhere.** `(certain)`

---

## 9. Sequence and Expert Parallelism — the two often missed

**Sequence / context parallelism** is the **only** strategy that shards **activations** rather than parameters — which makes it the answer to a question none of the others can touch. It splits the *token* dimension: GPU0 holds tokens 0–511, GPU1 holds 512–1023. Attention is inherently global (every token attends to every other), so there are two flavours:

- **Megatron-style sequence parallelism:** shard only the token-*independent* operations — layer norm, dropout, the residual adds. Look back at §8's "what stays whole" block: those DUPLICATED lines are exactly what this reclaims. It layers on top of tensor parallelism at no extra communication, converting TP's all-reduces into reduce-scatter/all-gather pairs.
- **Ring Attention:** shard attention itself, passing key/value blocks around a ring so each GPU eventually sees every key. This is what makes million-token context windows tractable.

🎯 **The diagnostic:** *"If sharding weights didn't help, your problem was never weights."* Long-context OOM is an **activation** problem — reach for activation checkpointing first (cheap, one flag), then sequence parallelism. `(likely — implementations move fast; verify the specific API before quoting one)`

**Expert parallelism** applies only to **Mixture-of-Experts** models: different experts live on different GPUs, and each MoE layer performs **two all-to-all** operations — one to route each token to the GPU holding its chosen expert, one to route the result back. All-to-all is the only collective where every rank sends *different* data to every other rank, which makes it uniquely sensitive to network topology. Its signature failure is **router load imbalance**: one popular expert's GPU becomes a straggler and every other rank blocks waiting for it, which is why production MoE training carries an auxiliary load-balancing loss. See [[Mixture of Experts]] for the routing mechanics themselves. `(certain)`

---

## 10. Composing Them — 3D parallelism and HSDP

**The assignment rule falls out of communication cost — don't memorize it, derive it:**

```
chattiest ──────────────────────────────────────────► least chatty
TENSOR                  ZeRO / FSDP            PIPELINE  /  DATA
4 all-reduces per       2 all-gathers +        activations at seams /
block, on the           1 reduce-scatter       1 all-reduce per step,
CRITICAL PATH           per block              fully overlapped
     ↓                        ↓                        ↓
NVLink, inside          within or near         across servers
one server              a node
```

![Three-dimensional parallelism on thirty-two GPUs — a grid where one axis is model or tensor parallelism, a second axis is pipeline parallelism across stages, and the third axis is ZeRO-backed data parallelism across replicas.](attachments/three-d-parallelism-32-gpus.png)
*Source: Microsoft DeepSpeed, "3D parallelism" blog figure, via the Hugging Face parallelism docs.*

**3D parallelism** is the frontier combination — e.g. 32 GPUs as **TP 8** (inside each server) × **PP 2** (across server pairs) × **DP 2** (the rest), almost always with **ZeRO-1 layered on the DP dimension** because it's free. A 70B+ pre-training run is essentially always some version of this.

**HSDP (hybrid sharding)** is the option most people don't know exists: **shard *within* a server, replicate *across* servers.** The frequent weight all-gathers stay on NVLink where they're cheap; the slow inter-server link sees exactly one overlapped all-reduce per step, precisely like DDP. You give up memory saving (`16P/8` instead of `16P/32`) and often gain a lot of throughput. In FSDP2 it's just a 2D `DeviceMesh`.

🎯 **When HSDP beats plain FSDP:** *"when FSDP's memory scaling is already sufficient but the network is your bottleneck."* That's the sentence — it's a **network** decision, not a memory one. `(certain)`

---

## 11. Worked Example — sizing a 70B run

**Task:** full fine-tune a 70B model. Hardware: nodes of 8×A100-80GB (NVLink inside, InfiniBand between).

**Step 1 — the raw requirement.**
```
state    = 70e9 × 16 B  =  1,120 GB   ≈ 1.12 TB
+ activations (batch × seq × d × layers)   ~100–300 GB depending on
                                            batch/seq/checkpointing
```

**Step 2 — rule out the cheap options, in order.**
- One GPU: 80 GB. **No** — off by 14×.
- DDP on 8 GPUs: still `1.12 TB` **per GPU**. **No** — DDP shards nothing (§5).
- ZeRO-1 on 8 GPUs: `70e9 × 7 = 490 GB` per GPU. **No**, but note it got you 56% for free.
- ZeRO-3 on 8 GPUs: `1120/8 = 140 GB` per GPU. **Still no.**
- ZeRO-3 on 16 GPUs (2 nodes): `70 GB` per GPU + activations. **Borderline** — tight, but activation checkpointing makes it plausible.
- ZeRO-3 on 32 GPUs (4 nodes): `35 GB` state per GPU, comfortable room for activations. **Yes.**

**Step 3 — the realistic production layout.** Pure ZeRO-3 across 32 GPUs works, but it all-gathers parameters across the *slow* inter-node link on every single block. Better:

```
TP = 8   within each node   (NVLink — the chatty dimension stays local)
PP = 2   across node pairs  (only 6 MB-scale activations cross nodes)
DP = 2   across the rest    (one all-reduce per step, overlapped)
                                    8 × 2 × 2 = 32 GPUs
```

**Step 3b — the cheaper thing to try first.** Before building a 3D config, try **HSDP**: FSDP within each 8-GPU node (`16P/8 = 140 GB` — too much here, but the right first question), replicating across nodes. It's a `DeviceMesh` argument rather than a model rewrite.

**Step 4 — the honest alternative.** Before any of this: **do you actually need a full fine-tune?** [QLoRA](Fine-Tuning%20LLMs.md#6-qlora--quantization-on-top) on a 70B fits on **two** A100s and, for most format/domain adaptation tasks, lands within noise of full fine-tuning. 🎯 *That's the answer that impresses an interviewer:* size the cluster correctly, **then** explain why you'd try hard not to need it.

---

## 12. Code / Implementation

**The most useful thing to know: you rarely write this yourself.** You configure it and launch with a runner.

### 12.1 DDP — what `accelerate`/`torchrun` does for you

```python
# Explicit DDP, to see the moving parts. In practice, use a launcher.
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")                 # NCCL = NVIDIA's GPU collective library
rank = int(os.environ["LOCAL_RANK"])            # set by the launcher, one process per GPU
torch.cuda.set_device(rank)

model = MyModel().to(rank)
model = DDP(model, device_ids=[rank])           # wraps backward with bucketed all-reduce

# CRITICAL: without DistributedSampler every GPU sees the SAME data —
# you'd compute the identical gradient N times and learn nothing extra.
loader = DataLoader(dataset, batch_size=8, sampler=DistributedSampler(dataset))

for epoch in range(epochs):
    loader.sampler.set_epoch(epoch)             # re-shuffles differently each epoch;
                                                # omit it and every epoch sees the same split
    for batch in loader:
        loss = model(batch).loss
        loss.backward()                         # all-reduce happens HERE, overlapped
        optimizer.step(); optimizer.zero_grad()
```

Launch: `torchrun --nproc_per_node=4 train.py`

### 12.2 FSDP2 — native PyTorch

```python
# FSDP2: fully_shard applied IN PLACE, per module. No wrapper object,
# no mangled parameter names — each param becomes a DTensor sharded on dim 0.
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (world_size,))

for block in model.model.layers:                # ← ONE TRANSFORMER BLOCK PER UNIT.
    fully_shard(block, mesh=mesh)               #   This is the memory/comms knob (§6).
fully_shard(model, mesh=mesh)                   # root unit catches embeddings + head

# reshard_after_forward=True (default) = ZeRO-3: free the gathered weights after
# forward, re-gather them in backward. False ≈ ZeRO-2: keep them, pay 372 MB not 558.
```

For **HSDP**, the only change is the mesh: `init_device_mesh("cuda", (num_nodes, gpus_per_node), mesh_dim_names=("replicate", "shard"))`.

### 12.3 FSDP2 / DeepSpeed via HF `Trainer` — the realistic path

Because `Trainer` and `SFTTrainer` are `accelerate`-backed, going multi-GPU is a **config change, not a code change**. Same training script:

```bash
accelerate config          # interactive: multi-GPU, FSDP or DeepSpeed, version, offload
accelerate launch train.py

# Already have an FSDP1 yaml? There's a converter:
accelerate to-fsdp2 --config_file config.yaml --output_file new_config.yaml
```

The FSDP2 yaml is in §6. The DeepSpeed equivalent:

```json
// pass with `--deepspeed ds_config.json`
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

> ⚠️ **The checkpoint gotcha that bites everyone at scale:** with sharded state dicts, saving "the model" writes `N` shard files, and naively calling `.save_pretrained()` tries to gather 70B parameters onto **rank 0**, which OOMs. Keep `SHARDED_STATE_DICT` during training (it's FSDP2's default) and consolidate once at the end — `stage3_gather_16bit_weights_on_model_save`, or an offline merge with `torch.distributed.checkpoint`. `(certain)`

---

## 13. When It Breaks

| Symptom | Cause | Fix |
|---|---|---|
| **Added GPUs, no speedup** | communication-bound, or micro-batch too small to saturate | raise per-device batch; profile the comms/compute overlap |
| **Loss worse after scaling out** | effective batch grew `N×`, LR not rescaled | `√N` scaling (Adam) + warmup |
| **Every GPU computes the same gradient** | missing `DistributedSampler` | add it; call `set_epoch()` each epoch |
| **GPU 0 at 90%, others at 40%** | you're on `DataParallel`, not DDP | switch to DDP |
| **Hang, then NCCL timeout (~30 min)** | a collective called on some ranks only | every rank hits every collective, in the same order |
| **OOM only when saving** | gathering the full state dict onto rank 0 | `SHARDED_STATE_DICT`; consolidate offline |
| **Throughput cliff at N nodes** | tensor parallelism crossed a node boundary | keep `TP ≤ GPUs-per-node` |
| **One slow GPU halves throughput** | straggler — thermal throttling, a bad card, uneven shards | the step ends when the slowest rank ends; monitor **per rank** |
| **Pipeline barely faster than 1 GPU** | too few micro-batches, bubble dominates | `M ≥ 4G` and 1F1B |
| **FSDP saves no memory** | wrapped the whole model as one unit | wrap at the transformer block |
| **Still OOM after ZeRO-3** | **activations** — ZeRO never sharded them | activation checkpointing, then sequence parallelism |
| **Non-deterministic results across runs** | all-reduce sums in nondeterministic order | expected; pin seeds, accept fp non-associativity |

**The debugging order that saves time:** confirm it's *actually distributed* (per-rank logs, distinct data) → confirm it's *correct* (loss curve matches single-GPU at matched effective batch) → **then** optimize throughput. Chasing MFU on a run that's silently training on identical data four times is a bad afternoon.

---

## 14. Production & LLMOps Notes

**Checkpoint the *full* training state.** Multi-day runs on many GPUs *will* be interrupted — a preempted spot instance, a failed NIC, an OOM at an unlucky sequence length. Checkpoint often enough to lose at most ~30 minutes, and save **model + optimizer + LR scheduler + dataloader position + RNG state**. A weights-only checkpoint silently restarts your LR schedule and re-serves data you already trained on, which is worse than no checkpoint because it looks fine.

**Monitor per rank, never per job.** Step time, MFU, GPU utilization, memory headroom, and the communication/compute ratio — all per rank. A job-level average hides the straggler that's costing you 30%.

**Report MFU (Model FLOPs Utilization)** — achieved FLOPs ÷ hardware peak. Well-tuned LLM training lands at **40–55%**. Scaling efficiency (`speedup(N)/N`) of 70–90% is good; **anyone quoting linear speedup is quoting a benchmark, not a training run.** `(likely)`

**Elastic and fault-tolerant training** (`torchrun --max-restarts`, elastic agents) lets a run survive losing a node. Configure it before you need it.

**Cost.** Multi-GPU is billed by the node-hour, so a run at 40% MFU costs 2× one at 80% for the same result. The cheapest wins are usually: bigger micro-batch (fewer, fatter kernels), `bf16`, FlashAttention, gradient checkpointing tuned rather than blanket-on, and fused optimizers. Spot/preemptible instances cut cost 60–90% and are viable *only* with solid checkpointing.

**Reproducibility.** Floating-point addition isn't associative, so all-reduce ordering changes results slightly run-to-run. Pin seeds and library versions, log the world size and parallelism config alongside the checkpoint, and treat exact bitwise reproduction as a non-goal unless you've deliberately paid for it.

**Before you build any of this:** the highest-leverage production decision is usually **not to**. [LoRA/QLoRA](Fine-Tuning%20LLMs.md#5-peft-and-lora--the-mechanism) on one or two GPUs solves most real adaptation needs. Distributed *training* is for pre-training, genuine full fine-tunes, and models too large to load — a specialist tool, not a default.

---

## 15. Interview Lens

**What the question is really testing:** whether you can match a *strategy* to a *bottleneck*, not whether you can list techniques. 🚩 **The failure mode is reciting "data, model, tensor, pipeline" as a list.** Lead with the diagnostic question, every single time.

**The 60-second spine — say this:**

> *"First I'd ask which problem we have. Training's slow but the model fits → that's a **throughput** problem → data parallelism: replicate, all-reduce the gradients once per step, overlapped with the backward pass. It doesn't fit → that's a **capacity** problem → ZeRO-3 or FSDP first, because it's still data parallelism — no model surgery — but each GPU stores only `1/N` of the weights, gradients and optimizer state. Still short → tensor parallelism inside the node, because its all-reduces need NVLink, and pipeline parallelism across nodes, because only activations cross that seam. At frontier scale you compose all three. And the ordering comes straight from communication cost: data parallelism communicates once per step and hides it behind the backward pass; tensor parallelism communicates four times per block on the critical path with nothing to hide behind."*

| They ask | Crisp answer |
|---|---|
| Data vs model parallelism? | Data splits the **batch**, fixes **speed**, gives **zero** memory relief. Model splits **layers**, fixes **fit**. |
| Model vs tensor parallelism? | Model/pipeline splits **between** layers; tensor splits **inside** one layer. |
| All-reduce vs reduce-scatter? | Same sum, different endpoint. All-reduce: everyone gets the whole result, `2(N−1)/N·S`. Reduce-scatter: everyone gets only their slice, half that. FSDP uses the latter because each rank owns only a shard and would discard the rest. |
| Model OOMs on one of 8 GPUs — plan? | ZeRO-1 first (56% free), then ZeRO-3/FSDP: same programming model, state → `16P/N`. TP inside the node only if that's still insufficient. And I'd ask whether QLoRA solves it on one GPU. |
| How do DDP replicas stay in sync? | They start identical and apply the same averaged gradient. **The all-reduce *is* the sync** — there's no separate weight-sync step. |
| What's the bubble? | `(G−1)/(M+G−1)` — `G−1` slots to fill the pipe, `G−1` to drain it. Fix with `M ≥ 4G` and 1F1B. |
| Why does DDP scale to thousands of GPUs? | Ring all-reduce is `2(N−1)/N·S` **per GPU** — flat in `N` — once per step, overlapped with backward. |
| Why does Megatron cut column-then-row? | Column first so **GeLU, which is nonlinear, can be applied locally with no communication**; row second so its input shape matches the column-cut's output, giving partial sums and exactly **one** all-reduce. |
| Why must TP stay in one node? | 4 all-reduces per block per step, on the critical path, unhideable. For GPT-3 that's 384 × 50 MB. NVLink 56 ms/step, Ethernet 2.8 s/step. |
| ZeRO stages? | 1 = optimizer states, 2 = + gradients, 3 = + weights. FSDP2 ≈ ZeRO-3. **Stage 1 gets you 56% for nearly free.** |
| Does 8 GPUs mean 8× faster? | No — communication, stragglers, bubbles. 70–90% scaling efficiency is good; report **MFU**, 40–55% is healthy. |
| How would you serve a 175B model? | **Not ZeRO** — it's a *training* technique; at inference there are no gradients and no optimizer state to shard. **Tensor or pipeline parallelism.** |
| Long context won't fit? | That's **activations**, not parameters — sharding weights does nothing. Activation checkpointing, then sequence parallelism. |
| Why can GPT-2 small use TP=4 but not TP=8? | TP degree must divide the head count, and it has 12 heads. |

### Master comparison

| | Data parallel | **ZeRO / FSDP** | Tensor | Pipeline | Sequence |
|---|---|---|---|---|---|
| Splits | the batch | **who *stores*** | one matrix, internally | layers | activations |
| Runs the whole model? | yes | **yes** | no | no | yes |
| Model code changes? | no | **no** | yes | yes | yes |
| Memory relief | **none** | state ÷ N | state ÷ N | state ÷ G | activations ÷ N |
| Cut mathematically meaningful? | — | **no — filing only** | yes | yes | yes |
| Payload | gradients | weights + gradients | **activations** | activations | activations |
| Volume vs DDP | 1× | 1.5× | ~1× but **unhideable** | tiny | moderate |
| Overlappable? | yes | mostly | **no** | yes | partly |
| Cross servers? | yes | with care (→ HSDP) | **never** | **best option** | within node |
| Works at inference? | n/a | **no** | **yes** | yes | yes |
| Helps if one layer won't fit? | no | **no** | **yes** | no | no |

---

## 16. Alternatives & How to Choose

**The decision tree — the first two questions are the ones that save the most money, and neither of them is about GPUs** ([RAG](RAG.md) and prompting beat training far more often than people expect):

```
Do you need weight updates at all, or will RAG / prompting do?
  └─ no → don't train

Do you need FULL-model updates, or will LoRA/QLoRA do?
  └─ LoRA, usually → most "fine-tuning" is format/domain adaptation

Does the model + training state fit on one GPU?
  ├─ YES → DDP (speed only)
  └─ NO  → ZeRO-1 first (free money)
           → ZeRO-3 / FSDP2   ← THE DEFAULT ANSWER
           → network-bound rather than memory-bound? → HSDP
           → still short? TP inside the node
           → still short? PP across nodes
           → still short? QLoRA / quantization

Long context won't fit but the parameters are fine?
  └─ that's ACTIVATIONS → checkpointing, then sequence parallelism
```

| Option | Use when | Don't when |
|---|---|---|
| **Single GPU + gradient accumulation** | the model fits; you just want a bigger effective batch | you're wall-clock bound |
| **[LoRA / QLoRA](Fine-Tuning%20LLMs.md)** | adapting a model to a task/domain/format — *most real work* | pre-training, or a genuine full-model update |
| **DDP** | model fits, training is too slow | the model doesn't fit — DDP will not help at any `N` |
| **ZeRO-1** | you're data-parallel and even slightly memory-tight | never — it's ~free, take it |
| **ZeRO-3 / FSDP2** | **the default for "doesn't fit on one GPU"** | your bottleneck is the inter-node network → HSDP |
| **HSDP** | FSDP's memory scaling is fine but the network is the bottleneck | you need maximum memory saving |
| **Tensor parallelism** | inside a node, over NVLink; **or for inference latency** | across nodes on commodity networking — ever |
| **Pipeline parallelism** | very deep model, slow links between nodes | you can't supply `M ≥ 4G` micro-batches |
| **Sequence parallelism** | long context, activations are the constraint | parameters are the constraint |
| **3D parallelism** | 70B+ pre-training on a real cluster | anything smaller — the complexity isn't free |
| **[Quantization](Model%20Quantization.md)** | shrink the bytes instead of splitting them — the only lever that also helps at **inference** | you need full-precision weight updates |
| **Rent a bigger GPU** | honestly, more often than people admit | you need more than one node's worth |

**The lecture's own version, which folds in the quantization branch:**

```
Model + training state fits on one GPU?
├─ YES ──→ DDP
└─ NO  ──→ Can you afford FULL training memory, spread across GPUs?
           ├─ YES ──→ FSDP / ZeRO-3
           └─ NO  ──→ LoRA  ──→  still doesn't fit?  ──→  QLoRA (4-bit NF4 base)
```

That last branch is where this note hands off to [Model Quantization](Model%20Quantization.md).

---

## 17. Formula Sheet

```
state_bytes        = 16 · P            (2 bf16 W + 2 bf16 G + 4 master + 4 m + 4 v)
params_per_block   = 12 · d²           (GPT-2 small: 12 × 768² = 7.08 M)
effective_batch    = per_device × grad_accum × num_gpus

ring all-reduce    = 2 · (N−1)/N · S   per GPU   (FLAT IN N)
reduce-scatter     =     (N−1)/N · S             (half an all-reduce)
all-reduce         = reduce-scatter + all-gather

ZeRO-1 per GPU     = 4P + 12P/N        ZeRO-2 = 2P + 14P/N        ZeRO-3 = 16P/N
FSDP volume        ≈ 1.5 × DDP         (the premium is the 2 weight all-gathers)

pipeline bubble    = (G−1)/(M+G−1)     rule: M ≥ 4G
TP all-reduces     = 4 per block per step   (2 forward + 2 backward)
TP degree          must divide the head count (KV-head count under GQA)

efficiency         = speedup(N)/N                    good: 0.7–0.9
MFU                = achieved FLOPs / peak FLOPs     healthy: 0.40–0.55
```

---

## 🧠 Self-Test

1. **Name the six buffers in GPU memory during mixed-precision Adam training and their byte counts. What's the total for GPT-2 XL, and which buffer is the one that will actually OOM you?**
   <details><summary>answer</summary>① bf16 <b>weights</b> 2 B, ② bf16 <b>gradients</b> 2 B, ③ fp32 <b>master weights</b> 4 B, ④ Adam <b>m</b> 4 B, ⑤ Adam <b>v</b> 4 B — <b>16 B/param persistent</b> — plus ⑥ <b>activations</b>, which scale with <code>batch × seq × d × layers</code>, <i>not</i> with <code>P</code>. GPT-2 XL: <code>1.5e9 × 16 = 24 GB</code> before a single activation. ③ exists because bf16 has only 7 stored mantissa bits, so <code>w − 1e-7·g</code> in bf16 is a no-op. <b>⑥ is the one that gets you:</b> the first five divide by N under sharding and activations don't, so post-FSDP an OOM is almost always activations.</details>

2. **Derive `2·(N−1)/N·S` from the identity `all-reduce = reduce-scatter + all-gather`, and say what the result implies.**
   <details><summary>answer</summary>A ring does each phase in <code>N−1</code> steps moving <code>S/N</code> bytes per step, so each phase costs <code>(N−1)/N · S</code> per GPU. All-reduce is both phases → <code>2·(N−1)/N·S</code>. The implication: <b>per-GPU traffic is flat in N</b> (it tends to <code>2S</code>, never growing), which is why data parallelism scales to thousands of devices. It also tells you FSDP's reduce-scatter is <b>half</b> the cost of DDP's all-reduce.</details>

3. **Derive the pipeline bubble formula from "fill takes `G−1` slots, drain takes `G−1` slots." Compute it for G=4 at M=1, 4, 16. What's the rule, and what does it cost?**
   <details><summary>answer</summary>Ideal is <code>M</code> slots with every stage busy; actual is <code>M + G − 1</code>. Bubble = <code>(M+G−1−M)/(M+G−1) = <b>(G−1)/(M+G−1)</b></code>. G=4: M=1 → 3/4 = <b>75%</b>; M=4 → 3/7 ≈ <b>43%</b>; M=16 → 3/19 ≈ <b>16%</b>. Rule: <code>M ≥ 4G</code>. Cost: more micro-batches in flight = more concurrently-live activation sets, so the bubble trades directly against memory — <b>1F1B</b> scheduling gets the same bubble while holding far fewer activations live.</details>

4. **Why does Megatron split the FFN column-first then row, rather than the other way round? Your answer must use the word *nonlinear*.**
   <details><summary>answer</summary>Column-splitting <code>W_fc</code> gives each GPU <i>complete but half-width</i> output features, and because <b>GeLU is elementwise</b>, <code>GeLU([H_left | H_right]) == [GeLU(H_left) | GeLU(H_right)]</code> — each GPU applies the activation to its own half with <b>zero communication</b>. Row-splitting instead gives each GPU a <i>partial sum</i>, and since GeLU is <b>nonlinear</b>, <code>GeLU(a+b) ≠ GeLU(a)+GeLU(b)</code> — you'd need an all-reduce in the <i>middle</i> of the block. Then <code>W_proj</code> is split row-wise because the column-cut's output shape is exactly the row-cut's input shape, so they chain for free and you pay <b>one</b> all-reduce at the end.</details>

5. **Why does DDP give zero memory relief, and what exactly does ZeRO-1 shard — how much does it buy?**
   <details><summary>answer</summary>DDP <b>replicates</b> everything — weights, gradients, master copy, <code>m</code> and <code>v</code> — on every GPU; it only splits the <i>batch</i>. So if 16·P doesn't fit on one card it doesn't fit on a thousand. <b>ZeRO-1 shards only the optimizer states</b> (the 12 fp32 bytes: master + m + v), taking per-GPU state from <code>16P</code> to <code>4P + 12P/N</code> — at N=4 that's 16 → 7 B/param, a <b>56% cut for essentially no extra communication</b>, because no GPU ever needs another GPU's optimizer state. It's the most underused setting in the stack.</details>

6. **Why can't tensor parallelism cross servers? Give the GPT-3 numbers — and be careful, the obvious answer is wrong.**
   <details><summary>answer</summary>Not because of volume — TP moves <b>18× less</b> data than data parallelism for GPT-3 (33.8 GB/step vs 613 GB/step). It's because TP fires <b>4 all-reduces per block per step</b> (2 forward + 2 backward) = <b>384 for GPT-3's 96 blocks</b>, each ~50 MB of activations (<code>1×2048×12288×2 B</code>), ring cost 88 MB/GPU — and every one sits <b>on the critical path</b> where the next matmul cannot start until it lands. NVLink @600 GB/s: 0.15 ms × 384 ≈ <b>56 ms/step</b>. Ethernet @12 GB/s: 7.3 ms × 384 ≈ <b>2.8 s/step</b> — dead. Plus ~10 µs launch latency × 384 ≈ 4 ms on its own. <b>Latency you can't hide beats bandwidth you can.</b></details>

7. **FSDP is running, memory is fine, but you can't push sequence length past 32K. What's the constraint, and what are the two levers?**
   <details><summary>answer</summary><b>Activations</b> — buffer ⑥. ZeRO/FSDP shards weights, gradients and optimizer states; it never touched activations, which scale with <code>batch × seq × d × layers</code>. Adding GPUs doesn't help because you spread more batch across them. Levers: <b>activation checkpointing</b> (store only block boundaries, recompute interiors — roughly a third more compute) and then <b>sequence/context parallelism</b> (shard the token dimension — Megatron-style for the token-independent ops, or Ring Attention for attention itself). 🎯 <i>"If sharding weights didn't help, your problem was never weights."</i></details>

8. **Why is ZeRO the wrong answer to "how would you serve a 175B model"?**
   <details><summary>answer</summary>Because ZeRO shards <b>training</b> state, and at inference there is none — no gradients, no fp32 master copy, no Adam <code>m</code>/<code>v</code>. Three of ZeRO's five buffers don't exist, and the fourth (gradients) doesn't either; you'd be sharding only the weights while paying an all-gather per block on every forward, which is pure added latency. The right answers are <b>tensor parallelism</b> (the main latency lever for a single request, since data parallelism can't help one request) and <b>pipeline parallelism</b> (throughput across stages), plus <a href="Model%20Quantization.md">quantization</a>.</details>

9. **Both tensor parallelism and ZeRO "split a weight matrix." Why is only one of those cuts mathematically meaningful?**
   <details><summary>answer</summary>Tensor parallelism cuts <code>W_fc</code> <b>by column</b>, so GPU0's slice is a set of <i>real, final output features</i> — it can compute with it immediately, and the cut was chosen precisely because GeLU is elementwise. ZeRO cuts <b>by row (dim 0)</b>, so GPU0 holds a <i>fragment</i> — a quarter of the rows of every matrix, from which it can compute <b>nothing</b>. ZeRO's cut is a <b>filing system</b>, not a decomposition, so it must be <b>undone with an all-gather before every single use</b>. 🎯 That's also why TP changes what each GPU <i>computes</i> while ZeRO changes only what each GPU <i>stores</i> — and hence why ZeRO is a config change and TP is a code change.</details>

10. **What's HSDP, and when does it beat plain FSDP?**
    <details><summary>answer</summary><b>Hybrid Sharded Data Parallel</b>: shard <i>within</i> a server, <b>replicate</b> across servers. The frequent weight all-gathers stay on NVLink where they're cheap, and the slow inter-server link sees exactly one overlapped all-reduce per step — exactly like DDP. You give up memory saving (<code>16P/8</code> rather than <code>16P/32</code>) and often gain a lot of throughput. In FSDP2 it's just a 2D <code>DeviceMesh</code>. <b>Use it when FSDP's memory scaling is already sufficient but your network is the bottleneck</b> — it's a network decision, not a memory one.</details>

11. **You move a working single-GPU run to 8 GPUs with DDP and the loss curve gets noticeably worse. Most likely cause?**
    <details><summary>answer</summary>The <b>effective batch multiplied by 8</b> (<code>per_device × grad_accum × num_gpus</code>), so you're taking fewer, less-noisy updates per epoch — a different optimization problem. Rescale the LR (<code>√N</code> for Adam-family; the linear rule is the SGD-era result) and add <b>warmup</b>. Then verify the sampler is actually giving each rank different data — a missing <code>DistributedSampler</code> produces the same symptom for a different reason.</details>

12. **Sketch a parallelism layout for a 70B full fine-tune on 4 nodes of 8×A100-80GB — and then argue against doing it.**
    <details><summary>answer</summary>State is <code>70e9 × 16 B ≈ 1.12 TB</code>, so ≥16 GPUs for state alone and realistically 32 with activations. Layout: <b>TP=8 within each node</b> (the chatty dimension rides NVLink), <b>PP=2</b> across node pairs (only ~6 MB activations cross the slow link), <b>DP=2</b> across the rest — 8×2×2 = 32, with ZeRO-1 on the DP dimension because it's free. <b>The argument against:</b> <a href="Fine-Tuning%20LLMs.md#6-qlora--quantization-on-top">QLoRA</a> on a 70B fits on ~2 A100s and matches full fine-tuning within noise for most format/domain adaptation. Unless you have a genuine full-model reason, the 32-GPU plan is excellent engineering aimed at the wrong question.</details>
