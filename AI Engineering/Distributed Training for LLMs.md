# Distributed Training for LLMs — what to split across GPUs, and which problem each split actually solves

> **TL;DR.** When one GPU isn't enough there are **three different problems**, and the mistake is treating them as one. *"Training is too slow"* → **data parallelism**: replicate the model on every GPU, give each a different mini-batch, **all-reduce the gradients** so every replica applies an identical update. *"The model doesn't fit"* → **model/pipeline parallelism**: put different **layers** on different GPUs, then split the batch into **micro-batches** so the GPUs aren't idle waiting in a chain (naive layer-splitting leaves ~75% of your GPU-time in the "bubble"). *"A single layer doesn't fit"* → **tensor parallelism**: split one weight matrix across GPUs and all-reduce *inside* every layer — enormous communication, so keep it inside one NVLink-connected node. *"I'm data-parallel but the optimizer state is eating my VRAM"* → **ZeRO / FSDP**: same math as data parallelism, but each GPU stores only `1/N` of the optimizer states, gradients, and eventually parameters. Real 70B+ runs combine all of them (**3D parallelism**). 🎯 The line that wins the question: *"Data parallelism scales **throughput**; model, pipeline and tensor parallelism scale **capacity** — you only reach for the second kind when the model genuinely doesn't fit, because every one of them buys memory with communication."* `(certain)`

**Where it fits:** The back half of the *Advanced Fine-Tuning* lecture, and the direct sequel to [Fine-Tuning LLMs](Fine-Tuning%20LLMs.md) — which ends at exactly the wall this note starts from: full fine-tuning a 7B needs ≈168 GB, a 70B needs **>1.1 TB**, and no single GPU exists at that size. [PEFT/LoRA](Fine-Tuning%20LLMs.md#5-peft-and-lora--the-mechanism) is how you *avoid* this problem; distributed training is how you *solve* it when avoiding isn't an option (pre-training, full fine-tuning, or a model too big to load at all). Its sequel is [Model Quantization](Model%20Quantization.md) — the **third** answer to the same wall: instead of splitting the bytes across GPUs, make every byte smaller.
**Prereqs:** [Fine-Tuning LLMs](Fine-Tuning%20LLMs.md) (the memory arithmetic, gradient accumulation), [Weight Initialization & Optimizers](../Machine%20Learning/Neural%20Networks/Weight%20Initialization%20&%20Optimizers.md) (what Adam stores), [RNN · LSTM · Transformers](../Machine%20Learning/NLP/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) (the layer stack you're about to cut up).

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../_STUDY%20LOOP.md))*

---

> ### ⚡ Fast Pass — 5 minutes
> **Model:** Four ways to cut up a training job, each fixing a different bottleneck. **Data** → split the *batch*. **Pipeline/model** → split the *layers*. **Tensor** → split *inside one layer*. **ZeRO/FSDP** → split the *optimizer state*.
> **Core:**
> - DDP: same weights everywhere, different batches, `ḡ = (1/N)Σgᵢ` via **all-reduce** → replicas never diverge.
> - `effective batch = per-device batch × grad-accum × num_gpus`.
> - GPipe bubble fraction = `(G−1)/(M+G−1)` — `G` stages, `M` micro-batches. G=4, M=4 → 43% idle; raise `M` to shrink it.
> - Ring all-reduce moves `2·(N−1)/N × model_bytes` per step — **independent of `N`** in per-GPU terms, which is why DDP scales.
> - ZeRO-1/2/3 shards optimizer states / +gradients / +parameters. FSDP ≈ ZeRO-3.
> **Traps:** ① using model parallelism when the model *does* fit (you just made it slower); ② tensor parallelism across nodes without NVLink/InfiniBand; ③ forgetting that DP multiplies your **effective batch**, so the LR needs rescaling; ④ quoting "N GPUs = N× faster" — communication and stragglers mean it never is.
> 🎯 **Kill-shot:** *"Data parallelism scales throughput; the other three scale capacity. Every capacity split buys memory with communication — so you take the cheapest one that makes the model fit, and no more."*

---

## Table of Contents
1. [Intuition / Mental Model — three different problems](#1-intuition--mental-model--three-different-problems)
2. [The Formal Core — memory, communication, and scaling efficiency](#2-the-formal-core--memory-communication-and-scaling-efficiency)
3. [Data Parallelism and DDP](#3-data-parallelism-and-ddp)
4. [Model and Pipeline Parallelism — and the bubble](#4-model-and-pipeline-parallelism--and-the-bubble)
5. [Tensor Parallelism — splitting inside a layer](#5-tensor-parallelism--splitting-inside-a-layer)
6. [ZeRO and FSDP — sharding the state](#6-zero-and-fsdp--sharding-the-state)
7. [Worked Example — sizing a 70B run](#7-worked-example--sizing-a-70b-run)
8. [Code / Implementation](#8-code--implementation)
9. [When It Breaks](#9-when-it-breaks)
10. [Production & LLMOps Notes](#10-production--llmops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose](#12-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model — three different problems

The lecture's framing was **Map-Reduce, but for gradients** — and that analogy is exactly right for *one* of the four strategies. The useful mental model is to first ask **which problem you actually have**:

```
"training takes 3 weeks"          →  a THROUGHPUT problem  →  data parallelism
"CUDA OOM loading the model"      →  a CAPACITY problem    →  pipeline / tensor parallelism
"OOM, but only on optimizer init" →  a STATE problem       →  ZeRO / FSDP
"one layer alone won't fit"       →  a LAYER problem       →  tensor parallelism
```

![Four ways to split a model across GPUs — data parallelism replicates the full model on every GPU and gives each a different batch, model or pipeline parallelism gives each GPU a contiguous slice of layers, tensor parallelism splits a single weight matrix column-wise across GPUs, and ZeRO or FSDP shards optimizer states then gradients then parameters across stages one to three.](attachments/parallelism-four-kinds.png)

| Strategy | What it splits | Fixes | Communication cost | Cost you pay |
|---|---|---|---|---|
| **Data (DDP)** | the batch | speed | gradient all-reduce, once per step | full model replica on every GPU |
| **Model / pipeline** | contiguous layers | fit | activations at stage boundaries — small | idle GPUs (the bubble) |
| **Tensor** | one matrix, internally | fit (a single layer) | all-reduce **twice per transformer block** — huge | needs NVLink; falls apart across nodes |
| **ZeRO / FSDP** | optimizer state, grads, params | fit + speed | all-gather params on demand | more communication than plain DDP |

**The one-sentence discriminator, worth memorizing:** *model parallelism splits **between** layers; tensor parallelism splits **inside** a layer; data parallelism doesn't split the model at all.* `(certain)`

---

## 2. The Formal Core — memory, communication, and scaling efficiency

### 2.1 What each GPU has to hold

Carrying the accounting over from [Fine-Tuning LLMs §4](Fine-Tuning%20LLMs.md#4-the-memory-wall--why-full-fine-tuning-is-off-the-table) — for `P` parameters in mixed-precision Adam training:

```
state_bytes  ≈ 16·P      (4 fp32 master weights + 8 Adam m,v + 4 gradients)
activations  ≈ f(batch × seq_len × layers)     — the part you control at runtime
```

*(The total is `16·P` either way, but the ZeRO paper splits it as 2P fp16 params + 2P fp16 grads + 12P fp32 states — which is why the per-stage numbers below apportion it differently.)*

Per-GPU memory under each strategy, with `N` GPUs:

| Strategy | Per-GPU state | Per-GPU activations |
|---|---|---|
| DDP | `16·P` — **no saving at all** | `batch_per_gpu` worth |
| ZeRO-1 | `4·P + 12·P/N` | same |
| ZeRO-2 | `2·P + 14·P/N` | same |
| ZeRO-3 / FSDP | `16·P/N` | same |
| Pipeline (G stages) | `16·P/G` | only its own layers' |
| Tensor (T ways) | `16·P/T` | sharded within the layer |

🚩 **The line people miss: plain data parallelism gives you zero memory relief.** Every GPU holds a *complete* replica — weights, gradients, and both Adam states. If a 7B full fine-tune doesn't fit on one A100, it doesn't fit on eight of them under DDP either. **That is precisely the gap ZeRO was invented to close.** `(certain)`

### 2.2 Communication — why DDP scales and tensor parallelism doesn't

**Ring all-reduce** (the standard algorithm, and what NCCL uses) moves:

```
bytes_per_gpu = 2 · (N−1)/N · model_bytes   ≈  2 · model_bytes  for large N
```

The remarkable property: **per-GPU traffic is essentially constant in `N`**. Doubling your GPU count doesn't double each GPU's communication — which is exactly why data parallelism scales to thousands of devices. It happens **once per optimizer step**, and modern DDP **overlaps** it with the backward pass (gradients are bucketed and all-reduced as soon as each bucket is ready, rather than waiting for the full backward).

Tensor parallelism is the opposite: **two all-reduces per transformer block**, on the activation tensors, *inside* the forward and backward pass, with nothing to overlap them with. At 80 layers that's 160+ synchronizations per step. Over NVLink (~600–900 GB/s) it's viable; over Ethernet it is catastrophic.

🎯 *"Data parallelism communicates once per step and hides it behind the backward pass; tensor parallelism communicates twice per layer and can't hide it. That's the whole reason TP stays inside a node."* `(certain)`

### 2.3 The NCCL collectives — the five primitives everything is built from

Every parallelism strategy in this note is assembled from a handful of **collective operations**, implemented by **NCCL** (NVIDIA Collective Communications Library) on GPU. Knowing the five by name — and knowing which one each strategy calls — is the fastest way to reason about communication cost, and it's a common interview probe.

```
                 GPU0   GPU1   GPU2   GPU3
  start          [1]    [2]    [3]    [4]

① Reduce         [10]    ·      ·      ·      combine → ONE destination rank
② All-Reduce     [10]   [10]   [10]   [10]    combine → EVERY rank gets the result
③ Broadcast      [w]    [w]    [w]    [w]     one rank's buffer copied to all
④ All-Gather     [1234][1234][1234][1234]     concatenate → every rank holds all pieces
⑤ Reduce-Scatter [10ₐ]  [10_b] [10_c] [10_d]  combine, but each rank keeps only ITS slice
```

| Collective | What it does | Who uses it |
|---|---|---|
| **Reduce** | sum/average across ranks, result on **one** rank | old parameter-server designs; debugging |
| **All-Reduce** | sum/average, result on **all** ranks | **DDP's gradient sync** — the workhorse |
| **Broadcast** | copy one rank's tensor to everyone | initial weight sync at startup, so all replicas start identical |
| **All-Gather** | each rank contributes a shard, everyone ends up with the whole | **FSDP/ZeRO-3** reconstructing a layer's parameters before its forward |
| **Reduce-Scatter** | reduce, then each rank keeps only the slice it owns | **FSDP/ZeRO** gradient sync — you only need the gradient for *your* shard |

**The identity worth memorizing:**

```
all-reduce  =  reduce-scatter  +  all-gather
```

This is not trivia — it's *why ring all-reduce has the cost it does* (it's literally implemented as those two phases, which is where the `2·(N−1)/N` factor comes from), and it explains the **FSDP communication premium**: DDP does one all-reduce per step, while ZeRO-3 does a reduce-scatter on gradients **plus** all-gathers of parameters on every layer in both forward and backward — roughly **1.5× DDP's volume**, which is the price of the memory you saved. `(certain)`

> 🚩 **The deadlock rule that follows from this:** collectives are **synchronizing** — every rank must call every collective, in the same order. A single `if rank == 0: dist.all_reduce(x)` hangs the entire job until the NCCL timeout fires, which is the single most common distributed-training bug (§9).

### 2.4 Scaling efficiency

```
speedup(N)  =  T(1) / T(N)          efficiency  =  speedup(N) / N
```

You will never get `efficiency = 1`. It's eaten by communication, by **stragglers** (the step ends when the *slowest* GPU finishes), and by the pipeline bubble. Good large-scale runs report 70–90% scaling efficiency; the honest headline metric is **MFU (Model FLOPs Utilization)** — achieved FLOPs ÷ hardware peak — where well-tuned LLM training lands around 40–55%. Anyone quoting linear speedup is quoting a benchmark, not a training run. `(likely)`

---

## 3. Data Parallelism and DDP

The most common strategy by a wide margin, and the one to reach for first.

![Distributed Data Parallel — four GPUs each hold the same weights and process a different mini-batch of eight samples, producing local gradients that are averaged by an all-reduce so every GPU applies an identical update, giving an effective batch of thirty-two.](attachments/ddp-gradient-all-reduce.png)

**The loop, in four steps:**

1. **Replicate** — every GPU starts with identical weights `W`.
2. **Scatter** — the global batch is split; GPU `i` gets mini-batch `i`.
3. **Compute** — each GPU runs forward + backward independently, producing a local gradient `gᵢ`.
4. **All-reduce** — gradients are averaged, `ḡ = (1/N)·Σᵢ gᵢ`, and **every GPU applies the same `ḡ`**.

**Why the replicas never drift:** they start identical and apply an *identical averaged gradient* every step. There is no sync-the-weights step — the sync is the gradient average. Break that invariant (a GPU that skips the all-reduce, a non-deterministic loss reduction) and the replicas silently diverge, which is a genuinely nasty bug because loss still goes down. `(certain)`

**`DataParallel` vs `DistributedDataParallel` — know the difference:**

| | `DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|---|---|---|
| Processes | **one** process, multi-threaded | **one process per GPU** |
| Bottleneck | Python GIL + rank-0 gathers all outputs | none of that |
| Gradient sync | gather to GPU 0, then scatter | ring all-reduce, overlapped with backward |
| Verdict | legacy; **don't use it** | the standard |

The classic symptom of `DataParallel` is **GPU 0 at 90% memory while the others sit at 40%** — because rank 0 gathers every replica's outputs to compute the loss.

**Effective batch, and the learning-rate consequence:**

```
effective_batch = per_device_batch × grad_accum_steps × num_gpus
                =        8         ×        1         ×    4     =  32
```

Going from 1 GPU to 8 **multiplies your effective batch by 8**, which changes the optimization problem — fewer, less-noisy updates per epoch. The standard corrections are the **linear scaling rule** (multiply LR by `N`) with a **warmup** (a large LR straight into an untrained state diverges), or `√N` scaling for Adam-family optimizers. Forgetting this is why "I added GPUs and my loss curve got worse" happens. `(certain)`

> **Gradient accumulation is data parallelism across *time* instead of *devices*** — same effective batch, one GPU, more wall-clock. See [Fine-Tuning LLMs §11](Fine-Tuning%20LLMs.md#11-production--llmops-notes). The two multiply, which is how you hit a target batch size on whatever hardware you have.

---

## 4. Model and Pipeline Parallelism — and the bubble

When the model genuinely doesn't fit, cut it **between layers**. With 12 layers and 4 GPUs:

```
GPU0: layers 1–3  ──▶  GPU1: layers 4–6  ──▶  GPU2: layers 7–9  ──▶  GPU3: layers 10–12
      activations flow forward along the chain; gradients flow back along it
```

Each GPU stores only its own slice's weights, gradients and optimizer state — so an 8B model that needs ~128 GB of state fits across 4×16 GB cards *for the weights*. Communication is tiny: only the **activation tensor at each stage boundary**, not the weights.

**And it's catastrophically inefficient if you stop there.** GPU1 cannot start until GPU0 finishes; GPU3 can't start until three stages have run. At any instant, **one GPU is working and `G−1` are idle**. This is the **bubble** — the lecture's "layer + micro-batches" note is the fix.

![Two Gantt charts comparing naive model parallelism with pipeline parallelism — naive processes one batch and leaves seventy-five percent of GPU-time idle in the bubble, while splitting the same batch into four micro-batches drops idle time to forty-three percent and finishes in 3.5 versus 8 time units.](attachments/pipeline-bubble-micro-batches.png)

**Pipeline parallelism (GPipe)** splits the batch into `M` **micro-batches** and streams them through the stages, so GPU0 starts micro-batch 2 while GPU1 works on micro-batch 1:

```
bubble_fraction = (G − 1) / (M + G − 1)
```

| `G` stages | `M` micro-batches | Bubble |
|---|---|---|
| 4 | 1 (naive) | 75% |
| 4 | 4 | 43% |
| 4 | 16 | 16% |
| 4 | 32 | 8.6% |

**So the design rule is `M ≫ G`** — a common heuristic is `M ≥ 4G`. The catch: more micro-batches means more concurrently-live activation sets to store, so the bubble trades directly against memory. **1F1B scheduling** (interleave one-forward-one-backward, as in PipeDream/Megatron) keeps the same bubble fraction while holding far fewer activations live — it's what production stacks actually use.

> ⚠️ **Pipeline parallelism and `batch_norm` don't mix**, and more relevantly for LLMs: any operation needing global batch statistics breaks when the batch is streamed as micro-batches. Transformers use [LayerNorm/RMSNorm](../Machine%20Learning/Neural%20Networks/Batch%20Normalization%20&%20Dropout.md) — which normalize *per-token, per-feature* — so this is a non-issue for LLMs, but it's the reason the technique reads differently in CNN literature. `(certain)`

---

## 5. Tensor Parallelism — splitting inside a layer

Pipeline parallelism assumes **one layer fits on one GPU**. At frontier scale that stops being true — a single `4096×4096` projection with its optimizer state, or an MLP with a 4× expansion, can exceed one card. Tensor parallelism (Megatron-LM style) splits **the matrix itself**.

For `Y = X·W` with `W` split **column-wise** into `[W₁ | W₂]`:

```
Y₁ = X·W₁  (GPU0)      Y₂ = X·W₂  (GPU1)      Y = [Y₁ | Y₂]     ← concatenate
```

Every GPU needs the *full* `X` but only its slice of `W`, and the outputs concatenate. Split **row-wise** instead and each GPU needs a slice of `X` but produces a full-shape *partial* `Y`, which must be **summed** via all-reduce.

**The trick Megatron uses** is to pair them so only one all-reduce is needed per block:
- **Attention:** shard `Wq/Wk/Wv` column-wise (each GPU owns whole attention *heads* — clean, since heads are independent), then the output projection `Wo` row-wise → one all-reduce at the end.
- **MLP:** first linear column-wise, second linear row-wise → one all-reduce at the end.

That's **two all-reduces per transformer block** (one per sub-layer), on activations, in both forward and backward.

> 🚩 **The practical rule: tensor parallelism stays inside a single node.** Those all-reduces sit on the critical path and cannot be overlapped. With NVLink between 8 GPUs in a box it works well; stretched across nodes on Ethernet, communication dominates and you go *slower* than a smaller configuration. Standard practice is **TP within a node, PP or DP across nodes.** `(certain)`

---

## 6. ZeRO and FSDP — sharding the state

The insight behind **ZeRO** (Zero Redundancy Optimizer, DeepSpeed) is embarrassingly simple in hindsight: under DDP, all `N` GPUs store **identical copies** of the optimizer states, gradients and parameters. That's `N`-fold redundancy for data that is bit-identical everywhere. So shard it, and gather each piece only when needed.

| Stage | Shards | Per-GPU state | Extra communication |
|---|---|---|---|
| **ZeRO-1** | optimizer states | `4P + 12P/N` | ~none over DDP |
| **ZeRO-2** | + gradients | `2P + 14P/N` | ~none over DDP |
| **ZeRO-3** | + parameters | `16P/N` | ~1.5× DDP volume |
| **+ offload** | to CPU/NVMe | lower still | much slower |

**ZeRO-3 / FSDP mechanics:** before a layer's forward pass, **all-gather** that layer's full parameters from the shards; compute; **immediately free** the non-owned shards; repeat for the backward. So at any instant a GPU holds full parameters for only *one layer*, plus its `1/N` shard of everything else.

Stated as the four-beat cycle that repeats for every unit, in both the forward and the backward pass:

```
  all-gather  →  compute  →  free  →  reduce-scatter
  (rebuild the    (fwd or    (drop the      (each rank keeps only
   full params)    bwd)       non-owned      the gradient slice
                              shards)        it owns)
```

**The knob this exposes is the FSDP *unit*** — the granularity at which you wrap the model (`auto_wrap_policy`, usually one transformer block per unit). It sets the memory/communication trade directly: **more units = smaller all-gathers = less peak memory, but more separate collectives**; wrapping the whole model as one unit degenerates to DDP's memory. One transformer block per unit is the sane default, and it's why FSDP can overlap the next unit's all-gather with the current unit's compute.

**FSDP (Fully Sharded Data Parallel)** is PyTorch's native implementation of essentially the ZeRO-3 algorithm. In practice: **FSDP** if you're in pure PyTorch/HF, **DeepSpeed** if you want the mature stage-by-stage config, CPU/NVMe offload, and its pipeline integration.

**The one-line contrast to keep:**

```
DDP :  same model everywhere  +  different data
FSDP:  sharded model (params + grads + optimizer states)  +  different data
       +  temporary reconstruction of one unit at a time
```

> 🎯 *"FSDP buys memory with communication, and the exchange rate is roughly 1.5× DDP's traffic — so it's the right call exactly when you're memory-bound and have the interconnect to pay for it (NVLink good, PCIe painful)."* `(certain)`

> **Why this is the modern default.** ZeRO-3 is *data* parallelism — the programming model is unchanged, no model surgery, no rewriting your layers — but with per-GPU memory falling as `1/N`. You get capacity scaling with the simplicity of DP. Reach for pipeline/tensor parallelism only when ZeRO-3 still isn't enough. `(certain)`

**3D parallelism** is the frontier combination: **TP inside a node** (8-way over NVLink) × **PP across a few nodes** × **DP across the rest**, usually with ZeRO-1 layered on the data-parallel dimension. A 70B+ pre-training run is essentially always some version of this.

---

## 7. Worked Example — sizing a 70B run

**Task:** full fine-tune a 70B model. Hardware: nodes of 8×A100-80GB (NVLink inside, InfiniBand between).

**Step 1 — the raw requirement.**
```
state    = 70e9 × 16 B  =  1,120 GB   ≈ 1.12 TB     (fp32 master + Adam m,v + grads)
+ activations (batch × seq)            ~100–300 GB depending on batch/seq/checkpointing
```

**Step 2 — rule out the cheap options.**
- One GPU: 80 GB. **No** — off by 14×.
- DDP on 8 GPUs: still `1.12 TB` **per GPU**. **No** — DDP shards nothing.
- ZeRO-3 on 8 GPUs: `1120/8 = 140 GB` per GPU. **Still no.**
- ZeRO-3 on 16 GPUs (2 nodes): `70 GB` per GPU + activations. **Borderline** — tight, but activation checkpointing makes it plausible.
- ZeRO-3 on 32 GPUs (4 nodes): `35 GB` state per GPU, comfortable room for activations. **Yes.**

**Step 3 — the realistic production layout.** Pure ZeRO-3 across 32 GPUs works but all-gathers parameters across the *slow* inter-node link on every layer. Better:

```
TP = 8   within each node   (NVLink — the chatty dimension stays local)
PP = 2   across node pairs  (only activations cross nodes)
DP = 2   across the rest    (one all-reduce per step, overlapped)
                                    8 × 2 × 2 = 32 GPUs
```

**Step 4 — the honest alternative.** Before any of this: **do you actually need a full fine-tune?** [QLoRA](Fine-Tuning%20LLMs.md#6-qlora--quantization-on-top) on a 70B fits on **two** A100s and, for most format/domain adaptation tasks, lands within noise of full fine-tuning. The 32-GPU plan is correct engineering for the wrong question unless you have a genuine full-model reason. 🎯 *That's the answer that impresses an interviewer:* size the cluster correctly, then explain why you'd try not to need it.

---

## 8. Code / Implementation

**The single most useful thing to know: you rarely write this yourself.** You configure it and launch with a runner.

### 8.1 DDP — what `accelerate`/`torchrun` does for you

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

### 8.2 FSDP / DeepSpeed via HF `Trainer` — the realistic path

Because `Trainer` and `SFTTrainer` are `accelerate`-backed, going multi-GPU is a **config change, not a code change**. Same training script:

```bash
accelerate config          # interactive: pick multi-GPU, FSDP or DeepSpeed, stage, offload
accelerate launch train.py
```

```yaml
# accelerate FSDP config — the ZeRO-3-equivalent knobs that matter
distributed_type: FSDP
mixed_precision: bf16                              # prefer bf16 over fp16: no loss-scaling drama
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD               # = ZeRO-3. SHARD_GRAD_OP = ZeRO-2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer   # ← wrap at the DECODER BLOCK.
                                                   # Wrap too coarsely and you gather the whole
                                                   # model at once, defeating the point.
  fsdp_offload_params: false                       # true → CPU offload; much slower, last resort
  fsdp_state_dict_type: SHARDED_STATE_DICT         # full state dict OOMs rank 0 at 70B
```

```json
// DeepSpeed equivalent — pass with `--deepspeed ds_config.json`
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

> ⚠️ **The checkpoint gotcha that bites everyone at scale:** with sharded state dicts, saving "the model" writes `N` shard files, and naively calling `.save_pretrained()` tries to gather 70B parameters onto **rank 0**, which OOMs. Use `SHARDED_STATE_DICT` during training and consolidate once at the end (`stage3_gather_16bit_weights_on_model_save`, or an offline merge script). `(certain)`

---

## 9. When It Breaks

| Failure | Cause | Fix |
|---|---|---|
| **Added GPUs, no speedup** | communication-bound, or micro-batch too small to saturate | raise per-device batch; check interconnect; profile the overlap |
| **Loss worse after scaling out** | effective batch grew `N×`, LR not rescaled | linear (or `√N`) LR scaling + warmup |
| **Every GPU computes the same gradient** | missing `DistributedSampler` | add it; call `set_epoch()` each epoch |
| **GPU 0 at 90%, others at 40%** | you're using `DataParallel`, not `DDP` | switch to DDP |
| **Hang at start / NCCL timeout** | rank mismatch, firewall, or a collective called on some ranks only | every rank must hit **every** collective — a conditional `if rank == 0: all_reduce()` deadlocks |
| **OOM only when saving** | gathering the full state dict on rank 0 | sharded state dict; consolidate offline |
| **Throughput drops off a cliff at N nodes** | tensor parallelism crossed a node boundary | keep TP ≤ GPUs-per-node |
| **One slow GPU halves throughput** | straggler — thermal throttling, a bad card, uneven shard sizes | the step ends when the slowest rank ends; monitor per-rank step time |
| **Non-deterministic results across runs** | all-reduce sums in nondeterministic order | expected; pin seeds and accept fp non-associativity, or use deterministic kernels (slower) |
| **Pipeline barely faster than one GPU** | too few micro-batches — the bubble dominates | `M ≥ 4G`; use 1F1B scheduling |

**The debugging order that saves time:** confirm it's *actually* distributed (per-rank logs, distinct data), then confirm it's *correct* (loss curve matches single-GPU at matched effective batch), *then* optimize throughput. Chasing MFU on a run that's silently training on identical data four times is a bad afternoon.

---

## 10. Production & LLMOps Notes

**Checkpointing is not optional.** Multi-day runs on many GPUs *will* be interrupted — a preempted spot instance, a failed NIC, an OOM at an unlucky sequence length. Checkpoint frequently enough that you lose at most ~30 minutes, and **checkpoint the full training state** (model, optimizer, LR-scheduler, dataloader position, RNG state) — a checkpoint with only weights silently restarts your LR schedule and re-serves data you already trained on.

**Elastic and fault-tolerant training** (`torchrun --max-restarts`, elastic agents) lets a run survive losing a node. Worth configuring before you need it, not after.

**Cost.** Multi-GPU is billed by the *node-hour*, so a run at 40% MFU costs 2× one at 80% for the same result. The cheapest optimizations are usually: bigger micro-batch (fewer, fatter kernels), `bf16`, FlashAttention, gradient checkpointing tuned rather than blanket-on, and fused optimizers. Spot/preemptible instances cut cost 60–90% and are viable *only* with solid checkpointing.

**Monitoring, per rank not per job:** step time, MFU, GPU utilization, memory headroom, and the communication/compute ratio. A job-level average hides the straggler that's costing you 30%.

**Reproducibility.** Distributed runs are harder to reproduce: floating-point addition isn't associative, so all-reduce ordering changes results slightly run-to-run. Pin seeds and library versions, log the world size and parallelism config alongside the checkpoint, and treat exact bitwise reproduction as a non-goal unless you've deliberately paid for it.

**Before you build any of this:** the highest-leverage production decision is usually **not to**. [LoRA/QLoRA](Fine-Tuning%20LLMs.md#5-peft-and-lora--the-mechanism) on one or two GPUs solves most real adaptation needs. Distributed training is for pre-training, genuine full fine-tunes, and models too large to load — a specialist tool, not the default.

---

## 11. Interview Lens

**What the question is really testing:** whether you can match a *strategy* to a *bottleneck*, instead of listing techniques. The failure mode is reciting "data, model, tensor, pipeline" without saying which problem each one solves.

| Question | Crisp answer |
|---|---|
| "Data vs model parallelism?" | 🎯 *"Data parallelism splits the **batch** and fixes **speed**; model parallelism splits the **layers** and fixes **fit**. Different problems — data parallelism gives you no memory relief at all."* |
| "Model vs tensor parallelism?" | Model/pipeline splits **between** layers; tensor splits **inside** one layer. Tensor is for when a single layer won't fit, and costs two all-reduces per block. |
| "I have 8 GPUs and a model that OOMs on one. Plan?" | ZeRO-3/FSDP first — same programming model, per-GPU memory `~1/N`. Only if that's insufficient add TP within the node, then PP. |
| "How do DDP replicas stay in sync?" | They start identical and apply the **same averaged gradient** every step. The all-reduce *is* the sync. |
| "What's the pipeline bubble?" | Idle time while stages wait in a chain: `(G−1)/(M+G−1)`. Fix with many micro-batches (`M ≥ 4G`) and 1F1B scheduling. |
| "Why keep tensor parallelism in one node?" | Its all-reduces are on the critical path and can't be overlapped — they need NVLink bandwidth. Across Ethernet, communication dominates. |
| "Does 4 GPUs mean 4× faster?" | No — communication, stragglers, and bubbles. Good runs hit 70–90% scaling efficiency; report **MFU** (~40–55% is healthy). |
| "What changes when you scale out?" | The **effective batch** multiplies by `N`, so the LR needs rescaling (linear or `√N`) with warmup. |
| "ZeRO stages?" | 1 = optimizer states, 2 = + gradients, 3 = + parameters. FSDP ≈ ZeRO-3. |

---

## 12. Alternatives & How to Choose

| Option | Use when | Don't when |
|---|---|---|
| **Single GPU + gradient accumulation** | the model fits; you just want a bigger effective batch | you're wall-clock bound |
| **[LoRA / QLoRA](Fine-Tuning%20LLMs.md)** | adapting a model to a task/domain/format — *most real work* | pre-training, or a genuine full-model update |
| **DDP** | model fits, training is too slow | the model doesn't fit — DDP won't help |
| **ZeRO-2 / FSDP `SHARD_GRAD_OP`** | slightly over memory | you need max capacity |
| **ZeRO-3 / FSDP `FULL_SHARD`** | **the default for "doesn't fit on one GPU"** | a single *layer* doesn't fit |
| **Tensor parallelism** | a single layer won't fit; NVLink available | across nodes on commodity networking |
| **Pipeline parallelism** | very deep model, limited interconnect between nodes | few micro-batches available (bubble eats it) |
| **3D parallelism** | 70B+ pre-training on a real cluster | anything smaller — the complexity isn't free |
| **[Quantization](Model%20Quantization.md)** | shrinking the bytes rather than splitting them — the only lever that also helps at **inference** | you need full-precision weight updates |
| **Rent a bigger GPU** | honestly, often | you need more than one node's worth |

**The decision in four questions:**
1. Do you need weight updates at all, or will [RAG](RAG.md)/prompting do? → don't train.
2. Do you need *full*-model updates, or will [LoRA/QLoRA](Fine-Tuning%20LLMs.md) do? → usually LoRA.
3. Does it fit on one GPU? → DDP for speed only.
4. It doesn't fit → ZeRO-3/FSDP → then TP within a node → then PP across nodes.

**The lecture's own version of that tree, which folds in the quantization branch:**

```
Model + training state fits on one GPU?
├─ YES ──→ DDP
└─ NO  ──→ Can you afford FULL training memory, spread across GPUs?
           ├─ YES ──→ FSDP / ZeRO-3
           └─ NO  ──→ LoRA  ──→  still doesn't fit?  ──→  QLoRA (4-bit NF4 base)
```

That last branch is where this note hands off to [Model Quantization](Model%20Quantization.md).

---

## 🧠 Self-Test

1. **You have a 7B model that OOMs on a single 80 GB A100 during full fine-tuning. Someone suggests "just use 8 GPUs with DDP." What's wrong with that, and what would you do instead?**
   <details><summary>answer</summary>DDP <b>replicates</b> the full model, gradients and optimizer states on <i>every</i> GPU — it shards nothing. If ~168 GB doesn't fit on one card, it doesn't fit on eight either; DDP only splits the <i>batch</i>, so it fixes throughput, not capacity. Use <b>ZeRO-3 / FSDP</b>, which keeps the data-parallel programming model but shards optimizer states, gradients and parameters so per-GPU state falls to ~<code>16P/N</code>. (Better still: ask whether <a href="Fine-Tuning%20LLMs.md">QLoRA</a> would do the job on one GPU.)</details>

2. **Distinguish model parallelism from tensor parallelism in one sentence each, and say when you'd need the second.**
   <details><summary>answer</summary><b>Model (pipeline) parallelism</b> splits the network <b>between</b> layers — GPU0 gets layers 1–3, GPU1 gets 4–6, and activations flow along the chain. <b>Tensor parallelism</b> splits <b>inside</b> a single layer — one weight matrix is cut column- or row-wise across GPUs and the partial results are concatenated or all-reduced. You need tensor parallelism when a <i>single layer</i> is too big for one GPU, which pipeline parallelism can't help with. It costs two all-reduces per transformer block, so it must stay within an NVLink node.</details>

3. **Give the pipeline bubble formula and compute it for 4 stages with 1, 4 and 16 micro-batches. What's the design rule, and what does it cost?**
   <details><summary>answer</summary><code>bubble = (G−1)/(M+G−1)</code>. For G=4: M=1 → 3/4 = <b>75%</b>; M=4 → 3/7 ≈ <b>43%</b>; M=16 → 3/19 ≈ <b>16%</b>. Rule: <code>M ≫ G</code>, commonly <code>M ≥ 4G</code>. Cost: more micro-batches in flight means more concurrently-live activation sets, so the bubble trades against memory — <b>1F1B</b> scheduling gets the same bubble fraction while holding far fewer activations live.</details>

4. **How do DDP replicas avoid drifting apart, and what's the classic bug that breaks it?**
   <details><summary>answer</summary>They begin identical and apply an <b>identical averaged gradient</b> (<code>ḡ = (1/N)Σgᵢ</code>, via all-reduce) every step — the all-reduce <i>is</i> the synchronization; there's no separate weight-sync. The classic bug is a <b>missing <code>DistributedSampler</code></b>: every GPU then reads the same data, so you compute the same gradient N times and gain nothing but heat. A subtler one is calling a collective on only some ranks (e.g. inside <code>if rank == 0</code>), which deadlocks — every rank must hit every collective.</details>

5. **Ring all-reduce moves about `2·(N−1)/N × model_bytes` per GPU per step. Why does that single fact explain the DDP-vs-tensor-parallelism split?**
   <details><summary>answer</summary>Per-GPU traffic is essentially <b>constant in N</b>, and it happens <b>once per step</b> — and modern DDP overlaps it with the backward pass by bucketing gradients. So data parallelism scales to thousands of GPUs. Tensor parallelism instead all-reduces <b>activations twice per transformer block</b>, inside the forward/backward, with nothing to hide the latency behind — 160+ synchronizations per step at 80 layers. That's viable over NVLink (~600–900 GB/s) and disastrous over Ethernet, which is why TP stays inside one node.</details>

6. **You move a working single-GPU run to 8 GPUs with DDP and the loss curve gets noticeably worse. Most likely cause?**
   <details><summary>answer</summary>The <b>effective batch size multiplied by 8</b> (<code>per_device × grad_accum × num_gpus</code>), so you're taking fewer, less-noisy updates per epoch — a different optimization problem. Rescale the learning rate (<b>linear scaling rule</b>, or <code>√N</code> for Adam-family) and add <b>warmup</b>, since a large LR applied immediately can diverge. Also verify the sampler is actually giving each rank different data.</details>

7. **What are the three ZeRO stages, and which one is FSDP?**
   <details><summary>answer</summary><b>Stage 1</b> shards optimizer states; <b>Stage 2</b> adds gradients; <b>Stage 3</b> adds parameters, so per-GPU state is ~<code>16P/N</code>. PyTorch <b>FSDP</b> with <code>FULL_SHARD</code> is essentially the ZeRO-3 algorithm (<code>SHARD_GRAD_OP</code> ≈ ZeRO-2). ZeRO-3 works by all-gathering a layer's parameters just before it's needed and freeing the non-owned shards immediately after — so a GPU only ever holds full parameters for one layer at a time.</details>

8. **Sketch a parallelism layout for a 70B full fine-tune on 4 nodes of 8×A100-80GB — and then argue against doing it.**
   <details><summary>answer</summary>State is <code>70e9 × 16 B ≈ 1.12 TB</code>, so you need ≥16 GPUs just for state and realistically 32 with activations. Layout: <b>TP=8 within each node</b> (the chatty dimension rides NVLink), <b>PP=2</b> across node pairs (only activations cross the slow link), <b>DP=2</b> across the rest — 8×2×2 = 32 GPUs, usually with ZeRO-1 on the DP dimension. <b>The argument against:</b> <a href="Fine-Tuning%20LLMs.md#6-qlora--quantization-on-top">QLoRA</a> on a 70B fits on ~2 A100s and matches full fine-tuning within noise for most format/domain adaptation. Unless you have a genuine full-model reason, the 32-GPU plan is excellent engineering aimed at the wrong question.</details>
