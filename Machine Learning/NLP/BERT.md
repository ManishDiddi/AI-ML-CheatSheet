# BERT — Bidirectional Encoder Representations from Transformers

> **TL;DR.** BERT is the **encoder** half of the Transformer, pretrained on huge *unlabeled* text with two **self-supervised** objectives — **MLM** (mask 15% of tokens, predict them using *both* left and right context) and **NSP** (does sentence B follow A?) — to produce deeply **contextual** token representations. Being bidirectional makes it a **natural-language-understanding** powerhouse (classification, **NER**, QA) but *not* a text generator (that's GPT, a left-to-right decoder). You use it by **transfer learning**: take the pretrained weights, bolt on a small task head (`[CLS]`→classify, per-token→NER, span→QA), and **fine-tune** end-to-end with a tiny learning rate. Inputs are **WordPiece** subwords wrapped in `[CLS] … [SEP]` with summed token + segment + position embeddings. Family: RoBERTa (drops NSP), DistilBERT (smaller/faster), BioBERT/ClinicalBERT (domain).

**Where it fits:** the model that made **transfer learning the default in NLP** — pretrain once on the internet, fine-tune cheaply per task. It's the contextual successor to static [Word Embeddings](Word%20Embeddings.md), the SOTA engine behind [NER](NER.md), sentiment ([Sentiment Analysis & Topic Modeling](Sentiment%20Analysis%20&%20Topic%20Modeling.md)), and the sentence-embedding models in [Embeddings](../../AI%20Engineering/Embeddings.md). Architecture/attention details live in [RNN · LSTM · Transformers](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md); its generative cousins in [LLM](../../AI%20Engineering/LLM.md).
**Prereqs:** [RNN · LSTM · Transformers](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) (self-attention, positional encoding, encoder blocks), [Word Embeddings](Word%20Embeddings.md) (static vs contextual), [Text Preprocessing](Text%20Preprocessing.md) (subword tokenization), transfer learning ([CV analogy](../Computer%20Vision/Transfer%20Learning.md)).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [Architecture & Input Representation](#2-architecture--input-representation)
3. [Pretraining — MLM & NSP](#3-pretraining--mlm--nsp)
4. [Bidirectionality — BERT vs GPT](#4-bidirectionality--bert-vs-gpt)
5. [WordPiece Tokenization](#5-wordpiece-tokenization)
6. [Using BERT — Fine-Tuning, Task Heads & a Worked Example](#6-using-bert--fine-tuning-task-heads--a-worked-example)
7. [When It Breaks](#7-when-it-breaks)
8. [The BERT Family](#8-the-bert-family)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

BERT is trained the way you'd teach language with a **fill-in-the-blank** exercise. Cover up words in billions of sentences and force a model to guess them from *everything around them* — do that at scale and it has to learn syntax, semantics, coreference, and world facts. The result is an encoder that turns each token into a **context-dependent** vector.

```
Static (word2vec):  "bank"  → ONE vector, always            (river? money? collapsed)
BERT (contextual):  "…power bank to charge…"  → vec_A       (device)
                    "…cash from the bank…"     → vec_B       (finance)   ← same word, different vectors
```

Two framings you must hold at once (the lecture stresses both):
- **BERT is an embedding model** — you can read out contextual vectors for words/sentences, like word2vec/GloVe/ELMo but context-aware.
- **BERT is a pretrained model** — exactly like `VGG`/`AlexNet` in vision: freeze the architecture + pretrained weights, add a task-specific output layer, and **fine-tune**. You almost always start from pretrained weights (random init would take days/weeks); random init is only for *pretraining a new domain model from scratch* (BioBERT for medicine, LegalBERT for law). 🎯

---

## 2. Architecture & Input Representation

BERT is a **stack of Transformer *encoder* blocks** (self-attention + feed-forward, [details](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)). No decoder, no causal mask — every token attends to every other token in both directions. Four canonical sizes:

| Variant | Encoder blocks (L) | Attention heads (A) | Hidden (H) | Cased? | Params |
|---|---|---|---|---|---|
| BERT-Base uncased | 12 | 12 | 768 | no | 110M |
| BERT-Base cased | 12 | 12 | 768 | yes | 110M |
| BERT-Large uncased | 24 | 16 | 1024 | no | 340M |
| BERT-Large cased | 24 | 16 | 1024 | yes | 340M |

**Input = three learned embeddings, summed per token:**
```
[CLS]  the   power  bank   [SEP]   near   the   river  bank   [SEP]   (input tokens)
 ─────────── token (WordPiece) embeddings ───────────
 + segment/token-type embeddings:  A A A A A   B B B B B      (which sentence: 0 vs 1)
 + position embeddings:            0 1 2 3 4    5 6 7 8 9      (LEARNED absolute positions)
```
- **`[CLS]`** — a special first token whose final hidden state is the **pooled sequence representation** used for classification tasks.
- **`[SEP]`** — separates/terminates segments (needed for sentence-pair tasks like NSP, QA, NLI).
- **Segment (token-type) embeddings** mark sentence A vs B; **position embeddings** are **learned** (BERT does *not* use the original Transformer's fixed sinusoidal encoding). Max sequence length is **512 tokens**.

Output: one **contextual vector per token**, plus the `[CLS]` vector as a whole-sequence summary.

---

## 3. Pretraining — MLM & NSP

BERT is pretrained on **Wikipedia (~2.5B words) + BookCorpus (~800M words)** with two objectives run **together** (final loss = MLM loss **+** NSP loss). It's **self-supervised** ("semi-supervised" in the lecture's words): no human labels — the labels are *generated from the raw text itself*.

**1. Masked Language Modeling (MLM).** Randomly pick **15%** of token positions and predict them from bidirectional context.
```
Actual : "Today morning, I went for a tooth removal to my dentist"
Input  : "Today morning, I went for a [MASK] removal to my dentist"   → predict: tooth
```
The detail the lecture glosses (**know this cold**): of the chosen 15%, **80% → `[MASK]`, 10% → a random token, 10% → left unchanged**. Why the 10/10 twist? `[MASK]` **never appears at fine-tuning time**, so always-masking would create a train/serve mismatch; mixing in random/unchanged tokens forces BERT to build a good representation for *every* token, not just masked ones. Loss is computed only on the selected 15%. `(certain)`

**2. Next Sentence Prediction (NSP).** Given sentences A and B, classify **IsNext?** — 50% of the time B truly follows A (positive), 50% B is a random sentence (negative). Combined length ≤ 512 tokens; the `[CLS]` vector feeds a binary classifier.
```
A: "Nadal won the 2022 Australian Open."   B: "He now has 21 Grand Slam titles."   → IsNext = Yes
A: "India won the match."                  B: "Obama served two terms."            → IsNext = No
```
*Why NSP?* MLM captures within-sentence context but not **inter-sentence** relationships, which QA and NLI need.

> ⚠️ **Correction to the lecture's framing:** later work (**RoBERTa**) found **NSP is nearly useless** — removing it and training on longer contiguous text *improved* results; **ALBERT** replaced it with **Sentence-Order Prediction (SOP)**. So modern encoders often drop NSP. MLM is the objective that carries BERT. `(certain)`

---

## 4. Bidirectionality — BERT vs GPT

This is *the* distinction interviewers probe:

```
BERT (encoder)   : "...went for a [MASK] removal..."   sees LEFT + RIGHT  → Masked LM (understanding)
GPT (decoder)    : "...went for a ___"                 sees LEFT only     → Autoregressive LM (generation)
```
- **BERT** conditions each prediction on **both directions**, which is why it's called a **Masked** Language Model and excels at **NLU** — classification, NER, QA, sentence similarity. It **can't generate** free text (no left-to-right factorization).
- **GPT** is a left-to-right **decoder** with a causal mask; it predicts the next token, so it **generates** but sees only the past.
- **T5/BART** use the full **encoder-decoder** for seq2seq (translation, summarization).

Rule of thumb: **encoder-only = understanding, decoder-only = generation, encoder-decoder = transformation.** Full comparison in [RNN · LSTM · Transformers §7](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) and [LLM](../../AI%20Engineering/LLM.md). 🎯

---

## 5. WordPiece Tokenization

BERT tokenizes with **WordPiece** — a subword scheme that splits rare/compound words into a **root + continuation subwords** marked with `##`:
```
paraglide    → ['para', '##gli', '##de']
paragliding  → ['para', '##gli', '##ding']
scubadiving  → ['scuba', '##di', '##ving']
```
This solves the two diseases of word-level tokenization at once: **huge vocabulary** and **out-of-vocabulary (OOV)** words — any word decomposes into known pieces, so there's no true OOV (same idea as fastText/BPE in [Word Embeddings](Word%20Embeddings.md)). The fixed vocab is ~30k WordPieces.

**Consequence for [NER](NER.md) / token tasks:** one word can become several subword tokens, so BIO labels must be **aligned** — label the **first** subword and mark the rest as "ignore" (`-100`), or the entity spans shift. This alignment step is where token-classification pipelines most often break.

---

## 6. Using BERT — Fine-Tuning, Task Heads & a Worked Example

**Two ways to use it:**
- **Feature extraction** — freeze BERT, take its (contextual) embeddings, feed a separate classifier. Cheap, but leaves accuracy on the table.
- **Fine-tuning** (the usual win) — add a thin task head and train **the whole thing** end-to-end with a **small learning rate** (`2e-5`–`5e-5`), **2–4 epochs**, a warmup schedule. Because pretraining did the heavy lifting, a little labeled data goes a long way.

**HuggingFace task heads** (each a different class):
| Task | Head | Uses |
|---|---|---|
| Sequence classification (sentiment, NLI) | `[CLS]` → Dense → softmax | one label per sequence |
| **Token classification (NER, POS)** | each token vector → Dense → softmax over BIO tags | one label per token |
| Question answering | two heads predict answer **start** & **end** positions | span extraction |

**Worked example — disease NER (the lecture's task, NCBI corpus).** Goal: tag disease mentions in medical abstracts. Since the *number* of diseases per article varies, it can't be fixed-label classification → it's **[NER](NER.md)** (token classification).
```python
from transformers import BertTokenizer, TFBertForTokenClassification
tok   = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)  # O, B-DISEASE, I-DISEASE

enc = tok(sentences, is_split_into_words=True, padding="max_length",
          truncation=True, max_length=128, return_tensors="tf")
# enc gives: input_ids, token_type_ids (segment A/B), attention_mask (1 real / 0 pad)
# Align BIO labels to WordPiece subwords: label first subword, set the rest to -100 (ignored in loss)
model.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss=..., metrics=[...])
model.fit(enc, labels, epochs=3)
```
At inference you decode per-token tag probabilities back into disease spans — exactly the BIO decoding from the [NER](NER.md) note, now powered by contextual BERT features instead of a BiLSTM-CRF.

---

## 7. When It Breaks

- **512-token limit.** Self-attention is `O(n²)`, so BERT caps at 512 tokens — long documents must be truncated or chunked. Long-context variants (**Longformer**, **BigBird**) use sparse attention to go further.
- **Not a generator.** Bidirectional MLM can't produce fluent left-to-right text; for generation use GPT/T5, not BERT.
- **Out-of-the-box sentence embeddings are *bad*.** Mean-pooling BERT token vectors (or using raw `[CLS]`) underperforms even GloVe for sentence **similarity/retrieval** — you need **SBERT** (Siamese fine-tuning). This is a top gotcha; see [Embeddings](../../AI%20Engineering/Embeddings.md). `(certain)`
- **Domain shift.** General BERT struggles on clinical/legal/scientific text → use **BioBERT/ClinicalBERT/SciBERT/LegalBERT** or continue-pretrain in-domain.
- **`[MASK]` pretrain–finetune gap.** `[MASK]` appears only in pretraining; the 80/10/10 trick mitigates but doesn't erase it (ELECTRA's replaced-token-detection avoids masking entirely).
- **Compute & latency.** 110M–340M params → GPU for training, non-trivial inference cost; distill/quantize for production (§9).
- **Subword–label misalignment** for token tasks (§5) silently shifts entities.
- **Catastrophic forgetting / instability.** Fine-tuning with too-high LR or too many epochs on small data destroys pretrained knowledge; keep LR small and epochs few.

---

## 8. The BERT Family

| Model | What changed | Why it matters |
|---|---|---|
| **RoBERTa** | more data, longer training, **dynamic masking**, **no NSP** | showed NSP is unnecessary; a stronger drop-in BERT |
| **ALBERT** | parameter sharing + factorized embeddings, **SOP** instead of NSP | far fewer params, similar accuracy |
| **DistilBERT** | knowledge **distillation** | ~40% smaller, ~60% faster, ~97% of BERT — the production default |
| **ELECTRA** | replaced-token detection (discriminator) instead of MLM | more sample-efficient pretraining, no `[MASK]` gap |
| **BioBERT / ClinicalBERT / SciBERT / FinBERT / LegalBERT** | domain-specific pretraining | large gains on specialized text (the pharma/medical case) |
| **mBERT / XLM-R** | multilingual pretraining | 100+ languages, cross-lingual transfer |
| **Longformer / BigBird** | sparse attention | documents ≫ 512 tokens |

All are used through the same fine-tuning + task-head recipe as §6.

---

## 9. Production & MLOps Notes

- **Fine-tune, don't pretrain.** Pretraining BERT costs thousands of GPU-hours; you almost always start from public weights and fine-tune (hours). Reserve from-scratch pretraining for a genuinely new domain/language with a large corpus.
- **Shrink for serving.** Latency/cost come down via **DistilBERT/TinyBERT**, **quantization** (INT8), **ONNX Runtime / TensorRT**, and batching. A distilled model is often the right production choice over BERT-Large.
- **Parameter-efficient fine-tuning (PEFT).** For many tasks or large models, **LoRA / adapters** fine-tune a tiny fraction of weights — cheaper, and you can hot-swap task adapters (see [RNN · LSTM · Transformers §9](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)).
- **Tokenizer is part of the model.** Always pair a fine-tuned model with its **exact** tokenizer/vocab; a mismatch corrupts inputs silently. Watch **truncation** (512) — long inputs lose their tails.
- **Evaluation & monitoring.** Benchmark on **GLUE/SuperGLUE** for general NLU; in prod, monitor task metrics ([precision/recall/F1](../Supervised%20ML/Classification%20Metrics.md), span-F1 for NER) and OOV/length distributions for drift.
- **Retrieval ≠ classification.** For semantic search/RAG you want **sentence embeddings (SBERT)** indexed in an ANN store, not vanilla BERT — see [Embeddings](../../AI%20Engineering/Embeddings.md).

---

## 10. Interview Lens

The question tests whether you can explain **why bidirectional pretraining works** and **how to adapt BERT to a task**.

🎯 **Kill-shots:**
- *"BERT is the Transformer **encoder**, pretrained self-supervised with **MLM** (predict 15% masked tokens from both sides) and **NSP**; being bidirectional makes it great at understanding but unable to generate."*
- *"You fine-tune it like a pretrained CV model: add a task head — `[CLS]` for classification, per-token for NER, start/end for QA — and train end-to-end with a small learning rate."*

**Likely follow-ups:**
- *Why is BERT bidirectional and GPT not?* BERT's MLM lets each token see left+right context; GPT's causal LM predicts the next token so it only sees the left — that's what makes GPT generative. `(certain)`
- *Explain the 15% masking — and the 80/10/10 split.* Mask 15% of tokens to predict; of those, 80% `[MASK]`, 10% random, 10% unchanged — to avoid a pretrain/finetune mismatch since `[MASK]` never appears at fine-tune time. `(certain)`
- *Is NSP important?* BERT used it for inter-sentence tasks, but **RoBERTa showed it's basically unnecessary**; ALBERT swapped in SOP. `(certain)`
- *What are the three input embeddings?* Token (WordPiece) + segment/token-type + **learned** position, summed; with `[CLS]`/`[SEP]`. `(certain)`
- *Why WordPiece?* Subwords kill OOV and shrink vocabulary; but they force BIO-label alignment for token tasks. `(certain)`
- *Why not use `[CLS]`/mean-pooling for sentence similarity?* It's poor out of the box — use SBERT. `(certain)`
- *BERT-Base dimensions?* 12 layers, 12 heads, hidden 768, 110M params. `(certain)`

---

## 11. Alternatives & How to Choose

| Need | Reach for |
|---|---|
| Understand/classify/tag/QA text | **BERT** (or RoBERTa/DistilBERT); domain-pretrained for specialized text |
| Generate text (chat, completion, summarization-by-LM) | **GPT-style decoder** / [LLM](../../AI%20Engineering/LLM.md) |
| Seq2seq (translation, summarization, text-to-text) | **T5 / BART** (encoder-decoder) |
| Sentence similarity / semantic search / RAG | **SBERT** sentence embeddings → [Embeddings](../../AI%20Engineering/Embeddings.md) |
| Tight latency / edge | **DistilBERT / TinyBERT**, quantized |
| Documents > 512 tokens | **Longformer / BigBird** |
| Specialized domain (medical/legal) | **BioBERT / ClinicalBERT / LegalBERT** |

**Decision rule:** default to **BERT (or a distilled/domain variant)** for any *understanding* task; switch to a **decoder/LLM** the moment the task is *generation*; use **SBERT** when the deliverable is *similarity/retrieval*, not a label. BERT is the encoder foundation; the generative side of the family lives in [LLM](../../AI%20Engineering/LLM.md).

---

## 🧠 Self-Test

1. What are BERT's two pretraining objectives, and what data was it trained on?
   <details><summary>answer</summary>**MLM** — mask 15% of tokens and predict them from bidirectional context; **NSP** — classify whether sentence B follows sentence A (50/50 positives/negatives). Self-supervised on **Wikipedia (~2.5B words) + BookCorpus (~800M words)**; final loss = MLM + NSP.</details>

2. Explain the 15% masking split and why it isn't 100% `[MASK]`.
   <details><summary>answer</summary>Of the 15% chosen tokens: 80% → `[MASK]`, 10% → a random token, 10% → unchanged. `[MASK]` never appears during fine-tuning, so always masking would cause a train/serve mismatch; the random/unchanged tokens force BERT to represent every token well, not just masked slots.</details>

3. Why is BERT bidirectional and unable to generate text, while GPT can generate?
   <details><summary>answer</summary>BERT's MLM conditions on left **and** right context (encoder, no causal mask), great for understanding but with no left-to-right factorization to generate from. GPT is a causal decoder predicting the next token from the left only — which is exactly what enables generation.</details>

4. What are the three summed input embeddings, and what is `[CLS]` for?
   <details><summary>answer</summary>**Token (WordPiece) + segment/token-type + learned position** embeddings, summed per token, wrapped in `[CLS] … [SEP]`. `[CLS]`'s final hidden state is the pooled sequence representation used as input to a classification head.</details>

5. How would you use BERT for disease NER, and what alignment issue arises?
   <details><summary>answer</summary>Token classification: add a per-token head over BIO tags (`O`, `B-DISEASE`, `I-DISEASE`) and fine-tune (`TFBertForTokenClassification`). Because WordPiece splits words into subwords, you must **align BIO labels to subwords** — label the first subword, set the rest to `-100` (ignored) — or entity spans shift.</details>

6. A colleague mean-pools BERT vectors for semantic search and gets poor results. Why, and what's the fix? Also: is NSP essential?
   <details><summary>answer</summary>Vanilla BERT (mean-pool or `[CLS]`) gives weak **sentence** embeddings — worse than GloVe for similarity — because it wasn't trained for that. Fix: **SBERT** (Siamese fine-tuning). And no — **RoBERTa showed NSP is essentially unnecessary**; MLM is what matters (ALBERT replaced NSP with SOP).</details>
