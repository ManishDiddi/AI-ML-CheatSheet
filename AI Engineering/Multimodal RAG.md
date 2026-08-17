# Multimodal RAG — Retrieval Over Images & Document Pages (CLIP, ColPali, VLMs)

> **TL;DR.** Multimodal RAG extends [RAG](RAG.md) to corpora where the meaning lives in **pixels, not text** — diagrams, charts, scanned pages, and image-heavy manuals. Two retriever families dominate: **CLIP** embeds text and images into **one shared vector space** so you can search across modalities (text→image, image→image, image→text) — cheap, great for a **POC**; **ColPali** skips OCR entirely, treats **each document page as an image**, and scores it with **late-interaction (MaxSim)** multi-vector matching — heavier, but the **production choice** for real documents because it keeps every diagram, table, and layout cue. The reader is a **VLM** (vision-language model) that answers from the *retrieved* text **and** images — and a RAG system **surfaces the real manual images verbatim; it never generates hypothetical ones**. The running case is **IKEA assembly manuals**: "how do I attach the legs?" → retrieve the exact page → VLM explains it and shows the real diagram.

**Where it fits:** The multimodal rung of [RAG](RAG.md). When your knowledge base is *visual* (product manuals, invoices, slide decks, medical scans, financial reports full of charts), text-only RAG throws away most of the signal. Builds directly on [RAG](RAG.md) (chunk → embed → index → retrieve → ground), [Embeddings](Embeddings.md) (bi-encoders, cosine, contrastive fine-tuning), and [LLM](LLM.md) (hallucination, context window).
**Prereqs:** [RAG](RAG.md) and [Embeddings](Embeddings.md) (dense vectors, cosine, bi- vs cross-encoder), plus the **contrastive / dual-encoder** idea from [Siamese Networks & Image Similarity](../Machine%20Learning/Computer%20Vision/Siamese%20Networks%20&%20Image%20Similarity.md) and the ViT/transformer backbone in [RNN · LSTM · Transformers](../Machine%20Learning/NLP/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md).

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../_STUDY%20LOOP.md))*

> ⚙️ *Format note: this adapts the vault's standard skeleton for a **pipeline with two competing retrievers** — "How It Works" fans out into §5 (CLIP), a **building-blocks primer** (§6), §7 (ColPali), and §8 (VLM generation), with the end-to-end run in §9. The **tabular** half of the old `[[Multimodal & Tabular RAG]]` placeholder wasn't taught here — it gets its own future note, `[[Tabular RAG]]`.*

---

## 🗺️ The whole system on one page

![Two-phase multimodal RAG pipeline: pre-production indexes the knowledge base with ColPali into an indexed store, and in-production a user query drives the retriever to fetch ranked top-k similar pages that become the VLM prompt, which the VLM turns into the answer](attachments/colpali-vlm-multimodal-rag-pipeline.png)

*Same two phases as text RAG. **Pre-production (offline):** embed the knowledge base once — with CLIP (embed text + images) or ColPali (render each page to an image, patch-embed it) — into a vector index. **In-production (online):** query → retriever scores it against the index → ranked top-k pages/images become the **VLM prompt** (retrieved text + the real page images) → the VLM generates the grounded answer. Swap "ColPali" for "CLIP" and the skeleton is identical; only the retriever changes.*

---

## Table of Contents
1. [Why Multimodal RAG Exists — When the Answer Is a Picture](#1-why-multimodal-rag-exists--when-the-answer-is-a-picture)
2. [OCR vs Vision-Based Extraction from PDFs](#2-ocr-vs-vision-based-extraction-from-pdfs)
3. [Intuition — One Shared Embedding Space](#3-intuition--one-shared-embedding-space)
4. [The Four Retrieval Directions (Cross-Modal Search)](#4-the-four-retrieval-directions-cross-modal-search)
5. [CLIP — The Dual-Encoder Contrastive Model](#5-clip--the-dual-encoder-contrastive-model)
6. [Building Blocks — The Five Pieces ColPali Is Made Of](#6-building-blocks--the-five-pieces-colpali-is-made-of)
7. [ColPali — Page-as-Image Retrieval with Late Interaction](#7-colpali--page-as-image-retrieval-with-late-interaction)
8. [Generation — VLMs That Show the Real Manual](#8-generation--vlms-that-show-the-real-manual)
9. [Worked Example — "How Do I Attach the Legs?" (IKEA)](#9-worked-example--how-do-i-attach-the-legs-ikea)
10. [Code / Implementation](#10-code--implementation)
11. [When It Breaks](#11-when-it-breaks)
12. [Production & LLMOps Notes](#12-production--llmops-notes)
13. [Interview Lens](#13-interview-lens)
14. [Alternatives & How to Choose](#14-alternatives--how-to-choose)

---

## 1. Why Multimodal RAG Exists — When the Answer Is a Picture

Text-only [RAG](RAG.md) assumes the knowledge is *in the text*. For a huge class of real documents that assumption is false:

- **IKEA assembly manuals** — almost **no prose**. The instructions *are* numbered exploded-view diagrams. OCR gets you "1", "2", "3" and a screw icon it can't name.
- **Financial reports, slide decks** — the answer is a **chart or a table**, not a sentence.
- **Invoices, forms, scans** — meaning lives in **layout** (which number sits in which box).
- **Medical / engineering** — X-rays, schematics, circuit diagrams.

The failure is concrete: run a text-RAG pipeline over an IKEA PDF and the retriever has **nothing meaningful to embed**, so it retrieves nothing useful, so the LLM answers from parametric memory and **bluffs**. Multimodal RAG fixes this by making **images first-class** in both retrieval (embed the picture, or the whole page) and generation (a VLM that can *see* the retrieved image).

🎯 *"Multimodal RAG is what you reach for the moment your corpus's information lives in diagrams, tables, or scanned layout — where OCR-then-text-RAG silently discards the very signal the user is asking about."*

---

## 2. OCR vs Vision-Based Extraction from PDFs

The first fork in any document-RAG design: **how do you turn a PDF page into something retrievable?**

```
                         ┌─ OCR path ──────────────────────────────────────┐
   PDF page (pixels) ──► │ Tesseract / AWS Textract / Google Document AI   │
                         │   page → OCR → plain text → chunk → text-embed  │──► text index
                         └─────────────────────────────────────────────────┘
                         ┌─ Vision path ───────────────────────────────────┐
   PDF page (pixels) ──► │ render page → image                             │
                         │   ColPali: patch-embed the image directly       │──► image/patch index
                         │   or VLM: caption the image → text-embed        │
                         └─────────────────────────────────────────────────┘
```

**OCR (Optical Character Recognition).** Detects and transcribes characters, converting the page to a text string you then run through normal text RAG.
- ✅ Cheap, mature, perfect when the page is **mostly clean digital text** (a contract, an article).
- ❌ **Throws away everything that isn't a character**: diagrams, icons, arrows, spatial layout. A table becomes a jumbled line of numbers with the row/column structure gone. A hand-drawn assembly step becomes empty. OCR errors on scans/handwriting **compound downstream** — a mis-read digit poisons the embedding. IKEA manuals are near-**unindexable** this way.

**Vision-based (image) extraction.** Render each page to a raster image and work with the *pixels*.
- **ColPali** embeds the page image **directly** — **OCR-free**. Nothing is transcribed; the diagram, the table grid, and the text-in-the-image are all preserved in the embedding.
- **VLM captioning** is the middle path: pass each page to a vision-language model, get a rich textual description ("exploded diagram showing four legs bolting to a tabletop with cam locks"), then embed *that* with a text model. Recovers diagram semantics, but adds an LLM call per page and can hallucinate the caption.

🎯 *"OCR asks 'what characters are on this page?'; vision-based retrieval asks 'what does this page look like?' — and for manuals, charts, and forms the second question is the one the user actually cares about."* `(certain)`

**Rule of thumb:** clean digital text → OCR + [text RAG](RAG.md) is cheapest and fine. Diagrams / tables / scans / layout → go vision-based (ColPali, or VLM captioning). In practice production systems often run **both** and merge — OCR text for exact keyword hits, image embeddings for the visual semantics.

---

## 3. Intuition — One Shared Embedding Space

**Start with the problem CLIP removes.** Before CLIP, text and images lived in **two unrelated vector spaces**. A text encoder mapped *"dog playing ball"* to one vector; an image encoder mapped 🐕 to another — but the two were trained **independently**, so their coordinates shared **no common meaning**. Comparing them is like comparing a latitude to a temperature:

```
   BEFORE CLIP — two separate spaces, no shared meaning:
     "dog playing ball" ─► text encoder   ─► (5, 8)
      🐕  dog image      ─► image encoder  ─► (100, 220)
     cosine( (5,8), (100,220) )  →  a number that means NOTHING
```

So text→image retrieval was simply **impossible**: there was no space in which a query and a picture could even *be* neighbours. CLIP's whole contribution is to build that one space.

The trick that makes *cross-modal* search possible: force a picture of the Eiffel Tower and the words "the Eiffel Tower" to land at **the same spot** in one vector space. Then a single cosine similarity works **regardless of modality** — text can find images, images can find text, images can find images.

![Contrastive training pulls a matched text-image pair together and pushes mismatched pairs apart in a shared representation space, so an anchor image of a dog sits near the text an image of a dog and far from an image of a cat](attachments/contrastive-shared-embedding-space.png)

*Contrastive learning in one picture: encode each modality, then **pull matched pairs together and push mismatched pairs apart** in a shared space. After training, "distance = semantic mismatch" holds **across** modalities, not just within one.*

```
   Two separate encoders, ONE shared space:

   "a photo of a dog" ─► text encoder  ─┐
                                        ├─► same ℝ^d ──► cosine works across modalities
    🐕  (dog image)    ─► image encoder ─┘

   matched (dog text ↔ dog image): pulled CLOSE
   mismatched (dog text ↔ cat image): pushed APART
```

Contrast this with a **bi-encoder for text** ([Embeddings §4](Embeddings.md#4-bi-encoder-vs-cross-encoder--the-two-architectures)): same idea (two towers, shared space, cosine), just **both towers are text**. Multimodal simply makes one tower an image encoder. The whole edifice rests on the shared space existing — which is exactly what **contrastive pretraining** builds.

---

## 4. The Four Retrieval Directions (Cross-Modal Search)

Once text and images share a space, four query→result directions fall out for free — pick by what the user *has* and what they *want*:

| Query (you have) | Result (you want) | Example | Who does it |
|---|---|---|---|
| **text → image** | reference pictures for a phrase | "aurora borealis" → photos of it | CLIP |
| **image → image** | visually similar images | upload a chair photo → similar chairs | CLIP |
| **image → text** | captions/descriptions for a picture | scan a landmark → its Wikipedia blurb | CLIP |
| **text → document-page** | the page that answers a question | "attach the legs" → the IKEA page image | ColPali |

The first three are **CLIP's** home turf — it's symmetric, so any modality queries any modality by embedding both sides and taking cosine. The fourth is where **ColPali** shines: the "document" is a rendered page image, and the query is text. (`text → text` is just ordinary [text RAG](RAG.md).)

🎯 *"A shared embedding space turns retrieval into a single cosine lookup that doesn't care which modality the query or the corpus is in — that's the whole superpower of CLIP-style models."*

---

## 5. CLIP — The Dual-Encoder Contrastive Model

**CLIP** = **C**ontrastive **L**anguage–**I**mage **P**retraining (Radford et al., OpenAI, 2021). It answers the §3 problem with one deceptively simple question:

> *Can we train an **image encoder** and a **text encoder** so that they output into the **same** semantic space — where the word "dog" and 🐕 the picture land at the same spot?*

Nail that, and §3's shared space stops being a wish and becomes a **trained artifact**. Everything below is how.

**Why two encoders (not one)?** Text and images are different **modalities** — a sequence of words vs a grid of pixels — so they need different feature extractors. CLIP uses **two independent towers** that meet only at the very end, in the shared space:

```
   "A dog running"  ─►  Text Encoder (Transformer)   ─►  text embedding ──┐
                                                                          ├─► cosine similarity
    🐕  dog image    ─►  Image Encoder (ViT / ResNet) ─►  image embedding ─┘
```

**Architecture.**
- **Text encoder** — a Transformer (max **77 tokens**). **Image encoder** — a ViT (e.g. ViT-B/32, see §6.1) or ResNet.
- Each projects to a shared **`d`-dim** space (`d = 512` for ViT-B/32 — the exact number you'll see in the code), then **L2-normalizes** the vector so **dot product = cosine similarity**.

![CLIP contrastive pre-training: a text encoder and an image encoder produce embeddings whose pairwise dot products form an N-by-N matrix, and training maximizes the blue diagonal of correct image-text pairs while minimizing all off-diagonal pairs](attachments/clip-contrastive-pretraining.png)

*Source: CLIP (Radford et al., 2021, OpenAI). A batch of N (image, text) pairs is encoded into N image vectors and N text vectors; the N×N dot-product matrix should be **bright on the diagonal** (the true pairs) and dark everywhere else. That single objective is what fuses the two modalities into one space.*

**How it's trained — contrastive learning on real pairs.** Take a batch of `(image, caption)` pairs from the web. For one image of a dog catching a frisbee:
- **Positive pair** (pull *together*): the image ↔ *"a dog catching a frisbee."*
- **Negative pairs** (push *apart*): that same image ↔ *"an airplane"*, ↔ *"pizza"*, ↔ *"a laptop"* — every *other* caption in the batch.

CLIP nudges the two encoders until **matched pairs are close and mismatched pairs are far**. Done over the whole batch at once, that's the **InfoNCE / symmetric cross-entropy** objective: build the `N×N` similarity matrix `S = I · Tᵀ` (scaled by a learned temperature `τ`), then make the **diagonal** (true pairs) bright and the **off-diagonal** dark, averaged over rows *and* columns:

```
L = ½ [ CE(softmax(S/τ), diagonal)  over images
      + CE(softmax(Sᵀ/τ), diagonal) over texts ]
```

That formula *is* the bright-diagonal picture above — and it's the same contrastive machinery behind [Siamese Networks & Image Similarity](../Machine%20Learning/Computer%20Vision/Siamese%20Networks%20&%20Image%20Similarity.md); CLIP is a **cross-modal siamese network at web scale**.

**Scale is the secret — 400M pairs, zero fixed classes.** CLIP wasn't trained on a fixed label set like ImageNet; OpenAI trained it on **~400 million (image, caption) pairs** scraped from the internet. So it never learned "class 37 = giraffe" — it learned what the *word* "giraffe" and giraffe *pixels* have in common. That buys **zero-shot** recognition: to detect a giraffe you just **embed the text "a giraffe"** and check which images sit nearest — no retraining, no predefined class list. (A vanilla CNN classifier can only output classes it was trained on; CLIP can score *any* phrase you write.)

**Why the dual encoder scales (and a cross-encoder can't).** This is the property that makes CLIP a *production* retriever, not just a clever model:

```
   DUAL ENCODER (CLIP)                          CROSS-ENCODER
   • encode 10M page-images ONCE, offline       • to score, feed (query + image) TOGETHER
     → store 10M vectors in a vector DB           through one Transformer → a score
   • per query: encode the query ONCE,          • per query: 10M forward passes
     then ANN nearest-neighbour search            (re-score every image from scratch)
   ⇒ ~O(1 encode) + fast ANN → millions ✅       ⇒ O(N) heavy forward passes → infeasible ❌
```

Because the towers are **independent**, image embeddings are precomputed once and reused for *every* query — the corpus is just vectors in a DB. A cross-encoder is more accurate per pair but must re-read the query *with* each candidate, so it can't scan millions. The production pattern: **CLIP retrieves top-k fast, then an optional cross-encoder/VLM re-ranks just those few** (see §14).

**The key limitation (and why ColPali exists).** CLIP squashes an **entire image or page into one global vector**. Perfect for "what is this a picture *of*?", but **lossy for documents**: a manual page with 12 diagram steps, a table, and a warning icon collapses to a single 512-d point, so fine-grained "which step shows the cam lock?" detail is **averaged away**. CLIP is also weak at **reading dense text inside an image** and capped at **77 text tokens**. `(certain)` The fix is to stop compressing the page into one vector — which is exactly ColPali's move (§7).

**The whole progression in one ladder** (say this in an interview):

```
   Text RAG  →  search WORDS                 (embed text chunks)
   CLIP      →  search WHOLE images / pages   (ONE vector per page — semantic, but global)
   ColPali   →  search PARTS of a page        (MANY patch vectors per page — keeps layout & detail)
```

🎯 *"CLIP earns its place by putting images and text in one space so a single cosine query retrieves across modalities, and — being a dual encoder — it scales to millions of pages; its ceiling is the one-vector-per-page compression, which is the exact gap ColPali closes."*

**Why it's still the POC choice:** ViT-B/32 is ~150 MB, runs on a **free-tier CPU/GPU**, embeds in milliseconds, and needs no exotic index — just cosine in any vector DB. For a demo or a natural-image corpus, CLIP is the fast, cheap win. **For document retrieval systems, ColPali is often the stronger production choice, whereas CLIP is an excellent proof-of-concept or lightweight baseline.** (To see *why* — and to decode "ColPali" itself — read the **building-blocks primer** in §6 before §7.)

---

## 6. Building Blocks — The Five Pieces ColPali Is Made Of

> **Read this before §7.** ColPali isn't one idea — it's a stack of five, and its **name literally spells two of them**: **Col**BERT + Pali**Gemma**. If ViT / SigLIP / VLM / PaliGemma / ColBERT are fuzzy, the ColPali section reads like alphabet soup. Here's each, ending with the one-line **dependency chain** that ties them together. Nothing here is exotic — it's the same "tokens + attention + shared space" machinery from earlier notes, pointed at pixels.

### 6.1 ViT — the Vision Transformer (the image encoder underneath all of this)

A **ViT** treats an image the way a language model treats a sentence (Dosovitskiy et al., 2020, *"An Image is Worth 16×16 Words"*):

![Vision Transformer architecture: the input image is split into fixed-size patches, each flattened and linearly projected into a patch embedding, combined with position embeddings and a prepended learnable class token, then processed by a Transformer encoder whose class-token output feeds an MLP head that predicts the label](attachments/vit-architecture.png)

*Source: Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (ICLR 2021), Figure 1.*

1. **Cut the image into fixed-size patches** (e.g. 16×16 px), non-overlapping — a 224×224 image → **196 patches**.
2. **Flatten + linearly project** each patch into a `d`-dim vector — a **patch embedding**, the visual equivalent of a word token.
3. Add a **position embedding** (so the model knows *where* each patch sat) and prepend one learnable **`[class]` token** (a slot whose job is to summarize the whole image).
4. Run the sequence through a standard **Transformer encoder** (multi-head self-attention) — **every patch attends to every other patch**.
5. For classification, the `[class]` token's output vector → **MLP head** → label.

The payoff is **global receptive field from layer one**: a patch in the corner (the cam lock) can directly influence a patch on the other side (the leg it clips into), whereas a [CNN](../Machine%20Learning/Computer%20Vision/Convolutional%20Neural%20Networks%20for%20Vision.md) only sees local neighborhoods until deep layers. The self-attention machinery itself is the same one from [RNN · LSTM · Transformers](../Machine%20Learning/NLP/RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md), just fed patches instead of words. **This ViT is the image encoder inside CLIP, SigLIP, and ColPali** — when §5 said CLIP's image side is "a ViT," *this* is that. `(certain)`

### 6.2 SigLIP — CLIP with a better loss (this is ColPali's actual vision encoder)

**SigLIP** = **Sig**moid **L**oss for Language–**I**mage **P**re-training (Zhai et al., Google DeepMind, ICCV 2023). Same dual-encoder skeleton as [§5 CLIP](#5-clip--the-dual-encoder-contrastive-model) — image encoder + text encoder → one shared space — but it swaps the **loss**:

- **CLIP** uses a **softmax / InfoNCE** loss: each pair's score is normalized against *every other pair in the batch* (the full `N×N` matrix). That global coupling makes it memory-hungry and sensitive to batch size.
- **SigLIP** uses a **pairwise sigmoid** loss: for each `(image, text)` pair *independently* it asks "do these two match — yes/no?" No global normalization, so one pair's loss doesn't depend on the rest of the batch.

Result: it trains well at **smaller batch sizes** and scales cleanly to huge ones — simpler and cheaper (SigLIP peaks around a 32k batch where softmax needed ~98k). For this note only one fact matters: **SigLIP's ViT is the vision encoder that PaliGemma — and therefore ColPali — is built on.** `(certain)`

### 6.3 VLM — a Vision-Language Model (sees images, writes text)

A **VLM** takes **image(s) + text in, and produces text out.** Almost every modern VLM is three parts bolted together:

![PaliGemma architecture: an image input is turned into patch tokens by the SigLIP image encoder, passed through a linear projection, concatenated with the text-input tokens, and fed to the Gemma language model which generates the text output](attachments/paligemma-architecture.png)

*Source: Hugging Face — "PaliGemma – Google's Cutting-Edge Open Vision Language Model" (huggingface.co/blog/paligemma).*

1. **Vision encoder** (a ViT — CLIP or SigLIP) → turns the image into **patch embeddings**.
2. **Projector / adapter** (a small **linear** layer, sometimes cross-attention) → maps those visual vectors into the **LLM's token space**, producing "**image tokens**" the language model can read alongside words.
3. **LLM** → consumes image tokens **+** text tokens together and generates the text answer.

You've already met VLMs twice in this note without the label: the **reader** in §8 (LLaMA-4-Scout / GPT-4o / Gemini / Claude) is a VLM, and — the part that matters here — **PaliGemma**, the model ColPali is built from, is a VLM. Same architecture, **two jobs**: one *reads* retrieved pages to answer; one *encodes* pages for retrieval. `(certain)`

### 6.4 PaliGemma-3B — the specific VLM ColPali fine-tunes

**PaliGemma** (Google, 2024, *"PaliGemma: A versatile 3B VLM for transfer"*) is a concrete, open **~3B**-parameter VLM — the diagram above *is* PaliGemma:

- **Vision encoder:** **SigLIP-So400m** (a ViT, patch size 14) → each patch → a **1152-dim** "soft token."
- **Projector:** a **linear projection** to **2048-dim**, matching Gemma's token width.
- **LLM:** **Gemma-2B** — the projected image tokens are **prepended** to the text tokens, then Gemma generates.

So the phrase from §7 — "**PaliGemma-3B (a SigLIP ViT vision encoder + a Gemma language model)**" — unpacks exactly as: *a ViT (SigLIP) turns the page into patch tokens → a linear layer maps them into Gemma's space → Gemma is the language model.* ColPali takes this pretrained model and **repurposes its per-patch representations for retrieval** instead of generation. `(certain)`

### 6.5 ColBERT — late interaction / MaxSim (the "Col" in ColPali)

**ColBERT** = **Contextualized Late Interaction over BERT** (Khattab & Zaharia, SIGIR 2020) is a *text* retrieval model, and it's the direct ancestor of ColPali's **scoring**. It sits between two extremes:

```
single-vector bi-encoder   →  ONE vector per doc     →  fast, but averages everything away   (this is CLIP)
cross-encoder              →  query+doc through BERT  →  accurate, but can't precompute → too slow to scan a corpus
ColBERT (late interaction) →  MANY vectors per doc    →  precompute docs offline + cheap match  ← the sweet spot
```

ColBERT encodes the query and the document **independently** into **one vector per token** (so document vectors are computed once, **offline**), then scores them with **late interaction / MaxSim**: for **each query token**, take the **max** similarity over all document tokens, then **sum** those maxima. Each query word finds its single best-matching document token — fine-grained, yet cheap because the interaction is just dot-products, no re-encoding.

![ColBERT late-interaction MaxSim scoring: a query encoder turns the query into one vector per token while a document encoder does the same offline for the document, then each query-token vector takes its maximum similarity (MaxSim) over all document-token vectors and those per-token maxima are summed into a single relevance score](attachments/colbert-late-interaction.png)

*Source: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (SIGIR 2020). Read it as the picture of §7: swap the blue document tokens for a page's ~1,000 image patches and you have ColPali.*

🎯 **ColPali = ColBERT's MaxSim, but the "document tokens" are PaliGemma's ~1,000 image-*patch* vectors and the query tokens are text.** That's the whole trick — and it's exactly the ASCII MaxSim diagram you'll see in §7. `(certain)`

### The dependency chain (say this in an interview)

```
ViT  ──(wrap in a dual encoder, train with a sigmoid loss)──►  SigLIP
SigLIP  +  linear projector  +  Gemma LM                    ──►  PaliGemma  ( = a VLM )
ColBERT's late-interaction MaxSim   ⊕   PaliGemma's patch tokens ──►  ColPali
```

🎯 *"ColPali is ColBERT's late interaction (MaxSim) run over the patch embeddings of a VLM (PaliGemma) whose vision encoder is a SigLIP ViT — so it inherits ViT's whole-page attention, SigLIP's efficient training, and ColBERT's token-level precision all at once."* That single sentence is the whole §7 in compressed form.

---

## 7. ColPali — Page-as-Image Retrieval with Late Interaction

**ColPali** = **Col**BERT-style late interaction + **Pali**Gemma (Faysse et al., 2024). It's built for **documents**, and it makes two moves CLIP doesn't.

**Move 1 — the page IS the document (no OCR, no chunking).** You render each PDF page to an image and index the **image**. No text extraction, no chunk-size tuning, no layout parser. The diagram, the table, the caption, the handwriting — all retained as pixels.

**Move 2 — multi-vector late interaction (MaxSim), not one global vector.** ColPali is built on **PaliGemma-3B** (a SigLIP ViT vision encoder + a Gemma language model). A page image is split into ~**1,000 patches**; ColPali emits **one contextualized vector per patch** (projected to ~128-d, ColBERT-style) — so a page is **~1,000 vectors**, not one. The query text becomes a handful of **token** vectors.

**One model encodes both sides — not two towers.** This is a sharp contrast with **CLIP**, which has **two independent towers** — a dedicated text Transformer and a ViT — that only meet in a shared space. ColPali instead has **one model, PaliGemma**, encode *both* the page and the query, through the **same ~128-d projection head** — which is exactly what makes the query-token and patch vectors comparable in the first place:

```
   CLIP — two independent towers          ColPali — one shared model (PaliGemma)
   ──────────────────────────────         ───────────────────────────────────────
   query text → text Transformer          page image → SigLIP ViT → Gemma
   image      → ViT                                   → per-patch vecs → 128-d
   (two separate encoders,                query text → Gemma-2B  (vision encoder
    meet only in a shared space)                       bypassed — no image)
                                                      → per-token vecs → 128-d
                                          ONE model, SAME 128-d projection head
```

- **Page (document):** image → **SigLIP ViT** → soft image tokens → **Gemma** → per-patch hidden states → **linear projection to ~128-d**.
- **Query (text):** text tokens → **Gemma-2B** (no image input, so the vision encoder is **bypassed**) → per-token hidden states → **the same ~128-d projection**.

Because both sides come out of the *same* model into the *same* space, their dot products are meaningful — exactly what MaxSim needs. There's **no separate CLIP-style text tower**. (Same trick as ColBERT's single BERT for query and doc; ColPali just swaps BERT for a VLM and runs the query through its **language-model path**.) `(certain)`

Scoring is **late interaction / MaxSim** (borrowed from **ColBERT** — "Contextualized Late Interaction over BERT"):

```
score(query, page) = Σ over query tokens q  [  max over page patches p  (q · p)  ]
                       └── each query word finds its BEST-matching patch, then sum ──┘
```

```
  query: "attach"  "legs"  "table"
             │        │        │
             ▼        ▼        ▼        (each query token scans ALL page patches,
   page patches: ▢▢▢▢▢▢▢▢▢▢▢▢…          keeps its single best match — MaxSim)
             │        │        │
           max      max      max   ──► Σ = relevance score
```

Because each query token can latch onto the **specific region of the page** that matches it, ColPali captures the fine-grained, spatially-local detail that CLIP's single vector blurs away. This is why it dominates the **ViDoRe** (Visual Document Retrieval) benchmark and beats OCR→text-RAG pipelines on real documents. `(likely)`

**A crucial clarification — MaxSim *scores*, it doesn't *retrieve*.** The single most common misread of late interaction is *"a 4-token query retrieves 4 patches."* It doesn't. Patches are an **internal scoring detail**; the thing you retrieve is a **page**:

- For each query token you compute its best (max) similarity to the page's patches and **keep only that number** — you never pull the patch out. Those per-token maxima are **summed into one relevance score for the whole page**.
- The unit that gets **retrieved is the page**, ranked by that summed score → **top-k pages, not top-k patches**. A 4-token query still returns *pages*, no matter how many patches were touched while scoring them.
- There is **no 1-to-1 token↔patch mapping**: the *same* patch can be the best match for several query tokens at once. If one patch covers the page region depicting "insert the table legs," then `attach`, `table`, **and** `legs` may all take their max on that single patch.

```
   query tokens:   How   attach   table   legs
                     │      │        │       │
        (each token's MAX similarity over ALL of THIS page's ~1,000 patches)
                     │      │        │       │
                   0.10   0.89     0.91    0.95
                     └───────────── Σ ───────────┘  = 2.85  ← ONE score for THIS page
                                                          (the winning patches are never returned)
   → repeat for every page → rank pages by score → retrieve the top-k PAGES
```

**Where ANN comes in (and why it must).** Brute-force scoring is `O(query_tokens × patches × pages)`. A 10M-page corpus at ~1,000 patches/page is **~10 billion patch vectors** — far too many to MaxSim against on every query. So production late-interaction systems (ColBERT's **PLAID**, and the vector DBs that copy it) run retrieval in **two stages**:

```
   query token ─► ANN index over ALL patch vectors ─► candidate patches
                     └── their parent PAGES become the shortlist ──┘
                                   │
                                   ▼
        exact MaxSim ONLY on the shortlisted pages ─► final page ranking
```

Each query token does an approximate-nearest-neighbour lookup to gather a handful of candidate patches; the pages those patches belong to form a small candidate set; **exact MaxSim then re-scores only that set**. You get late-interaction precision without ever touching every patch — the same "cheap shortlist → MaxSim rerank" pattern flagged in §11 and §12. (The catch: candidate generation is *approximate*, so a page whose patches never surface as candidates can be missed — the usual recall/latency knob.)

🎯 *"Late interaction retrieves pages, not patches: MaxSim keeps each query token's best-patch **score** and sums them into one **page** score — so n query tokens never mean n retrieved patches, and one patch can win several tokens at once."* `(certain)`

**The costs are real.** ~1,000 vectors/page means the index is **~1,000× larger** than a single-vector store (96 IKEA pages ≈ **~100k vectors**), MaxSim is **more compute** than one dot product, and the 3B model needs a **GPU** (the notebook 4-bit-quantizes it just to fit a free T4's ~15 GB). That storage/latency bill is the price of the accuracy — and exactly why the instructor's rule is **ColPali for production, CLIP for POC**.

🎯 *"CLIP gives a page one vector and asks 'what is this page about?'; ColPali gives it a thousand and asks 'which patch answers each word of the query?' — late interaction is what buys document-grade precision."*

---

## 8. Generation — VLMs That Show the Real Manual

Retrieval hands you the **top-k real page images** (plus any text). The reader is a **Vision-Language Model** (VLM) — LLaMA-4-Scout via Groq in the notebook, or GPT-4o / Gemini / Claude — that ingests a **multimodal prompt**: the question, the retrieved text context, and the retrieved **images** (as base64 `image_url` parts). It *sees* the diagram and explains it.

**The non-negotiable rule: surface the retrieved images verbatim — never generate new ones.** This is the single most important design point for a manual/product RAG:

- A RAG system's job is to be **grounded in ground truth**. The correct assembly diagram already exists in the retrieved page — so the UI **displays that exact retrieved image**.
- You must **not** hand the query to an image-*generation* model (DALL·E / diffusion). A generated "assembly diagram" is a **plausible hallucination** — it will invent screws, steps, and orientations that don't match the real product. For a manual, that's not a cosmetic error; it's **wrong instructions that break the furniture or injure the user**.

```
   RIGHT:  query ─► retrieve REAL page image ─► VLM explains it ─► UI shows the SAME real image
   WRONG:  query ─► VLM/diffusion GENERATES a new "diagram" ─► hallucinated, unsafe
```

So the VLM's role is **explain + point at**, not **draw**. Prompt discipline mirrors [text RAG](RAG.md)'s grounded prompt (see [Prompt Engineering](Prompt%20Engineering.md)): *"Answer only from the provided context and images; if the images don't show it, say so; cite the page."* The generative model produces **words**; the **images in the answer are the retrieved originals**, passed through untouched.

🎯 *"In multimodal RAG the model generates the explanation, but the images it shows are retrieved, not invented — a manual assistant that draws its own diagrams is a safety bug, not a feature."* `(certain)`

---

## 9. Worked Example — "How Do I Attach the Legs?" (IKEA)

Straight from the ColPali notebook — five IKEA manuals, no OCR anywhere:

1. **Ingest.** Download 5 assembly PDFs (MALM, BILLY, BOAXEL, ADILS, MICKE) → `pdf2image` renders **96 pages** to PIL images.
2. **Index (pre-production).** Feed each page image to ColPali → multi-vector patch embeddings, one page at a time, embeddings moved to CPU to free GPU memory. Result: **96 page-embeddings** (each itself ~1,000 patch vectors).
3. **Query (in-production).** `"How do I attach the legs to the table?"` → ColPali embeds the query into token vectors.
4. **Score (MaxSim).** `score_multi_vector` runs late interaction of the query tokens against every page's patches → a relevance score per page → `top-k`:

```
   #1  BILLY.pdf  page 3   score 17.41
   #2  ADILS.pdf  page 7   score 17.39   ← ADILS is literally a table-leg product
   #3  BILLY.pdf  page 4   score 17.38
```

5. **Fetch the real images.** Map `(doc_id, page_num)` back to the actual page images.
6. **Generate.** Send those page images + the question to the VLM → it reads the exploded diagram and answers *"Align each leg with the pre-drilled corner holes and fasten with the provided screws, as shown"* — and the app **displays page 7 of ADILS itself**. The observation the notebook prints says it all: *"ColPali retrieved pages that visually show leg attachment instructions, even without OCR text extraction."*

Note the retrieval landed on **ADILS (a leg product)** and **BILLY** pages purely from the **visual** match between the query and the diagram regions — no page contained clean extractable prose describing "attach the legs."

---

## 10. Code / Implementation

**Path A — CLIP (POC): shared-space cross-modal search + VLM answer.**

```python
import clip, torch, chromadb
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)   # ~150MB, dim=512

def embed_text(txt):                                        # text tower
    tok = clip.tokenize([txt], truncate=True).to(device)    # 77-token cap
    with torch.no_grad(): f = model.encode_text(tok)
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy() # L2-norm ⇒ dot = cosine

def embed_image(img):                                       # image tower, SAME space
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(): f = model.encode_image(x)
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

# One index, cosine space — text and image vectors are interchangeable (cross-modal)
col = chromadb.Client().create_collection("kb", metadata={"hnsw:space": "cosine"})
col.add(ids=["eiffel"], embeddings=embed_image(Image.open("eiffel.jpg")).tolist())
hits = col.query(query_embeddings=embed_text("iron tower in Paris").tolist(), n_results=3)
#     ↑ text query retrieves an IMAGE — the shared space makes this "just work"
```

**Path B — ColPali (production): OCR-free page retrieval with late interaction.**

```python
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import BitsAndBytesConfig
from pdf2image import convert_from_path
import torch

# 4-bit quantize the 3B model so it fits a free T4 (~15GB); T4 has no bf16 → fp16 compute
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = ColPali.from_pretrained("vidore/colpali-v1.2", quantization_config=bnb,
                                device_map="cuda:0", torch_dtype=torch.float16).eval()
proc  = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

pages = convert_from_path("ADILS.pdf")                      # PDF → page images (NO OCR)
doc_embeddings = []
for pg in pages:                                            # one page at a time = tiny GPU footprint
    batch = proc.process_images([pg]).to(model.device)
    with torch.no_grad(): emb = model(**batch)              # multi-vector: ~1000 patch vectors/page
    doc_embeddings.extend(list(emb.to("cpu").float()))      # offload to CPU, free GPU for next page

q = proc.process_queries(["How do I attach the legs to the table?"]).to(model.device)
with torch.no_grad(): q_emb = model(**q)
scores = proc.score_multi_vector(q_emb.cpu().float(), doc_embeddings)[0]  # ← MaxSim late interaction
top = torch.topk(scores, k=3)                               # best pages, as images to hand the VLM
```

**Path C — VLM generation (both paths converge here).** Pack retrieved **text + real images** into one multimodal message; the answer's images are the retrieved originals:

```python
from groq import Groq
def image_to_data_url(img):   # resize to cap tokens, then base64
    import base64, io; b = io.BytesIO(); img.convert("RGB").save(b, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

content = [{"type": "text", "text": f"Answer ONLY from the context and images. Cite the page.\n\nQ: {query}"}]
for pg in retrieved_page_images:                            # the REAL pages, not generated ones
    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(pg)}})

resp = Groq().chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",      # a VLM: it can SEE the images
    messages=[{"role": "user", "content": content}], max_tokens=500, temperature=0.3)
# UI then displays retrieved_page_images verbatim alongside resp — never a generated diagram
```

---

## 11. When It Breaks

**CLIP.**
- **Single global vector = lost detail.** Fine on natural photos, weak on **dense documents** — multi-step diagrams and tables blur into one point. Don't use raw CLIP as a document retriever in production.
- **Can't read text-in-image well** and is capped at **77 text tokens** — long queries get truncated.
- **Domain shift.** CLIP trained on web photos; IKEA line-drawings, radiology, or schematics are out-of-distribution, so cosine gets noisy. Note the notebook's own Great-Barrier-Reef pair scored only **0.116** — CLIP genuinely struggled on that image. Fine-tune or switch models for a specialized domain.

**ColPali.**
- **Storage & latency.** ~1,000 vectors/page explodes the index and makes MaxSim costlier than a single dot product; naïvely it's `O(query_tokens × page_patches)` per page. Mitigations: **PLAID**-style approximate late interaction, vector **pooling/quantization**, a cheap first-stage filter then MaxSim rerank.
- **Needs a GPU** and non-trivial indexing time; page-render **resolution** matters (too low → the diagram's fine lines vanish).

**VLM generation.**
- **Hallucination if the prompt isn't grounded** — it'll confidently describe a step that isn't in the image. Enforce "answer only from provided images; else say you can't."
- **Image tokens are expensive** — each retrieved page costs hundreds of tokens; retrieving top-10 full pages blows the context/budget. Cap `k`, downscale images.
- **The cardinal sin:** wiring a **generative image model** into the answer path → invented, unsafe "instructions." Always show **retrieved** images.

**System-level.** Retrieval **evaluation is harder** — there's no clean text ground truth, so you lean on human-labeled page relevance and [Embeddings §7](Embeddings.md#7-evaluating-retrieval--precision-recall-mrr-ndcg) metrics (Recall@k, MRR, nDCG) computed over *pages*. Deeper end-to-end faithfulness scoring is deferred to `[[RAG Evaluation]]`.

---

## 12. Production & LLMOps Notes

- **The core decision — ColPali vs CLIP (the instructor's rule):** **ColPali for production product/document RAG** (manuals, invoices, decks — anywhere layout and diagrams carry the meaning); **CLIP for POCs and natural-image corpora** where compute is the constraint. CLIP is cheap and CPU-friendly; ColPali needs a GPU and a fat index but *retrieves what actually matters* on documents. `(certain — stated by instructor from experience)`
- **Storage & index.** Budget for multi-vector blow-up: quantize (PQ/scalar), pool patch vectors, or two-stage retrieve (single-vector shortlist → MaxSim rerank). A single-vector CLIP store fits in memory trivially; a ColPali store may not.
- **Latency.** ColPali's late interaction is the bottleneck; precompute/cache page embeddings **offline** (they never change), keep only query encoding + scoring online. Cache frequent queries.
- **Serving the VLM.** Cost scales with **image tokens** — downscale retrieved pages (e.g. long side ≤ 512–1024 px), cap `k`, and consider a cheap text-first pass that only escalates to the VLM when images are needed.
- **Ingestion.** Page-render DPI is a real hyperparameter; version your renderer + model. **Incremental indexing** — new pages add without a full rebuild (same as [RAG §6](RAG.md#6-vector-databases--ann-indexing-hnsw)).
- **Guardrails.** Hard-enforce "**show retrieved images, never generate**." Log which page each answer cites so you can audit groundedness and catch drift when the manual catalog updates.
- **Hybrid is often best.** OCR text (exact keyword/serial-number hits) **+** image/patch embeddings (visual semantics), merged with RRF — the multimodal analogue of hybrid search in [RAG §5.4](RAG.md#5-stage-2--retrieval--generation-online).

---

## 13. Interview Lens

The question behind the questions: *do you know when meaning lives in pixels, and which retriever pays for itself?*

- **"Why not just OCR the PDF and use normal RAG?"** 🎯 *Because OCR only transcribes characters — it discards diagrams, tables, icons, and layout, which is exactly where a manual's or a report's information lives; you retrieve nothing useful and the LLM bluffs.*
- **"CLIP vs ColPali — when each?"** 🎯 *CLIP = one global vector per image, cheap, CPU-friendly → POCs and natural images. ColPali = ~1,000 patch vectors per page + late-interaction MaxSim, OCR-free, GPU-heavy → production document retrieval where layout/diagrams matter.*
- **"What is late interaction / MaxSim?"** For each query token, take its **max** similarity over all document patches, then **sum** across query tokens — ColBERT's idea, applied to image patches. It preserves token-level, region-level detail a single pooled vector destroys. `(certain)`
- **Follow-up (the trap): "So a 5-token query retrieves 5 patches?"** 🎯 *No — MaxSim is a **scoring** mechanism, not a retrieval one. Each query token keeps only its best-patch **similarity score**; those are summed into one page score, and the system retrieves the top-k **pages**. One patch can be the best match for several tokens, so there's no token↔patch correspondence.* At scale, an **ANN index shortlists candidate patches/pages first**, then exact MaxSim re-scores only the shortlist. `(certain)`
- **"How do you stop the system inventing assembly diagrams?"** 🎯 *You never put an image-generation model in the answer path. Retrieval returns the real page image; the VLM only explains it; the UI shows the retrieved original. A drawn diagram is a hallucination, and for a manual that's a safety bug.*
- **"What makes the shared embedding space possible?"** Contrastive pretraining (CLIP's InfoNCE) that pulls matched text–image pairs together and pushes mismatched apart — after which a single cosine works across modalities. `(certain)`
- **Follow-up: "ColPali's downside?"** Multi-vector storage and MaxSim latency — mitigate with pooling/quantization, PLAID, or a single-vector first stage. `(likely)`

---

## 14. Alternatives & How to Choose

| Approach | Retrieval unit | Best when | Cost |
|---|---|---|---|
| **OCR + text RAG** | extracted text chunks | clean **digital text**, exact keyword needs | cheapest |
| **CLIP** (single-vector) | one vector / image | **POC**, natural images, cross-modal search, tight compute | low (CPU-OK) |
| **VLM captioning + text RAG** | LLM-written captions | want text-search but need diagram *semantics* | +1 LLM call/page |
| **ColPali** (multi-vector) | ~1,000 patch vectors / page | **production** docs: manuals, invoices, decks, tables | high (GPU, big index) |
| **Unified multimodal embeddings** | one vector, both modalities | one API for text+image, managed | API cost |

- **Unified embedding models** worth knowing by name: **Cohere Embed v4**, **Jina-CLIP**, **Nomic Embed Vision**, **SigLIP**, **Voyage multimodal** — single models that embed text and images into one space (CLIP's idea, productized). Good default when you don't want to run ColPali yourself.
- **ColBERT / ColQwen2** — the same late-interaction family; ColQwen2 swaps PaliGemma for a Qwen2-VL backbone and often tops ViDoRe. `(likely)`
- **GraphRAG / long-context** — orthogonal; they change *how you reason over* retrieved context, not how you retrieve images.
- **Decision one-liner:** *digital text → OCR; visual documents in production → ColPali; a quick cross-modal demo or a compute budget → CLIP; want diagram meaning in a text index → VLM-caption then embed.*

Structured **tables/spreadsheets** are their own problem (SQL/text-to-SQL, row-serialization, schema-linking) — deferred to `[[Tabular RAG]]`. Retrieval metrics live in [Embeddings §7](Embeddings.md#7-evaluating-retrieval--precision-recall-mrr-ndcg); end-to-end faithfulness in `[[RAG Evaluation]]`.

---

## 🧠 Self-Test

1. Why does OCR-then-text-RAG fail on an IKEA manual, and what does the vision-based path do instead?
   <details><summary>answer</summary> The manual's information is in <b>diagrams and layout</b>, not characters — OCR transcribes "1, 2, 3" and loses the exploded-view instructions, so the retriever has nothing meaningful to match. The vision path <b>renders each page to an image and embeds the pixels</b> (ColPali) or captions them with a VLM, preserving diagrams, tables, and spatial cues. OCR-free retrieval.</details>

2. What single objective lets CLIP put "a photo of a dog" and a dog image at the same point in space?
   <details><summary>answer</summary> <b>Contrastive pretraining (InfoNCE)</b>: over a batch of N image–text pairs, maximize the similarity of the N <b>matched</b> pairs (the diagonal of the N×N matrix) and minimize all off-diagonal mismatches, symmetric over rows and columns. After training, both encoders share one space where cosine = semantic match across modalities.</details>

3. What are the four cross-modal retrieval directions, and which model owns each?
   <details><summary>answer</summary> <b>text→image, image→image, image→text</b> — all CLIP (symmetric shared space). <b>text→document-page</b> — ColPali (query text vs rendered page image). (text→text is plain text RAG.)</details>

4. Explain late-interaction MaxSim and why ColPali beats CLIP on documents.
   <details><summary>answer</summary> CLIP pools a page into <b>one vector</b>, blurring fine detail. ColPali emits <b>one vector per image patch</b> (~1,000/page) and scores by <b>MaxSim</b>: for each query token, take its max dot-product over all page patches, then sum across query tokens. Each query word latches onto the specific page region that matches it, capturing layout/diagram/table detail a single pooled vector destroys — at the cost of a much larger index and more compute.</details>

5. A teammate wants to call an image-generation model to draw the assembly step for the user. What do you say?
   <details><summary>answer</summary> No. A RAG system must be <b>grounded in ground truth</b> — the correct diagram already exists in the retrieved page, so <b>display that retrieved image verbatim</b>. A generated diagram is a plausible <b>hallucination</b> that invents screws/steps/orientations; for a manual that's wrong, unsafe instructions. The VLM <i>explains</i> the real image; it never draws a new one.</details>

6. The instructor's rule: ColPali for production, CLIP for POC — why?
   <details><summary>answer</summary> CLIP is a ~150 MB single-vector model, CPU-friendly, millisecond cosine, trivial index → perfect for demos and natural-image corpora on a compute budget. ColPali is a quantized 3B model needing a GPU with a ~1,000×-larger multi-vector index and costlier MaxSim — but it <b>retrieves the layout/diagram detail that real documents depend on</b>, so it wins in production despite the cost.</details>
