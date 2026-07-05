# Evaluating LLMs — How You Know an LLM or AI App Actually Works

> **TL;DR.** Classic ML eval is easy because there's *one right answer* and *one number* (accuracy, AUC). LLM eval is hard because outputs are **open-ended text** with *many valid answers*, quality is **multi-dimensional** (correct + fluent + faithful + safe), the model is **non-deterministic**, and you often have **no reference answer at all**. So you split evaluation two ways: **reference-based** (compare output to a gold answer — Exact Match, F1, Accuracy, **BLEU**, **ROUGE**, **BERTScore**) vs **reference-less** (judge quality with no gold answer — **LLM-as-Judge**, perplexity, heuristics, human rating). And you evaluate at two levels: the **model** (benchmarks like MMLU / HumanEval / Chatbot Arena) and the **application** (the whole pipeline, component by component, plus live production signals). Pick the metric that matches the *task shape*, never a single leaderboard number.

**Where it fits:** The measurement layer of AI engineering — how you choose a model, catch regressions, and prove an LLM app is good enough to ship. Builds directly on [[LLM]] (decoding, hallucination) and [[RNN · LSTM · Transformers]] (tokenization, embeddings for BERTScore).
**Prereqs:** [[LLM]] (what the model outputs and why it's stochastic), classic classification metrics (accuracy / precision / recall / F1), embeddings & cosine similarity.

> ⚙️ *Format note: this note adapts the vault's standard topic skeleton for an **AI-engineering** topic — evaluation is a discipline, not a single model, so the "Formal Core / When It Breaks" slots become metric-math and pitfall sections, and Benchmarks / Judge-bias / Online-eval get their own homes.*

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [LLM Evaluation vs Classic ML Evaluation](#2-llm-evaluation-vs-classic-ml-evaluation)
3. [What You Evaluate — Model vs Application (Components)](#3-what-you-evaluate--model-vs-application-components)
4. [Objectives & Goals of LLM Evaluation](#4-objectives--goals-of-llm-evaluation)
5. [Challenges in Evaluating an LLM](#5-challenges-in-evaluating-an-llm)
6. [Evaluation Methods — Reference-based vs Reference-less](#6-evaluation-methods--reference-based-vs-reference-less)
7. [Reference-based Metrics — BLEU · ROUGE · BERTScore · EM/F1 · Accuracy · Perplexity](#7-reference-based-metrics--bleu--rouge--bertscore--emf1--accuracy--perplexity)
8. [LLM-as-Judge — the Reference-less Flagship](#8-llm-as-judge--the-reference-less-flagship)
9. [Benchmarks & Leaderboards](#9-benchmarks--leaderboards)
10. [Beyond Overlap — Safety, Bias, Hallucination & Cost/Latency](#10-beyond-overlap--safety-bias-hallucination--costlatency)
11. [Code / Implementation (SST-2, SQuAD, Opik multi-LLM)](#11-code--implementation-sst-2-squad-opik-multi-llm)
12. [When It Breaks — Metric & Eval Pitfalls](#12-when-it-breaks--metric--eval-pitfalls)
13. [Production & Online Evaluation](#13-production--online-evaluation)
14. [Interview Lens](#14-interview-lens)
15. [Alternatives & How to Choose a Metric](#15-alternatives--how-to-choose-a-metric)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

```
CLASSIC ML:   input → model → ONE label → compare to ONE gold label → accuracy      (easy: it's ==)
LLM:          prompt → model → OPEN-ENDED TEXT → ??? → compare to ??? → ???          (hard: what is "correct"?)
```

The whole difficulty is that **"The capital of France is Paris." and "Paris." and "It's Paris, the French capital." are all correct**, but a naive `prediction == reference` check scores two of them as *wrong*. Evaluation is the art of measuring "is this output good?" when good is fuzzy, multi-dimensional, and often has no single gold answer.

Three questions organize everything below:
- **What** are you judging? (a model, or a whole app pipeline — §3)
- **Against what?** (a reference answer, or no reference — §6)
- **By whom?** (a formula, another LLM, or a human — §7–8)

```
                        ┌───────────── EVALUATION ─────────────┐
      reference-based ──┤  BLEU · ROUGE · BERTScore · EM · F1   │── automated / cheap / reproducible
                        │                                        │
      reference-less  ──┤  LLM-as-Judge · perplexity · heuristics│── scales to open-ended & production
                        │                                        │
        human        ──┤  Likert · pairwise preference · Arena  │── gold standard, slow & costly
                        └────────────────────────────────────────┘
```

---

## 2. LLM Evaluation vs Classic ML Evaluation

The single most-asked framing. Six axes:

| Axis | Classic ML | LLM |
|---|---|---|
| **Output space** | Fixed & discrete (a class, a number) | Open-ended free text — combinatorially infinite |
| **Ground truth** | One correct label per input | Many valid answers; often *no* reference at all |
| **Metric** | One scalar (accuracy, RMSE, AUC) | Many dims at once: correctness, fluency, faithfulness, safety, format, tone |
| **Determinism** | Same input → same output | **Stochastic** (temperature/sampling) → same input → different outputs |
| **Correctness test** | `pred == label` works | `pred == ref` fails on paraphrase; need semantic/graded scoring |
| **Where quality lives** | In the model weights | In the *whole system* — prompt, context, retrieval, tools, decoding |

🎯 **Kill-shot:** *"ML evaluation compares one prediction to one label with one number; LLM evaluation has to score open-ended text that has many valid forms, along several quality dimensions at once, produced non-deterministically — so a single accuracy number is usually the wrong tool."*

**A subtle trap:** an LLM *can* be evaluated the classic way **when you force the task closed-ended** — e.g. sentiment classification (§11, SST-2) collapses the output to Positive/Negative, so plain **accuracy** applies. The eval difficulty scales with how open-ended you let the output be. `(certain)`

---

## 3. What You Evaluate — Model vs Application (Components)

Two levels, and conflating them is a classic mistake.

```
MODEL EVALUATION (is GPT-4 better than Llama-3?)   PRODUCT / SYSTEM EVALUATION (is MY app good?)
   → benchmarks: MMLU, HumanEval, TruthfulQA          → end-to-end task success on YOUR data, plus each
   → generic, task-agnostic, public                     COMPONENT: prompt, RAG retrieval, workflow, API,
                                                         business logic, safety guardrails
```

**The LLM systems you'd be evaluating** (instructor's four types — complexity rising):
```
1. SIMPLE PROMPT-BASED   input → prompt → LLM → output          e.g. product-description generator
2. RAG                   input → retrieve docs → LLM → answer   e.g. support bot over a knowledge base
3. AGENTIC               multi-step reasoning + tools + planning + memory   e.g. auto-debugging assistant
4. MULTI-PROMPT CHAINS   prompts run in sequence                e.g. generate → improve style → fact-check
```
The more components a system has, the more you must evaluate each **stage** separately, not just the final output. A real AI app is a **pipeline**, and end-to-end score alone can't tell you *where* it failed:

```
user query
   │
   ├─▶ [prompt / router]        → did it pick the right task/tool?
   ├─▶ [retriever]  (if RAG)    → did it fetch the RIGHT context?      (retrieval quality)
   ├─▶ [LLM generation]         → given that context, is the answer correct, grounded, well-formed?
   └─▶ [output parser / guard]  → valid JSON? safe? within policy?
                    │
                    ▼
             end-to-end answer  → overall task success + user satisfaction
```
If the end-to-end answer is wrong, component evals tell you whether the **retriever** fed garbage or the **LLM** ignored good context — you can't fix what you can't localize. `(certain)`

### High-level: evaluating a RAG *system* vs evaluating an LLM

You haven't covered [[RAG]] yet, so just the shape of the difference (deep-dive belongs in a future RAG note):

```
Evaluating an LLM alone        →  ONE thing to judge: the generated text.
                                  "Given the prompt, is the output good?"

Evaluating a RAG SYSTEM        →  TWO stages to judge separately, because either can fail:
   1. RETRIEVAL — did it fetch relevant, sufficient context?   (a search problem)
   2. GENERATION — is the answer FAITHFUL to that context, i.e.
      grounded in it and not hallucinated, and does it answer the question?
```

The new axis RAG introduces is **faithfulness / groundedness**: not "is the answer true in general?" but "is it *supported by the retrieved context?*" A perfectly true answer the model knew from pretraining but that *isn't* in the retrieved docs still signals a retrieval failure. 🎯 *"For a bare LLM you ask 'is the answer good?'; for RAG you also ask 'did we retrieve the right context, and is the answer faithful to it?' — retrieval and grounding are extra failure modes a standalone LLM doesn't have."* (Frameworks like RAGAS score exactly this — faithfulness, answer relevance, context precision/recall — you'll meet them with [[RAG]].) `(likely)`

---

## 4. Objectives & Goals of LLM Evaluation

At bottom, every evaluation answers one of two kinds of question (instructor's split) — the goal is output that is **accurate, safe, and aligned with user needs**:
```
CAPABILITY questions        RISK questions
   Is it correct?              Is it safe?
   Is it useful?               Is it leaking data?
   Is it complete?             Is it biased / toxic?
```

*Why* you evaluate — each goal picks different methods:

1. **Model selection** — pick the best model *for your task* (not the top of a public leaderboard). Objective, comparative eval on your own dataset.
2. **Regression testing / CI for prompts** — a prompt tweak or model upgrade must not silently break existing behavior. Fixed eval set → automated metrics on every change.
3. **Quality assurance before shipping** — meet a bar on correctness, safety, format across a representative test set.
4. **Continuous improvement loop** — eval is the feedback signal that tells you *what to fix*; without it you're tuning prompts blind.
5. **Guardrails & safety sign-off** — prove the system won't emit toxic/unsafe/PII-leaking output (§10).
6. **Monitoring in production** — catch drift, degradation, and new failure modes on live traffic (§13).

🎯 *"Evaluation isn't a one-time score — it's the test suite of an AI system: model selection up front, regression gating on every change, and monitoring in production."*

---

## 5. Challenges in Evaluating an LLM

Why this is genuinely hard (interviewers want the list):

- **No single ground truth.** Open-ended tasks (summarize, chat, write) have many valid answers → `==` and even n-gram overlap under-credit correct paraphrases.
- **Multi-dimensional quality.** One output can be factually right but verbose, or fluent but hallucinated. A scalar can't capture correctness + faithfulness + fluency + safety + format at once.
- **Non-determinism.** Sampling (temperature > 0) means the *same prompt scores differently across runs* → you need multiple runs, fixed seeds/`temperature=0`, and statistical treatment, not a single number.
- **Prompt sensitivity.** Reword the prompt or few-shot examples and the score swings — so you're partly evaluating the *prompt*, not the model.
- **Huge input diversity.** Real users bring typos, multiple languages, ambiguous requests, and even jailbreak / adversarial prompts → your test set must *cover many scenarios*, not just the happy path, or the eval lies about production.
- **Subjectivity.** "Helpful," "coherent," "on-brand" resist formulas; humans disagree (low inter-annotator agreement).
- **Reference cost.** Gold answers are expensive to write and go stale; many real tasks never have them.
- **Data contamination.** If the benchmark leaked into pretraining, a high score measures *memorization*, not capability (§9).
- **The judge problem.** Using an LLM to grade LLMs is scalable but inherits the judge's biases (§8).
- **Cost & latency of eval itself.** Running big test sets through large models (or LLM judges) is slow and expensive → you trade coverage vs cost.

---

## 6. Evaluation Methods — Reference-based vs Reference-less

The core taxonomy your metrics slot into.

```
REFERENCE-BASED  (a.k.a. ground-truth / referenced)
   You HAVE a gold answer. Score = similarity(output, reference).
   Metrics: Exact Match, F1, Accuracy, BLEU, ROUGE, BERTScore, METEOR, COMET.
   ✅ objective, cheap, reproducible, automatable in CI
   ❌ needs labeled data; punishes valid answers that differ in wording;
      impossible for open-ended tasks with no single "right" answer.

REFERENCE-LESS  (a.k.a. reference-free)
   You have NO gold answer. Score = quality(output | input, criteria).
   Methods: LLM-as-Judge, perplexity, learned quality models, heuristics
            (valid JSON? contains required field? within length?), human rating.
   ✅ scales to open-ended generation & live production where refs don't exist
   ❌ subjective; judge/human bias; can be costly (a judge call per example).
```

Rule of thumb: **closed-ended or has a canonical answer → reference-based; open-ended or in production → reference-less.** Most serious eval stacks use *both* — reference-based on a labeled regression set, reference-less (judge + human) on open-ended and live traffic.

---

## 7. Reference-based Metrics — BLEU · ROUGE · BERTScore · EM/F1 · Accuracy · Perplexity

All answer "how close is the output to the reference?" — they differ in *what* "close" means, on a ladder of strictness:

```
exact string  →  overlapping words (n-grams)  →  meaning (embeddings)
   EM              BLEU / ROUGE / F1               BERTScore
 (harshest)        (surface overlap)             (semantic — most forgiving)
```
Every metric below is worked end-to-end on a tiny example so the formula turns into something you can *see*.

### 7.1 Accuracy (closed-ended tasks — SST-2)
When you force the output into fixed classes, LLM eval *is* classic ML eval: `accuracy = correct / total`.
```
Task: classify 5 movie reviews.   (SST-2-style, §11)
gold  = [pos, neg, pos, pos, neg]
LLM   = [pos, neg, NEG, pos, neg]        ← 3rd one wrong (said neg, truth pos)
accuracy = 4 / 5 = 0.80
```
Use precision/recall/F1 instead when classes are imbalanced (e.g. 95% negative — a model that always says "negative" scores 0.95 accuracy but is useless). This is exactly the SST-2 sentiment demo.

### 7.2 Exact Match (EM) & token-F1 (extractive QA — SQuAD)
SQuAD's **official** metrics (not BLEU/ROUGE):
- **EM** = fraction of predictions that match a gold answer *exactly* after normalization (lowercase, strip punctuation & articles a/an/the). Binary, harsh.
- **F1** = token-overlap F1 between prediction and gold: `precision = shared / |pred tokens|`, `recall = shared / |gold tokens|`, `F1 = 2PR/(P+R)`; take the **max over gold answers**, average over the set.

```
question: "Who won Super Bowl 50?"
gold = "Denver Broncos"            pred = "the Denver Broncos won"
                                   → normalize: drop "the" → "denver broncos won"

EM = 0                             (extra token "won" → not an exact match)
F1:  pred tokens = {denver, broncos, won} (3)   gold tokens = {denver, broncos} (2)
     shared = {denver, broncos} = 2
     precision = 2/3 = 0.67   recall = 2/2 = 1.0
     F1 = 2·(0.67·1.0)/(0.67+1.0) = 1.33/1.67 = 0.80
```
🎯 EM is unforgiving (one extra word → 0); **F1 gives partial credit** for overlapping spans → 0.80. Report both.

### 7.3 BLEU — precision of n-gram overlap (machine translation)
**BLEU** = **B**i**l**ingual **E**valuation **U**nderstudy. It measures **precision**: of the n-grams the candidate produced, how many appear in the reference?
```
BLEU = BP · exp( Σₙ wₙ · log pₙ )         n = 1..4 (unigram→4-gram), wₙ = 1/4
  pₙ = CLIPPED n-gram matches / n-grams in candidate      (a candidate n-gram can't be counted more times than it appears in the reference)
  BP = brevity penalty = 1 if c>r else exp(1 − r/c)       (only docks output that's too SHORT; c,r = candidate/reference length)
```

**Worked example** (the instructor's — one word swapped, `sat → slept`):
```
Generated (candidate): "cat sat on the mat"
Reference:             "cat slept on the mat"
```
| n-gram | candidate n-grams | matches in reference | precision pₙ |
|---|---|---|---|
| 1 (unigram) | cat, sat, on, the, mat | cat, on, the, mat | **4/5 = 0.80** |
| 2 (bigram)  | cat·sat, sat·on, on·the, the·mat | on·the, the·mat | **2/4 = 0.50** |
| 3 (trigram) | cat·sat·on, sat·on·the, on·the·mat | on·the·mat | **1/3 = 0.33** |
| 4 (4-gram)  | cat·sat·on·the, sat·on·the·mat | — | **0/2 = 0.00** |

- Lengths are equal (5 = 5) → **BP = 1** (no brevity penalty here).
- BLEU **multiplies** the precisions (geometric mean), so the single **p₄ = 0 collapses standard BLEU-4 to 0**. That's why real BLEU applies **smoothing** — add a tiny constant (e.g. `0.01`) to zero counts so one missing 4-gram doesn't nuke the score. Smoothed, this lands around the geometric mean of the non-zero precisions ≈ **0.5**.

**Rough interpretation bands** (the instructor's rule of thumb):
```
0.00–0.25  poor      0.25–0.50  okay      0.50–0.75  good/strong      0.75–1.00  near-identical
```
🎯 BLEU = **precision + a brevity penalty** — "of what you generated, how much matches, and don't reward being too short." Corpus-level; built for translation.

### 7.4 ROUGE — recall of n-gram overlap (summarization)
**ROUGE** = **R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation — the flip of BLEU. It measures **recall**: of the reference's n-grams, how many did the candidate cover? Variants:
```
ROUGE-1    unigram overlap            ROUGE-2    bigram overlap
ROUGE-L    Longest Common Subsequence (order-aware, gap-tolerant)
ROUGE-Lsum ROUGE-L aggregated sentence-by-sentence over a whole summary
```

**Worked example** (the instructor's — an *extra-word* candidate):
```
Reference:             "the cat sat on the mat"            (6 unigrams)
Generated (candidate): "the cat sat on the mat happily"   (7 unigrams — one extra word)
```
- **ROUGE-1 recall** = matched / *reference* unigrams = **6/6 = 1.0** — the candidate **covered the whole reference**.
- **ROUGE-1 precision** = matched / *candidate* unigrams = 6/7 = **0.86** — docked for the extra "happily".
- **ROUGE-2 recall** = 5/5 = 1.0;  **ROUGE-L recall** = LCS("the cat sat on the mat") = 6 → 6/6 = **1.0**.

**Precision vs recall as FP/FN** (the way the instructor drew it):
```
   Precision = TP / (TP + FP)                 Recall = TP / (TP + FN)
   FP = a token the model GENERATED that is NOT in the reference  → hurts PRECISION (BLEU)   e.g. "happily"
   FN = a reference token the model did NOT generate              → hurts RECALL   (ROUGE)
```
🎯 **Same idea, opposite lens:** BLEU asks *"was what you said right?"* (precision — punishes extra/wrong words); ROUGE asks *"did you cover what mattered?"* (recall — punishes missing words). Summarization uses ROUGE because covering the reference is what counts.

### 7.5 BERTScore — semantic similarity + IDF weighting
BLEU/ROUGE only match **identical tokens**, so a correct *paraphrase* scores low. BERTScore compares **meaning**: embed every token with a contextual model (BERT), then greedily match each token to its most-similar token in the other sentence by **cosine similarity**.

**Worked example** (the instructor's paraphrase — `cat→feline`, `mat→carpet`):
```
Reference:             "cat sat on the mat"
Generated (candidate): "feline sat on the carpet"
```
Keyword metrics first: only *sat, on, the* overlap (cat≠feline, mat≠carpet) → BLEU/ROUGE drop sharply **even though the meaning is identical**. BERTScore instead sees the embeddings are close:
```
cat↔feline ≈ 0.80,  mat↔carpet ≈ 0.80,  sat↔sat = 1.0,  on↔on = 1.0,  the↔the = 1.0

Recall    = (Σ_{x∈ref}  max_{y∈cand} cos(x,y)) / |ref|     (each REFERENCE token → best match in candidate)
Precision = (Σ_{y∈cand} max_{x∈ref}  cos(x,y)) / |cand|
F1        = 2PR/(P+R)

Unweighted recall = (0.80 + 1.0 + 1.0 + 1.0 + 0.80) / 5 = 4.6/5 = 0.92     ← paraphrase correctly credited
```

**IDF weighting (what your instructor stressed).** The 0.92 was propped up by *perfect but meaningless* matches on common words (`sat/on/the` = 1.0). Weight each token by how **rare/informative** it is, using **inverse document frequency**:
```
idf(w) = log( N / c(w) )     N = # reference docs,  c(w) = # docs that contain w
   → common words (the, on) appear everywhere → LOW idf;   rare content words (feline, carpet) → HIGH idf

IDF-weighted recall = Σ_x idf(x)·max_y cos(x,y)  /  Σ_x idf(x)
   idf:  cat=0.9, sat=0.2, on=0.1, the=0.1, mat=0.9        ← illustrative weights
   = (0.9·0.80 + 0.2·1.0 + 0.1·1.0 + 0.1·1.0 + 0.9·0.80) / (0.9 + 0.2 + 0.1 + 0.1 + 0.9)
   = (0.72     + 0.20    + 0.10    + 0.10    + 0.72)     / 2.2
   = 1.84 / 2.2
   = 0.84
```
🎯 **What IDF did:** the score moved 0.92 → **0.84** because it stopped rewarding the trivial stopword matches and now reflects how well the **content** words (cat→feline, mat→carpet) were captured. **IDF stops a model gaming BERTScore by nailing common words** — rare, meaning-bearing tokens count most. (Precision is computed the same way with candidate-side IDF; F1 combines them.) `(certain)`

### 7.6 The three side by side — why they disagree

| Example (Reference → Candidate) | BLEU (precision) | ROUGE-1 (recall) | BERTScore (meaning) |
|---|---|---|---|
| "cat slept on the mat" → "cat sat on the mat" (one word swapped) | ~0.5 | 0.80 | ~0.95 |
| "cat sat on the mat" → "feline sat on the carpet" (paraphrase) | ~0.3 | 0.60 | 0.92 → 0.84 (IDF) |

🎯 **The lesson of the whole metrics unit:** n-gram metrics (BLEU/ROUGE) are cheap and reproducible but **blind to meaning** — they punish valid paraphrases and reward lexical mimicry. BERTScore captures semantics (and IDF makes it weight the words that matter) but still needs a reference and can over-credit fluent-but-wrong text. **None of them judge factual correctness** — which is why open-ended and production eval moves to LLM-as-Judge (§8).

### 7.7 Perplexity — the intrinsic LM metric (know it exists)
`PPL = exp( −(1/N) Σ log P(xᵢ | x_<ᵢ) )` — how "surprised" the model is by held-out text; lower = better.
```
If the model assigns the true next token an average probability of 0.1:
   PPL = exp(−ln 0.1) = exp(2.30) = 10  →  "as unsure as picking among ~10 equally-likely words each step."
```
**Caveat:** it's **tokenizer-dependent** (can't compare across models with different vocabularies) and measures fluency/likelihood, *not* task correctness. Rarely the headline metric for an app; useful for pretraining/domain-fit checks. `(certain)`

---

## 8. LLM-as-Judge — the Reference-less Flagship

Use a strong LLM to grade another LLM's output against a rubric — the only method that scales to open-ended quality with **no reference**. The instructor's mental model: **content creation vs critique** — the model being judged is the *player*; the judge LLM is the *commentator/critic*. Same domain skill, different job.
```
Candidate LLM  →  generates output  →  Judge LLM  →  score / verdict
```

### 8.1 Three evaluation scenarios
```
1. SINGLE OUTPUT (binary / pointwise)  → grade one answer: "Is it safe? correct? polite?"  → score or yes/no
2. PAIRWISE COMPARISON (A vs B)        → "Which output is better, A or B?"  → preference (powers MT-Bench / Arena)
3. FULL-CONVERSATION evaluation        → grade a whole multi-turn dialogue, not just one turn
```
Two ways to give the judge a target:
- **Reference-free** — judge from criteria alone (safe? relevant? hallucinated?). No gold answer needed.
- **With reference / context** — also hand the judge a source document and ask "is the answer *supported* by this?" → this is how you do **hallucination detection**.

### 8.2 The 4-step process (how you actually build one)
```
1. DEFINE the evaluation scenario   → single / pairwise / full-conversation (§8.1)
2. PREPARE an evaluation dataset    → real + SYNTHETIC examples, chosen to be REPRESENTATIVE of production
3. CREATE the evaluation prompt     → the rubric + scale as a prompt template — this prompt IS your metric
4. EVALUATE & ITERATE               → run the judge on a LABELLED set, compare to ground truth, refine the prompt
                                       (calibrating the judge against gold labels this way is the "G-Eval" idea)
```
Step 3 is where **[[Prompt Engineering]]** pays off — the rubric prompt *is* the metric, so a clear scale, few-shot examples, and chain-of-thought make the judge sharper (that's the natural next topic after this lecture).

### 8.3 Three judge metrics from the demo (Opik built-ins)
The instructor's live judge used three concrete metrics — memorize the **ranges and directions**:

| Metric | Range | Good direction | Asks |
|---|---|---|---|
| **Relevance** (Answer-Relevance) | 0–1 | **higher** | does the answer actually address the prompt? |
| **Hallucination** | 0–1 | **lower** | is anything unsupported / made up? |
| **Moderation** | binary | safe = pass | explicit content? hate? unsafe / policy violation? |

(Opik ships these as `AnswerRelevance`, `Hallucination`, `Moderation` — see §11. A judge can also test a *reasoning* claim, e.g. "is `2 + 2 = 5` relevant/correct?" — the score should catch it.)

### 8.4 Judge biases you MUST name (wins the follow-up)
- **Position bias** — favors whichever answer is shown first (or second). *Fix:* run both orders, require consistency.
- **Verbosity (length) bias** — prefers longer answers regardless of quality. *Fix:* control for length.
- **Self-enhancement bias** — a model rates *its own* outputs higher. *Fix:* judge ≠ generator, or use a panel.
- **Sycophancy / format bias** — swayed by confident tone or nice formatting over substance.

### 8.5 Making judges reliable
Give an explicit rubric + scale; use a **strong** judge model; **calibrate against human labels** and report agreement (Cohen's κ); for high stakes use a **panel/jury of judges** and majority-vote. 🎯 *"An LLM judge is only trustworthy once you've measured its agreement with humans and de-biased for position and verbosity — otherwise you've automated a biased grader."* Trade-off: flexible and cheap-ish vs humans, but adds **cost + latency** (an extra LLM call per example) and can be gamed. `(certain)`

---

## 9. Benchmarks & Leaderboards

Standardized *test suites* for **model-level** comparison — the vocabulary of every model release:

| Benchmark | Measures | Scoring |
|---|---|---|
| **MMLU** | Broad knowledge, 57 subjects (multiple-choice) | Accuracy |
| **HellaSwag / ARC / WinoGrande** | Commonsense reasoning | Accuracy |
| **TruthfulQA** | Resistance to falsehoods | Accuracy / truthful% |
| **GSM8K / MATH** | Math word problems | Exact answer match |
| **HumanEval / MBPP** | Code generation | **pass@k** (fraction where ≥1 of k samples passes unit tests) |
| **IFEval** | Instruction following | Rule-based pass rate |
| **MT-Bench** | Multi-turn chat quality | LLM-as-judge (pairwise) |
| **Chatbot Arena** | Human preference, head-to-head | **Elo / Bradley-Terry** ranking |
| **HELM / BIG-bench** | Holistic multi-metric suites | Many axes |

**The two things to say about benchmarks:**
- **Contamination** — if test items leaked into pretraining, the score measures **memorization**, not capability. Suspiciously high numbers + public benchmarks = check for leakage.
- **Saturation & Goodhart** — models are tuned *to* popular benchmarks, so scores cluster near the ceiling and stop discriminating. Human-preference **Arena (Elo)** and *your own private eval set* resist this better. 🎯 *"Leaderboards rank models in general; only your own eval set ranks them for your task — and watch for contamination."*

---

## 10. Beyond Overlap — Safety, Bias, Hallucination & Cost/Latency

Quality ≠ correctness alone. Ship-blocking dimensions the overlap metrics never touch:

- **Hallucination / factuality** — fluent, confident, wrong. Detect via faithfulness checks (grounded in source?), fact-verification, or judge prompts. The #1 open-ended failure mode.
- **Toxicity & safety** — offensive/unsafe content (classifiers like Perspective API; moderation endpoints).
- **Bias & fairness** — systematic differences across gender/race/etc. (e.g. BBQ, stereotype sets).
- **Robustness / red-teaming** — jailbreaks, **prompt injection**, adversarial inputs that override the system prompt. Evaluate by *attacking* the system, not just testing happy paths.
- **PII / data leakage** — does it emit secrets or training data?
- **Format validity** — for structured output: JSON parses? schema valid? function-call args correct? (Cheap deterministic checks — always include them.)

**Non-quality axes (first-class in production):** **cost** ($/1M tokens), **latency** (time-to-first-token, tokens/sec), **throughput**, **context length**. A slightly worse model that's 10× cheaper and 5× faster often *wins* the deployment. 🎯 *"Eval isn't just 'is it right?' — it's 'is it right, safe, well-formed, fast, and affordable enough to ship?'"*

---

## 11. Code / Implementation (SST-2, SQuAD, Opik multi-LLM)

### 11.1 Reference-based metrics with HuggingFace `evaluate`
```python
import evaluate

# ---- Sentiment / SST-2: closed-ended → ACCURACY (classic ML eval on an LLM) ----
acc = evaluate.load("accuracy")
# You prompt each LLM: "Classify sentiment as Positive/Negative: <sentence>", parse to 1/0
preds  = [1, 0, 1, 1]          # parsed from LLM text output
labels = [1, 0, 0, 1]          # SST-2 gold labels
print(acc.compute(predictions=preds, references=labels))   # {'accuracy': 0.75}

# ---- Q&A / SQuAD: official metrics are EXACT MATCH + F1 (not BLEU/ROUGE) ----
squad = evaluate.load("squad")
predictions = [{"id": "1", "prediction_text": "Denver Broncos won"}]
references  = [{"id": "1", "answers": {"text": ["Denver Broncos"], "answer_start": [0]}}]
print(squad.compute(predictions=predictions, references=references))  # {'exact_match': 0.0, 'f1': 80.0}

# ---- Generation quality: compare generated output vs reference (the §7 paraphrase pair) ----
ref  = [["cat sat on the mat"]]                  # BLEU takes a LIST of references per prediction
cand = ["feline sat on the carpet"]
print(evaluate.load("bleu").compute(predictions=cand, references=ref))       # low  (cat≠feline, mat≠carpet)
print(evaluate.load("rouge").compute(predictions=cand, references=[r[0] for r in ref]))  # ROUGE-1 ~0.6
print(evaluate.load("bertscore").compute(predictions=cand,
        references=[r[0] for r in ref], lang="en"))                          # F1 ~0.92  ← semantics win
```

### 11.2 LLM-as-Judge (reference-less) — the prompt is the metric
```python
JUDGE_PROMPT = """You are grading an answer. Rate 1-5 for CORRECTNESS and HELPFULNESS.
Think step by step, then output strictly: {{"reasoning": "...", "score": <1-5>}}

Question: {question}
Answer:   {answer}"""
# Call a strong model with temperature=0, parse the JSON score.
# De-bias: for pairwise, run BOTH orders (A,B) and (B,A) and require agreement.
```

### 11.3 Opik — evaluate multiple LLMs over a dataset (the instructor's multi-LLM demo)
Opik (open-source, by Comet) = **datasets + experiments + built-in metrics/judges + a comparison UI**. Shape of it (API sketch):
```python
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Equals, AnswerRelevance, Hallucination, Moderation  # heuristic + LLM-judge

client  = Opik()
dataset = client.get_or_create_dataset(name="sst2-eval")
dataset.insert([{"input": "The movie was a masterpiece.", "expected_output": "Positive"}, ...])

def task(item, model):                          # run ONE model on one dataset row
    out = call_llm(model, item["input"])        # your model call
    return {"output": out}

# The instructor's four models: Llama-3.1 & Llama-3.3 via a hosted API (Groq), Phi & Qwen via HuggingFace
for model in ["llama-3.1-8b", "llama-3.3-70b", "phi-3", "qwen-2.5"]:   # SAME dataset, MANY LLMs
    evaluate(
        dataset=dataset,
        task=lambda item, m=model: task(item, m),
        scoring_metrics=[Equals(), AnswerRelevance(), Hallucination(), Moderation()],  # reference-based + judge
        experiment_name=f"sst2-{model}",
    )
# Opik logs per-example traces + aggregate scores → compare models side by side in the UI.
```
The pattern: **one dataset, many models, the same metric suite, compared in one place** — exactly what "evaluate multiple LLMs" means in practice. `(likely — Opik's API surface shifts across versions; the workflow is stable)`

---

## 12. When It Breaks — Metric & Eval Pitfalls

```
❌ BLEU/ROUGE on open-ended text → punish valid paraphrases, reward lexical mimicry (see §7.6).
❌ Reporting one run → sampling variance makes it noise; use temperature=0 or average N runs + CIs.
❌ Trusting a single leaderboard number → contamination + saturation; evaluate on YOUR data.
❌ Unbiased-judge assumption → position/verbosity/self-enhancement bias inflate scores (§8).
❌ Only end-to-end score on a pipeline → can't localize which component failed (§3).
❌ Wrong metric for the task → BLEU on SQuAD, accuracy on open-ended chat, EM on a summary.
❌ No factuality/safety check → high BERTScore on a fluent hallucination; overlap ≠ truth.
❌ Tiny/unrepresentative eval set → passes CI, fails on real traffic; curate a "golden set" that mirrors prod.
❌ Comparing perplexity across different tokenizers → not comparable.
```

---

## 13. Production & Online Evaluation

Offline metrics ≠ live behavior. The full loop:

```
OFFLINE (pre-ship)                          ONLINE (post-ship, on real traffic)
  golden eval set + reference-based           implicit feedback: 👍/👎, regeneration rate,
  metrics + LLM-judge, run in CI on            edit distance of accepted answer, task completion
  every prompt/model change (regression)      A/B tests between prompt/model variants
                                              LLM-judge sampled on live outputs
                                              guardrails + drift & quality monitoring
```

- **Implicit signals** are your cheapest, largest-scale reference-less metric — thumbs, "regenerate" clicks, and whether users edit or accept the output.
- **A/B testing** is the production gold standard: route traffic to variant A vs B, measure a real outcome (resolution rate, conversion), test for **statistical significance**.
- **Guardrails** run *inline* (block toxic/PII/off-policy output before it reaches the user) — eval that's also enforcement.
- **Drift & monitoring** — input distribution and quality shift over time (new topics, model provider updates); sample-and-judge live traffic to catch degradation.
- **Observability tooling** — **Opik**, LangSmith, Phoenix, Langfuse: log every trace, attach metrics/judges, dashboard the aggregates.
- **Statistical rigor** — because outputs are stochastic, report **confidence intervals / bootstrap** and check that a "win" is significant, not sampling noise.

🎯 *"Offline eval gates the release; online eval (implicit feedback + A/B + monitoring) is the only thing that proves it works on real users."*

---

## 14. Interview Lens

> ⚡ **Golden rule:** match the metric to the **task shape** and the **presence of a reference** — the wrong metric is the classic tell.

**"How is LLM evaluation different from ML evaluation?"** → §2: open-ended output, many valid answers / no single ground truth, multi-dimensional quality, non-deterministic — so one accuracy number is usually wrong.

**"BLEU vs ROUGE vs BERTScore?"** → BLEU = n-gram *precision* (translation); ROUGE = n-gram *recall* (summarization); both are surface-overlap and blind to paraphrase; BERTScore = embedding *cosine similarity* → captures meaning. All three need a reference; none judge factual truth.

**"Reference-based vs reference-less?"** → have a gold answer → EM/F1/BLEU/ROUGE/BERTScore (objective, cheap, needs labels); no gold answer / open-ended / production → LLM-as-Judge / perplexity / heuristics / human.

**"What metric for extractive QA / for sentiment?"** → SQuAD → **Exact Match + token-F1**; SST-2 → **accuracy** (forcing closed-ended output makes classic metrics apply).

**"Problems with LLM-as-Judge?"** → position, verbosity, self-enhancement, sycophancy bias; fix by swapping order, controlling length, judge≠generator, panel-of-judges, and calibrating agreement with humans.

**"How would you set up an LLM judge?"** → pick a scenario (single/pairwise/full-conversation), prepare a representative (incl. synthetic) dataset, write the rubric as a prompt, then calibrate against gold labels (G-Eval). Common metrics: Relevance (↑), Hallucination (↓), Moderation (safe/unsafe).

**"Why not just trust MMLU / the leaderboard?"** → contamination (memorization) + saturation (Goodhart); evaluate on your own private set and human-preference Arena.

**"How do you evaluate an AI app, not just a model?"** → component-wise down the pipeline (retriever, generation, parser) + end-to-end + online (implicit feedback, A/B, monitoring).

---

## 15. Alternatives & How to Choose a Metric

**The instructor's decision tree** — walk it top-down to land on the right metric:
```
                        ┌─ Do we have a REFERENCE (ground truth)? ─┐
                       YES                                         NO
                        │                                          │
        ┌─ Only ONE correct answer? ─┐              ┌─ Objective QUALITY criteria exist? ─┐
       YES                          NO             YES                                    NO
        │                           │               │                                     │
 ┌ Predictive task? ┐     ROUGE · BLEU ·      Valid-code checks ·              LLM-AS-A-JUDGE
YES               NO       BERTScore ·        structured-output / JSON          (context, readability,
 │                 │       LLM-as-Judge         validation                       length, tone…)
Hit-rate ·    Exact Match ·  (+ BLEURT)
Precision ·   string match ·
Recall · F1   keyword match
```

Same thing as a table — pick by task shape:

| Task | Primary metric | Why |
|---|---|---|
| Classification / sentiment (SST-2) | **Accuracy, F1** | Closed-ended → classic ML eval applies |
| Extractive QA (SQuAD) | **Exact Match + token-F1** | Short canonical spans; F1 gives partial credit |
| Machine translation | **BLEU**, COMET/BLEURT | Overlap works; learned metrics correlate better with humans |
| Summarization | **ROUGE** + BERTScore + faithfulness | Recall of key content + semantic + not-hallucinated |
| Open-ended chat / writing | **LLM-as-Judge + human/Arena** | No reference; multi-dim quality |
| Code generation | **pass@k** | Correctness = passes unit tests, not text overlap |
| RAG app | retrieval metrics + **faithfulness** | Two failure stages ([[RAG]]) |
| Production system | **implicit feedback + A/B + online judge** | Real users, no static reference |

**Meta-rule:** prefer the **cheapest metric that's valid for the task**, and *layer* — deterministic checks (format/EM) → overlap/semantic (F1/BERTScore) → judge → human — using the expensive ones only where the cheap ones can't discriminate. 🎯 *"There is no universal LLM metric; there's a metric per task shape, and a good eval stack combines several."*

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. Give the one-sentence reason LLM eval is harder than classic ML eval.
   <details><summary>answer</summary>Outputs are open-ended text with many valid forms and often no single reference, quality is multi-dimensional (correct + fluent + faithful + safe), and generation is non-deterministic — so a single accuracy number can't capture it.</details>
2. Define reference-based vs reference-less and give two metrics of each.
   <details><summary>answer</summary>**Reference-based:** compare to a gold answer — EM, F1, BLEU, ROUGE, BERTScore, accuracy. **Reference-less:** judge quality with no gold answer — LLM-as-Judge, perplexity, heuristics (valid JSON?), human rating.</details>
3. BLEU vs ROUGE vs BERTScore — the one-line distinction for each.
   <details><summary>answer</summary>BLEU = n-gram **precision** (translation); ROUGE = n-gram **recall** (summarization); BERTScore = **cosine similarity of contextual embeddings** → captures meaning, not just surface overlap.</details>
4. Reference "cat sat on the mat", candidate "feline sat on the carpet": why does BERTScore ≫ BLEU, and what does IDF weighting change?
   <details><summary>answer</summary>Only *sat, on, the* overlap as exact tokens (cat≠feline, mat≠carpet) → BLEU/ROUGE drop sharply — n-gram metrics are blind to paraphrase. BERTScore embeds tokens, so cat≈feline (0.80) and mat≈carpet (0.80) match → recall ~0.92. **IDF weighting** (`idf(w)=log(N/c(w))`) down-weights the trivial stopword matches (sat/on/the = 1.0) and up-weights rare content words (feline, carpet), dropping it to ~0.84 so the score reflects meaning-bearing tokens, not stopword mimicry.</details>
5. What are the official SQuAD metrics, and why report both?
   <details><summary>answer</summary>**Exact Match** (harsh, binary) and **token-level F1** (partial credit for overlapping spans). EM alone under-credits answers that are correct but include extra/missing tokens.</details>
6. Name three LLM-as-Judge biases and a fix for each.
   <details><summary>answer</summary>**Position** (swap order / average), **verbosity/length** (control for length), **self-enhancement** (judge ≠ generator / panel). Also calibrate the judge's agreement with human labels.</details>
7. Why not rely on public benchmarks alone?
   <details><summary>answer</summary>**Contamination** (test data leaked into pretraining → measures memorization) and **saturation/Goodhart** (models tuned to the benchmark). Evaluate on your own private set + human-preference Arena.</details>
8. High-level: what does evaluating a RAG system add over evaluating a bare LLM?
   <details><summary>answer</summary>Two stages instead of one — **retrieval** (did it fetch the right context?) and **generation faithfulness/groundedness** (is the answer supported by that context, not hallucinated?). A standalone LLM has neither failure mode.</details>
9. Offline vs online evaluation — what does each catch?
   <details><summary>answer</summary>**Offline:** reference-based + judge on a golden set in CI → gates releases, catches regressions. **Online:** implicit feedback (👍/👎, regen), A/B tests, live judging, drift monitoring → proves it works on real traffic.</details>
10. Name the three LLM-as-Judge evaluation scenarios and the three judge metrics from the demo (with their good direction).
    <details><summary>answer</summary>**Scenarios:** single-output / binary, pairwise (A vs B), full-conversation. **Metrics:** Relevance 0–1 (higher better), Hallucination 0–1 (**lower** better), Moderation (binary — safe/unsafe). Build a judge in 4 steps: define scenario → prepare representative dataset → create the eval-prompt rubric → evaluate & iterate against gold labels.</details>
11. In the instructor's BLEU example ("cat sat on the mat" vs ref "cat slept on the mat"), why can standard BLEU-4 read 0, and what's the fix?
    <details><summary>answer</summary>Unigram precision 4/5, bigram 2/4, trigram 1/3, but **4-gram = 0/2 = 0**; BLEU multiplies the precisions, so one zero collapses the product to 0. Fix = **smoothing** (add a small constant like 0.01 to zero counts). Rough bands: <0.25 poor, 0.25–0.5 okay, 0.5–0.75 good, 0.75–1 near-identical.</details>

---

*Covers: LLM vs ML evaluation · model vs product evaluation · the 4 LLM system types · component/pipeline eval (+ high-level RAG-vs-LLM contrast) · objectives (capability vs risk) · challenges (incl. input diversity) · reference-based vs reference-less methods · metrics (accuracy, EM/F1, BLEU, ROUGE, BERTScore + IDF, perplexity) each worked on the instructor's examples, with BLEU interpretation bands & smoothing · LLM-as-Judge — 3 scenarios, 4-step build, Relevance/Hallucination/Moderation metrics, biases · benchmarks + contamination/saturation · safety/bias/hallucination + cost/latency · SST-2 / SQuAD / Opik 4-model code · metric pitfalls · production & online eval · interview lens · metric-choice decision tree.*
