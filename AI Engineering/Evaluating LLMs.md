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
MODEL-LEVEL (is GPT-4 better than Llama-3?)      APPLICATION / SYSTEM-LEVEL (is MY app good?)
   → benchmarks: MMLU, HumanEval, MT-Bench          → end-to-end task success on YOUR data
   → generic, task-agnostic, public                 → plus each COMPONENT of the pipeline
```

A real AI app is a **pipeline**, and end-to-end score alone can't tell you *where* it failed. You evaluate **component by component**:

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
```
BLEU = BP · exp( Σₙ wₙ · log pₙ )         typically n=1..4, wₙ = 1/4
  pₙ = CLIPPED n-gram matches / n-grams in candidate      (PRECISION: of what you SAID, how much is right?)
  BP = brevity penalty = 1 if c>r else exp(1 − r/c)       (c=cand len, r=ref len; punishes too-short output)
  "clipped" = a candidate n-gram can't be counted more times than it appears in the reference
```

**Worked example** (used again for ROUGE in §7.4, so you see precision vs recall on the *same* data):
```
Reference (R): "the cat is on the mat"     r = 6 tokens   ("the" appears twice)
Candidate (C): "the cat on the mat"        c = 5 tokens   ← correct but DROPPED the word "is"
```
*Step 1 — unigram precision p₁:* candidate words = {the, cat, on, the, mat}. Every one appears in R (clipped: "the" allowed twice, R has two) → **5 of 5 match** → `p₁ = 5/5 = 1.0`.
*Step 2 — bigram precision p₂:* candidate bigrams = {the·cat, cat·on, on·the, the·mat} (4). In R? the·cat ✓, cat·on ✗ (R has cat·is), on·the ✓, the·mat ✓ → **3 of 4** → `p₂ = 3/4 = 0.75`.
*Step 3 — brevity penalty:* c=5 < r=6 → `BP = exp(1 − 6/5) = exp(−0.2) = 0.82`.
*Step 4 — combine (BLEU-2):* `BP · exp(½·ln1.0 + ½·ln0.75) = 0.82 · 0.866 = ` **`0.71`**.

🎯 **Read it:** the candidate said *nothing wrong* (precision 1.0) yet BLEU is only **0.71** — the brevity penalty docks it for being too short. That penalty is *why* BLEU exists: without it you'd game precision by outputting one word you're sure of. (Real BLEU-4 also multiplies in p₃, p₄ and uses smoothing to avoid a single zero collapsing the score.)

### 7.4 ROUGE — recall of n-gram overlap (summarization)
```
ROUGE-N recall = matching n-grams / n-grams in the REFERENCE   (RECALL: of what SHOULD be there, how much did you COVER?)
ROUGE-L        = based on Longest Common Subsequence (order-aware, gap-tolerant):
                 R = LCS/|ref|,  P = LCS/|cand|,  F = 2PR/(P+R)
```

**Same pair as §7.3** — now measured by recall:
```
R = "the cat is on the mat" (6 unigrams)    C = "the cat on the mat" (5 unigrams)
```
- **ROUGE-1 (recall):** matched unigrams / *reference* unigrams = **5/6 = 0.83**. (C covers everything except "is".)
- **ROUGE-2 (recall):** matched bigrams / reference bigrams = 3/5 = **0.60**.
- **ROUGE-L:** LCS = "the cat on the mat" = 5 tokens → recall = 5/6 = **0.83**, precision = 5/5 = 1.0.

🎯 **Same candidate, different lens:** BLEU (precision + brevity) = **0.71**; ROUGE-1 (recall) = **0.83**. **BLEU asks "was what you said right?"; ROUGE asks "did you cover what mattered?"** — that's the whole BLEU-vs-ROUGE distinction, and here you can see them diverge on one example.

### 7.5 BERTScore — semantic similarity + IDF weighting
The fatal flaw of BLEU/ROUGE: they only match **identical tokens**, so a correct *paraphrase* scores near zero. BERTScore fixes this by comparing **meaning**: embed every token with a contextual model (BERT/RoBERTa), then greedily match each token to its most-similar token in the other sentence by **cosine similarity**.
```
Recall    = (Σ_{x∈ref}  max_{y∈cand} cos(x, y)) / |ref|         (each REFERENCE token → its best match in the candidate)
Precision = (Σ_{y∈cand} max_{x∈ref}  cos(x, y)) / |cand|        (each CANDIDATE token → its best match in the reference)
F1        = 2PR/(P+R)
```

**Worked example** — a pure paraphrase, where n-gram metrics fail:
```
Reference (R): "the dog is running"       [the, dog, is, running]
Candidate (C): "a puppy is sprinting"     [a,  puppy, is, sprinting]
   n-gram check first: only "is" overlaps → ROUGE-1 recall = 1/4 = 0.25, BLEU ≈ 0.  ❌ badly under-credits a correct paraphrase.
```
Contextual cosine similarities (illustrative): `puppy↔dog = 0.85`, `sprinting↔running = 0.88`, `is↔is = 1.0`, `a↔the = 0.90`.

*Unweighted recall* (each R token → best C token):
```
the→a 0.90,  dog→puppy 0.85,  is→is 1.00,  running→sprinting 0.88
Recall = (0.90 + 0.85 + 1.00 + 0.88) / 4 = 3.63/4 = 0.91
```
BERTScore ≈ **0.91** vs n-gram ≈ 0.25 — it correctly sees the paraphrase. `(certain)`

**Now the IDF weighting (what your instructor stressed).** Notice the unweighted 0.91 was propped up by *easy* matches on common words — `is↔is = 1.0` and `a↔the = 0.90` — which carry almost no information. A system could look good just by reproducing stopwords. **IDF (inverse document frequency) fixes this: weight each token by how rare/informative it is** — common words (the, is, a) get a *small* weight, rare content words (dog, running) get a *large* one — so the score depends on matching the words that actually carry meaning:
```
IDF-weighted Recall = Σ_x ( idf(x) · max_y cos(x,y) )  /  Σ_x idf(x)

idf:  the=0.1,  dog=0.8,  is=0.1,  running=0.9     ← stopwords down-weighted, content words up-weighted
    = (0.1·0.90 + 0.8·0.85 + 0.1·1.00 + 0.9·0.88) / (0.1 + 0.8 + 0.1 + 0.9)
    = (0.09    + 0.68     + 0.10     + 0.792   ) / 1.9
    = 1.662 / 1.9
    = 0.87
```
🎯 **What IDF did:** the score dropped 0.91 → **0.87** because it stopped rewarding the trivial `is↔is`/`a↔the` matches and now reflects how well the *content* words (dog→puppy, running→sprinting) were captured. **IDF weighting stops a model from gaming BERTScore with common words** — it makes rare, meaning-bearing tokens count most. (Precision is computed the same way with candidate-side IDF; F1 combines them → also ≈ 0.87 here.)

### 7.6 The three side by side — why they disagree
Running the same two examples through all three:

| Example (Ref → Cand) | BLEU (precision) | ROUGE-1 (recall) | BERTScore (meaning) |
|---|---|---|---|
| "the cat is on the mat" → "the cat on the mat" (dropped a word) | 0.71 | 0.83 | ~0.96 |
| "the dog is running" → "a puppy is sprinting" (paraphrase) | ~0.00 | 0.25 | 0.87 (IDF) |

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

Use a strong LLM to grade outputs against a rubric — the only method that scales to open-ended quality with no reference. Three modes:

```
1. POINTWISE (single-answer grading)  → "Score this answer 1–5 for helpfulness & correctness."
2. PAIRWISE  (A vs B)                  → "Which answer is better, A or B?"  (most reliable; powers MT-Bench/Arena)
3. REFERENCE-GUIDED                    → "Here is a gold answer; grade the candidate against it."  (hybrid)
```

**Judge biases you MUST name** (this wins the follow-up):
- **Position bias** — favors whichever answer is shown first (or second). *Fix:* run both orders, average / require consistency.
- **Verbosity (length) bias** — prefers longer answers regardless of quality. *Fix:* control for length, instruct otherwise.
- **Self-enhancement bias** — a model rates *its own* outputs higher. *Fix:* judge ≠ generator, or use a panel.
- **Sycophancy / format bias** — swayed by confident tone or nice formatting over substance.

**Make judges reliable:** give an explicit rubric + scale; **G-Eval** style (chain-of-thought reasoning then a score, probability-weighted); use a strong judge model; **calibrate against human labels** and report agreement (Cohen's κ); for high stakes use a **panel/jury of judges** and majority-vote. 🎯 *"An LLM judge is only trustworthy once you've measured its agreement with humans and de-biased for position and verbosity — otherwise you've automated a biased grader."*

Trade-off: judges are flexible and cheap-ish vs humans, but add **cost + latency** (an extra LLM call per example) and can be gamed. `(certain)`

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
ref  = [["the dog is running"]]                  # BLEU takes a LIST of references per prediction
cand = ["a puppy is sprinting"]
print(evaluate.load("bleu").compute(predictions=cand, references=ref))       # ~0    (only "is" overlaps)
print(evaluate.load("rouge").compute(predictions=cand, references=[r[0] for r in ref]))  # ROUGE-1 ~0.25
print(evaluate.load("bertscore").compute(predictions=cand,
        references=[r[0] for r in ref], lang="en"))                          # F1 ~0.87  ← semantics win
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
from opik.evaluation.metrics import Equals, Hallucination, AnswerRelevance  # heuristic + LLM-judge metrics

client  = Opik()
dataset = client.get_or_create_dataset(name="sst2-eval")
dataset.insert([{"input": "The movie was a masterpiece.", "expected_output": "Positive"}, ...])

def task(item, model):                          # run ONE model on one dataset row
    out = call_llm(model, item["input"])        # your model call
    return {"output": out}

for model in ["gpt-4o", "claude-sonnet-4-6", "llama-3.1-8b"]:   # loop the SAME dataset over MANY LLMs
    evaluate(
        dataset=dataset,
        task=lambda item, m=model: task(item, m),
        scoring_metrics=[Equals(), AnswerRelevance(), Hallucination()],  # mix reference-based + judge
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

**"Why not just trust MMLU / the leaderboard?"** → contamination (memorization) + saturation (Goodhart); evaluate on your own private set and human-preference Arena.

**"How do you evaluate an AI app, not just a model?"** → component-wise down the pipeline (retriever, generation, parser) + end-to-end + online (implicit feedback, A/B, monitoring).

---

## 15. Alternatives & How to Choose a Metric

Decision table — pick by task shape:

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
4. Reference "the dog is running", candidate "a puppy is sprinting": why does BERTScore ≫ BLEU, and what does IDF weighting change?
   <details><summary>answer</summary>Only "is" overlaps as an exact token → BLEU ≈ 0, ROUGE-1 ≈ 0.25 — n-gram metrics are blind to paraphrase. BERTScore embeds tokens, so puppy≈dog (0.85) and sprinting≈running (0.88) match → ~0.91. **IDF weighting** down-weights the trivial stopword matches (is↔is, a↔the) and up-weights rare content words (dog, running), dropping it to ~0.87 so the score reflects meaning-bearing tokens, not stopword mimicry.</details>
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

---

*Covers: LLM vs ML evaluation · model-level vs application/component-level (+ high-level RAG-vs-LLM contrast) · objectives/goals · challenges · reference-based vs reference-less methods · metrics (accuracy, EM/F1, BLEU, ROUGE, BERTScore, perplexity) with formulas + a worked paraphrase example · LLM-as-Judge modes & biases · benchmarks/leaderboards + contamination/saturation · safety/bias/hallucination + cost/latency · SST-2 / SQuAD / Opik multi-LLM code · metric pitfalls · production & online eval · interview lens · metric-choice decision table.*
