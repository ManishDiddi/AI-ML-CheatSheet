# Prompt Engineering — Steering a Frozen LLM With Context Alone

> **TL;DR.** Prompt engineering is programming an LLM **through its input**, not its weights — you shape the context so the frozen next-token predictor "continues the pattern" the way you want. There's a **ladder of techniques** in rising power/cost: **simple → zero-shot → one-shot → few-shot → Chain-of-Thought (CoT) → ReAct**. Few-shot teaches format/task from in-prompt examples (in-context learning, no training); **CoT** makes the model *show its reasoning* ("let's think step by step") to unlock multi-step problems; **ReAct** adds a **Thought → Action → Observation** loop so the model can **call tools** and pull in external, fresh information. Reach for the *cheapest rung that works*: prompt first, then RAG for knowledge, then fine-tuning for behavior.

**Where it fits:** The cheapest, first-reach lever in AI engineering — you adapt an LLM's behavior with words before touching data or weights. Builds directly on [LLM](LLM.md) (next-token prediction, instruct-tuning, decoding params); pairs with [Evaluating LLMs](Evaluating%20LLMs.md) (how you *measure* whether a prompt is better) and points forward to [[RAG]] (grounding) and agents.
**Prereqs:** [LLM](LLM.md) (base vs instruct models, chat roles, temperature/top-p), and the idea that generation is autoregressive.

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [Why Prompting Works — Instruct Models & In-Context Learning](#2-why-prompting-works--instruct-models--in-context-learning)
3. [The Prompting Ladder — Simple · Zero · One · Few-Shot](#3-the-prompting-ladder--simple--zero--one--few-shot)
4. [Chain-of-Thought (CoT)](#4-chain-of-thought-cot)
5. [ReAct — Reason + Act](#5-react--reason--act)
6. [Prompt Design — Best Practices & Pitfalls](#6-prompt-design--best-practices--pitfalls)
7. [Code / Implementation](#7-code--implementation)
8. [When It Breaks](#8-when-it-breaks)
9. [Prompting vs RAG vs Fine-Tuning — Where Prompt Engineering Ends](#9-prompting-vs-rag-vs-fine-tuning--where-prompt-engineering-ends)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose a Technique](#12-alternatives--how-to-choose-a-technique)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

```
An LLM is a next-token predictor:  P(next token | everything so far, θ)
Prompt engineering = choosing "everything so far" so the continuation is what you want.
θ (the weights) never changes — you only change the CONTEXT.
```

The model is a pattern-continuation machine (see [LLM](LLM.md)). You don't *ask* it in the human sense; you **set up a pattern it will complete**. "Translate to French: cat →" completes with "chat" because that's the most likely continuation. Every technique below is a different way of arranging the context so the most-likely continuation *is* the answer you need.

```
   fine-tuning  →  change the WEIGHTS      (expensive, permanent, needs data)
   prompting    →  change the CONTEXT      (free, instant, per-request)   ← this note
```

🎯 **Kill-shot:** *"Prompt engineering steers a frozen model through its context window — no gradients, no training; you're arranging the prompt so the model's next-token continuation is the behavior you want."*

---

## 2. Why Prompting Works — Instruct Models & In-Context Learning

Two things make prompting effective, and both are worth stating precisely:

**(a) Instruct models.** A *base* model just continues text; ask it a question and it may reply with more questions. An **instruct model** (base + SFT + RLHF, see [LLM](LLM.md)) has been tuned to **follow instructions** and respect **system/user roles**. Prompt engineering leans on an instruct model — that's why "Classify the sentiment…" works reliably at all. The instructor's demo ran **instruction-tuned Llama-3.1/3.3** (via Groq) for exactly this reason — versus **Phi-2** (a small *base* model via HuggingFace), which handles simple prompts thanks to its training data but follows instructions *less* reliably than an instruct-tuned model.

**(b) In-context learning (ICL).** When you put **examples** in the prompt, the model adapts to the pattern **within the forward pass — no weights update**. The "learning" is transient, living only in the context window.
```
few-shot prompt:  [example 1] [example 2] [example 3] [your input] →  model infers the rule and continues it
                  ↑ this is "learning" with ZERO gradient steps — it vanishes when the context does
```
ICL is an **emergent** ability of large models — small models can't do it well; it appears with scale. `(certain)`

🎯 *"Few-shot 'learning' is in-context learning: the model infers the task from examples in the prompt during inference — nothing is trained, nothing persists."*

---

## 3. The Prompting Ladder — Simple · Zero · One · Few-Shot

The techniques form a ladder of rising power **and** rising token cost. Climb only as high as you need.

```
SIMPLE      →  ZERO-SHOT   →  ONE-SHOT    →  FEW-SHOT    →  CoT (§4)   →  ReAct (§5)
plain ask      task + no       task + 1       task + k       + show          + tools &
               examples        example        examples       reasoning       observations
```

### 3.1 Simple prompting
Just ask. Best for **factual recall, basic calculation, simple completion/generation**.
```
"What is the primary function of mitochondria in cells?"        (temperature≈0.3 for factual)
"Calculate 15% of 240. Provide only the number."               (temperature=0 for determinism)
```
**Limit:** no control over format, no help on anything multi-step; quality rides entirely on how the model happens to continue.

### 3.2 Zero-shot prompting
Give the **task instruction** but **no examples** — rely on the instruct model's tuning.
```
Classify the sentiment of this review as Positive, Negative, or Neutral.
Review: "The product arrived damaged and customer service was unhelpful."
Sentiment:            ← model completes: "Negative"
```
Use low temperature (0–0.1) and cap output length for clean, deterministic labels. Works well for common tasks the model saw during instruction-tuning.

### 3.3 One-shot & few-shot prompting
Add **one** (one-shot) or **several** (few-shot) worked examples. This is the workhorse: examples pin down **the exact output format and the decision boundary** the instruction alone leaves ambiguous.

The instructor's few-shot **classification** pattern (label after a `//` delimiter):
```
Classify the financial sentiment as Positive, Negative, or Neutral.

Examples:
"Revenue grew by 15% in Q2, exceeding projections."     // Positive
"The acquisition discussions were terminated..."         // Negative
"The CEO will retire in December as scheduled."          // Neutral

Now classify this:
"The company reported strong earnings and raised guidance." //      ← completes: "Positive"
```
And the classic few-shot **math** example (from the CoT paper) — note standard few-shot gives the *answer only*:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?
A: The answer is 11.
Q: The cafeteria had 23 apples, used 20, bought 6 more. How many now?
A:                    ← completes: "The answer is 9."
```
**Few-shot design levers that matter:** how many examples (`k`), which examples (representative + cover edge classes), their **order**, and a **consistent delimiter/format** the model can mimic. More examples = more tokens = more cost/latency — so use the fewest that lock the behavior.

🎯 *"Zero-shot tells the model the task; few-shot shows it — examples fix the format and the boundary that instructions alone leave fuzzy, via in-context learning."*

---

## 4. Chain-of-Thought (CoT)

**The move:** make the model **generate its reasoning steps before the final answer**. Because it's autoregressive, the reasoning tokens it writes become context that conditions the answer — so "thinking on paper" measurably improves **multi-step** problems.

```
Direct:   "A store has 50 apples, sells 12 then 18. How many left?"  → "20"   (often wrong, no trace)
CoT:      "...Let's think step by step."  → "Start 50. Sold 12 → 38. Sold 18 → 20. Answer: 20."
```

### 4.1 When CoT earns its cost
The instructor's list — reach for CoT on:
- **Multi-step reasoning** / sequential logic
- **Mathematical / word problems**
- **Strategic approach & decision-making**
- **Logical reasoning**
- **Debugging / diagnosing model behavior** (the visible trace shows you *where* it went wrong)

Don't use it for simple factual/lookup tasks — you pay extra tokens and latency for nothing.

### 4.2 Two flavors
```
ZERO-SHOT CoT   → just append a TRIGGER phrase; no example reasoning needed.   "Let's think step by step."
FEW-SHOT CoT    → the examples THEMSELVES contain worked reasoning, so the model imitates the reasoning style.
```
**CoT trigger phrases** (the instructor's set — all provoke a reasoning chain):
```
✓ "Let's think step by step"                    ← the canonical one
✓ "Work through this strategically / methodically"
✓ "First do this, then [next] step"
✓ "Analyze step by step"
```
A **structured CoT** goes further and *names* the steps ("1. identify pattern → 2. compute rate → 3. hypothesize cause → 4. predict"), which makes the reasoning auditable and repeatable.

### 4.3 Direct vs Implicit vs Explicit — the trade-off (instructor's table)
```
                 SPEED     RISK of wrong    DEBUGGABLE?   CONSISTENT?
 Direct          fast      risky            no            no            ← answer only, no trace
 Implicit CoT    medium    lower            some          some          ← "let's think step by step" (zero-shot)
 Explicit CoT    slow      low ✓            yes ✓         yes ✓         ← few-shot / structured reasoning
```
🎯 *"CoT trades latency and tokens for accuracy, debuggability, and consistency on multi-step problems — the reasoning trace is both the accuracy boost and your window into failures."*

### 4.4 Self-consistency (the enrichment worth knowing)
Sample **several** CoT chains at non-zero temperature and **majority-vote the final answers**. Different reasoning paths that converge on the same answer are more trustworthy — a reliable accuracy bump over a single greedy chain, at the cost of **N× inference** (you run the model N times). `(certain)`

---

## 5. ReAct — Reason + Act

**CoT reasons; ReAct reasons *and acts*.** It interleaves thinking with **tool calls**, so the model can fetch information it doesn't have (fresh data, private DBs, exact math) instead of hallucinating it. The loop:

```
        ┌─────────────────────────────────────────────────────┐
        │   THOUGHT   → model analyzes the situation, decides what to do next
        │      ↓
        │   ACTION    → calls a TOOL  (e.g. web_search(...), calculator(...))   ← beats the knowledge cutoff
        │      ↓
        │   OBSERVATION → receives the tool's result; feeds it back into context
        └──────────↑  repeat until ──►  FINAL ANSWER
```

The model emits a fixed text format the runtime parses:
```
Thought: I need Tokyo's current weather.
Action:  get_weather("Tokyo")
Observation: Sunny, 22°C, humidity 45%      ← the RUNTIME injects this after actually calling the tool
Thought: I have what I need.
FinalAnswer: It's sunny and 22°C in Tokyo.
```

### 5.1 CoT vs ReAct
```
              REASONING       KNOWLEDGE SOURCE        TOOLS?   SHAPE
 CoT          internal steps  the model's own memory   no      sequential, one pass
 ReAct        internal steps  + EXTERNAL via tools      yes     looping / iterative
```

### 5.2 When to use ReAct (instructor's four)
1. You **need external information** (beyond the model's knowledge/cutoff)
2. **Multiple lookups** are required
3. **Multi-step research** (chain several tool calls)
4. **Tool calling** — DB, calculator, internet/web API

### 5.3 The paper's canonical comparison
The ReAct paper's HotpotQA example (Apple-Remote question) is the one to remember: **Standard** answers straight (wrong), **CoT/Reason-only** reasons but can't check facts (wrong), **Act-only** searches without reasoning (wrong), and **ReAct** interleaves reason + search and lands the correct answer. That's the whole pitch: *reasoning tells you what to look up; acting gets the facts; together they beat either alone.* `(certain)`

🎯 *"ReAct = CoT + tools in a Thought→Action→Observation loop — use it the moment the answer needs information the model doesn't already contain."*

---

## 6. Prompt Design — Best Practices & Pitfalls

The techniques above are *what*; this is *how to write the actual prompt well* (audited in as a senior would — these are the levers that separate a flaky prompt from a robust one):

- **Be specific & unambiguous.** Vague in → vague out. State the task, constraints, and audience ("explain to a 3rd grader").
- **Specify the output format explicitly.** Ask for JSON / a single label / "only the number." Pair with low temperature. This is what makes outputs *parseable* downstream (and testable — see [Evaluating LLMs](Evaluating%20LLMs.md)).
- **Use delimiters** to separate instructions from data (` ``` `, `###`, XML tags, the `//` label separator). Prevents the model from confusing your data for instructions — and blunts **prompt injection**.
- **Assign a role / persona** via the system message ("You are a financial research assistant"). Steers tone and domain.
- **Decompose** hard tasks into steps (structured CoT) or multiple prompts (prompt chaining: generate → critique → refine).
- **Give the model an "out"** ("If the context doesn't contain the answer, say 'I don't know'") — cuts hallucination.
- **Prefer positive instructions** ("respond in ≤2 sentences") over negative ("don't be verbose"); models follow *do* better than *don't*.
- **Control decoding to the task:** temperature 0 for extraction/classification/math; higher for creative generation (see [LLM](LLM.md) decoding).
- **Iterate against an eval set**, not vibes — prompts are sensitive; measure changes (§10, [Evaluating LLMs](Evaluating%20LLMs.md)).

**Common pitfalls:** overstuffed prompts (bury the instruction), inconsistent few-shot formats, examples that leak the wrong pattern, no format spec (unparseable output), and assuming a prompt that works on one model transfers to another (it often doesn't).

---

## 7. Code / Implementation

The instructor's notebook uses **Groq** (an OpenAI-compatible API) for Llama models and **HuggingFace** for Phi-2, with **Opik** tracking each call. Condensed patterns:

```python
# --- two backends behind one interface (Groq = OpenAI-compatible; HF = local) ---
from openai import OpenAI
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])

def generate_openai(prompt, system_prompt=None, temperature=0.3, max_tokens=200):
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
           [{"role": "user", "content": prompt}]
    return client.chat.completions.create(model="llama-3.1-8b-instant",
                                          messages=msgs, temperature=temperature,
                                          max_tokens=max_tokens).choices[0].message.content
```

```python
# --- FEW-SHOT via a reusable template (§3.3): fill {statement}, keep the // format ---
template = """Classify the financial sentiment as Positive, Negative, or Neutral.
Examples:
"Revenue grew by 15% in Q2." // Positive
"Acquisition talks were terminated." // Negative
"The CEO will retire in December." // Neutral
Now classify this:
"{statement}" // """
generate_openai(template.format(statement="The board authorized a $2B buyback."), temperature=0.2)
```

```python
# --- ZERO-SHOT CoT (§4.2): the trigger IS the technique ---
cot = f"Let's think step by step:\n{problem}"          # vs a plain  f"{problem}\nAnswer:"
generate_openai(cot, temperature=0.3, max_tokens=250)
```

```python
# --- ReAct loop (§5): parse Thought/Action, run the tool, feed back Observation ---
tools = {"calculator": calculator, "web_search": web_search, "prime_checker": prime_checker}
def react_agent(query, max_steps=5):
    convo = [f"User Query: {query}"]
    for _ in range(max_steps):
        out = generate_openai("\n".join(convo), system_prompt=REACT_FORMAT, temperature=0.3)
        if "FinalAnswer:" in out:
            return out.split("FinalAnswer:")[-1].strip()
        tool, arg = parse_action(out)                  # extract  Action: tool("arg")
        obs = tools[tool](arg)                         # ACTUALLY call the tool
        convo += [out, f"Observation: {obs}"]          # feed result back → next Thought
```
The `REACT_FORMAT` system prompt is what enforces the `Thought:/Action:/FinalAnswer:` structure and lists the available tools with call examples — the prompt *is* the agent's contract.

---

## 8. When It Breaks

```
❌ Prompt sensitivity     — rewording / reordering examples swings results; a prompt is not a stable API.
❌ Model-specific         — a prompt tuned on Llama may flop on GPT/Phi; re-tune per model.
❌ CoT cost & latency     — reasoning tokens multiply output length → slower, pricier; overkill on simple tasks.
❌ Reasoning ≠ truth      — a fluent CoT can be confidently wrong; the trace can rationalize a bad answer.
❌ ReAct failure modes    — tool errors, malformed Action strings, infinite loops (cap max_steps), bad observations.
❌ Prompt injection       — untrusted input ("ignore previous instructions…") overrides your system prompt.
❌ Knowledge limits       — prompting can't add facts the model lacks or that changed after its cutoff → use tools/RAG.
❌ Long-context dilution  — huge few-shot blocks push the real question far down; models lose the middle.
```

---

## 9. Prompting vs RAG vs Fine-Tuning — Where Prompt Engineering Ends

Prompting adapts **behavior and format**, but it can't inject **knowledge** the model doesn't have. The instructor closed the lecture by introducing the next tool for exactly that gap — **RAG**:

```
RAG (Retrieval-Augmented Generation) = restrict/ground the answer to a DEFINED KNOWLEDGE BASE (a DB of docs).
   Solves what prompting can't:
     • hallucination            (answers are grounded in retrieved text)
     • questions beyond the knowledge CUTOFF date
     • UNTRAINED / private domains
     • cost-effective vs fine-tuning (no retraining; just index documents)
```
You met RAG as a *system* to evaluate in [Evaluating LLMs](Evaluating%20LLMs.md); the full retrieval + grounding mechanism gets its own note → **[[RAG]]**. The decision rule:

```
Need better FORMAT / behavior / reasoning?   → PROMPT ENGINEERING (this note): few-shot, CoT, ReAct
Need external / fresh / private KNOWLEDGE?    → RAG (retrieve) or ReAct (call a tool)
Need a permanent new STYLE/skill the base model can't do even with good prompts?  → FINE-TUNING
```
🎯 *"Don't fine-tune to add facts — prompt for behavior, retrieve (RAG/tools) for knowledge, and fine-tune only for a durable skill/style prompting can't reach."* `(certain)`

---

## 10. Production & MLOps Notes

The part study notes skip — prompts in a real system:

- **Version & template your prompts.** Treat them as code: store templates, version them, diff changes. A prompt edit is a deploy.
- **Regression-test every change** against a fixed eval set ([Evaluating LLMs](Evaluating%20LLMs.md)) — prompts are sensitive, so gate changes in CI, don't ship on vibes.
- **Token budget = cost & latency.** Few-shot examples and CoT traces are *input/output tokens you pay for on every call*. Trim examples to the minimum; cap `max_tokens`; consider CoT only where it moves the metric.
- **Prompt caching.** Providers cache stable prefixes (long system prompts, fixed few-shot blocks) → put the invariant part first to cut cost/latency.
- **Structured output & validation.** Ask for JSON/schema, then *validate* it; retry or repair on parse failure. Function/tool-calling APIs make this robust for ReAct.
- **Security.** Sanitize and delimit untrusted input; keep secrets/tools behind an allowlist; assume users will attempt **prompt injection / jailbreaks** and add guardrails.
- **Observability.** Log prompt, response, tokens, latency, and tool traces (the notebook uses **Opik**) so you can debug and monitor drift in production.

---

## 11. Interview Lens

> ⚡ **Golden rule:** name the *cheapest* technique that solves the task, and know exactly when to climb the ladder.

**"Zero-shot vs few-shot?"** → Zero-shot = task instruction, no examples (relies on instruct-tuning); few-shot = k in-prompt examples that fix format + decision boundary via **in-context learning** (no weight updates).

**"What is Chain-of-Thought and why does it help?"** → Prompt the model to emit reasoning steps before the answer; since it's autoregressive, those tokens condition a better answer on multi-step problems. Zero-shot CoT = "let's think step by step"; few-shot CoT = examples with worked reasoning. Trade-off: more tokens/latency for accuracy + debuggability + consistency.

**"CoT vs ReAct?"** → CoT reasons using the model's own knowledge, single pass, no tools. ReAct interleaves **Thought → Action → Observation**, calling tools for external/fresh info in a loop — use it when the answer needs information the model doesn't contain.

**"Prompting vs fine-tuning vs RAG?"** → Prompt for behavior/format, RAG/tools for knowledge, fine-tune for a durable skill prompting can't reach. Don't fine-tune to add facts.

**"Why does in-context learning not count as 'training'?"** → No gradients, no weight change; the model infers the pattern within the forward pass and it vanishes with the context.

**"How do you make prompt outputs reliable in production?"** → Specify output format + low temperature, delimit untrusted input, version prompts, and regression-test on an eval set; validate structured output and cap tokens/steps.

---

## 12. Alternatives & How to Choose a Technique

Climb only as high as the task demands:

| Situation | Technique | Why |
|---|---|---|
| Factual recall, simple gen | **Simple / zero-shot** | Instruct model handles it; no example cost |
| Need an exact format / boundary | **Few-shot** | Examples pin format via in-context learning |
| Multi-step math / logic / decisions | **Chain-of-Thought** | Reasoning trace boosts accuracy + debuggability |
| Needs fresh / external / exact-computed info | **ReAct (tools)** | Thought→Action→Observation fetches real data |
| Grounding in a fixed document corpus | **[[RAG]]** | Restricts answers to a knowledge base; kills hallucination |
| Durable new style/skill the base model lacks | **Fine-tuning** | Only weight changes give a permanent new behavior |

**Meta-rule:** start at the **bottom** of the ladder and climb only when an eval shows you must — every rung adds tokens, latency, and failure surface. 🎯 *"The best prompt is the simplest one that passes your eval; reach for CoT/ReAct/RAG only when a cheaper rung provably falls short."*

---

## 🧠 Self-Test
*Cover the answers; retrieve first.*

1. In one sentence, what is prompt engineering doing to the model?
   <details><summary>answer</summary>Shaping the **context** of a frozen next-token predictor so its most-likely continuation is the desired behavior — no weights change (no gradients).</details>
2. List the prompting ladder from cheapest to most powerful.
   <details><summary>answer</summary>Simple → zero-shot → one-shot → few-shot → Chain-of-Thought → ReAct.</details>
3. Why do few-shot examples work without any training?
   <details><summary>answer</summary>**In-context learning** — the model infers the task/format from the examples *within the forward pass*; nothing is trained and the "learning" disappears when the context does. It's an emergent ability of large models.</details>
4. Give the zero-shot CoT trigger and say when CoT is worth its cost.
   <details><summary>answer</summary>"**Let's think step by step.**" Worth it on multi-step reasoning, math/word problems, strategic decisions, logical reasoning, and debugging model behavior — not on simple factual lookups.</details>
5. Fill the trade-off: Direct vs Explicit CoT on speed / risk / debuggability / consistency.
   <details><summary>answer</summary>**Direct:** fast, risky, not debuggable, inconsistent. **Explicit CoT:** slow, low-risk, debuggable, consistent (Implicit/zero-shot CoT sits in between).</details>
6. Name the three steps of the ReAct loop and what each does.
   <details><summary>answer</summary>**Thought** (reason about what to do next) → **Action** (call a tool) → **Observation** (receive the tool's result, feed it back) → repeat until **FinalAnswer**.</details>
7. When do you pick ReAct over plain CoT?
   <details><summary>answer</summary>When the answer needs **external information** (beyond the model's knowledge/cutoff), **multiple lookups**, **multi-step research**, or **tool calling** (DB, calculator, web). CoT only uses the model's own knowledge; ReAct pulls in fresh facts via tools in a loop.</details>
8. Prompting can't fix which problem, and what do you reach for instead?
   <details><summary>answer</summary>It can't add **knowledge** the model lacks or that post-dates its cutoff → use **tools (ReAct)** or **[[RAG]]** to ground answers in a defined knowledge base; fine-tune only for a durable new skill/style.</details>

---

*Covers: prompting as context-steering of a frozen model · instruct models & in-context learning · the ladder (simple/zero/one/few-shot) with the instructor's classification & math examples · CoT — triggers, zero-shot vs few-shot vs structured, Direct/Implicit/Explicit trade-off, self-consistency · ReAct — Thought/Action/Observation loop, CoT-vs-ReAct, when-to-use, the HotpotQA comparison · prompt-design best practices & pitfalls · Groq/HF/Opik code (few-shot template, CoT trigger, ReAct agent loop) · failure modes · prompting vs RAG vs fine-tuning (+ RAG teaser) · production (versioning, tokens, caching, structured output, injection, observability) · interview lens · technique-choice table.*
