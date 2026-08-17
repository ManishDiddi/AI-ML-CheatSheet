# Agentic Workflows — self-improvement and the five control-flow patterns between one prompt and a full agent

> **TL;DR.** Once you've built a bare [ReAct](AI%20Agents%20%E2%80%94%20Foundations.md) agent, the next lever isn't a bigger model — it's **structure**. Two ideas do the work. **Self-improvement**: let the system *look at its own output before shipping it* — **self-reflection** (is this the right answer to the question asked?) and **self-correction** (this failed, here's the error, fix it) — which is what turns a one-shot 70%-correct generator into a ~95%-correct one at 2–4× the calls. **The five workflow patterns**: **prompt chaining** (sequential, with gates), **routing** (classify then dispatch), **parallelization** (scatter–gather: sectioning + voting), **evaluator–optimizer** (generate → critique → retry, bounded), and **orchestrator–worker** (the model writes the plan at runtime). The dividing line that organizes all of it: **who holds the steering wheel** — in a *workflow* the developer fixes the control flow at design time; in an *agent* the LLM picks its own actions from feedback. Real systems are hybrids, and the patterns **compose** (router in front, chain on the cheap branch, evaluator guarding the exit). Start with the simplest thing that works; every pattern you add is cost, latency, and a new failure surface you're buying on purpose. `(certain)`

**Where it fits:** Lecture 2 of the *Advanced AI Agents* track, sitting directly on top of [AI Agents — Foundations](AI%20Agents%20%E2%80%94%20Foundations.md). Foundations answered *"what is an agent"*; this answers *"what shape should the system around it be"*. It is the design vocabulary you need before [Model Context Protocol](Model%20Context%20Protocol.md) (how tools plug in) and before any orchestration framework — LangGraph, CrewAI and friends are just implementations of these five shapes. `(certain)`
**Prereqs:** [AI Agents — Foundations](AI%20Agents%20%E2%80%94%20Foundations.md) (ReAct, function calling, the Text2SQL build), [Prompt Engineering](Prompt%20Engineering.md) (system prompts, structured JSON output), [LLM](LLM.md) (context window, temperature, cost per token).

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../_STUDY%20LOOP.md))*

> ⚙️ *Format note: this follows the vault's standard skeleton, with two lecture-specific homes added — **§3 Self-Improvement** (the conceptual engine behind the patterns) and **§8 Cost, Latency & Model Routing** (the "cost of LLMs" thread the lecture kept returning to), because both are load-bearing for production and neither fits cleanly in another section.*

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — Workflow vs Agent](#2-the-formal-core--workflow-vs-agent)
3. [Self-Improvement — Reflection, Correction, and the Feedback Source](#3-self-improvement--reflection-correction-and-the-feedback-source)
4. [The Five Patterns](#4-the-five-patterns)
5. [Worked Example — Improving the Text2SQL Agent](#5-worked-example--improving-the-text2sql-agent)
6. [Code / Implementation](#6-code--implementation)
7. [When It Breaks](#7-when-it-breaks)
8. [Cost, Latency & Model Routing](#8-cost-latency--model-routing)
9. [Production & LLMOps Notes](#9-production--llmops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Think of it as a **spectrum of who's driving**, not a binary.

- **Single-shot** — one call, one answer, no tools, no check. One chance to be right.
- **Static chain** — a fixed sequence of calls; step N feeds step N+1. No way back.
- **Agentic workflow** — generate, check, adapt: loops, branches, gates — but **the developer sets the shape**.
- **Agent** — the model picks the tools and decides when to stop. Open-ended control.

![Spectrum of LLM system complexity — four boxes from single-shot to static chain to agentic workflow to agent, with an axis underneath running from control fixed by the developer on the left to control ceded to the model on the right, annotated reliability up, cost and latency up, predictability down.](attachments/llm-system-complexity-spectrum.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

Moving right buys you **reliability on open-ended tasks** and pays in **cost, latency, and predictability**. The five patterns in this note all live in the *agentic workflow* band: they add control loops without giving up the developer's hand on the process shape.

🎯 **Kill-shot:** *"'Agentic' isn't a property of the model, it's a property of the **control flow**. Ask one question: who decided the sequence — the developer at design time, or the model at runtime? Everything else follows from that."* `(certain)`

---

## 2. The Formal Core — Workflow vs Agent

**Workflows** orchestrate models and tools through code paths *the developer wrote* — the sequence is decided at design time. **Agents** hand the model control over which tools to call, in what order, and when to stop. Anthropic's framing splits it three ways, and it's worth memorising because it's the diagram most interviewers have seen:

![Workflows versus Agent — the left two panels show LLM calls embedded in predefined code paths (prompt chaining, parallelization) and LLMs directing control flow through predefined code paths (orchestrator-worker, evaluator-optimizer, routing), while the right panel shows an agent where a single LLM call chooses an action, invokes a tool, and loops on environmental feedback.](attachments/workflows-vs-agents.png)
*Source: Anthropic, "Building Effective Agents" (Dec 2024), as reproduced in the lecture.*

| | Who decides the steps | Loop? | Example |
|---|---|---|---|
| **LLM embedded in code paths** | developer, at design time | no | prompt chaining, parallelization |
| **LLM directs flow through code paths** | a model, but only among *pre-built branches* | sometimes | routing, evaluator–optimizer, orchestrator–worker |
| **Agent** | the model, freely, from environmental feedback | yes, open-ended | the ReAct Text2SQL agent |

**The trade is always the same one.** A single call might be correct ~70% of the time; a workflow that generates, critiques, and retries can push that toward ~95% — at **two to four times the calls per request**. The only real engineering question is whether that exchange is worth it for the job in front of you. `(likely — the exact numbers are illustrative, the shape of the trade is not)`

### 2.1 The five patterns at a glance

| # | Pattern | Control flow | Who decides the steps | Iterates | Calls per task | Core failure mode |
|---|---------|--------------|----------------------|----------|----------------|-------------------|
| 01 | **Prompt chaining** | sequential | developer, design time | no | fixed = #steps | error propagation |
| 02 | **Routing** | branching, one path taken | a classifier, per input | no | 1 classify + handler | misclassification |
| 03 | **Parallelization** | concurrent, all paths run | developer, design time | no | fan-out width + 1 | bad aggregation, correlated errors |
| 04 | **Evaluator–optimizer** | closed loop | developer; loop length by the evaluator | **yes** | 2 → 2 × max-retries | weak critic, or an unbounded loop |
| 05 | **Orchestrator–worker** | plan, delegate, synthesize | the orchestrator, **at runtime** | no | 1 plan + n workers + 1 synthesis | a bad plan — nothing downstream fixes it |

### 2.2 Seven rules that hold across all five

1. **Start with the simplest thing that could work.** Every pattern is a cost and a failure surface you're adding on purpose.
2. **Bound every loop.** A retry ceiling and an escalation path are not optional extras.
3. **Make intermediate output structured.** Prose between stages is prose you cannot check.
4. **Prefer a deterministic check to a model-based one** wherever a deterministic check exists.
5. **Pass forward only what the next step needs.** Accumulated context is how careful decompositions collapse back into one bloated prompt.
6. **Log every stage.** A workflow you can't replay is a workflow you can't improve.
7. **Measure the end-to-end task, not the individual calls.** Steps that each look fine can still produce a bad answer together.

---

## 3. Self-Improvement — Reflection, Correction, and the Feedback Source

This is the concept the lecture opened with, and it's the engine inside pattern 04. **Self-improvement** = the system inspects its own output *before* the user sees it, and gets another shot.

It splits into two distinct moves that people constantly conflate:

| | **Self-reflection** | **Self-correction** |
|---|---|---|
| Question asked | *"Is this the right answer to the question that was asked?"* | *"This failed — what's the error, and how do I repair it?"* |
| Trigger | proactive, always runs | reactive, fires on a failure signal |
| Text2SQL example | the query returns `Salary` but the question asked for **`Annual_Salary`** — semantically wrong, executes fine | `You have an error in your SQL syntax near 'GROUP'` → syntactic repair |
| Catches | *logic* errors — the silent, dangerous kind | *syntax / execution* errors — loud and cheap to detect |

🎯 **Kill-shot:** *"Self-correction catches the query that **crashes**; self-reflection catches the query that **runs and returns the wrong number**. Only the second one is scary in production, and only the second one needs an LLM — for the first, the database is a free and perfect critic."* `(certain)`

### 3.1 Where the feedback comes from — rank your critics

The word "self" in *self-improvement* is misleading: the best feedback isn't from the model at all. Rank your critic by how objective it is:

```
CHEAPEST + MOST OBJECTIVE                                MOST EXPENSIVE + MOST SUBJECTIVE
        │                                                                 │
  ┌─────┴──────┐   ┌────────────┐   ┌──────────────┐   ┌──────────┐  ┌────┴──────┐
  │ programmatic│   │  compiler / │   │ execution    │   │ separate │  │ the same  │
  │ check       │──►│  linter /   │──►│ against the  │──►│ LLM      │─►│ LLM that  │
  │ regex,schema│   │ SQL EXPLAIN │   │ real system  │   │ critic   │  │ generated │
  └─────────────┘   └────────────┘   └──────────────┘   └──────────┘  └───────────┘
   free, exact       free, exact      truth, has cost     costs a call   ← weakest
```

**The self-critique blind spot:** a single call asked to both generate *and* critique **anchors on its own output and rarely contradicts itself**. That's the whole reason evaluator–optimizer splits the roles into two calls: the evaluator has no investment in the draft it's reading. `(certain)`

### 3.2 Self-Refine and Reflexion — the named versions

- **Self-Refine** (Madaan et al., 2023, `arXiv:2303.17651`) — one model, three roles: **generate → feedback → refine**, looped until a stopping condition. No extra training, no extra model.
- **Reflexion** (Shinn et al., 2023, `arXiv:2303.11366`) — "verbal reinforcement learning": after a failure the agent writes a **natural-language lesson** into an episodic memory, and that memory is prepended to future attempts. The distinction that matters: **Self-Refine improves *this* answer; Reflexion improves the *next* one.**

The lecture's `SelfImprovingSQLAgent` (§6.4) implements both — a bounded refine loop *plus* an `error_memory` list whose last three corrections are injected as few-shot context into the next question's prompt.

> ⚠️ **The honest caveat:** intrinsic self-correction — a model fixing reasoning errors with *no external signal* — has been shown to often make answers **worse**, not better, because the model can't reliably tell a good answer from a bad one it just produced. Self-improvement earns its keep when the feedback is **grounded** (a DB error, a failing test, a schema mismatch, a checkable rule), not when it's "please review your work." `(likely)`

---

## 4. The Five Patterns

### 4.1 Prompt chaining — sequential, with gates

Cut a complex task into a fixed sequence of smaller calls. Each call has **one responsibility**, and its output becomes the next call's input. **Validation gates sit between the steps.**

![Prompt chaining schematic — an input brief flows through Extract claims plus CTA, then Draft English copy, then Localize to de-DE with a glossary, then a send-ready output, with a diamond gate between every pair of steps whose fail branch routes to halt, repair, or re-run this step only, never letting a bad intermediate flow downstream.](attachments/prompt-chaining-gates.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

**Gates are the whole point.** They can be programmatic checks, regex, schema validation, or a cheap LLM judge — anything that catches a bad intermediate *before step N+1 inherits it*.

**Why it improves quality.** Instead of one prompt asked to understand, reason, generate and verify at once, each step carries one concern: per-step load drops, intermediates become structured and inspectable, and a failure at step 3 is debugged *at step 3* rather than by re-reading a monolithic prompt.

**What it is not:** a workflow, not an agent — no dynamic tool selection, no model-driven planning.

| ✅ Use it when | ❌ Skip it when |
|---|---|
| the task splits into stages you can name in advance, **and** you can write a check for each stage's output | the right sequence depends on the input, or one well-written prompt already clears your quality bar |

**Costs:** latency is the *sum* of every step, never less. Errors compound — a flawed step 2 poisons everything after it. Nuance leaks at every hand-off if you pass context sloppily; **over-decomposition costs more and reads worse than one good prompt**.

### 4.2 Routing — classify, then dispatch

Classify the incoming request, then send it to the handler built for that kind of request. **Routing decides *where* work goes; it never decides *how* the work gets done.**

![Routing schematic — an inbound message hits a classifier built from rules or a small LLM, which dispatches to one of four specialised handlers: refunds with policy RAG and a strict tool allow-list, technical support with docs retrieval and ticket creation, account security with no automated action and immediate human escalation, and a low-confidence fallback that asks one clarifying question.](attachments/routing-classifier-fallback.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

**Why it exists.** A prompt tuned for refund requests degrades on security incidents. Routing buys **separation of concerns** — each downstream path is optimised for its own input type without being compromised to accommodate the others — and it buys **cost control**: trivial lookups go to a cheap model, hard reasoning goes to an expensive one (§8).

**The box everyone forgets is the fallback.** Unroutable input is normal, not exceptional. Below a confidence threshold (the lecture used `0.7`), don't guess — ask one clarifying question or route to a general handler.

| ✅ Use it when | ❌ Skip it when |
|---|---|
| inputs fall into distinct categories that genuinely want different prompts, tools, or models — and you can classify them reliably | categories overlap heavily, or one handler already performs well across the whole input distribution |

**Costs:** a **misroute is unrecoverable inside the routing layer** — nothing downstream checks it. Multi-intent messages ("I was double-charged *and* I think my account was hacked") don't fit one category. The classifier becomes its own model to evaluate, monitor and maintain, and categories drift and multiply until the taxonomy is unmanageable.

> **Rule-based vs LLM classifier:** keywords/regex are fast, deterministic and free but brittle; an LLM handles ambiguity but adds a call *before any real work begins*. A common production shape is **rules first, LLM only on the miss**. `(likely)`

### 4.3 Parallelization — scatter–gather

Fan the same input out to several **concurrent** calls, then gather the results into one answer. Two distinct variants:

![Parallelization schematic in two variants — A, sectioning: one claim packet fans out to three different jobs, extract policy number and coverage limits, itemize claimed damages, and screen the narrative for fraud signals, which merge into one claim record; B, voting: the same code diff goes to three reviewers running the same job with varied temperature and phrasing, and a threshold flags an issue only when at least two agree.](attachments/parallelization-sectioning-voting.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

- **A · Sectioning — different jobs, one input.** Independent concerns run at once and merge. Canonical production use: run a **guardrail screen in parallel with the response being generated**, and block delivery if the screen fails (§6.3).
- **B · Voting — same job, several times.** Vary temperature or phrasing, then threshold: report a finding only when ≥ k runs agree. Drops false positives sharply on noisy judgement tasks.

**The prerequisite.** Subtasks must be **genuinely independent**. If B needs A's output, that's sequential and belongs in a chain. Faking independence to get concurrency produces workers that each hold half the picture.

**Aggregation is a design decision, not an afterthought:** concatenate, compare-and-select-best, majority vote, or hand everything to a synthesizer model. Pick it deliberately — the aggregator is a new single point of failure.

| ✅ Use it when | ❌ Skip it when |
|---|---|
| several distinct concerns apply to the same input, or the task is high-stakes enough that redundant judgements are worth paying for | subtasks depend on each other, or one careful pass is already reliable — you'd be buying variance, not accuracy |

**Costs:** cost scales **linearly with width** (3 workers ≈ 3× spend) while wall-clock is only the slowest branch. And the deep one: **voting with one model repeats that model's blind spots, not independent errors** — correlated failures don't cancel. This is exactly the [bagging](../Machine%20Learning/Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md) analogy the lecture drew, *including* its limitation: variance reduction is bounded by the correlation `ρ` between voters, so diversify prompt, temperature, **and ideally the model** — not just the seed. `(certain)`

### 4.4 Evaluator–optimizer — the bounded improvement loop

One call generates, a **second** call judges it against explicit criteria. On failure, structured feedback flows back to the generator for a **bounded** number of retries. This is the only pattern in the list that **iterates** — the first one that genuinely *improves* an artifact rather than just producing it carefully.

![Evaluator-optimizer schematic — a source document feeds a Generator that writes a claim, producing draft v of n which goes to an Evaluator returning PASS or FAIL plus reasons; pass exits to approved, while fail loops itemized violations back to the generator with n incremented, bounded to stop at n equals 3 and escalate to a human.](attachments/evaluator-optimizer-loop.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

**Two rules keep this pattern honest:**
1. **The evaluator returns a structured verdict, not prose.** `{"verdict":"FAIL","violations":[...]}` — a vague quality score gives you no stopping condition.
2. **The loop has a hard ceiling** and an escalation path.

**The evaluator doesn't have to be a model at all.** A test suite, a linter, a JSON-schema validator, or a SQL `EXPLAIN` is a free, objective, perfectly deterministic critic. Reach for an LLM judge only for the criteria a program can't express.

| ✅ Use it when | ❌ Skip it when |
|---|---|
| you can **articulate the criteria**, output visibly improves given specific feedback, and one pass isn't reliably good enough | "good" is subjective, or the latency budget is tight — you're paying double or triple for every request |

**Costs:** cost and latency multiply with every iteration; **the critic sets the ceiling** (a poor evaluator makes output *worse*); loops can oscillate or plateau; fixing one criterion often breaks one that already passed; worst-case latency is unpredictable from the caller's side.

**Free upside people miss:** the verdict log is an **evaluation dataset**. Every `{input, draft, verdict, violations}` tuple you write is labelled data for offline evals later — see [Evaluating LLMs](Evaluating%20LLMs.md).

### 4.5 Orchestrator–worker — the plan is written at runtime

An orchestrator model reads the task, decides **at runtime** what subtasks it needs, delegates each to a worker, and synthesizes the returns. **The decomposition itself is model-generated, not hardcoded.**

![Orchestrator-worker schematic — a report request goes to an Orchestrator that reads the brief and writes the section plan, which fans out over dashed edges to workers for market sizing, competitive landscape, regulatory risk and however many more the plan names, all feeding a Synthesize step that de-duplicates and unifies voice into the final report; workers see their own slice only.](attachments/orchestrator-worker.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

**The dashed edges are the point:** neither the number of workers nor their assignments exist until the orchestrator writes the plan *for this specific request*.

**Against its neighbours.** Chaining fixes the stages at design time. Routing makes one classification and dispatches once. Parallelization runs a *predefined* set of workers regardless of input. Only here does the model choose **both the subtasks and how many of them there are** — and unlike evaluator–optimizer, it decomposes **once** rather than refining repeatedly.

| ✅ Use it when | ❌ Skip it when |
|---|---|
| the number and nature of subtasks depend on the input, and one prompt keeps missing aspects of complex requests | you can name the subtasks in advance — **if the plan comes out the same every time, you've written an expensive parallelization** |

**Costs:** everything rests on the plan — **a missed subtask is never noticed**. Cost per request is variable and hard to forecast. Workers lack global context, producing gaps, overlap and tonal drift (the synthesizer is what makes it one document). Non-deterministic decomposition makes regression testing hard.

> A close cousin you already use: **"fix this bug"** where the number of files to change is unknown until the code is read.

### 4.6 The patterns compose — real systems nest them

![The patterns compose — a router classifies input, sending simple work to a chain of extract, draft and check, and complex work to an orchestrator that fans out to parallel workers; both branches converge on an evaluator returning PASS or FAIL, where FAIL loops back to regenerate and PASS ships.](attachments/agentic-patterns-compose.png)
*Source: Scaler lecture reference — "Five Workflows for LLM Agents".*

Nothing forces a single choice. **A router in front, a chain on the cheap branch, an orchestrator on the expensive one, and one evaluator loop guarding the exit** is a common production shape — and it's the answer to "how would you architect this?" in an interview.

---

## 5. Worked Example — Improving the Text2SQL Agent

Take the plain ReAct Text2SQL agent from [Foundations](AI%20Agents%20%E2%80%94%20Foundations.md) (`list_tables → get_table_schema → execute_sql`) and add structure. Each pattern buys something different.

**5.1 Chaining — the lecture's Text2SQL chain.** The bare agent generates SQL and runs it. Insert gates:

```
Prompt Generation ──► Validation Prompt ──► Self-Reflection ──► Execute on DB ──► output
                       (syntax check)        (does this answer
                                              the question asked?)
      ▲                      │ fail                    │ fail
      └──────────────────────┴─────────────────────────┘
                      repair this step only
```

The **syntax verifier** is the cheap deterministic gate; the **self-reflection** step is the expensive LLM gate that catches `Salary` when the question said `Annual_Salary`. Note the ordering — cheap check first, so you never pay for reflection on a query that won't even parse.

**5.2 Routing — by question difficulty.** BIRD labels every question `simple / moderate / challenging`, and the lecture mapped that straight onto SQL shape:

| Tier | SQL shape | Model |
|------|-----------|-------|
| simple | `SELECT … FROM … WHERE` | cheapest model |
| moderate | `… GROUP BY …` | mid-tier |
| complex | `… GROUP BY … HAVING …`, multi-table joins | frontier model |

A small classifier reads the question, predicts the tier, and dispatches. Most questions are simple, so most requests never touch the expensive model — see §8 for the arithmetic.

**5.3 Parallelization — the guardrail screen.** Run `screen_for_safety(question)` concurrently with `generate_sql(question)`. Same wall-clock as generation alone; block delivery if the screen fails. (Sectioning, not voting — two different jobs on one input.)

**5.4 Evaluator–optimizer — the self-refine loop.** This is the lecture's headline build:

```
question ──► generate SQL ──► critique (LLM → JSON)  ──┐
                  ▲                                    ├──► both OK? ──► ACCEPT
                  │            execute on DB ──────────┘        │
                  │            (ground-truth critic)            │ no
                  └──────── refine(critique + db_error) ◄───────┘
                             bounded: max_retries = 3
                                    │
                             error_memory (last 3 corrections)
                             prepended to the NEXT question
```

Two critics, deliberately different: the **LLM critic** returns `{syntax_ok, schema_ok, logic_ok, confidence, issues[], suggestion}` — it catches *logic*; the **database** returns a real error or real rows — it catches *truth*. The refine prompt gets **both**.

**A subtlety worth stealing:** the loop has a *pragmatic accept* branch — if the query **executed successfully** and confidence ≥ 0.6, accept even when a checkbox is unticked. Without it, a fussy critic burns retries on queries that are already right. The trace from the lecture's run:

```
📝 Initial SQL: SELECT COUNT(*) AS customer_count FROM customers WHERE Currency = 'EUR';
🔍 Attempt 0: confidence=1.00, syntax=✓ schema=✓ logic=✓
🔄 Refined SQL: SELECT COUNT(*) AS customer_count FROM customers WHERE Currency = 'EUR';
🔍 Attempt 1: confidence=1.00, syntax=✓ schema=✓ logic=✓
✅ Accepted on attempt 1        → customer_count = 2002
```

🚩 **Read that trace critically — it shows a real bug.** Attempt 0 passed every check *and* executed fine, yet the agent still refined. The cause: the initial generation returned SQL wrapped in a ```` ```sql ```` fence, so the *first* execute failed on the fence; the fence-stripping only happens on the **refine** path. It cost one wasted LLM call on a query that was correct from the start. **Strip fences at generation time, and assert your accept-condition fires on the happy path** — this class of "loop that never accepts on attempt 0" is the single most common evaluator–optimizer bug. `(certain — visible in the notebook output)`

**5.5 Orchestrator–worker — when it's actually warranted.** For a single SQL question, it isn't: the subtasks *are* knowable (discover → inspect → write → run), so you've written an expensive parallelization. It earns its place at the next level up — *"produce a Q3 revenue report"* — where the orchestrator decides this particular report needs five queries plus a chart, and a different brief needs two.

---

## 6. Code / Implementation

All snippets use one thin wrapper so the pattern, not the SDK, is what you read:

```python
from openai import OpenAI
import json
client = OpenAI()

def llm_call(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """One call. temperature=0 because workflows need reproducible stages."""
    r = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
    )
    return r.choices[0].message.content.strip()
```

> The lecture used `gpt-4o-mini`. **Don't hardcode a model name into a pattern** — the point of §8 is that the tier is a routing decision, and model names go stale fast. Swap in whatever your cheap/mid/frontier tiers are today.

### 6.1 Prompt chaining — copy → translate → validate

```python
# STEP 1 — generate. One concern: write the English copy.
step1_copy = llm_call(
    system_prompt="You are a B2B technical copywriter. Write a 60-80 word product "
                  "description. Professional, technically precise. No buzzwords.",
    user_prompt=product_brief,
)

# STEP 2 — translate. NOTE: input is ONLY step 1's output, not the original brief.
#   Passing the brief forward too is how a chain silently becomes one bloated prompt.
step2_de = llm_call(
    system_prompt="Translate to German. Preserve technical terminology (PM2.5, "
                  "LoRaWAN, IP67 stay as-is). Return ONLY the translation.",
    user_prompt=step1_copy,
)

# STEP 3 — the GATE. Structured JSON verdict, never prose, so it's machine-checkable.
gate = json.loads(llm_call(
    system_prompt="""Localization QA. Compare original and translation. Return ONLY JSON:
{"tone_preserved": bool, "technical_terms_correct": bool,
 "meaning_preserved": bool, "issues": ["..."]}""",
    user_prompt=f"Original:\n{step1_copy}\n\nGerman:\n{step2_de}",
))
if not all([gate["tone_preserved"], gate["technical_terms_correct"], gate["meaning_preserved"]]):
    raise ValueError(gate["issues"])   # halt here — never let it flow downstream
```

The same shape with a **deterministic** gate (always prefer this where it exists):

```python
REQUIRED = ["reporter_email", "account_id", "issue_summary", "error_type", "onset_date"]

def gate_extraction(obj: dict) -> list[str]:
    """Free, exact, instant. No LLM needed to check a field is non-empty."""
    missing = [f for f in REQUIRED if not str(obj.get(f, "")).strip()]
    if not re.fullmatch(r"AC-\d{5}", obj.get("account_id", "")): missing.append("account_id:format")
    return missing
```

### 6.2 Routing — classify then dispatch

```python
def classify_ticket(text: str) -> dict:
    raw = llm_call(
        system_prompt="""Categorize into exactly ONE of:
- billing: payments, invoices, charges, subscriptions, refunds
- technical: bugs, errors, outages, performance, integrations
- general: product questions, feature requests, how-to
Return ONLY JSON: {"category": "...", "confidence": 0.0-1.0, "reasoning": "one sentence"}""",
        user_prompt=text)
    try:
        return json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except json.JSONDecodeError:
        # Fail SAFE, not silent: unparseable → lowest confidence → hits the fallback.
        return {"category": "general", "confidence": 0.0, "reasoning": "parse failure"}

# Each handler is a *specialised* prompt — that's the whole payoff of routing.
SUPPORT_ROUTES = {"billing": handle_billing, "technical": handle_technical, "general": handle_general}

def route_support_ticket(text: str, threshold: float = 0.7) -> dict:
    c = classify_ticket(text)
    if c["confidence"] < threshold:                    # the box everyone forgets
        return {"route": "fallback", "response": ask_one_clarifying_question(text)}
    handler = SUPPORT_ROUTES.get(c["category"], handle_general)
    return {"route": c["category"], "classification": c, "response": handler(text)}
```

### 6.3 Parallelization — sectioning (guardrail) and voting (moderation)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── A · SECTIONING: generate and screen at the same time ──
def respond_with_guardrail(user_query: str) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_resp   = pool.submit(generate_response, user_query)
        f_safety = pool.submit(screen_for_safety, user_query)   # independent job, same input
        response, safety = f_resp.result(), f_safety.result()
    # Wall-clock = max(gen, screen), not gen + screen. The screener is NOT anchored
    # by the answer text, because it never sees it.
    if safety.get("safe", False):
        return {"delivered": True, "response": response, "safety": safety}
    return {"delivered": False, "response": "I'm unable to process that request.", "safety": safety}
```

```python
# ── B · VOTING: three DIFFERENT lenses, not three identical runs ──
REVIEWER_PERSPECTIVES = [
    {"name": "policy_reviewer",  "prompt": "Check hate speech, PII, threats. JSON {flagged, reason}"},
    {"name": "tone_reviewer",    "prompt": "Check aggressive/passive-aggressive/unprofessional tone. JSON {flagged, reason}"},
    {"name": "factual_reviewer", "prompt": "Check unverifiable claims, unguaranteeable promises, libel. JSON {flagged, reason}"},
]

def moderate_content(content: str, threshold: int = 2) -> dict:
    with ThreadPoolExecutor(max_workers=len(REVIEWER_PERSPECTIVES)) as pool:
        futures = [pool.submit(run_single_review, content, p) for p in REVIEWER_PERSPECTIVES]
        reviews = [f.result() for f in as_completed(futures)]
    flags = [r for r in reviews if r.get("flagged")]
    return {"blocked": len(flags) >= threshold, "flag_count": len(flags), "reviews": reviews}
```

Real output on adversarial copy — note that **the lenses disagree**, which is exactly why diversity beats redundancy:

```
--- "Our competitor's product is a complete disaster… we guarantee you'll never have an incident" ---
[Voting] Reviews: 3 | Flagged: 2 | Threshold: 2 | Decision: BLOCK
  [FLAG] tone_reviewer:    aggressive and disparaging towards a competitor
  [OK]   policy_reviewer:  no hate speech, PII, or explicit threats detected
  [FLAG] factual_reviewer: potentially libelous + unguaranteed security promise
```

Three *identical* reviewers would have voted 0–3 or 3–0 together and told you nothing new. `(certain)`

### 6.4 Evaluator–optimizer — the self-improving SQL agent

```python
CRITIQUE_SYSTEM_PROMPT = """You are a SQL quality reviewer. Return ONLY valid JSON:
{"syntax_ok": bool, "schema_ok": bool, "logic_ok": bool,
 "confidence": 0.0-1.0, "issues": ["specific"], "suggestion": "one-line fix or ''"}
- schema_ok: do ALL table/column names exist in the provided schema?
- logic_ok:  does it answer the question? Check GROUP BY, JOINs, aggregation.
- Be specific: "column 'dept' does not exist, use 'department'", not "wrong column"."""

def execute_sql_safe(query: str) -> dict:
    """The DB is the ground-truth critic. Capture the exception as a FEEDBACK SIGNAL,
    don't crash — the error string is the most useful token the loop will see."""
    try:
        return {"success": True,  "result": execute_sql(query), "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


class SelfImprovingSQLAgent:
    def __init__(self, max_retries=3, confidence_threshold=0.7):
        self.max_retries = max_retries                 # HARD CEILING — non-negotiable
        self.confidence_threshold = confidence_threshold
        self.schema_info  = get_schema_summary()
        self.error_memory = []                         # Reflexion-style, survives across questions

    def _build_memory_context(self) -> str:
        """Last 3 corrections as few-shot context. Capped: unbounded memory = context bloat."""
        return "\n".join(
            f"Q: {e['question']}\nBad SQL: {e['failed_sql']}\nError: {e['error_reason']}\nFixed: {e['corrected_sql']}"
            for e in self.error_memory[-3:])

    def run(self, question: str) -> dict:
        sql = self._generate_initial_sql(question)     # memory context injected here
        iterations = []
        for attempt in range(self.max_retries + 1):
            critique = critique_sql(question, sql, self.schema_info)   # critic #1: the LLM
            execution = execute_sql_safe(sql)                           # critic #2: the DB
            all_pass = (critique["syntax_ok"] and critique["schema_ok"] and critique["logic_ok"]
                        and critique["confidence"] >= self.confidence_threshold)
            iterations.append({"attempt": attempt, "sql": sql,
                               "critique": critique, "execution": execution})

            if all_pass and execution["success"]:
                break                                   # ACCEPTED
            if execution["success"] and critique["confidence"] >= 0.6:
                break                                   # PRAGMATIC ACCEPT — it runs, ship it
            if attempt == self.max_retries:
                break                                   # ESCALATE: return best effort, flag it

            sql = strip_fences(refine_sql(question, sql, critique,
                                          self.schema_info, execution["error"]))
        if len(iterations) > 1:                         # only remember actual corrections
            self.error_memory.append({"question": question,
                                      "failed_sql": iterations[0]["sql"],
                                      "error_reason": "; ".join(iterations[0]["critique"]["issues"]),
                                      "corrected_sql": sql})
        return {"final_sql": sql, "iterations": iterations, "total_attempts": len(iterations)}
```

Three things to notice, because they generalise beyond SQL: **(1)** the loop has *two* critics with different failure coverage; **(2)** there are **three** exits — accept, pragmatic-accept, escalate — and every one is reachable; **(3)** `error_memory` is capped at 3, because a memory that grows forever becomes the context bloat it was meant to prevent.

### 6.5 Orchestrator–worker — the shape

```python
PLAN_SCHEMA = {"sections": [{"title": str, "scope": str, "success_criterion": str}]}

def orchestrate(brief: str) -> str:
    plan = json.loads(llm_call(                                  # 1 planning call
        "Return ONLY JSON matching: " + json.dumps(PLAN_SCHEMA) +
        "\nName only the sections THIS brief needs — two is fine, seven is fine.", brief))
    with ThreadPoolExecutor(max_workers=8) as pool:              # n worker calls, concurrent
        drafts = list(pool.map(lambda s: write_section(s, shared_context=brief), plan["sections"]))
    return llm_call(                                             # 1 synthesis call
        "Remove overlap, resolve contradictions, unify voice, add an executive summary.",
        "\n\n".join(drafts))
```

The **plan is a readable artifact** — log it, and let a human override it. That's the main thing that makes this pattern debuggable at all.

---

## 7. When It Breaks

| Failure | Pattern | What it looks like | Fix |
|---|---|---|---|
| **Error propagation** | chaining | step 2 drops a field; step 4's output is confidently wrong | a gate after *every* step; halt/repair, never flow downstream |
| **Over-decomposition** | chaining | 6 calls, 6× latency, worse output than one good prompt | merge steps until each still has one clear concern |
| **Context creep** | chaining | you pass the whole history forward "just in case" — the chain is now a monolith with extra latency | pass only the slice the next step needs |
| **Misroute** | routing | a security incident answered by the refunds prompt | fallback path + confidence threshold + per-category metrics |
| **Taxonomy drift** | routing | 3 categories become 19, half overlapping | review the confusion matrix monthly; merge rare categories |
| **Correlated errors** | parallel voting | 3 runs of one model agree on the same hallucination | diversify prompt **and** model; voting bounds variance, not bias |
| **Aggregator failure** | parallel | contradictory sections merged without a tie-break rule | define the tie-break *before* you fan out |
| **Weak critic** | evaluator | the evaluator rubber-stamps everything; quality unchanged at 2× cost | test the critic on **known-bad** inputs; prefer deterministic checks |
| **Unbounded / oscillating loop** | evaluator | attempt 7; fixing criterion A breaks criterion B forever | hard `max_retries` + escalate; make criteria non-conflicting |
| **Never accepts on attempt 0** | evaluator | passes all checks yet still refines (see §5.4) | assert the happy path fires; normalise output (strip fences) *before* the first check |
| **Self-critique anchoring** | any | the model that wrote it says it's great | split generator and evaluator into **separate calls** |
| **Bad plan** | orchestrator | a whole required section simply never existed | validate the plan against a checklist; let a human edit it |
| **Cost blowout** | any | 4× calls on 100% of traffic to fix the 15% that were wrong | route: apply the expensive pattern only on the branch that needs it |

**The meta-failure:** reaching for a pattern because it's sophisticated. Every one of these is a cost and a failure surface you added **on purpose** — be able to say what it bought.

---

## 8. Cost, Latency & Model Routing

The lecture's "cost of LLMs" thread deserves its own section because it's the argument that gets a workflow *approved*.

**The arithmetic.** Let `p` = fraction of traffic that's simple, `c_cheap` and `c_frontier` the per-request costs. Single-model cost is `c_frontier`; routed cost is `p·c_cheap + (1−p)·c_frontier + c_classify`. With a typical `p ≈ 0.7` and a cheap tier ~20× cheaper, the routed system lands near **~1/3 of the flat-frontier bill** — provided the classifier itself is cheap or rule-based. If your classifier is a frontier-model call, you've added cost before doing any work.

**Latency composes differently per pattern — know which one you're paying:**

| Pattern | Wall-clock | Calls |
|---|---|---|
| chaining | **sum** of steps | fixed = #steps |
| routing | classify + 1 handler | 2 |
| parallelization | **max** of branches | fan-out + 1 |
| evaluator–optimizer | 2 × (attempts) — **unpredictable from the caller's side** | 2 → 2·max_retries |
| orchestrator–worker | plan + max(workers) + synthesis | 1 + n + 1 |

**Practical levers, in order of payoff:**
1. **Route first.** Cheapest structural saving; nothing else is close.
2. **Deterministic gates before LLM gates.** A regex that rejects 20% of drafts is 20% of an evaluator call you never make.
3. **Prompt caching** on the long, stable prefix (system prompt + schema dump). The schema summary in §6.4 is re-sent on *every* critique — cache it.
4. **Parallelize what's independent** — it converts a latency problem into a cost problem, which is usually the better problem to have.
5. **Cap retries by value.** `max_retries=3` on a report; `max_retries=1` on an autocomplete.

🎯 **Kill-shot:** *"A workflow's cost isn't the sum of its calls — it's the sum of its calls **times the fraction of traffic you send down that path**. Routing is the only lever that changes the multiplier; everything else just makes the path cheaper."* `(likely)`

---

## 9. Production & LLMOps Notes

**Structured intermediates, always.** Every hand-off between stages should be JSON with a schema you validate (Pydantic). Prose between stages is prose you cannot gate, diff, or replay.

**Trace every stage.** Emit one span per LLM call with `{stage, model, prompt_hash, tokens_in/out, latency, verdict}`. A workflow you can't replay is a workflow you can't improve — and when quality drops, the question is always *"which stage?"*. Standard tooling: OpenTelemetry-based LLM tracing (LangSmith, Langfuse, Phoenix). `(likely)`

**Evaluate end-to-end, not per call.** Stages that each look fine can still produce a bad answer together. Hold a fixed regression set of `{input, expected_outcome}` and score the **final** output; per-stage metrics are for debugging, not for the go/no-go. See [Evaluating LLMs](Evaluating%20LLMs.md).

**Idempotency and partial failure.** Parallel branches fail independently — decide *before* shipping whether a missing branch degrades gracefully or fails the request. Never let `as_completed` silently drop an exception into a `None` you then format into a prompt.

**Determinism knobs.** `temperature=0` for every gate, classifier and extractor; keep sampling only where you *want* variance (the voting pattern). Pin model versions — a silent model upgrade is a silent workflow regression.

**Concurrency and rate limits.** `ThreadPoolExecutor` is fine for a handful of branches, but fan-out width is also your rate-limit exposure. Bound `max_workers`, add jittered retry on 429, and remember a voting pattern of width 5 multiplies your TPM burn by 5.

**Guardrails belong on their own branch.** Running the safety screen *in parallel* with generation (§6.3) means the guardrail can never be talked out of firing by the answer text — a real defence-in-depth property, not just a latency trick. Pair it with the controls in [Prompt Security](Prompt%20Security.md).

**Human escalation is a feature.** Every bounded loop needs a defined "we gave up" path — a queue, a flag, a fallback answer. Silent best-effort output from a loop that hit `max_retries` is the worst possible outcome because it looks identical to success.

**Version prompts like code.** Each stage's prompt is a deployable artifact with its own eval. Changing step 2's prompt can break step 3's gate.

---

## 10. Interview Lens

The trade-off these questions are really testing: **how much structure to buy, and what each unit of structure costs you.** Anyone can list five patterns; the signal is knowing the failure mode and the price of each.

- 🎯 *"Workflow or agent is decided by **who writes the control flow** — the developer at design time, or the model at runtime. Most production systems are hybrids: deterministic scaffolding with pockets of model-driven decisions inside it."*
- 🎯 *"Self-correction catches the query that crashes; **self-reflection catches the query that runs and returns the wrong number**. The second is what actually hurts in production."*
- 🎯 *"Split the generator and the evaluator into **two calls**. One call asked to write and critique anchors on its own output and almost never contradicts itself."*
- 🎯 *"Prefer a deterministic critic. A test suite, a linter, a schema validator or a SQL `EXPLAIN` is free, objective, and can't be talked out of its verdict."*
- 🎯 *"Voting with one model three times buys you **variance reduction, not independence** — you're re-sampling the same blind spots. Diversify the lens, not just the seed."*
- 🎯 *"If your orchestrator produces the same plan every time, you've written an expensive parallelization."*

**Likely follow-ups:**
- *"When would you NOT add an evaluator loop?"* → when criteria are subjective (taste, not rules), or the latency budget can't absorb 2–3× worst case. `(certain)`
- *"Chaining vs orchestrator–worker?"* → chaining fixes the stages at design time; orchestrator decides the number and nature of subtasks per input. If you can name them in advance, chain. `(certain)`
- *"How do you stop an evaluator loop oscillating?"* → hard ceiling + require the evaluator to return the *same structured criteria set* each time so you can detect a criterion flipping back and forth, then escalate. `(likely)`
- *"Routing: rules or an LLM?"* → rules where the signal is lexical (free, deterministic, zero added latency); LLM where it's semantic. Rules first, LLM on the miss. `(likely)`
- *"Parallelization made it slower — why?"* → the subtasks weren't independent, so you serialized on a shared resource (DB connection pool, rate limit), or the aggregator became the bottleneck. `(likely)`
- *"How do you know the workflow actually helped?"* → A/B the *end-to-end* task metric against the single-prompt baseline, reporting cost and p95 latency alongside quality. A quality win you can't price isn't a decision. `(certain)`
- *"What does Reflexion add over Self-Refine?"* → persistence: Self-Refine improves *this* answer, Reflexion writes a lesson into memory that improves the *next* one. `(certain)`

---

## 11. Alternatives & How to Choose

Climb this ladder only when the rung below can't reach:

| Rung | Reach for it when | Don't |
|------|-------------------|-------|
| **Single prompt** ([Prompt Engineering](Prompt%20Engineering.md)) | one-shot transform, already reliable | the task needs live data or a check |
| **+ Deterministic gate** | you can *program* the correctness check | the criterion is semantic |
| **Prompt chaining** | nameable stages, each checkable | the sequence depends on the input |
| **Routing** | distinct input categories wanting different prompts/models | categories overlap heavily |
| **Parallelization** | genuinely independent concerns, or high-stakes redundancy | subtasks depend on each other |
| **Evaluator–optimizer** | criteria are writable and output improves on feedback | "good" is subjective; tight latency budget |
| **Orchestrator–worker** | the subtask set is unknowable in advance | the plan is always the same |
| **Full agent** ([AI Agents — Foundations](AI%20Agents%20%E2%80%94%20Foundations.md)) | the path is unknown and branchy, tool choice must be dynamic | a flowchart would do — code the flowchart |

**The framework question.** LangGraph, CrewAI, AutoGen, the OpenAI Agents SDK and friends don't invent new patterns — they give you a graph runtime with state, retries, checkpointing and tracing so you don't hand-roll them. Adopt one when your hand-rolled orchestration starts needing durable state and resumability; not before. The related but orthogonal question — *how tools get plugged in at all* — is [Model Context Protocol](Model%20Context%20Protocol.md). `(likely)`

---

## 🧠 Self-Test

1. **What single question separates a workflow from an agent, and why is "it uses an LLM" the wrong test?**
   <details><summary>answer</summary> <b>Who decides the control flow.</b> A workflow's sequence is fixed by the developer at design time; an agent's is chosen by the model at runtime from environmental feedback. A workflow can absolutely call an LLM at a fixed step and still be a workflow — the LLM is <i>embedded in</i> the code path, not directing it. Most production systems are hybrids. </details>

2. **Distinguish self-reflection from self-correction with a Text2SQL example of each. Which one needs an LLM?**
   <details><summary>answer</summary> <b>Self-correction</b> is reactive repair of a detected failure — <code>syntax error near 'GROUP'</code> → fix the SQL. <b>Self-reflection</b> is proactive: the query executed fine but returned <code>Salary</code> when the question asked for <code>Annual_Salary</code> — semantically wrong, silently. Correction needs no LLM (the DB/parser is a free, exact critic); reflection does, because "does this answer the question asked" isn't programmatically checkable. </details>

3. **Why does evaluator–optimizer use two separate calls instead of one call that self-critiques?**
   <details><summary>answer</summary> The <b>self-critique blind spot</b>: a single call asked to generate and judge anchors on its own output and rarely contradicts itself. Splitting the roles creates real adversarial tension — the evaluator has no investment in the draft. Two more rules keep it honest: the evaluator returns a <b>structured verdict</b> (not prose, so you have a stopping condition) and the loop has a <b>hard ceiling</b> with an escalation path. And the best evaluator is often not a model at all: a test suite, linter, schema validator or SQL <code>EXPLAIN</code>. </details>

4. **You run the same review prompt 3× at different temperatures and take a majority vote. What reliability gain do you actually get, and what don't you?**
   <details><summary>answer</summary> You reduce <b>variance</b> — run-to-run noise on a high-variance judgement. You do <b>not</b> reduce <b>bias</b>: three runs of one model share its blind spots, so correlated errors don't cancel (exactly the ρ-floor from bagging). Fix by diversifying the <i>lens</i> — different reviewer perspectives, different phrasing, ideally a different model — not just the seed. Also: cost scales linearly with width, and the aggregator becomes a new single point of failure needing an explicit tie-break rule. </details>

5. **Your evaluator loop passes every check on attempt 0 but still refines, costing an extra call. What's the likely bug?**
   <details><summary>answer</summary> The output isn't <b>normalised before the first check</b> — classically, the generator returns SQL wrapped in a <code>```sql</code> fence, the first execute fails on the fence, and fence-stripping only happens on the refine path. The accept branch therefore can't fire on attempt 0. Fix: normalise (strip fences, trim) at generation time and add a test asserting the happy path accepts at attempt 0. </details>

6. **When is orchestrator–worker the wrong choice, and what's the tell?**
   <details><summary>answer</summary> When you can name the subtasks in advance. <b>The tell: the plan comes out identical every run</b> — you've paid for a planning call to reproduce a fixed fan-out, i.e. an expensive parallelization. Its real cost is that a missed subtask is never noticed (nothing downstream checks the plan) and per-request cost is unforecastable. Use it when the number and nature of subtasks genuinely depend on the input. </details>

7. **Sketch a production shape that composes several patterns, and justify each piece.**
   <details><summary>answer</summary> <b>Router in front</b> (send the ~70% simple traffic to a cheap model — the only lever that changes the cost multiplier), <b>chain with deterministic gates on the cheap branch</b> (nameable stages, free checks), <b>orchestrator + parallel workers on the complex branch</b> (subtasks unknown in advance, breadth costs little wall-clock), <b>one bounded evaluator loop guarding the exit</b> (last line of quality defence), plus a <b>guardrail screen running in parallel with generation</b> so it can't be anchored by the answer text. Log every stage; measure the end-to-end outcome, not the individual calls. </details>

---

*Sources for version-sensitive facts: Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Dec 2024) — origin of the five-pattern taxonomy and the workflows-vs-agents diagram; Self-Refine ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)); Reflexion ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)). Lecture: Scaler, Advanced AI Agents — "Advanced Agent Concepts" + the "Five Workflows for LLM Agents" pattern reference + the `Agent2` Text2SQL notebook. Model names in code are the lecture's; treat the tier, not the name, as the design decision.*
