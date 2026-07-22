# 📐 Topic Note Format — The Standard Template

*How every topic note in this vault is structured. Copy the skeleton below into a new note and fill it in. Optimised for fast revision **and** deep mastery across three goals: engineering, interviews, and real-world problem solving.*

**Calibration locked with the author:**
- **Shape:** Comprehensive layered — intuition → math → mechanism → example → code → limits → production → interview → alternatives.
- **Math depth:** *Math-aware, not proof-heavy.* Show the formulas and explain **why** they hold; derive something only when the derivation itself builds intuition. No proofs for their own sake.
- **Recall aids:** A **TL;DR** at the very top and a **Self-Test** at the very bottom. (No flashcard/cloze lines unless asked.)
- **Completeness by integration (no "gaps" section):** acting as a senior engineer, you still audit for what's missing for production/interview/real-world mastery — but you **teach it inline in the section where it belongs**, not in a terminal flag list. If a missing concept deserves its own home, add a section for it (e.g. *Interpretability* for SHAP). For a big adjacent topic that merits its *own future note*, drop an inline `[[wiki-link]]` pointer at the natural spot rather than a separate list.
- **Voice:** Second-person, opinionated, dense — "no filler." Reuse the house conventions: 🎯 marks *the single line that wins the question*; confidence tags `(certain)` / `(likely)` / `(guessing)` flag how solid a claim is.
- **Images:** embed **1–3+ real figures** — at least one, and **no upper cap** (add more wherever a visual speeds understanding; a dense note may want 5–8+) — architecture diagrams, mechanism illustrations, annotated plots, **not** prettier ASCII. Best-image-wins, **licensing is not a gate** (personal educational vault): source *notebook-first* (the topic's Scaler notebook), then any authoritative canonical figure (Wikimedia/papers/blogs/course sites — scraping allowed), then self-made. **Attribution mandatory for third-party images** (one-line italic source credit); quality gate (no watermarks/inappropriate content) still applies. Syntax `![caption-as-alt-text](attachments/kebab.png)`, never `![[ ]]`; keep the ASCII diagram as a companion by default, or **replace it with the image when the image is clearly clearer**. Full process — extractor, sourcing order, quality gate — in **`_IMAGE EMBEDDING WORKFLOW.md`**.

---

## The Skeleton — copy this

```markdown
# <Topic> — <one line: what it is & why it matters>

> **TL;DR.** 2–3 sentences: the essence, plus when you'd reach for it and when you wouldn't.

**Where it fits:** one line placing it in the ML landscape / which problem class it solves.
**Prereqs:** [[linked]] concepts worth knowing first.

---

## 1. Intuition / Mental Model
Plain-language analogy first, then an ASCII diagram of the core idea. The reader should "get it" before any math.

## 2. The Formal Core
Key formulas as backticked pseudo-math. Define every symbol. State the assumptions the math rides on.

## 3. How It Works
Step-by-step mechanism — the pipeline from input to output. Number the steps.

## 4. Worked Example
Tiny concrete numbers/data run end-to-end, so the formula in §2 becomes something you can see.

## 5. Code / Implementation
Library version (sklearn / torch / statsmodels / …) + a from-scratch snippet where it teaches.
Inline comments framed by ML relevance, not generic API description.

## 6. When It Breaks
Assumptions that fail, limitations, failure modes, and the classic bugs/traps.

## 7. Production & MLOps Notes
Data & scale, train vs inference cost, latency, monitoring & drift, retraining triggers, deployment pitfalls. The part most study notes skip — don't.

## 8. Interview Lens
The trade-off the question is really testing. 🎯 kill-shot line(s). Likely follow-ups with crisp answers. Confidence tags where a claim is an inference.

## 9. Alternatives & How to Choose
What you'd use instead, and the decision criteria that pick between them.

---

## 🧠 Self-Test
4–6 active-recall questions. Cover the answer, force retrieval, then check.

1. <question>
   <details><summary>answer</summary> … </details>
```

> The senior-engineer **completeness audit happens while drafting**, not in a closing section: as you write each part, ask "what would an interviewer or a production system also expect here that the author hasn't covered?" — then add that explanation in the right section (or a new one). The note should leave no obvious hole, not a list of holes.

---

## Rules that keep the vault coherent
- **Not every note needs all 9 sections.** Drop one only when it's genuinely N/A (e.g. no meaningful "code" for a pure-theory topic) — and say so in one line rather than leaving a blank heading.
- **Keep numbering + any ToC in sync** when you add/remove sections.
- **Link liberally** with `[[wiki-links]]` so the Obsidian graph stays connected; a link to a note that doesn't exist yet is a fine TODO marker.
- **Formulas inline** as backticked pseudo-math (`s = (b − a) / max(a, b)`), never LaTeX/MathJax.
- **Long notes (≳150 lines)** get a compact anchored Table of Contents right under the TL;DR.
- **Fill coverage holes inline, every time.** The author's explicit goal is complete mastery, so when you spot something important that's missing (a metric, a failure mode, an interpretability tool, a production concern), *explain it in the section it belongs to* — never leave it as a bare flag, and never silently skip it.
