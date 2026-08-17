# 🔁 The Study Loop — how to actually get through this

*Your operating manual. Read this once, then follow the tables. Designed 2026-08-15 against the real constraint: **3 Scaler lectures/week, ~6 hrs/week outside them, ~2 hrs per topic.***

---

## Why this exists

You were running **four full passes per topic** — lecture → read the note → notebook → assignment — at 3 topics/week. That's 12 units of work against 6 hours. Survival mode was the correct output of that system, not a discipline failure.

Three things were structurally wrong, and this loop fixes each:

| Problem | Fix |
|---|---|
| A 3,800-word reference note read start-to-finish, ~30 min, ~25% of the topic budget on the **lowest-retention** activity | **Test-first entry** — enter every note at its Self-Test, read only what you missed |
| MCQ assignments test **recognition**, the weakest retrieval — you pass and still feel it draining, because passing was never evidence | **Free recall** (blank-page Self-Test + one-cell rebuild) is the only thing that counts as evidence |
| No WIP limit — unclosed topics roll forward and compound | **Two tiers**, chosen deliberately on Monday. Not every topic gets the full hour |

🎯 **The one-line version:** *confidence comes from a passed test, never from a completed read.*

---

## The week

| When | What | Time |
|---|---|---|
| **Lecture day ×3** | Lecture → **MCQ immediately after**, log hesitations | +20 min |
| **Weekday ×2** (non-lecture) | Tier-1 notebook session — [Protocol B](#protocol-b--the-notebook-hour-tier-1) | 1 hr each |
| **Weekend day 1** | Self-tests for all 3 topics + targeted reading of failures only | ~1.5 hr |
| **Weekend day 2** | [Review queue](_REVIEW%20QUEUE.md) — 2 old topics, self-test only — **+ buffer** | ~1.5 hr |

**The buffer is load-bearing.** A schedule with no slack fails the first week something goes wrong. Do not fill it in advance.

---

## The tiering rule — decide Monday, in 30 seconds

At ~6 hrs against 3 topics you **cannot** deeply master all three. Any plan that claims otherwise breaks on your first bad week. So choose on purpose:

- **Tier 1 — "build it" · 1 topic/week.** Full protocol including the notebook hour. Pick the topic with the most real **AI-Engineering leverage** — anything you'd actually build with, anything with a live API surface.
- **Tier 2 — "know it" · 2 topics/week.** MCQ + Fast Pass + Self-Test, ~45 min. **No notebook.** Deliberately, not as failure.

You are *already* making this tradeoff — currently by accident, at 11pm, on whichever topic ran out of week. Making it on Monday is the entire difference between a system and survival.

---

## Protocol A — Lecture day (+20 min)

1. **Do the MCQ immediately after the lecture**, while it's fresh. Deadline cleared, nagging gone.
2. **Log every question you hesitated on** — one line each, into the topic's tracker row.
3. Stop. No note, no notebook. Lecture days are already long; don't stack.

> The MCQ is not your goal — it's your **diagnostic**. Hesitations are a free, precise map of your weak spots, and they're what tells you which note sections to read later. You stop reading notes linearly and start reading them *targeted*.

---

## Protocol B — The notebook hour (Tier 1)

Passively reading a notebook is as low-value as passively reading a note. Make it produce evidence:

| # | Move | Time |
|---|---|---|
| 1 | **Run it top to bottom**, unmodified. Just get it green. | 10 min |
| 2 | **Predict-then-run.** Change a hyperparameter, *write down what you expect*, then run. | 20 min |
| 3 | **Break one thing** deliberately. Read the actual error. | 10 min |
| 4 | **One-cell rebuild** — blank cell, reimplement one core cell **from memory**. | 20 min |

🎯 Step 2 being *wrong* is the highest-information event of your week — that's a mental model correcting itself. Don't skip past it when it happens; write down why you were wrong.

Step 4 is the self-test for code. It is what "practical exposure" actually means — not having read the cell, but being able to produce it.

---

## Protocol C — Tier 2 topic (~45 min)

1. **Fast Pass** the note — the top block only. **Never the whole note.** (5 min)
2. **Self-Test cold**, answers covered. (15 min)
3. Read **only** the sections you missed. (20 min)
4. Tracker: mark confidence, log what beat you. (5 min)

---

## Protocol D — Weekend review (~20 min)

Open [`_REVIEW QUEUE.md`](_REVIEW%20QUEUE.md), take the **2 stalest 🔴/🟡 topics**, and run **self-test only**.

**Do not re-read the note.** If you fail a question, read *that section* and nothing else. Your 316 existing recall questions are the entire content — there is nothing new to write.

---

## Test-first entry — the habit that makes this work

Every note now carries a **"Start here"** line at the top pointing at its Self-Test.

> **Open the note at the Self-Test. Attempt every question cold. Then read only what you missed.**

A 30-minute read becomes 5 targeted minutes, and you finish holding evidence — *"4 of 6, and I know exactly which 2 were weak"* — instead of a vague, accurate dread. That inversion is the point of this whole document.

---

## Rules that keep this alive

- **One Tier 1 per week. Maximum.** Two is how you end up back here.
- **Never read a note linearly.** If you catch yourself at the top scrolling down, stop and jump to the Self-Test.
- **A skipped weekday is fine.** That's what the weekend buffer absorbs. Don't repay it by doubling up.
- **A topic left at Tier 2 is closed, not owed.** It'll come back through the review queue. There is no debt.
- **If you fall behind: drop Tier 1 for the week**, keep the three self-tests. Retrieval is the last thing to cut, never the first.
- **Update the tracker every session**, even if only one character. The queue is worthless if it's stale.

---

## Related

- [`_REVIEW QUEUE.md`](_REVIEW%20QUEUE.md) — what's due, what's weak, what's next
- [`_TEMPLATE — Topic Note Format.md`](_TEMPLATE%20%E2%80%94%20Topic%20Note%20Format.md) — how notes get built
- [`_IMAGE EMBEDDING WORKFLOW.md`](_IMAGE%20EMBEDDING%20WORKFLOW.md) — figure sourcing process
