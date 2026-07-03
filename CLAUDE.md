# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An **Obsidian vault** (`.obsidian/` config present) of AI/ML study material. There is no application, build system, tests, or runnable code — the "codebase" is Markdown content authored for the author's own interview prep and reference. Work here is reading, writing, and editing `.md` notes. Do not look for a package manager, lint, or test command; none exist.

## Content lives in two distinct genres — match the one you're editing

**Cheatsheets** (`Python/Numpy/`, `Python/Pandas/`) — exhaustive references for "experienced AI/ML engineers and senior data scientists."
- Structure: H1 title, italic H3 subtitle, an `import` block, a numbered Table of Contents with anchor links, then numbered `##` sections.
- The substance is large `python` code blocks where **the inline `# comment` carries the teaching** — and comments are framed by *ML relevance* (e.g. `np.triu(np.ones((n,n)), k=1)  # causal mask for transformers`), not generic API description. Many sections end with a `| Function | ML Use Case |` table.
- When extending one, keep the ToC, anchors, and section numbering in sync with what you add.

**Interview revision notes** (`Interview Prep/`) — dense, exam-oriented walkthroughs of one topic each (CNN, NLP/RNN-LSTM-Transformers, SARIMAX, Bagging & Boosting).
- Subtitle convention is "Dense, no filler. Every concept an interviewer will probe." Honor it: no padding, every line earns its place.
- Heavy use of ASCII-diagram fenced blocks for mental models, and explicit "interview trap" / "follow-up question" / "what wins the question" framing.

`Interview Prep/Interview Questions.md` is special: a meta-layer that drills *answer delivery*. Each answer follows a fixed **6-step template (Direct answer → Formula → Mechanism → Example → Limitation → Alternative)**, the 🎯 line marks the single sentence that wins the question, and confidence is tagged `(certain)`/`(likely)`/`(guessing)`. Preserve this template, the 🎯 markers, and the tags when editing it.

## Authoring NEW topic notes — the standard format (use this every time)

The author is studying ML day by day (Scaler course) and wants one note per topic. **Folder placement:** classic ML / DL / interview topics go in `Interview Prep/`; LLM / generative-AI / AI-engineering topics go in `AI Engineering/` (e.g. `AI Engineering/LLM.md`). Note also that Obsidian auto-renames files to match their H1 heading, and wiki-links resolve by note name regardless of folder — so `[[LLM]]` works across folders. There is a **locked canonical format** for these notes — the full spec + copyable skeleton lives in **`_TEMPLATE — Topic Note Format.md`** at the vault root; read it before writing a new note. The retrofitted `Interview Prep/Bagging_Boosting.md` is the reference exemplar of the format applied. The contract:

- **Comprehensive layered shape**, sections in this order: TL;DR → Where-it-fits/Prereqs → 1. Intuition/Mental Model → 2. Formal Core → 3. How It Works → 4. Worked Example → 5. Code/Implementation → 6. When It Breaks → (Interpretability or other topic-specific sections as warranted) → Production & MLOps Notes → Interview Lens → Alternatives & How to Choose → 🧠 Self-Test. Renumber sections to fit; drop one only if genuinely N/A, and say so.
- **Depth = math-aware, not proof-heavy.** Show formulas and *why* they hold; derive only when the derivation builds intuition. No proofs for their own sake.
- **Recall aids = TL;DR at top + Self-Test (4–6 active-recall questions with collapsed `<details>` answers) at the bottom.** Not flashcards/cloze unless asked.
- **Completeness by integration (NO separate "gaps" section).** The author's standing goal is *complete* coverage for production + interviews + real-world mastery. Acting as a senior ML engineer, audit for what's missing **while drafting** and then *teach it inline in the section where it belongs* — e.g. SHAP/feature-importance pitfalls become an **Interpretability** section, probability calibration and monotonic constraints go into **Production & MLOps**. Add a new numbered section when a missing concept deserves its own home. Only for a big adjacent topic that merits its *own future note* do you drop an inline `[[wiki-link]]` pointer instead of explaining it. Never leave a bare flag list, and never silently skip a known hole. (The author explicitly rejected a terminal "Gaps Flagged" section in favor of this.)
- Reuse house conventions throughout: 🎯 = the line that wins an interview question; `(certain)`/`(likely)`/`(guessing)` confidence tags; ASCII diagrams for mechanisms; `[[wiki-links]]`; backticked pseudo-math (no LaTeX); a compact anchored ToC on notes ≳150 lines.
- **Images:** embed with **standard markdown relative links** — `![alt](attachments/file.png)` — NOT Obsidian `![[file.png]]` wiki-embeds. The repo is published to GitHub, whose renderer ignores `![[…]]` and shows nothing; the standard form renders in both Obsidian and GitHub. Keep images in an `attachments/` subfolder next to the note (Obsidian is configured to save them there).

**Status of the four legacy Interview Prep notes:** all four (`Bagging_Boosting.md`, `CNN.md`, `NLP.md`, `SARIMAX.md`) have been **retrofitted** to this format — they are the reference exemplars. Each demonstrates completeness-by-integration, e.g. an inline **Interpretability** section (SHAP in Bagging, Grad-CAM in CNN, attention-caveat in NLP), calibration/constraints in Production (Bagging), tokenization/KV-cache/decoding/LoRA in Production (NLP), and walk-forward CV/prediction-intervals/global-models in Production (SARIMAX). New topic notes should match this depth and structure.

## Conventions that span the vault

- **Author's voice is second-person and opinionated** ("your documented gap," "the sentence that proves you've built one"). Keep that register; don't rewrite into neutral encyclopedia prose.
- Formulas are written inline as backticked pseudo-math (`s = (b − a) / max(a, b)`), not LaTeX/MathJax.
- Wiki-links (`[[note]]`) and the Obsidian graph are in play — when adding a note that relates to an existing one, link it so the graph stays connected.
- Prefer precision over hedging in the *content*, but the interview notes deliberately flag uncertainty with the confidence tags above — those are a feature, not sloppiness.

## Git

`main` is the working branch and commits go straight to it. Commit messages are terse topic labels (`NLP`, `CNN`, `SARIMAX`) — one per note/topic added. `.DS_Store` and `.obsidian/workspace.json` show up as noise in `git status`; they are local Obsidian/macOS state, not content — don't commit them as part of a content change.
