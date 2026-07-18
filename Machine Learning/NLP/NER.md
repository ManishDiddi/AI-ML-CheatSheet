# NER — Named Entity Recognition: Finding & Typing Entity Spans in Text

> **TL;DR.** NER is the **information-extraction** task of locating **named-entity spans** (proper names of real-world things) and classifying them into types — `PERSON`, `ORG`, `LOCATION`, `DATE`, `DRUG`, `DISEASE`, … It's framed as **sequence labeling**: tag every token with a **BIO** label (`B-`egin / `I-`nside / `O`utside) so multi-word spans and boundaries are recoverable. The approach ladder climbs from **dictionary/rule-based** (gazetteers + patterns — brittle, no generalization) → **ML with CRF** (models label *transitions*, so illegal tag sequences can't happen) → **BiLSTM-CRF** (learned features + char-level morphology) → **pretrained transformers** (BERT token-classification, SOTA) and **domain models** (scispaCy/Stanza/BioBERT for biomedical). Evaluate at the **entity/span level** with F1 — never token accuracy (the `O` class swamps it).

**Where it fits:** the workhorse of **information extraction** — the first step that turns unstructured text into structured records. It powers PII masking, résumé/invoice parsing, search, question answering, and (the lecture's case) **pharmacovigilance**: pull `DRUG` and `ADVERSE-EVENT` spans from patient reviews, then link them. Uses the same tokenization/POS machinery as [Text Preprocessing](Text%20Preprocessing.md) and the sequence models in [RNN · LSTM · Transformers](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md).
**Prereqs:** [Text Preprocessing](Text%20Preprocessing.md) (tokenization, POS), [Word Embeddings](Word%20Embeddings.md) (token features), sequence models & [BERT](BERT.md) (for the DL tiers), [Classification Metrics](../Supervised%20ML/Classification%20Metrics.md) (precision/recall/F1).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — Sequence Labeling & CRF](#2-the-formal-core--sequence-labeling--crf)
3. [How It Works — The Approach Ladder](#3-how-it-works--the-approach-ladder)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Evaluation — Score at the Span Level](#7-evaluation--score-at-the-span-level)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Alternatives & How to Choose](#10-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

NER is a subtask of **Information Extraction** — *"find the limited, relevant, factual bits in a big blob of text and make them structured."* You're highlighting spans and stamping each with a type:

```
"Dr. Sharma prescribed aspirin , but the patient reported severe nausea ."
   └── PERSON ──┘            └DRUG┘                         └─ PROBLEM ─┘
```

Two things happen at once: **detection** (where does the entity start and end?) and **classification** (what type is it?). The business payoff in the lecture is **pharmacovigilance** — a data scientist at a pharma company extracts `DRUG` and `ADVERSE-EVENT` mentions from reviews so they can associate *which drug caused which side effect*. (Strictly, tying the drug to the event is **relation extraction** built *on top of* NER — NER is step one; §8.)

Because entities are usually **multi-word** (`New York City`, `Dr. A. P. J. Abdul Kalam`, `myocardial infarction`), you can't just classify each word independently — you need a scheme that marks **span boundaries** and respects that labels depend on their neighbors. That scheme is **BIO tagging**, and that dependency is why **CRFs** exist. 🎯

---

## 2. The Formal Core — Sequence Labeling & CRF

**NER = sequence labeling.** Given tokens `x = (x₁…xₙ)`, predict tags `y = (y₁…yₙ)`, one per token.

**BIO / IOB2 tagging** encodes spans in per-token labels:
```
token:   Dr.    Sharma   prescribed   aspirin   for    severe   nausea
BIO:     B-PER  I-PER     O            B-DRUG    O      B-PROB   I-PROB
         │      │                      │                │        │
         begin  inside                 begin            begin    inside
```
`B-TYPE` starts an entity, `I-TYPE` continues it, `O` is outside any entity. A `k`-type problem needs `2k+1` labels. Variants: **IOB1** (`B-` only to separate adjacent same-type entities), **BILOU/BIOES** (adds `L`/`E` = last, `U`/`S` = unit/single) — the extra boundary signals often **improve accuracy** by making the model commit to where entities end.

**Why not classify each token independently?** Tags are **not independent**: `I-PER` may only follow `B-PER`/`I-PER`; `O → I-DRUG` is illegal. A per-token softmax has no way to forbid that. You need to score the **whole label sequence jointly** — enter the CRF.

**Linear-chain CRF** models the conditional distribution of the label sequence:
```
P(y | x) = (1 / Z(x)) · exp( Σₜ [ Σₖ λₖ · tₖ(yₜ₋₁, yₜ, x, t)   ← transition features
                                 + Σⱼ μⱼ · sⱼ(yₜ, x, t) ] )     ← state/emission features
```
- `sⱼ` = **emission** features (does this token look like a drug? capitalized? suffix "-mycin"?). `tₖ` = **transition** features (is `B-PER → I-PER` allowed/likely?).
- `Z(x)` = **global** normalizer summing over *all* possible tag sequences (computed by the forward algorithm) — this global normalization is what lets the CRF penalize illegal transitions, and it avoids the **label-bias** problem of locally-normalized MEMMs. `(certain)`
- **Decoding** = find the highest-scoring sequence with **Viterbi** (dynamic programming).

Contrast: **HMM** is *generative* (`P(x,y)`, strong independence assumptions); **CRF** is *discriminative* (`P(y|x)`, arbitrary overlapping features) — which is why CRF beat HMM for NER. A **BiLSTM-CRF** simply replaces hand-built emission features with a neural network's per-token scores while keeping the CRF's transition matrix + Viterbi decode.

---

## 3. How It Works — The Approach Ladder

**A. Classical**
- **Dictionary / gazetteer** — hold a vocabulary list per type; string-match it against the text. Simplest possible NER. *Cons:* you must maintain the dictionary, it can't generalize to unseen names, and it can't disambiguate by context.
- **Rule-based** — hand-written rules of two kinds: **pattern-based** (morphology/regex: `Xxxx Inc.` → ORG; capitalized-word runs) and **context-based** (surrounding words: *"a title (Dr./Mr.) followed by a proper noun ⇒ that proper noun is a PERSON"*). Great for structured formats (PII masking). *Cons:* misses anything outside the rules (a name with no title slips through); brittle and high-maintenance.

**B. Machine Learning (feature + CRF)**
- Treat it as **multi-class classification per token** at first (entity types are the labels), trained on **annotated** documents, then applied to raw text. But independent classification ignores label dependencies and **entity ambiguity** ("Washington" = person/place/org depending on context) → upgrade to a **CRF** over features like the word, POS, prefixes/suffixes, capitalization/shape, and gazetteer hits. `sklearn-crfsuite` is the classic implementation.

**C. Deep Learning**
- **BiLSTM-CRF** — a bidirectional LSTM reads the sentence and emits per-token scores; a CRF layer adds transition scores and Viterbi-decodes. The standard strong pre-transformer architecture.
- **+ Char-level CNN/BiLSTM** (BiLSTM-CNN-CRF) — encode each word from its **characters** first. This captures morphology and **handles OOV/rare words, capitalization, and suffixes** (`-mab`, `-vir`, `-itis`) that word embeddings miss — crucial for biomedical/entity-heavy text.
- **Pretrained language models (ELMo, [BERT](BERT.md))** — the current SOTA: run the transformer, take each token's contextual vector, and add a linear **token-classification** head over BIO tags (optionally with a CRF on top). Context resolves ambiguity that classical methods can't.

**D. Hybrid** — combine a learned model with rules/gazetteers (e.g., spaCy's statistical NER **+** an `EntityRuler`) to inject known entities and guardrails. Common in production.

---

## 4. Worked Example

Sentence: `"Dr. Sharma prescribed aspirin but the patient reported severe nausea."`

BIO gold tags:
```
Dr.=B-PER  Sharma=I-PER  prescribed=O  aspirin=B-DRUG  but=O  the=O
patient=O  reported=O  severe=B-PROB  nausea=I-PROB  .=O
```
- **Rule-based** catches `Dr. Sharma` (title + proper noun) but would **miss** a name with no title, and knows nothing about `aspirin`/`nausea` unless they're in a gazetteer.
- **CRF/BiLSTM-CRF** shines here: suppose the emission scores momentarily favor `O` for `nausea` after `B-PROB severe`. The CRF's **transition** term makes `B-PROB → I-PROB` cheap and `B-PROB → O`(then losing the entity) or an illegal `O → I-PROB` expensive, so Viterbi keeps the two-word span `severe nausea` together. That's the whole point of modeling transitions — **the label sequence is decoded jointly, not token-by-token.**
- **Downstream** (the pharma goal): NER gives spans `{aspirin: DRUG}` and `{severe nausea: PROBLEM}`; a **relation-extraction** step then asserts *aspirin —causes→ severe nausea*.

---

## 5. Code / Implementation

**spaCy — pretrained NER + a hybrid rule layer** (fast, production default):
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion in 2024.")
for ent in doc.ents:
    print(ent.text, ent.label_, ent.start_char, ent.end_char)   # Apple ORG, U.K. GPE, $1 billion MONEY, 2024 DATE

# Hybrid: force known entities with rules (EntityRuler runs before/after the statistical NER)
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([{"label": "DRUG", "pattern": "aspirin"},
                    {"label": "DRUG", "pattern": [{"LOWER": "acetyl"}, {"LOWER": "salicylic"}]}])
```
*(Recall from [Text Preprocessing](Text%20Preprocessing.md) that spaCy tagged `ML → ORG` — NER is probabilistic and makes mistakes.)*

**Domain-specific biomedical NER** (the lecture's pharma need — don't use general English models on clinical text):
```python
# scispaCy or Stanza with biomedical models trained on BC5CDR / i2b2 / NCBI-disease
import stanza
nlp = stanza.Pipeline("en", package="bc5cdr", processors={"ner": "bc5cdr"})  # CHEMICAL, DISEASE
doc = nlp("Aspirin can cause gastric ulcers and nausea.")
print([(e.text, e.type) for s in doc.sentences for e in s.ents])
```

**Transformer token-classification** (SOTA; mind the subword→label alignment):
```python
from transformers import pipeline
ner = pipeline("token-classification", model="dslim/bert-base-NER",
               aggregation_strategy="simple")   # merges B-/I- subwords back into whole entity spans
ner("Elon Musk founded SpaceX in California.")   # PER: Elon Musk, ORG: SpaceX, LOC: California
# Under the hood: each token's BERT vector → softmax over BIO tags; a word split into subwords is
# labeled on its FIRST subword (rest set to -100/ignored) so labels align to words.
```

---

## 6. When It Breaks

- **Ambiguity / polysemy** — `Washington` (person, city, state, org), `Apple` (fruit vs company), `Jordan` (person vs country). Only **context-aware** models resolve these; dictionaries and rules can't.
- **Boundary errors** — is it `Bank of America` or `Bank` + `America`? `New York Times` (ORG) vs `New York` (GPE)? Span boundaries are the hardest part; BILOU tagging and CRF transitions help.
- **Nested & overlapping entities** — `[University of [California]LOC]ORG`. Flat BIO can't represent nesting; you need **span-based** or layered models.
- **Novel / OOV entities** — new drugs, products, usernames never seen in training; char-level features and subword transformers cope best, gazetteers fail.
- **Domain shift** — a `CoNLL`-trained (news) model collapses on clinical, legal, or financial text. Use **domain-pretrained** models (BioBERT/scispaCy/Stanza) or annotate in-domain.
- **Capitalization over-reliance** — models leaning on casing break on all-caps, all-lowercase, or social text; cased vs uncased matters.
- **Tokenization / subword misalignment** — transformer subwords must be re-aligned to word-level labels, or your entities shift by a token.
- **Annotation inconsistency** — human labelers disagree on boundaries/types; noisy labels cap achievable F1 (the real ceiling is often the data, not the model).
- **Multilingual / code-switching** — English models and casing cues fail; use multilingual models and language routing.

---

## 7. Evaluation — Score at the Span Level

**Use entity-level precision / recall / F1, not token accuracy.** A prediction counts as correct only if the **span boundaries AND the type both exactly match** the gold entity.
```
Precision = correct entities / predicted entities      Recall = correct entities / gold entities
F1 = 2PR / (P + R)          # CoNLL-style exact-match F1 is the standard headline number
```
- **Why token accuracy lies:** `O` is ~90%+ of tokens, so a model that predicts everything `O` scores high accuracy while finding **zero** entities. Span-level F1 exposes that.
- **Partial/boundary credit:** exact-match is strict; predicting `severe` when gold is `severe nausea` is a *miss* on both P and R (some evals give partial credit — state which you use).
- **Tooling & benchmarks:** `seqeval` computes entity-level F1 from BIO sequences. Standard sets: **CoNLL-2003** (PER/ORG/LOC/MISC), **OntoNotes 5** (18 types), and biomedical **BC5CDR** (chemical/disease), **NCBI-disease**, **i2b2** (clinical).

---

## 8. Production & MLOps Notes

- **Annotation is the bottleneck**, not modeling. Labeling entity spans is slow and expensive. Mitigate with **weak supervision** (bootstrap labels from gazetteers + rules), **active learning** (label the examples the model is least sure about), and pre-annotation with a pretrained model that humans correct.
- **Start from a domain-pretrained model.** General-English NER on clinical/legal/financial text underperforms badly — reach for **scispaCy / Stanza / BioBERT / ClinicalBERT / FinBERT** and fine-tune, rather than training from scratch.
- **Speed vs accuracy:** spaCy's statistical NER is fast/cheap and often good enough; transformer NER is more accurate but needs GPU + batching. Match to your latency SLA; a hybrid (rules + small model) is frequently the sweet spot.
- **NER is step one — plan the pipeline.** Real value usually needs **entity linking** (map `aspirin` → a drug-database ID, resolving aliases) and **relation extraction** (assert *drug —causes→ adverse-event*, the pharma goal). Budget for those, not just tagging.
- **Confidence & thresholds:** expose per-entity confidence; in high-stakes settings (PII redaction, clinical) tune for **recall** (missing an entity is worse than a false positive) and keep a human in the loop.
- **Drift:** new entities (drugs, products, slang) appear constantly → rising "novel entity" miss-rate is your retraining signal; refresh gazetteers and re-fine-tune periodically.

---

## 9. Interview Lens

The question tests whether you frame NER as **sequence labeling** and know **why CRF/transitions matter** and **how to evaluate it**.

🎯 **Kill-shots:**
- *"NER is sequence labeling with **BIO tags** — B/I/O per token so multi-word spans and boundaries are recoverable."*
- *"You use a **CRF** (or a CRF head on a BiLSTM/BERT) because tags depend on neighbors — it scores the whole sequence jointly and forbids illegal transitions like O→I-PER, which a per-token softmax can't."*
- *"Always report **entity-level F1**, not token accuracy — `O` dominates, so token accuracy is meaningless."*

**Likely follow-ups:**
- *BIO vs BILOU?* Both encode spans; BILOU adds explicit Last/Unit tags, giving the model clearer end-of-entity signals and often better F1. `(certain)`
- *Why CRF over softmax on a BiLSTM?* The CRF adds a learned transition matrix + Viterbi decoding, enforcing valid label sequences and modeling `y`-dependencies the softmax treats independently. `(certain)`
- *How does BERT do NER?* Token-classification head: each token's contextual vector → softmax over BIO tags; align subwords to words (label the first subword); optionally add a CRF layer. `(certain)`
- *How do you handle a new domain (medical)?* Use domain-pretrained models (scispaCy/BioBERT/Stanza) and annotate in-domain; general models suffer domain shift. `(certain)`
- *Nested entities?* Flat BIO can't represent them — use span-based/layered models. `(likely)`
- *NER found the drug and the symptom — are you done?* No — associating them is **relation extraction** on top of NER; NER only detects and types spans. `(certain)`

---

## 10. Alternatives & How to Choose

| Approach | Needs labeled data? | Strength | Reach for it when |
|---|---|---|---|
| **Dictionary / gazetteer** | no | trivial, precise on known lists | fixed, closed entity sets (product codes, known drug list) |
| **Rule-based** | no | precise on structured patterns | regular formats, PII masking, bootstrapping labels |
| **CRF (sklearn-crfsuite)** | small | strong with feature engineering, cheap | limited labels, interpretable features, low compute |
| **BiLSTM-CRF (+char-CNN)** | medium | learns features + morphology/OOV | enough labeled data, pre-transformer setups/edge |
| **Transformer (BERT) token-classification** | medium–large | contextual, SOTA, resolves ambiguity | accuracy matters and you can afford GPU |
| **Domain-pretrained (scispaCy/Stanza/BioBERT)** | little (fine-tune) | strong on specialized text out-of-the-box | biomedical/legal/financial domains |

**Decision rule:** closed known list → **gazetteer**; structured patterns / no labels → **rules** (also to bootstrap); some labels, want cheap+explainable → **CRF**; general accuracy → **fine-tuned transformer**; specialized domain → **domain-pretrained model**. In production, a **hybrid** (rules/gazetteers guarding a statistical/transformer model) usually wins.

---

## 🧠 Self-Test

1. Why is NER framed as sequence labeling, and what does BIO tagging encode?
   <details><summary>answer</summary>Entities are multi-word spans whose token labels depend on neighbors, so you tag each token in sequence. **BIO** encodes spans + boundaries: `B-TYPE` begins an entity, `I-TYPE` continues it, `O` is outside — letting you recover exact multi-word entity spans (`k` types → `2k+1` labels).</details>

2. Why add a CRF layer instead of a plain per-token softmax?
   <details><summary>answer</summary>Tags aren't independent (`I-PER` can't follow `O`). A CRF adds a learned **transition** matrix and normalizes over the whole sequence (Viterbi decode), so it forbids illegal sequences and models label dependencies. A per-token softmax scores each token in isolation and can emit invalid tag sequences.</details>

3. Walk the NER approach ladder from simplest to SOTA.
   <details><summary>answer</summary>Dictionary/gazetteer (string match) → rule-based (pattern + context rules) → ML with **CRF** over hand features → **BiLSTM-CRF** (+char-CNN for morphology/OOV) → **pretrained transformers** (ELMo/BERT token-classification), plus **domain-pretrained** models (scispaCy/BioBERT) for specialized text; hybrids combine rules + models.</details>

4. A model scores 94% token accuracy but is useless. What happened and how should you measure NER?
   <details><summary>answer</summary>`O` is ~90%+ of tokens, so predicting mostly `O` yields high token accuracy while finding almost no entities. Measure **entity/span-level precision, recall, and F1** (exact boundary + type match), e.g. with `seqeval` / CoNLL eval.</details>

5. Your CoNLL-news NER model fails on clinical drug reviews. Why, and what do you do?
   <details><summary>answer</summary>**Domain shift** — vocabulary, entity types (DRUG/DISEASE), and style differ from news. Switch to a **domain-pretrained** model (scispaCy/Stanza/BioBERT, trained on BC5CDR/i2b2) and fine-tune on in-domain annotations.</details>

6. You've tagged `aspirin` (DRUG) and `nausea` (PROBLEM). Is the pharmacovigilance task solved?
   <details><summary>answer</summary>No. NER only **detects and types** the spans. Asserting that *aspirin causes nausea* is **relation extraction** (and mapping `aspirin` to a canonical drug ID is **entity linking**) — separate steps built on top of NER.</details>
