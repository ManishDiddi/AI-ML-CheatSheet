# Sentiment Analysis & Topic Modeling — What People Feel and What They Talk About

> **TL;DR.** Two complementary text-mining tasks over the same reviews/tweets/tickets. **Sentiment analysis** answers *how* people feel (positive/negative/neutral, or fine-grained emotions); **topic modeling** answers *what* they're talking about (latent themes). Sentiment has three tiers: **lexicon-based** (TextBlob/VADER/AFINN/NRC — no training, transparent, but brittle on negation, sarcasm, and domain slang), **supervised ML** (TF-IDF + a classifier — needs labels, learns your domain), and **transformer** (fine-tuned BERT / zero-shot LLM — contextual, SOTA). Topic modeling's workhorse is **LDA** — model each document as a *mixture of topics* and each topic as a *distribution over words* — with modern embedding-based **BERTopic** as the upgrade. The naive "count words / wordcloud" approach fails because stopwords and generic praise dominate; you climb from POS-noun extraction → noun phrases → LDA to get real topics.

**Where it fits:** the two most common **text-analytics** deliverables on unstructured feedback. They consume clean tokens from [Text Preprocessing](Text%20Preprocessing.md); sentiment is a text **classification/scoring** task (baseline: [Naive Bayes](../Supervised%20ML/Naive%20Bayes.md)); topic modeling is **unsupervised** theme discovery. Both are upgraded by the representations in [Word Embeddings](Word%20Embeddings.md) and [Embeddings](../../AI%20Engineering/Embeddings.md).
**Prereqs:** [Text Preprocessing](Text%20Preprocessing.md) (tokenize, stopwords, POS), TF-IDF & bag-of-words, [Classification Metrics](../Supervised%20ML/Classification%20Metrics.md) (for evaluating ML sentiment), a feel for probability distributions (for LDA).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [Sentiment — The Three Approaches](#2-sentiment--the-three-approaches)
3. [Lexicon-Based Sentiment in Depth](#3-lexicon-based-sentiment-in-depth)
4. [Supervised & Transformer Sentiment](#4-supervised--transformer-sentiment)
5. [Worked Example — Sentiment](#5-worked-example--sentiment)
6. [Topic Modeling — Why Naive Approaches Fail](#6-topic-modeling--why-naive-approaches-fail)
7. [LDA — The Formal Core](#7-lda--the-formal-core)
8. [LDA Implementation & Modern Alternatives](#8-lda-implementation--modern-alternatives)
9. [When It Breaks](#9-when-it-breaks)
10. [Production & MLOps Notes](#10-production--mlops-notes)
11. [Interview Lens](#11-interview-lens)
12. [Alternatives & How to Choose](#12-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Two orthogonal questions about a pile of text:

```
                 "I love the cappuccino but the place is too noisy."
   SENTIMENT  →  how do they feel?   positive about coffee, negative about noise
   TOPIC      →  what's discussed?   {coffee/drinks}, {ambience/noise}
```

- **Sentiment** = a **scoring/classification** problem: map text → polarity (`−1…+1`), a class (`pos/neu/neg`), or an emotion vector (joy/anger/fear…). Supervised-flavored (you have a notion of "right answer").
- **Topic modeling** = an **unsupervised clustering** problem: discover latent themes *without labels*, and describe each document as a blend of them.

The coffee-shop line also exposes the ceiling of the simple versions: real opinions are **aspect-based** (positive on one aspect, negative on another), which a single document-level score can't express — that's where **ABSA** and transformers come in (§4, §9). 🎯

---

## 2. Sentiment — The Three Approaches

| Approach | How it works | Needs labels? | Strengths | Weaknesses |
|---|---|---|---|---|
| **Lexicon / rule-based** | look words up in a sentiment dictionary, aggregate scores | **No** | zero training, transparent, instant baseline | misses negation/sarcasm/domain slang, no learning |
| **Supervised ML** | TF-IDF / embeddings → classifier (NB, LogReg, SVM) | **Yes** | learns *your* domain, calibratable, strong | needs labeled data, retrain per domain |
| **Transformer / LLM** | fine-tune BERT or prompt an LLM | fine-tune: yes; zero-shot: no | contextual, handles negation/sarcasm, SOTA | compute/cost, heavier ops |

**The decision:** no labels & need it today → lexicon (VADER for social text). Have labels & a fixed domain → supervised ML. Need top accuracy / aspect-level / multilingual and can afford it → transformer. Teams often ship a **lexicon baseline first**, then graduate to ML once labels accumulate.

---

## 3. Lexicon-Based Sentiment in Depth

A **sentiment lexicon** is a dictionary mapping words → sentiment values. You aggregate the matched words into a document score. Four you should know:

- **TextBlob** — returns **polarity ∈ [−1, +1]** (negative↔positive) and **subjectivity ∈ [0, 1]** (factual↔opinionated). Built on the `pattern` lexicon; averages word polarities with *some* handling of negation and intensifiers.
  `"The movie was a masterpiece, but too long and slightly boring"` → polarity ≈ **0.1** (mildly positive — "masterpiece" pulled down by "boring/too long"), subjectivity ≈ **0.75** (opinionated). `"The sun is 93 million miles from Earth"` → polarity **0.0**, subjectivity **0.0** (pure fact).
- **AFINN** — each word has an integer **valence from −5 to +5** (`fantastic +4`, `terrible −3`); the document score is the **sum** of matched words. Fast and simple; great for short, informal text (tweets), no emotion breakdown.
- **NRC Emotion Lexicon (EmoLex)** — ~**14,000 words**, each tagged (binary) with **8 emotions** (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) + **2 sentiments** (positive, negative). **Multi-label** — a word can carry several emotions. Gives `raw_emotion_scores` (counts) and `affect_frequencies` (each emotion's share of all emotional tags). Use it when you need *which emotion*, not just good/bad. `"I love the design but I'm scared about durability"` → joy/trust **and** fear.
- **VADER** *(the tool the lecture skips — know it)* — a rule-based analyzer **tuned for social media**. It's the robust lexicon because it explicitly handles **negation** ("not good"), **intensifiers/boosters** ("very", "extremely"), **ALL-CAPS**, **punctuation** ("good!!!" > "good"), and **emoji/slang**. Returns `pos/neu/neg` proportions plus a normalized **`compound ∈ [−1,+1]`** (the headline score; `≥0.05` positive, `≤−0.05` negative). `(certain)`

**The formal weakness of naive lexicons.** A pure sum/average, `score(doc) = Σ valence(wᵢ)`, is **bag-of-words** — it can't see word order, so `"not good"` scores like `"good"`. TextBlob and VADER patch this with negation/booster *rules*; AFINN/NRC don't. This is exactly why supervised and transformer models win on hard text.

| Lexicon | Output | Handles negation? | Best for |
|---|---|---|---|
| TextBlob | polarity + subjectivity | partially | quick doc-level polarity + objectivity |
| AFINN | summed valence | no | short informal text, speed |
| NRC | 8 emotions + 2 sentiments | no | emotion analysis (not just polarity) |
| VADER | pos/neu/neg + compound | **yes** (rules) | social media, tweets, reviews |

---

## 4. Supervised & Transformer Sentiment

The lecture stops at lexicons; production sentiment usually doesn't. Fill the gap:

**Supervised ML (the standard baseline).**
1. Get **labeled** examples (review → pos/neg/neu). 2. [Preprocess](Text%20Preprocessing.md) and vectorize with **TF-IDF** (add bigrams so `"not good"` is a feature). 3. Train a classifier — **[Naive Bayes](../Supervised%20ML/Naive%20Bayes.md)**, Logistic Regression, or linear SVM. 4. Evaluate with **[precision/recall/F1](../Supervised%20ML/Classification%20Metrics.md)**, watching **class imbalance** (reviews skew positive).
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
clf = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=3),   # bigrams capture negation
                    LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(train_texts, train_labels)                                   # learns YOUR domain's sentiment words
```
Why it beats lexicons: it learns domain-specific polarity (`"unpredictable"` is bad for a car, good for a thriller) and, with n-grams, local negation.

**Transformer / LLM (SOTA).** Fine-tune **[BERT](BERT.md)** (`AutoModelForSequenceClassification`) on labeled data, or **zero/few-shot prompt an LLM**. These read the whole sentence in context, so they nail negation, intensity, and much sarcasm — and enable **Aspect-Based Sentiment Analysis (ABSA)**: extract *(aspect, sentiment)* pairs like *(cappuccino, +)*, *(noise, −)* from the coffee-shop review, which no document-level score can do. See [RNN · LSTM · Transformers](../RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) and [BERT](BERT.md).

🎯 *"Lexicons need no data but can't learn negation or domain; supervised TF-IDF+classifier learns your domain from labels; a fine-tuned transformer reads context and enables aspect-level sentiment."*

---

## 5. Worked Example — Sentiment

Five sentences, three tools, watch where lexicons crack:

| Sentence | AFINN (sum) | VADER compound | Note |
|---|---|---|---|
| "The service was amazing and the staff wonderful!" | +7 | +0.87 | easy positive |
| "The food was horrible and I hated the experience." | −6 | −0.83 | easy negative |
| "The hotel is okay, nothing special." | 0 | ≈ −0.05 | neutral; AFINN sees no lexicon words |
| **"The movie was not good."** | +3 (`good`=+3) ❌ | −0.34 ✅ | **naive sum misreads negation**; VADER's rule catches it |
| **"Yeah, great. Another delayed flight."** | +3 (`great`) ❌ | +0.6 ❌ | **sarcasm** — both lexicons fail; needs ML/transformer |

Takeaway: AFINN's `Σ valence` flips on `"not good"`; VADER's negation rule fixes that case but **neither** handles sarcasm — the ceiling that motivates supervised/transformer models.

---

## 6. Topic Modeling — Why Naive Approaches Fail

Business case: *"You're a data scientist at Amazon; from thousands of Musical-Instruments reviews, tell sellers **what is being talked about**."* The lecture climbs a ladder, each rung fixing the last one's failure — memorize this progression:

```
1. Count words + wordcloud        → top words are STOPWORDS (the, is, and) → useless
2. Remove stopwords + wordcloud   → now dominated by SENTIMENT words ("good","liked") → only "guitar","strings" useful
3. POS-tag, keep NOUNs (spaCy)    → "guitar, strings, price, sound" → nouns are what's talked ABOUT
4. Noun PHRASES (noun_chunks)     → "pop sounds","battery life" → phrases > single nouns (constituency/CFG: NP→Noun Noun)
5. Dependency parse (attach head) → context via parent word → limited extra signal
6. LDA                            → actual latent TOPICS, directly
```

The insight: **what people talk about lives in the nouns/noun-phrases**, and sentiment/stopwords are noise for *this* question. POS tagging (via spaCy's pretrained model — an HMM+Viterbi or neural sequence tagger trained on a treebank) and **noun-phrase extraction** get you usable terms, but only **LDA** yields general *themes*.

---

## 7. LDA — The Formal Core

**Latent Dirichlet Allocation** is a **generative** model. It assumes each document was *written* like this:

```
For each document d:
   θ_d ~ Dirichlet(α)          # document's mix over K topics   (e.g. 50% guitar + 30% amp + 20% price)
   For each word position:
       z ~ Categorical(θ_d)    # pick a topic
       w ~ Categorical(β_z)    # pick a word from that topic's word-distribution
   β_k ~ Dirichlet(η)          # each topic is a distribution over the vocabulary
```

- `K` = number of topics (you choose). `θ_d` = **document–topic** distribution (row sums to 1). `β_k` = **topic–word** distribution. A "topic" is literally a **probability distribution over words** — no human label; you name it by eyeballing its top words.
- **α** (document–topic Dirichlet) controls how *many topics per document* — low α → each doc is about few topics. **η/β** (topic–word Dirichlet) controls how *many words per topic* — low η → each topic concentrates on few words. Both are sparsity priors.
- **Learning** *inverts* this story: given the observed words, infer the θ and β that most likely generated them (via **variational inference** or **collapsed Gibbs sampling**; think KL-divergence minimization between the generated and observed word distributions).

**Basket analogy** (the lecture's): one basket picks *topic balls* for each document (governed by α); `K` more baskets each hold *word balls* for one topic (governed by η). You "regenerate" documents by drawing topics then words, and tune the baskets until regenerated docs match the real ones.

> **LDA is a bag-of-words model** — order is ignored — and it needs a *cleaned, mostly-noun* vocabulary to produce coherent topics (garbage/ stopwords in → mushy topics out).

**LSA/LSI connection:** the count-based cousin is **Latent Semantic Analysis** — TF-IDF matrix → **SVD** → latent dimensions as topics. That's the *same SVD* you saw building [Word Embeddings](Word%20Embeddings.md); LDA is its probabilistic upgrade. `(certain)`

---

## 8. LDA Implementation & Modern Alternatives

**gensim LDA** (the lecture's path — tokenized, stopword-removed docs):
```python
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
dictionary = Dictionary(tokenized_reviews)                 # word ↔ id vocabulary
corpus     = [dictionary.doc2bow(doc) for doc in tokenized_reviews]   # each doc → bag-of-words counts
lda = LdaModel(corpus, num_topics=10, id2word=dictionary,  # α, η default to symmetric priors
               passes=10, random_state=42)
lda.print_topics()      # each topic = weighted word list; weight = word's contribution to the topic
```
Each printed topic is like `0.04*"guitar" + 0.03*"string" + 0.02*"tune" + …` — you read the top words and label it "Guitars/Strings."

**Choosing `K`.** Don't eyeball — sweep `num_topics` and pick the peak **coherence score** (`c_v` or UMass via gensim's `CoherenceModel`); coherence measures how semantically related a topic's top words are. **Perplexity** is available but correlates poorly with human judgment. Visualize/debug with **pyLDAvis** (inter-topic distance + top terms).

**Modern alternatives (know these):**
- **NMF** — factor the TF-IDF matrix into non-negative topic × word parts; often crisper topics than LDA on short texts, deterministic.
- **LSA/LSI** — SVD on TF-IDF (§7); fast, but topics can have negative weights and are less interpretable.
- **BERTopic / Top2Vec** — embed documents with [sentence embeddings](../../AI%20Engineering/Embeddings.md), cluster (HDBSCAN), then extract topic words (c-TF-IDF). **Handles short text, needs no `K` upfront, and gives far more coherent topics** — the current default when you can afford embeddings. `(likely)`

---

## 9. When It Breaks

**Sentiment:**
- **Negation** — "not good", "hardly recommend" flip meaning; naive lexicon sums miss it (bigrams / VADER rules / transformers fix it).
- **Sarcasm & irony** — "Yeah, great, another delay" reads positive lexically; only context-aware models cope, and even they struggle.
- **Domain polarity** — "unpredictable" is negative for a car, positive for a thriller; lexicons are domain-blind, supervised models learn it.
- **Aspect mixing** — one review, opposite sentiments per aspect; a single doc score is wrong by construction → use ABSA.
- **Neutral class & thresholds** — collapsing to pos/neg hides the large neutral mass; calibrate the compound/probability threshold.
- **Class imbalance** — reviews skew positive; accuracy misleads, use F1 / balanced metrics.
- **Multilingual / code-switching / emoji** — English lexicons fail; use multilingual transformers.

**Topic modeling:**
- **Choosing `K`** — too few → blended mush; too many → fragmented/duplicate topics. Use coherence, not guesswork.
- **Incoherent topics** — from dirty vocabulary (stopwords, rare tokens, no noun-filtering) or too little data; clean harder, filter extremes (`no_below`/`no_above`).
- **Short documents** (tweets, titles) — LDA's word co-occurrence signal is too thin; prefer NMF or **BERTopic**.
- **Non-determinism** — LDA is stochastic; fix `random_state` and `passes` for reproducibility.
- **Topics ≠ labels** — the model gives word clusters; a human still has to *name* and validate them.

---

## 10. Production & MLOps Notes

- **Sentiment ladder in prod:** ship **VADER/lexicon** as a day-one baseline, log predictions, collect corrections → train a **TF-IDF + classifier**, then a **fine-tuned transformer** once accuracy matters. Always keep a human-labeled **gold set** to measure against; sentiment "accuracy" claims are meaningless without one.
- **Monitor drift:** new slang, products, and events shift sentiment vocabulary and topic mixes. Track OOV rate, class-distribution drift, and (for topics) topic prevalence over time; retrain on a schedule or trigger.
- **Aspect-level in practice:** most business value is ABSA ("delivery is slow but product is great"), not one score — plan for it if feedback drives product decisions.
- **Topic modeling ops:** LDA supports **online/streaming** updates (`LdaModel.update`) for growing corpora; recompute **coherence** each run and watch for **topic drift** (a stable topic's word list changing) — often the earliest signal of a shifting conversation. Persist the `Dictionary` + model together (the id↔word map is part of the model).
- **Labeling topics:** auto-name topics from top words (or an LLM) but keep a human in the loop; wrong labels silently mislead stakeholders.
- **Cost/latency:** lexicon & LDA are cheap CPU; transformers/BERTopic need GPU and batching. Match the tool to the SLA.

---

## 11. Interview Lens

The question tests whether you can **pick the right tool for the data you actually have** and explain LDA cleanly.

🎯 **Kill-shots:**
- *"Lexicon methods need no training but can't handle negation, sarcasm, or domain-specific polarity; supervised TF-IDF+classifier learns your domain from labels; transformers add context and aspect-level sentiment."*
- *"LDA models each document as a Dirichlet mixture of topics and each topic as a distribution over words, then infers those distributions from the observed words."*

**Likely follow-ups:**
- *Why does a naive lexicon fail on "not good"?* It's bag-of-words — it sums `good`'s positive score and never sees the preceding "not". Fix with bigrams, VADER's negation rules, or a contextual model. `(certain)`
- *TextBlob polarity vs subjectivity?* Polarity `[−1,1]` = how positive/negative; subjectivity `[0,1]` = opinion vs fact. `(certain)`
- *How do you choose the number of LDA topics?* Sweep `K`, pick the peak **coherence** (c_v); perplexity is unreliable. `(certain)`
- *LDA vs LSA vs NMF vs BERTopic?* LSA = SVD on TF-IDF; LDA = probabilistic/generative; NMF = non-negative factorization (crisper on short text); BERTopic = embeddings + clustering (best coherence, no fixed K). `(likely)`
- *What is aspect-based sentiment?* Extracting sentiment *per aspect* ("battery good, screen bad") instead of one doc-level label. `(certain)`
- *Why extract nouns before topic modeling?* Nouns/noun-phrases are *what's discussed*; stopwords and sentiment words are noise for the "what" question. `(certain)`

---

## 12. Alternatives & How to Choose

**Sentiment:**

| Method | Reach for it when |
|---|---|
| **VADER / TextBlob / AFINN / NRC** | no labels, need a baseline now; social text (VADER); emotion breakdown (NRC) |
| **TF-IDF + [Naive Bayes](../Supervised%20ML/Naive%20Bayes.md)/LogReg/SVM** | you have labels and a fixed domain; want calibrated, cheap, explainable |
| **Fine-tuned [BERT](BERT.md) / LLM** | top accuracy, negation/sarcasm, aspect-based, multilingual, and you can afford it |

**Topic modeling:**

| Method | Reach for it when |
|---|---|
| **LDA (gensim)** | medium-to-long documents, interpretable probabilistic topics, the standard baseline |
| **NMF / LSA** | short docs (NMF) or a fast linear-algebra baseline (LSA = SVD on TF-IDF) |
| **BERTopic / Top2Vec** | short/noisy text, want best coherence and no preset `K`, can afford embeddings |

**Decision rule:** sentiment → climb *lexicon → supervised → transformer* as labels and accuracy needs grow; topic modeling → **LDA** to start, **BERTopic** when you have embeddings and short/noisy text. The content-based **recommendation / distance-metrics** material bundled in the V2 lecture is a different problem — see [Recommendation Systems](../Recommendation%20Systems/Recommendation%20Systems.md) and the cosine-vs-dot treatment in [Embeddings](../../AI%20Engineering/Embeddings.md).

---

## 🧠 Self-Test

1. Sentiment analysis vs topic modeling — one line each, and which is supervised-flavored?
   <details><summary>answer</summary>Sentiment = *how* people feel (polarity/emotion), a scoring/classification task with a notion of ground truth. Topic modeling = *what* they discuss (latent themes), an **unsupervised** discovery task with no labels.</details>

2. Name the four lexicons from the lecture and what each outputs; which handles negation?
   <details><summary>answer</summary>**TextBlob** → polarity `[−1,1]` + subjectivity `[0,1]` (partial negation handling). **AFINN** → summed word valence `−5…+5` (no). **NRC** → 8 emotions + 2 sentiments, multi-label (no). **VADER** → pos/neu/neg + compound, **rule-based negation/intensifier/emoji handling** (yes) — the go-to for social media.</details>

3. Why does `Σ valence(wᵢ)` misclassify "The movie was not good," and give two fixes.
   <details><summary>answer</summary>It's bag-of-words: it adds `good`'s positive score and never sees "not," so it flips positive. Fixes: (1) TF-IDF with **bigrams** so "not good" is its own feature, or use VADER's negation rules; (2) a **contextual/transformer** model that reads word order.</details>

4. Explain LDA's generative story and what α and η control.
   <details><summary>answer</summary>Each document draws a topic mixture `θ ~ Dirichlet(α)`; each word picks a topic from `θ`, then a word from that topic's word distribution `β ~ Dirichlet(η)`. Learning infers θ and β from observed words. **α** controls topics-per-document sparsity; **η** controls words-per-topic sparsity.</details>

5. You count words in reviews and the top terms are "the, is, good, liked." Walk the fixes to get real topics.
   <details><summary>answer</summary>Remove stopwords (kills "the/is"); then keep **nouns/PROPN** via POS tagging (drops sentiment words "good/liked"); extract **noun phrases** for multi-word themes ("battery life"); finally run **LDA** (or BERTopic) for latent topics. Nouns/phrases capture *what's discussed*.</details>

6. How do you choose the number of LDA topics, and when would you drop LDA for BERTopic?
   <details><summary>answer</summary>Sweep `num_topics` and pick the peak **coherence** (c_v/UMass), not perplexity. Switch to **BERTopic** for short/noisy documents (tweets, titles) where LDA's co-occurrence signal is too thin, or when you want higher coherence and no preset `K` and can afford embeddings.</details>
