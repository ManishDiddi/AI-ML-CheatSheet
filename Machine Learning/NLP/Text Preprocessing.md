# Text Preprocessing — Turning Raw Text into Model-Ready Tokens

> **TL;DR.** Models eat numbers, not words. Preprocessing is the pipeline that turns messy raw text into clean, discrete units (tokens) and then into features: **clean → segment (tokenize) → normalize (stopwords, stemming/lemmatization) → vectorize (BoW/TF-IDF/embeddings)**. The single most important idea is that **preprocessing is task- and model-dependent**: a classical bag-of-words model needs *heavy* cleaning (lowercase, strip punctuation, remove stopwords, stem), while a modern **transformer needs almost none** — you feed nearly-raw text to the model's own **subword tokenizer** and let contextual embeddings do the rest. Over-cleaning silently deletes signal ("not", "!", casing, numbers). NLTK = flexible/educational; spaCy = fast/production; Hugging Face tokenizers = subword for transformers.

**Where it fits:** The **first stage of every NLP pipeline** — the bridge from raw documents to anything downstream: [Naive Bayes](../Supervised%20ML/Naive%20Bayes.md) text classification, [Text Classification with Neural Networks](../Neural%20Networks/Text%20Classification%20with%20Neural%20Networks.md), sequence models ([RNN · LSTM · Transformers](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md)), sentiment, [NER](NER.md), and topic modeling.
**Prereqs:** basic Python + regular expressions; a feel for high-dimensional sparse vectors. Downstream representation lives in [Word Embeddings](Word%20Embeddings.md).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core](#2-the-formal-core)
3. [How It Works — the Pipeline](#3-how-it-works--the-pipeline)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [When It Breaks](#6-when-it-breaks)
7. [Production & MLOps Notes](#7-production--mlops-notes)
8. [Interview Lens](#8-interview-lens)
9. [Alternatives & How to Choose](#9-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Text is **unstructured and noisy** — variable length, mixed casing, punctuation, typos, HTML, emojis, five spellings of the same idea. A model wants **fixed, discrete, numeric** input. Preprocessing is the funnel that gets you there:

```
raw text  "Machine Learning ROCKS!! Visit http://x.com in 2023 :)"
   │  clean       lowercase · strip URLs/HTML · (maybe) punctuation/numbers
   ▼
"machine learning rocks visit in"
   │  tokenize    split into units (words / subwords / sentences)
   ▼
["machine","learning","rocks","visit","in"]
   │  normalize   remove stopwords · stem OR lemmatize · POS/NER (optional)
   ▼
["machine","learn","rock","visit"]        # stemmed, stopword-free
   │  vectorize   BoW · TF-IDF · embeddings
   ▼
[0, 1, 1, 0, 2, ...]  →  a number the model can train on
```

🎯 **The framing that wins the question:** *"How much you preprocess depends on the model."* Classical ML (Naive Bayes, SVM, logistic regression on BoW/TF-IDF) is a **bag of independent words** — it has no idea "run" and "running" are related, so you *must* stem/lemmatize and drop stopwords to reduce a huge sparse vocabulary. A **transformer** already models context and morphology via subwords, so you **skip** the classical steps and hand it near-raw text through its matched tokenizer. Same raw input, opposite pipelines.

---

## 2. The Formal Core

**Vocabulary & the feature space.** Let the corpus have `N` documents. Collect the set of unique tokens → **vocabulary** `V` (size `|V|`, often 10⁴–10⁶). Every classical text vector lives in `ℝ^|V|` — huge and **sparse** (mostly zeros). Preprocessing is largely an exercise in **shrinking `|V|`** (lowercasing, stemming, stopword removal, `min_df`) so the model has fewer, denser features to learn from.

**Bag of Words (BoW).** Represent document `d` as a length-`|V|` vector of counts: `BoW(d)[t] = count(t in d)`. Order is thrown away ("dog bites man" == "man bites dog").

**TF-IDF** — down-weight ubiquitous words, up-weight distinctive ones:
```
tf-idf(t, d) = tf(t, d) · idf(t)
   tf(t, d)  = count of term t in doc d           (often √ or log-scaled)
   idf(t)    = log( N / df(t) )                    df(t) = # docs containing t
```
A term in *every* document has `df = N → idf = log(1) = 0` (killed automatically — TF-IDF is a *soft, data-driven stopword remover*). sklearn uses a **smoothed** idf `log((1+N)/(1+df(t))) + 1` and **L2-normalizes** each row so document length doesn't dominate.

**n-grams.** BoW loses order; adding **bigrams/trigrams** (`"not good"`, `"new york"`) recovers a little local order at the cost of an exploding `|V|`. Interviewers love: *"BoW can't see 'not good' as negative — bigrams fix the most damaging cases."*

**Zipf's law** — word frequency ≈ inversely proportional to rank (`freq ∝ 1/rank`). Consequence: a *tiny* set of words (the, of, is) covers most tokens → **stopwords**; and a *huge long tail* of rare words each appears once → **OOV / sparsity** problems that subword tokenization exists to solve.

**Stemming vs lemmatization** (formal distinction):
- **Stem** = chop affixes by heuristic rule. Fast, deterministic, **may not be a real word** (`studies → studi`, `caring → care` but `caress → caress`). Porter/Snowball/Lancaster.
- **Lemma** = the dictionary base form via a lexicon + morphology, **needs the POS tag** to be correct (`better →(ADJ) good`, `saw →(VERB) see` vs `saw →(NOUN) saw`). Slower, always a valid word.

---

## 3. How It Works — the Pipeline

Numbered stages. You pick which to run based on the model (§1).

1. **Acquire & decode.** Read from `.txt`/`.csv` (pandas) / `.pdf` (`pdfminer`, `PyPDF2`) / `.docx` (`docx2txt`, `python-docx`) / web (`requests` + `BeautifulSoup`). Fix **encoding** (decode as UTF-8) and apply **Unicode normalization** (`unicodedata.normalize("NFKC", …)`) so `"café"` and `"café"` (composed vs decomposed) become one token; optionally `unidecode` to strip accents.

2. **Clean (task-dependent).** Lowercase; strip HTML tags, URLs, emails, and control chars with regex; decide *deliberately* whether to remove **punctuation**, **numbers**, and **emojis** — each can be signal (see §6). `str.maketrans('', '', string.punctuation)` is the fast punctuation stripper; `re.sub(r'\s+', ' ', t).strip()` collapses whitespace.

3. **Segment / tokenize.**
   - **Sentence tokenization** (`nltk.sent_tokenize`, spaCy `doc.sents`) — needed for sentence-level tasks; the hard cases are abbreviations (`U.K.`, `Dr.`) and decimals, which statistical tokenizers handle better than a naïve split on `.`.
   - **Word tokenization** (`nltk.word_tokenize`, spaCy tokens). spaCy handles contractions and attached punctuation more gracefully (`"it's" → ["it", "'s"]`) and each token carries POS/lemma/`is_stop`.
   - **Subword tokenization** (the modern default for transformers) — **BPE**, **WordPiece** (BERT), **SentencePiece/Unigram** (T5, LLaMA). Learns a fixed vocab of frequent character chunks so any word decomposes into known pieces → **no true OOV** (`"tokenization" → token ##ization`). This one step *replaces* stemming, stopword removal, and OOV handling. It's owned by the model; deeper treatment in [RNN · LSTM · Transformers](RNN%20%C2%B7%20LSTM%20%C2%B7%20Transformers.md) and [BERT](BERT.md).

4. **Normalize (classical models only).**
   - **Stopword removal** — drop high-frequency low-signal words (`nltk.corpus.stopwords`, spaCy `token.is_stop`). *Audit the list* — it contains `not`, `no`, `against` (deadly for sentiment).
   - **Stemming** (`PorterStemmer`) or **lemmatization** (spaCy `token.lemma_`, or `WordNetLemmatizer` **with** a POS map). Pick one, not both.

5. **Linguistic enrichment (optional).** **POS tagging** (`token.pos_` coarse / `token.tag_` fine) → grammatical category; **chunking / noun-phrase extraction** (`doc.noun_chunks`) → shallow parsing into phrases; **[NER](NER.md)** (`doc.ents`) → typed entities (PERSON/ORG/GPE/DATE). These feed feature engineering, information extraction, and downstream models.

6. **Vectorize.** BoW / TF-IDF (`sklearn` `CountVectorizer` / `TfidfVectorizer`) for classical models; **dense embeddings** (Word2Vec/GloVe/fastText → [Word Embeddings](Word%20Embeddings.md); contextual → BERT) for neural. This is the handoff out of "preprocessing" into "representation."

---

## 4. Worked Example

Run one messy paragraph through the classical pipeline (numbers are the *actual* library outputs).

Input (two texts joined): a clean ML definition + `"Machine learning rocks! It's revolutionizing the world in 2023 (and beyond!). Visit our site: http://example.com … We collected 1,234 data points. Softbank and Google are major players."`

| Stage | Effect on the noisy line |
|---|---|
| lowercase | `machine learning rocks! it's revolutionizing the world in 2023 …` |
| remove URLs (regex) | `visit our site:  for more info.` |
| remove punctuation | `machine learning rocks its revolutionizing the world in 2023 and beyond` |
| remove numbers | `… collected  data points` (the `1234` is gone) |
| collapse whitespace | single-spaced, `len = 1186` chars |
| word tokenize | `['machine','learning','ml','is','a','field',…]` |
| stopword removal | `['machine','learning','ml','field','study','artificial',…]` (dropped is/a/of/in) |

Now **stem vs lemma** on the stopword-free tokens — the money comparison:

```
tokens :  machine  learning  study    artificial  intelligence  concerned  algorithms  data
Porter :  machin   learn     studi    artifici    intellig      concern     algorithm   data
spaCy  :  machine  learning  study    artificial  intelligence  concern     algorithm   datum
```
Read it: the stemmer produces **non-words** (`machin`, `studi`, `artifici`) — fine for matching, ugly for humans. The lemmatizer keeps **valid words** and even does `data → datum` (correct Latin singular — a classic "gotcha" that surprises people). Both collapse `algorithms → algorithm`, which is the *point*: fewer vocabulary entries, denser features.

**NER on the raw text** shows the tools are imperfect: spaCy tags `ML → ORG` (wrong — it's an abbreviation), `2023 → DATE`, `1,234 → CARDINAL`, `Google → ORG`. Great illustration that enrichment steps are *probabilistic*, not oracle.

---

## 5. Code / Implementation

**NLTK + spaCy — the explicit pipeline** (educational; you see each step):
```python
import re, string, nltk, spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nlp   = spacy.load("en_core_web_sm")          # integrated pipeline: tok+POS+lemma+NER
stops = set(stopwords.words("english"))
stem  = PorterStemmer()

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)            # numbers (drop only if noise for your task!)
    return re.sub(r"\s+", " ", text).strip()

doc      = nlp(clean(raw))                     # one pass gives everything below
lemmas   = [t.lemma_ for t in doc if not t.is_stop]     # spaCy: lemma, no manual POS
stems    = [stem.stem(t.text) for t in doc if not t.is_stop]  # NLTK stemmer for contrast
entities = [(e.text, e.label_) for e in doc.ents]       # NER for free
```

**scikit-learn — the production one-liner** (does tokenize + lowercase + stopwords + n-grams + TF-IDF + L2-norm in a single fitted object):
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(lowercase=True, stop_words="english",
                      ngram_range=(1, 2),     # unigrams + bigrams → catches "not good"
                      min_df=2, max_df=0.9,    # drop ultra-rare AND ultra-common terms
                      sublinear_tf=True)       # 1+log(tf), dampens frequent-word dominance
X_train = vec.fit_transform(train_texts)       # FIT on train only — vocabulary is learned here
X_test  = vec.transform(test_texts)            # TRANSFORM test — never refit (leakage!)
```

**Transformers — you barely preprocess** (use the model's matched subword tokenizer; do *not* stem/stopword/lowercase):
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok(raw_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
# handles subword splitting, [CLS]/[SEP], attention_mask — matched to the pretrained weights
```

---

## 6. When It Breaks

- **Over-cleaning deletes signal.** Removing `not`/`no`/`n't` flips sentiment (`"not good"` → `"good"`). Stripping `!!!` and lowercasing erases emphasis/emotion. Deleting numbers wrecks tasks where numbers matter (finance, dates, dosages). **Clean for the task, not by reflex.** 🎯
- **Stemming is crude.** Over-stemming conflates unrelated words (`universe`, `university → univers`); under-stemming misses (`ran` stays `ran`). It's rule-based and language-specific.
- **Lemmatization without POS defaults to noun** → `WordNetLemmatizer().lemmatize("running")` returns `"running"`, not `"run"`; you must pass `pos='v'`. spaCy avoids this by using context.
- **Lowercasing hurts NER & meaning.** `US` (country) → `us` (pronoun); `Apple` (company) vs `apple` (fruit); `IT` (field) → `it`. Cased models exist for exactly this.
- **Stopword lists are blunt.** Standard lists remove negations and domain-critical tokens; some drop `"computer"`'s neighbors you actually need. Always inspect and customize.
- **Train/serve skew.** If inference cleans text differently than training (different regex, tokenizer version, stopword list), features silently shift → quiet accuracy loss. Ship *one* pipeline object used both places.
- **Vocabulary leakage.** Calling `fit_transform` on test (or on train+test together) leaks corpus statistics; always `fit` on train, `transform` on the rest.
- **Language & script assumptions.** English stopwords/tokenizers fail on other languages; **Chinese/Japanese/Thai have no spaces** (need dedicated segmenters); RTL scripts, emojis, and code-switching break naïve regex.
- **The big modern trap:** running a *classical* heavy pipeline before a transformer. Stemming/stopword-removal/lowercasing **strip information the model was pretrained to use**, and hand-tokenizing breaks alignment with the model's subword vocabulary. Rule: **match the model's own tokenizer; preprocess minimally.** 🎯

---

## 7. Production & MLOps Notes

- **Persist the fitted vectorizer/tokenizer with the model** (`joblib.dump`) and **version** it — the vocabulary *is* part of the model. A retrained model with a new vocab needs its vectorizer redeployed together.
- **One deterministic pipeline, train and serve.** Wrap cleaning + vectorizer in an `sklearn.Pipeline` (or a spaCy pipeline) so there is literally no way for the two environments to diverge.
- **Speed at scale:** use `nlp.pipe(texts, batch_size=…, n_process=…)` for batched/multiprocessed spaCy, and `nlp.select_pipes(disable=[…])` to switch off components you don't use (parser/NER) — often a 5–10× speedup. Preprocessing is cheap CPU; don't pay for pipes you ignore.
- **Control vocabulary growth:** `min_df`, `max_df`, `max_features`, and hashing (`HashingVectorizer`) bound memory for streaming/online settings.
- **Monitor OOV / vocabulary drift.** A rising **out-of-vocabulary rate** at inference is an early, cheap signal of distribution shift (new slang, new products, a new language showing up) — alert on it and use it as a retraining trigger.
- **Robustness:** guard against decoding errors, empty strings after aggressive cleaning (→ all-zero vectors → NaNs), and mixed languages (route via `langdetect`/`fasttext-langid`).
- **Reproducibility:** pin NLTK data and spaCy model versions; a silent model-version bump changes lemmas/NER and thus features.

---

## 8. Interview Lens

The question is really testing whether you know **preprocessing is a modeling decision, not a checklist**.

🎯 **Kill-shots:**
- *"Stemming chops affixes by rule — fast but can produce non-words; lemmatization maps to a real dictionary base form using the word's POS — slower but valid. Use lemmatization when the output must be human-readable or POS-aware, stemming when you only need matching."*
- *"For transformer models I skip classical preprocessing and use the model's own subword tokenizer — stemming, stopword removal, and lowercasing would delete signal the model was pretrained on."*

**Likely follow-ups (crisp answers):**
- *BoW vs TF-IDF?* BoW = raw counts; TF-IDF re-weights by `log(N/df)` so common words fade and distinctive words stand out — a soft, automatic stopword remover. `(certain)`
- *Why remove stopwords — and when not to?* Shrinks the sparse vocabulary and denoises classical models; **don't** for sentiment (kills `not`), for transformers (they use context), or when phrases matter. `(certain)`
- *How do you handle OOV / rare words?* Classical: `min_df`, an `<UNK>` token, or char n-grams (fastText). Modern: **subword tokenization** decomposes any word into known pieces → no true OOV. `(certain)`
- *Why did subword tokenization win?* Fixed vocab + full coverage + morphology awareness in one learned step, matched to the model's weights. `(likely)`
- *Where does the pipeline usually break in prod?* **Train/serve skew** — different cleaning/tokenizer at inference than training. `(certain)`

---

## 9. Alternatives & How to Choose

| Tool | Best at | Reach for it when |
|---|---|---|
| **NLTK** | breadth, teaching, corpora (WordNet, VADER) | learning, research, custom/algorithm-swapping pipelines |
| **spaCy** | fast, accurate, integrated pipeline (tok+POS+lemma+NER+chunks) | production classical NLP, information extraction at scale |
| **scikit-learn** | `CountVectorizer` / `TfidfVectorizer` end-to-end | classical ML models (NB, SVM, LogReg) on BoW/TF-IDF |
| **Hugging Face `tokenizers`** | subword (BPE/WordPiece/SentencePiece) | anything feeding a transformer / [BERT](BERT.md) / LLM |
| **gensim** | `simple_preprocess`, phrase detection, LDA/Word2Vec | topic modeling, training your own word embeddings |

**Decision rule:** *Classical model?* → clean heavily with spaCy/NLTK, vectorize with sklearn TF-IDF (bigrams, `min_df`). *Transformer?* → minimal cleaning, feed the model's matched subword tokenizer, skip stemming/stopwords/lowercasing. When in doubt, **preprocess as little as the model allows** and let training discover what matters.

---

## 🧠 Self-Test

1. Why does the *amount* of preprocessing depend on the model?
   <details><summary>answer</summary>Classical BoW/TF-IDF models treat words as independent, unaware that "run"/"running" relate or that context flips meaning — so you must stem/lemmatize, drop stopwords, and shrink the vocabulary. Transformers already capture morphology (subwords) and context (attention), so classical steps *remove* signal; you feed near-raw text to the model's matched tokenizer.</details>

2. Stemming vs lemmatization — define both and give the deciding factor.
   <details><summary>answer</summary>Stemming = rule-based affix chopping: fast, deterministic, may yield non-words (`studi`, `artifici`). Lemmatization = dictionary base form via lexicon + **POS**: slower, always valid words (`better→good`, `data→datum`). Choose lemmatization when output must be readable/POS-correct; stemming when you only need matching and want speed.</details>

3. What does TF-IDF's `idf = log(N/df(t))` accomplish, and what happens to a word in every document?
   <details><summary>answer</summary>It down-weights common words and up-weights distinctive ones. A word in all `N` docs has `df=N`, so `idf=log(1)=0` — it's zeroed out automatically. TF-IDF is thus a soft, data-driven stopword remover.</details>

4. You lowercase, strip punctuation, and remove stopwords before a sentiment model and accuracy drops. Why?
   <details><summary>answer</summary>You deleted signal: stopword lists contain **negations** (`not`, `no`, `n't`) so `"not good"` becomes `"good"`; stripping `!!!` and casing removes emphasis/emotion. Sentiment needs negation and intensity — clean gently (keep negations, consider bigrams, or use a transformer).</details>

5. What is subword tokenization and which classical problems does it eliminate?
   <details><summary>answer</summary>Learning a fixed vocab of frequent character chunks (BPE/WordPiece/SentencePiece) so any word decomposes into known pieces (`tokenization → token ##ization`). It eliminates **OOV** (full coverage), and folds **stemming/normalization** into the representation — one learned step matched to the model's weights.</details>

6. Name two production failures unique to text pipelines and how you prevent each.
   <details><summary>answer</summary>(1) **Train/serve skew** — different cleaning/tokenizer at inference; prevent by shipping one `Pipeline` object used in both places and pinning tokenizer/model versions. (2) **Vocabulary leakage** — `fit`ting the vectorizer on test/train+test; prevent by `fit` on train only, `transform` elsewhere. Bonus: monitor **OOV rate** as a drift signal.</details>
