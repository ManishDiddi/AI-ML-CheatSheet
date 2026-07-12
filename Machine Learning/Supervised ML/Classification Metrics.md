# Classification Metrics — How You Know a Classifier Actually Works

> **TL;DR.** Accuracy lies under class imbalance (predict "all majority" and score 99%). Everything trustworthy is built from the **confusion matrix**: **precision** = of what you flagged positive, how much was right (punishes false positives); **recall** = of the real positives, how many you caught (punishes false misses); **F1** = their harmonic mean (punishes either one dropping). These are all measured at *one threshold*; **ROC-AUC** and **PR-AUC** summarise performance across *all* thresholds. Pick the metric from the *cost of errors*: recall for cancer/fraud (a miss is deadly), precision for spam-to-inbox (a false alarm is costly), F1/PR-AUC when positives are rare.

**Where it fits:** The measurement layer for every classifier — [Logistic Regression](Logistic%20Regression.md), [trees](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md), [[Naive Bayes]], [[KNN]]. Choosing the wrong metric is how good-looking models fail in production.
**Prereqs:** [Logistic Regression](Logistic%20Regression.md) (probabilities, thresholding), the notion of positive/negative class.

---

## Table of Contents
1. [Intuition / Mental Model — the Confusion Matrix](#1-intuition--mental-model--the-confusion-matrix)
2. [The Formal Core — every metric from four cells](#2-the-formal-core--every-metric-from-four-cells)
3. [The Precision–Recall Tradeoff](#3-the-precisionrecall-tradeoff)
4. [Worked Example — bank fraud](#4-worked-example--bank-fraud)
5. [ROC, AUC, and PR-AUC](#5-roc-auc-and-pr-auc)
6. [Multiclass & Probability Metrics](#6-multiclass--probability-metrics)
7. [When It Breaks — the traps](#7-when-it-breaks--the-traps)
8. [Production & MLOps Notes](#8-production--mlops-notes)
9. [Interview Lens](#9-interview-lens)
10. [Which Metric When — the decision guide](#10-which-metric-when--the-decision-guide)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model — the Confusion Matrix

Every classification metric is a ratio of four counts. Fix a **positive class** (the thing you're detecting: spam, fraud, cancer), then bucket every prediction:

```
                        PREDICTED
                    Positive     Negative
              ┌──────────────┬──────────────┐
   ACTUAL  P  │  TP          │  FN          │   ← actual positives = TP + FN
              │  (caught it) │ (missed it)  │      = row 1
              ├──────────────┼──────────────┤
           N  │  FP          │  TN          │   ← actual negatives = FP + TN
              │ (false alarm)│ (correct no) │      = row 2
              └──────────────┴──────────────┘
                 predicted pos    predicted neg
                 = TP + FP        = FN + TN
```

- **TP / TN** — correct (the diagonal). **FP** = false alarm = **Type I error**. **FN** = miss = **Type II error**.
- 🎯 The whole game: **which error hurts more?** A false alarm (FP) or a miss (FN)? That single business judgment picks your metric. `(certain)`
- Memory hook: **precision reads a column** (of what I *predicted* positive…), **recall reads a row** (of what's *actually* positive…). Neither uses TN — that's why both survive when negatives swamp the data.

---

## 2. The Formal Core — every metric from four cells

```
Accuracy    = (TP + TN) / (TP+TN+FP+FN)      # overall correctness — misleading under imbalance
Precision   = TP / (TP + FP)                  # of predicted-positive, how many are right (↓ FP)
Recall/TPR  = TP / (TP + FN)                  # of actual-positive, how many caught (↓ FN)
             (a.k.a. Sensitivity)
Specificity/TNR = TN / (TN + FP)              # of actual-negative, how many correctly cleared
FPR         = FP / (TN + FP) = 1 − Specificity # of actual-negative, how many falsely flagged
F1          = 2·P·R / (P + R)                 # harmonic mean of precision & recall
Fβ          = (1+β²)·P·R / (β²·P + R)          # β>1 favours recall, β<1 favours precision
```

**Why F1 uses the harmonic (not arithmetic) mean.** The arithmetic mean forgives a collapse: precision 0.9, recall 0.1 → AM 0.5, which looks fine despite recall being terrible. The **harmonic mean is dominated by the smaller value** (0.9, 0.1 → F1 ≈ 0.18), so F1 is only high when *both* are high. It "finds the harmony" between the two — a single number you can rank models by. `(certain)`

**Fβ generalises F1.** `β = 2` (F2) weights recall 2× — right for fraud/medical where misses cost most; `β = 0.5` weights precision — right when false alarms cost most. F1 is the neutral `β = 1`. `(certain)`

All of these (except accuracy) are defined **relative to the positive class** and range `[0, 1]`.

---

## 3. The Precision–Recall Tradeoff

Precision and recall pull against each other, and the **threshold** is the dial between them. A classifier outputs a probability; you call "positive" when `p̂ ≥ t`:

```
lower t  (e.g. 0.3)  → flag more positives → catch more (↑ recall) but more false alarms (↓ precision)
higher t (e.g. 0.7)  → flag fewer          → fewer false alarms (↑ precision) but more misses (↓ recall)
t = 1.0              → flag nothing         → precision undefined, recall 0
t = 0.0              → flag everything      → recall 1, precision = base rate
```

So **there's no single precision/recall — there's a curve** as `t` sweeps 0→1. You choose the operating point from the cost of errors:

- **Cancer screening** (positive = cancer): a **miss (FN) can be fatal**, a false alarm just means more tests → **maximise recall** → lower the threshold.
- **Important-mail-to-spam** (positive = spam): a **false positive (FP)** buries a real email → **maximise precision** → raise the threshold.
- **Loan repayment / fraud**: both errors cost money → balance with **F1** (and handle FPs downstream with human review).

---

## 4. Worked Example — bank fraud

10,000 transactions, 300 fraud (positive), 9,700 legit. A model produces this confusion matrix:

```
              Predicted Fraud   Predicted Legit
Actual Fraud       TP = 100         FN = 200
Actual Legit       FP = 700         TN = 9,000
```

```
Accuracy  = (100 + 9000) / 10000            = 0.91   ← looks great…
Recall    = 100 / (100 + 200)               = 0.333  ← …but we miss 2/3 of fraud
Precision = 100 / (100 + 700)               = 0.125  ← and 7/8 of our alarms are wrong
F1        = 2·0.125·0.333 / (0.125+0.333)   = 0.182  ← the honest score
FPR       = 700 / (9000 + 700)              = 0.072  ← 7.2% of legit txns falsely flagged
```

**Now the manager's "better" dumb model** predicts *everything legit*: `TP=0, FP=0, FN=300, TN=9700`.

```
Accuracy = 9700/10000 = 0.97   ← HIGHER accuracy…
Precision = Recall = F1 = 0    ← …but it catches zero fraud. Useless.
```

🎯 This is the entire reason accuracy is banned for imbalanced problems: **the model that does nothing scores highest.** Precision/recall/F1 correctly call it out (all 0).

---

## 5. ROC, AUC, and PR-AUC

The metrics above judge **one threshold**. To judge the model *independent of threshold*, sweep `t` from 1→0 and plot the trajectory.

**ROC curve** = **TPR (recall) vs FPR** at every threshold.

```
TPR │        ┌──────  ● perfect model (0,1): TPR=1, FPR=0
1.0 │      ╱ ╱
    │    ╱ ╱   ← better models bow toward the top-left corner
    │  ╱ ╱
    │╱ ╱   ← diagonal = random guessing (AUC 0.5)
0.0 └───────────── FPR 1.0
```

- Curve starts at `(0,0)` (threshold 1: flag nothing) and ends at `(1,1)` (threshold 0: flag everything).
- **AUC = area under the ROC curve** — one number in `[0,1]`. 🎯 **Interpretation:** the probability the model ranks a random positive above a random negative. `1.0` = perfect ranking, `0.5` = coin flip, `<0.5` = worse than random (just invert your predictions → `1 − AUC`). `(certain)`

**Key properties of ROC-AUC:** `(certain)`
- **Threshold-free** and **rank-based** — depends only on the *ordering* of scores, not their absolute values. Two models `[0.95,0.92,…]` and `[0.2,0.1,…]` with the same ordering have the same AUC.
- **Weak under severe imbalance** — FPR has the huge TN count in its denominator, so lots of false positives barely move FPR, and AUC stays flatteringly high on a mediocre model.

**PR curve & PR-AUC** = **precision vs recall** across thresholds. Because neither precision nor recall uses TN, **PR-AUC is the honest summary under severe imbalance** (fraud, rare disease). Its baseline isn't 0.5 — it's the positive base rate. Prefer PR-AUC (a.k.a. Average Precision) when positives are rare and you care about them. `(certain)`

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(y_test, y_proba)            # threshold-free ranking quality
average_precision_score(y_test, y_proba)  # PR-AUC — use under heavy imbalance
```

---

## 6. Multiclass & Probability Metrics

**Multiclass averaging** — compute precision/recall/F1 per class, then aggregate:
- **Macro** = unweighted mean across classes → every class counts equally → **use when you care about rare classes** (they aren't drowned out).
- **Micro** = pool all TP/FP/FN globally → dominated by frequent classes → equals accuracy for single-label problems.
- **Weighted** = mean weighted by class support → a compromise.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))   # per-class precision/recall/F1 + macro & weighted
```

**Metrics on probabilities, not just labels:**
- **Log-loss (cross-entropy)** — penalises confident wrong probabilities; the training loss itself, and a metric for *probability quality*.
- **Brier score** = mean squared error of predicted probabilities — a simple calibration check (lower is better).
- **Calibration** — do the probabilities mean what they say (of things predicted 0.8, are ~80% positive)? Check with a reliability curve; fix with Platt/isotonic. Logistic regression is calibrated by default; boosted trees/SVMs often aren't.

**Robust single-number metrics for imbalance:**
- **Matthews Correlation Coefficient (MCC)** — a balanced correlation between predictions and truth using *all four* cells; `+1` perfect, `0` random, `−1` inverse. Often the most honest single number under imbalance. `(likely)`

---

## 7. When It Breaks — the traps

- **Accuracy under imbalance** — the headline trap; the do-nothing model wins (§4). Never report accuracy alone on skewed data.
- **ROC-AUC under *severe* imbalance** — looks great because FPR is diluted by huge TN. Switch to **PR-AUC** when positives are <~5%.
- **Reporting a threshold-metric without the threshold** — precision/recall/F1 are meaningless without saying *at what threshold*; the same model gives wildly different values at `t=0.5` vs `t=0.3`.
- **Optimising the metric you don't ship** — if the business cost is asymmetric, don't optimise F1 by habit; optimise Fβ or expected cost.
- **Metric vs loss confusion** — 🎯 the **loss** (log-loss) is what you *train* on: it must be differentiable so gradient descent can optimise it. A **metric** (accuracy, F1, AUC) is what you *report*: it need not be differentiable, and often isn't. You train on a smooth surrogate (log-loss) and evaluate on the business metric (F1/recall). `(certain)`
- **Micro-averaging hides rare-class failure** — on imbalanced multiclass, micro-F1 can look fine while the class you care about is ignored; use **macro** to expose it.

---

## 8. Production & MLOps Notes

- **Pick the operating threshold deliberately.** Turn error costs into a choice on the PR/ROC curve (e.g. "recall ≥ 0.9 at best achievable precision"), or use Youden's J (max `TPR−FPR`) for balanced cost. Ship *that* threshold, and revisit it when class balance drifts.
- **Monitor the metric that matches the cost**, on labelled feedback — track recall/PR-AUC for fraud, not accuracy. Watch **score drift** and **calibration drift** separately from label drift.
- **Handle the false positives operationally** — in fraud, high recall means many FPs; those are absorbed by automated verification (OTP, confirmation calls) rather than blocking customers.
- **Report with confidence intervals** — evaluate with cross-validated metrics (`cross_val_score(..., scoring='roc_auc')`) so a lucky test split doesn't flatter you.
- **Threshold ≠ retrain.** Often the fastest production win isn't a new model but re-tuning the decision threshold to the current cost/base-rate.

---

## 9. Interview Lens

**"Your model has 97% accuracy — ship it?"** → 🎯 *"Not until I see the class balance and the confusion matrix. If positives are 3%, a do-nothing model also scores 97%. I'd look at precision, recall, and PR-AUC for the positive class and pick the metric from the cost of a miss vs a false alarm."*

**"Precision or recall — how do you choose?"** → 🎯 *"By which error is worse. If a miss is catastrophic (cancer, fraud) I maximise recall; if a false alarm is costly (flagging real mail as spam) I maximise precision; if both matter I use F1 or Fβ tuned to the cost ratio."*

**Likely follow-ups:**
- *Why harmonic mean for F1?* → It's dominated by the smaller of precision/recall, so F1 is high only when both are — the arithmetic mean would hide a collapse.
- *What does AUC actually measure?* → P(model scores a random positive above a random negative); threshold-free, rank-based.
- *ROC-AUC vs PR-AUC?* → ROC can look great under heavy imbalance (TN dilutes FPR); PR-AUC ignores TN, so it's the honest choice when positives are rare.
- *Sensitivity vs specificity?* → Sensitivity = recall = TP/(TP+FN); specificity = TN/(TN+FP); FPR = 1 − specificity.
- *Type I vs Type II error?* → Type I = FP (false alarm); Type II = FN (miss).
- *Metric vs loss?* → Loss is the differentiable thing you train on (log-loss); metric is the (possibly non-differentiable) thing you report (F1/AUC).
- *One metric for imbalanced binary?* → MCC or PR-AUC — both stay honest when classes are skewed.

---

## 10. Which Metric When — the decision guide

| Situation | Use | Why |
|---|---|---|
| Balanced classes | **Accuracy** | simple and meaningful when base rates are even |
| Need probability quality | **Log-loss / Brier** | scores the calibrated probability, not just the label |
| Imbalanced, false positives costly | **Precision** (or Fβ<1) | punishes false alarms |
| Imbalanced, misses costly | **Recall** (or F2) | punishes missed positives |
| Imbalanced, both matter | **F1** | harmonic balance of P and R |
| Care about both classes, threshold-free ranking | **ROC-AUC** | overall separability across thresholds |
| **Severe** imbalance (rare positives) | **PR-AUC** / **MCC** | ignore the huge TN count that flatters ROC-AUC |
| Multiclass with rare classes you care about | **Macro-F1** | weights every class equally |

**Decision rule:** start from *"what does each error cost?"*, not from a default. Balanced → accuracy is fine. Imbalanced → go to precision/recall/F1 chosen by cost, and summarise across thresholds with PR-AUC (rare positives) or ROC-AUC (roughly balanced). Train on log-loss; *report* the business metric.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. Define precision and recall from the confusion matrix, and say which error each fights.
   <details><summary>answer</summary>Precision = TP/(TP+FP) — fights false positives (false alarms). Recall = TP/(TP+FN) — fights false negatives (misses). Neither uses TN.</details>
2. A model has 97% accuracy on a 3%-positive dataset but catches no positives. What's its precision, recall, F1?
   <details><summary>answer</summary>All 0 — it predicts only the majority class, so TP=0. Accuracy is high only because negatives dominate; this is why accuracy is banned under imbalance.</details>
3. Why does F1 use the harmonic mean instead of the arithmetic mean?
   <details><summary>answer</summary>The harmonic mean is dominated by the smaller value, so F1 is high only when both precision and recall are high. The arithmetic mean would report 0.5 for (0.9, 0.1), hiding the recall collapse.</details>
4. What does ROC-AUC represent, and when does it mislead?
   <details><summary>answer</summary>The probability the model ranks a random positive above a random negative (threshold-free). It misleads under severe imbalance because FPR's huge TN denominator hides false positives — use PR-AUC instead.</details>
5. You must not miss any fraud, even at the cost of false alarms. Which metric and which way do you move the threshold?
   <details><summary>answer</summary>Maximise recall (or F2); lower the threshold so more transactions are flagged as fraud, accepting more false positives.</details>
6. What's the difference between a loss function and a performance metric?
   <details><summary>answer</summary>The loss (log-loss) is the differentiable objective you train on via gradient descent; the metric (accuracy/F1/AUC) is what you report and needn't be differentiable. You optimise a surrogate loss and evaluate the business metric.</details>
7. Macro vs micro averaging for multiclass — when does the choice matter?
   <details><summary>answer</summary>Under class imbalance: macro treats every class equally (exposes rare-class failure), micro pools all counts (dominated by frequent classes, ≈ accuracy). Use macro when rare classes matter.</details>

---

*Covers: confusion matrix & Type I/II errors, accuracy trap, precision/recall/sensitivity/specificity/FPR, F1 & Fβ & why harmonic mean, precision–recall tradeoff & thresholding, ROC/AUC (meaning + imbalance weakness), PR-AUC, multiclass macro/micro/weighted, log-loss/Brier/MCC/calibration, metric-vs-loss, and a which-metric-when guide. Sourced from the Scaler Classification Metrics lecture; model context in [Logistic Regression](Logistic%20Regression.md).*
