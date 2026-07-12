# Decision Trees — If-Else Rules Learned From Data

> **TL;DR.** A decision tree carves the feature space into boxes with **axis-parallel splits**, choosing each split to make the resulting nodes as **pure** (single-class) as possible — measured by **entropy** or **Gini impurity**, maximised via **information gain**. Predict by dropping a point down the tree to a leaf and taking the **majority class** (classification) or **mean** (regression). It's the most **interpretable** model there is (it's literally a flowchart), needs **no feature scaling**, and handles mixed data types — but a single deep tree **overfits badly** (high variance), which is exactly why we wrap trees in [ensembles](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md).

**Where it fits:** Supervised **classification and regression** on tabular data. Captures non-linear boundaries that [Logistic Regression](Logistic%20Regression.md) can't, and unlike [[KNN]] it does its work at *training* time so inference is cheap. The building block of Random Forests & Gradient Boosting.
**Prereqs:** [Classification Metrics](Classification%20Metrics.md) (confusion matrix), the [bias–variance tradeoff](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md), basic probability (for entropy).

---

## Table of Contents
1. [Intuition / Mental Model](#1-intuition--mental-model)
2. [The Formal Core — impurity & information gain](#2-the-formal-core--impurity--information-gain)
3. [How It Works — the greedy splitting algorithm](#3-how-it-works--the-greedy-splitting-algorithm)
4. [Worked Example](#4-worked-example)
5. [Code / Implementation](#5-code--implementation)
6. [Overfitting & Pruning](#6-overfitting--pruning)
7. [Regression Trees](#7-regression-trees)
8. [When It Breaks](#8-when-it-breaks)
9. [Production & MLOps Notes](#9-production--mlops-notes)
10. [Interview Lens](#10-interview-lens)
11. [Alternatives & How to Choose](#11-alternatives--how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition / Mental Model

Instead of one straight boundary (logistic regression), keep asking **yes/no questions** that slice the space into rectangles until each rectangle holds mostly one class:

```
Age < 35 ?
├── yes → OverTime ≥ 2.5h ?
│         ├── yes → LEAVES  (mostly churn)     ← a leaf: predict "churn"
│         └── no  → STAYS
└── no  → STAYS  (mostly retained)

     Each question = an axis-parallel cut; the stack of cuts = a flowchart = the tree.
     root node (top) → internal nodes (questions) → leaf nodes (predictions).
```

- **Why trees, not a line?** Real boundaries are often non-linear; a line underfits. Trees stack simple cuts to fit any shape. `(certain)`
- **Why not [[KNN]], which also fits non-linear data?** KNN stores *all* training data and searches it at every prediction → you can't productionise it at scale. A tree learns compact rules at training time, so inference is a handful of comparisons. `(certain)`
- 🎯 The killer feature is **interpretability**: "employees under 35 who work >2.5h overtime tend to churn" is a sentence a business acts on — no other model gives you that for free. `(certain)`
- **Training = finding the questions.** Given the features, learning a tree means discovering *which feature and threshold to split on, in what order*. The objective: each split should make child nodes **purer** than the parent.

---

## 2. The Formal Core — impurity & information gain

**Why purity?** A leaf predicts the **majority class**, and the class proportion is its **confidence** (a leaf with 90 −ve / 10 +ve predicts −ve with 0.9 probability). Pure leaves = confident, correct predictions. So we need to *quantify* impurity.

**Entropy** — impurity from information theory (`k` classes, `pᵢ` = class fraction):

```
H = − Σᵢ pᵢ · log₂(pᵢ)          # binary: H = −p·log₂p − (1−p)·log₂(1−p)
```

- Pure node (`p = 0` or `1`) → `H = 0`. Maximally mixed (`p = 0.5`) → `H = 1`. Ranges `[0, log₂k]`. `(certain)`

**Gini impurity** — the same idea, cheaper to compute:

```
G = 1 − Σᵢ pᵢ²                   # binary: G = 2p(1−p)
```

- Pure → `G = 0`; 50/50 → `G = 0.5`. 🎯 **Gini avoids the `log`**, so it's faster to evaluate at every candidate split — which is why it's sklearn's **default** (`criterion='gini'`). Entropy and Gini almost always pick the same splits; the difference is speed, not quality. `(certain)`

**Information Gain — the split score.** Combine children by a **weighted average** (weight = fraction of samples in each child, so a 98-vs-2 split isn't judged as if the children were equal), then measure the drop:

```
IG = I(parent) − Σₖ (nₖ / n) · I(childₖ)      I = entropy or Gini
```

At each node the tree tries every feature (and threshold) and **picks the split with maximum information gain** = biggest impurity drop. `(certain)`

---

## 3. How It Works — the greedy splitting algorithm

```
build(node):
  1. if node is pure (or a stop rule hits) → make it a leaf, store majority class
  2. for every feature, for every candidate split:
        compute information gain
  3. pick the split with the HIGHEST information gain
  4. partition the data into children by that split
  5. recurse on each child
```

- **Greedy & recursive.** It takes the *locally* best split at each step — never backtracks, never guarantees a globally optimal tree. That greed is why trees are fast to build but unstable (§8). `(certain)`
- **Categorical features:** split into one child per category (or, in sklearn's **CART**, binary yes/no groupings). `(likely)`
- **Numerical features — threshold search:** sort the values, try each midpoint as a threshold (`f < t`), score the information gain, keep the best `t`. Cost is `O(n)` candidate thresholds per feature → **binning** the values (e.g. Age → [20,30), [30,40)…) cuts that down and is what histogram-based boosters exploit at scale. `(certain)`
- **Prediction:** route the point down the questions to a leaf → **majority class** (classification) or **mean target** (regression).

> Note: sklearn implements **CART** (binary splits, Gini default). The classic ID3/C4.5 algorithms use entropy/gain-ratio and multi-way splits — same principle, different bookkeeping. `(likely)`

---

## 4. Worked Example

Root: **100 points, 50 +ve / 50 −ve** → maximally impure. Consider a candidate split into two children of 50 each: **left = 40+/10−**, **right = 10+/40−**.

```
Parent entropy  H(50/50) = 1.0

Left child (p=0.8):  H = −0.8·log₂0.8 − 0.2·log₂0.2 = 0.257 + 0.464 = 0.722
Right child (p=0.2): H = 0.722   (symmetric)

Weighted child entropy = (50/100)·0.722 + (50/100)·0.722 = 0.722
Information Gain        = 1.0 − 0.722 = 0.278
```

Compare with **Gini** on the same split:

```
Parent Gini  = 1 − 0.5² − 0.5² = 0.5
Child Gini   = 1 − 0.8² − 0.2² = 0.32   (both children)
Weighted     = 0.32   →   Gini IG = 0.5 − 0.32 = 0.18
```

The tree computes this IG for **every** feature/threshold and keeps the highest. A 50/50 split into 60/40 children would score a *lower* IG than this 80/20 split — so the tree prefers the split that makes nodes purer.

---

## 5. Code / Implementation

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(
    criterion="gini",      # or "entropy" — near-identical results, gini is faster
    max_depth=4,           # the single biggest anti-overfit lever (see §6)
    min_samples_leaf=20,   # no leaf smaller than 20 → smoother, less noise-fitting
    class_weight="balanced" # handle imbalance without resampling
).fit(X_train, y_train)

plt.figure(figsize=(14, 8)); plot_tree(clf, filled=True, feature_names=cols)  # it's readable!
clf.feature_importances_    # normalized information gain per feature (see caveat in §9)
```

**No scaling, no encoding headaches:** trees are invariant to monotonic transforms of a feature (they only compare thresholds), so **standardization is unnecessary** — a rare and convenient property. `(certain)`

**Tune depth with cross-validation** (the honest way to pick the bias-variance sweet spot):

```python
from sklearn.model_selection import cross_val_score
for d in [3, 4, 5, 7, 9, 11]:
    acc = cross_val_score(DecisionTreeClassifier(max_depth=d), X_train, y_train, cv=10).mean()
    print(d, acc)   # pick the depth with the best validation score, not the deepest
```

---

## 6. Overfitting & Pruning

**A tree left unchecked overfits — hard.** Grow until every leaf is pure and you get **train accuracy = 1.0, test ≈ 0.76** — the textbook symptom. `(certain)`

```
Why deep = overfit:  as depth ↑, each leaf holds fewer points → eventually a leaf
                     memorises noise/outliers → LOW bias, HIGH variance.
Why shallow = underfit: too few splits → too few boundaries → HIGH bias, LOW variance.
depth = 0 → a single node ("decision stump"); depth huge → memorised training set.
```

So **depth is a hyperparameter** you tune. Two ways to control growth:

- **Pre-pruning (early stopping)** — cap growth *while building*: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `max_features`. Fast, the common default.
- **Post-pruning (cost-complexity pruning, `ccp_alpha`)** — grow the full tree, then **snip back** branches that don't pay for their complexity, penalising leaf count by `α`. Often generalises better than guessing `max_depth`; sweep `α` via `clf.cost_complexity_pruning_path(...)`. `(likely)`

🎯 *"Pruning is removing branches that don't improve validation performance — pre-pruning stops growth early (max_depth/min_samples_leaf), post-pruning grows fully then cuts back (ccp_alpha)."*

---

## 7. Regression Trees

Same tree, two swaps: `(certain)`

- **Impurity → variance / MSE.** Entropy and Gini need class probabilities, which regression doesn't have. Instead score a node by the **variance (MSE) of its target values**; a good split reduces total variance.
- **Prediction → the mean** of the target values in the leaf (not a majority vote).

Consequence: a regression tree outputs a **piecewise-constant** surface (one value per leaf box). It therefore **cannot extrapolate** beyond the training range — predict outside it and you get the nearest leaf's flat mean, never a rising trend. (This is why trees struggle with time-series trends; see [[Time Series]].) `(certain)`

---

## 8. When It Breaks

- **High variance / instability** — the headline flaw. Because splitting is greedy, changing a few training rows can flip the top split and reshape the whole tree. A single tree rarely wins on accuracy. 🎯 **The fix is ensembling:** average many decorrelated trees ([Random Forest](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)) or boost them sequentially — the single biggest reason trees matter in practice. `(certain)`
- **Axis-parallel only** — splits are `feature < threshold`, so a diagonal boundary is approximated by a staircase of steps; correlated/rotated boundaries need many splits.
- **Outliers hit deep trees, not shallow ones** — in a deep tree an outlier can end up alone in its own leaf; a shallow tree buries it among many points. `(certain)`
- **Class imbalance** — a tree can ignore a 1% class; use `class_weight='balanced'`, or resampling/SMOTE.
- **Feature-importance bias** — built-in `feature_importances_` (normalized information gain) is **biased toward high-cardinality / continuous features** (more thresholds to split on). Don't ship decisions on it alone — prefer permutation importance or SHAP (see [Interpretability in the Ensemble note](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)). `(certain)`
- **Can't extrapolate** (regression, §7).

---

## 9. Production & MLOps Notes

- **Cheap, fast inference & tiny footprint.** Space is `O(nodes)`, prediction is `O(depth)` comparisons, and build time is roughly `O(n·f·log n)`. A trained tree is a compact set of rules — trivial to serve, embed, even hand-code. `(likely)`
- **Interpretability is the reason to keep a single tree** — regulated settings (credit, healthcare) that need an auditable, human-readable decision path may prefer a shallow tree over a black-box ensemble, trading some accuracy for explainability.
- **No preprocessing pipeline** to drift: no scaler/normaliser to fit, one fewer thing to version and monitor.
- **Feature-importance caveat** (from §8) matters in prod dashboards — a spurious high-cardinality ID column can top the chart; validate with permutation/SHAP before anyone acts on it.
- **Monitor & retrain** like any model, but note a single tree's instability means small data shifts can change the rules a lot — another argument for ensembles in production.

---

## 10. Interview Lens

**"How does a decision tree decide where to split?"** → 🎯 *"It picks the feature and threshold that **maximise information gain** — the drop in impurity (entropy or Gini) from parent to the weighted average of the children — greedily at each node."*

**"Entropy vs Gini?"** → 🎯 *"Both measure node impurity and usually choose the same splits; Gini skips the logarithm so it's faster to compute, which is why it's the default."*

**Likely follow-ups (Scaler's own list):**
- *Does a tree need feature scaling?* → No — splits compare thresholds, and impurity depends on class counts, not on the feature's scale. `(certain)`
- *Why does a deep tree overfit?* → Leaves get tiny and memorise noise → low bias, high variance. Fix with depth limits / min_samples_leaf / pruning.
- *What is pruning?* → Removing branches that don't improve generalisation; pre-pruning (early stop) vs post-pruning (`ccp_alpha`).
- *Impact of an outlier — deep or shallow tree?* → Worse on a deep tree (outlier can occupy its own leaf); shallow trees dilute it.
- *Imbalanced data?* → Yes it's affected; use `class_weight`, resampling, or SMOTE.
- *Space/time complexity?* → Space `O(nodes)`, inference `O(depth)`, build `O(n·f·log n)`.
- *Can the same feature be used in multiple splits?* → Yes, especially numerical features with different thresholds down a path.
- *Multiclass?* → Yes — the leaf predicts whichever class is the majority there.
- *Decision tree for regression?* → Split on variance/MSE reduction; predict the **mean** of the leaf's targets.
- *Why do we ensemble trees?* → A single tree is high-variance; bagging/boosting many trees cuts variance and wins on accuracy.

---

## 11. Alternatives & How to Choose

| Situation | Reach for | Why |
|---|---|---|
| Need a human-readable, auditable model | **Single (shallow) Decision Tree** | it *is* a flowchart; no other model explains itself so directly |
| Best tabular accuracy, variance is hurting you | **[Random Forest / Gradient Boosting](Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md)** | many trees average out a single tree's instability |
| Linear-ish boundary, want probabilities | **[Logistic Regression](Logistic%20Regression.md)** | simpler, calibrated, fewer ways to overfit |
| Smooth/continuous relationship, extrapolation needed | **[Linear Regression](Linear%20Regression.md)** | trees are piecewise-constant and can't extrapolate |
| Small data, non-linear, latency not a concern | **[[KNN]]** | no training, but can't productionise at scale |

**Decision rule:** use a single decision tree when **interpretability is the deliverable**; the moment accuracy matters more than explainability, move to a tree **ensemble** — same building block, dramatically lower variance. Everything you learn here (impurity, information gain, splitting) *is* the machinery inside Random Forests and XGBoost.

---

## 🧠 Self-Test
*Cover the answers; force retrieval first.*

1. What quantity does a tree maximise at each split, and how is it defined?
   <details><summary>answer</summary>Information gain = impurity(parent) − weighted-average impurity(children), where impurity is entropy or Gini. It picks the feature/threshold with the highest IG.</details>
2. Entropy vs Gini — practical difference?
   <details><summary>answer</summary>Both measure impurity and usually pick the same splits; Gini avoids the logarithm so it's faster to compute — hence sklearn's default.</details>
3. Train accuracy 1.0, test 0.76. Diagnose and give two fixes.
   <details><summary>answer</summary>Overfitting (deep tree, pure leaves memorising noise → high variance). Fix with pre-pruning (max_depth, min_samples_leaf) or post-pruning (ccp_alpha); ultimately, ensemble the trees.</details>
4. Does a decision tree need standardized features? Why or why not?
   <details><summary>answer</summary>No. Splits only compare a feature to a threshold and impurity depends on class counts, not magnitudes — monotonic rescaling doesn't change the chosen splits.</details>
5. How does a tree handle a numerical feature, and what's the cost?
   <details><summary>answer</summary>Sort the values, try each midpoint as a threshold, pick the one with max information gain — `O(n)` candidate splits per feature; binning reduces the cost.</details>
6. How do regression trees differ from classification trees?
   <details><summary>answer</summary>Impurity becomes variance/MSE (no probabilities), and the leaf prediction is the **mean** of its target values, not a majority vote. Output is piecewise-constant, so it can't extrapolate.</details>
7. Why is a single tree rarely the final model, and what's the standard fix?
   <details><summary>answer</summary>Greedy splitting makes it high-variance/unstable. Ensembling — bagging (Random Forest) or boosting — averages/combines many trees to cut variance and boost accuracy.</details>

---

*Covers: axis-parallel splitting & tree anatomy, entropy vs Gini, information gain & weighted child impurity, greedy recursive building, categorical vs numerical (threshold) splits, overfitting & pre/post-pruning (ccp_alpha), regression trees (variance + mean), feature-importance bias, complexity, and the tree→ensemble bridge. Sourced from the Scaler Decision Tree lecture (incl. its interview-Q section).*
