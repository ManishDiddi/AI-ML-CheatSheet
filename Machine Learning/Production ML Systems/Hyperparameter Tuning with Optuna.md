# Hyperparameter Tuning with Optuna — Spending a Fixed Budget of Fits Well

*Dense, no filler. Every concept an interviewer will probe.*

> **TL;DR.** A grid search multiplies (`3^k` combinations), forces continuous parameters into hand-picked lists, and learns nothing from the results sitting right next to it. **Optuna** replaces it with a plain Python function: ask for values with `trial.suggest_*`, return **one number**, and a `study` decides what to try next from what it has already seen. Two orthogonal knobs do the work — the **sampler** picks the next configuration (once per trial), the **pruner** kills a trial that is already losing (at every step you report). And the rule that comes before all of it: 🎯 *check that your metric can tell two models apart before you spend an hour optimising it* — on a skewed target, R² moved **0.18** on a model that did not change.

**Where it fits:** Session 2 of the Scaler **MLOps** track (first half) — the last step before a model is worth shipping. This is Stage 3 of the lifecycle in [Introduction to MLOps](Introduction%20to%20MLOps.md), and the second half of the same lecture — putting the tuned artifact in front of a human — is [Streamlit and Gradio](Streamlit%20and%20Gradio.md).
**Prereqs:** [Classification Metrics](../Supervised%20ML/Classification%20Metrics.md) and regression metrics (MAE/RMSE/R²), cross-validation, and gradient boosting — [Ensemble Methods that Trade Off Bias vs Variance](../Supervised%20ML/Ensemble%20Methods%20that%20Trade%20Off%20Bias%20vs%20Variance.md). The MLP in §6 leans on [Weight Initialization & Optimizers](../Neural%20Networks/Weight%20Initialization%20&%20Optimizers.md).

> 🧠 **Start here, not at the top.** Jump straight to the [Self-Test](#-self-test), answer cold, then read **only** the sections you missed — a 5-minute pass instead of 30. *([why](../../_STUDY%20LOOP.md))*

---

> ### ⚡ Fast Pass — 5 minutes
>
> **Model.** An Optuna **objective** is a function that takes a `trial`, asks it for values, and returns **one number to minimise or maximise**. A **study** collects trials and decides what to ask for next. The search space is *code*, so it can contain `if` — that's **define-by-run**.
>
> **Core.** `trial.suggest_int / suggest_float(..., log=True) / suggest_categorical` · `study = optuna.create_study(direction="minimize", sampler=..., pruner=..., storage="sqlite:///study.db", load_if_exists=True)` · `study.optimize(objective, n_trials=N)` · `study.best_params`. **Sampler** = which config next (TPE models `l(x)/g(x)`, good-density over bad-density; first `n_startup_trials=10` are **random**). **Pruner** = kill this running trial, via `trial.report(v, step)` + `if trial.should_prune(): raise optuna.TrialPruned()`.
>
> **Traps.** ① Tuning a metric that is unstable to the *split* — R² has a denominator that changes with the test set. ② A **12-trial TPE study is a 10-trial random search** with two informed guesses stapled on. ③ A trial budget is not a time budget — good configs are usually *expensive* configs, so trials get slower as the search succeeds.
>
> 🎯 **Kill-shot.** *"Grid search's cost multiplies with every parameter you add and it learns nothing from trial 26 when choosing trial 27; Optuna's cost is the trial count you set regardless of dimensionality, and it samples the next configuration from a model of which regions have been scoring well."*

---

## Table of Contents

- [1. Intuition — The Loop Everyone Writes First](#1-intuition--the-loop-everyone-writes-first)
- [2. Before You Tune — Can Your Metric Tell Two Models Apart?](#2-before-you-tune--can-your-metric-tell-two-models-apart)
- [3. The Formal Core — Trial, Study, Sampler, Pruner](#3-the-formal-core--trial-study-sampler-pruner)
- [4. How It Works — Define-by-Run, and Why a Grid Cannot](#4-how-it-works--define-by-run-and-why-a-grid-cannot)
- [5. Worked Example — Against Grid Search, Measured](#5-worked-example--against-grid-search-measured)
- [6. Pruning — Stop Training the Ones Already Losing](#6-pruning--stop-training-the-ones-already-losing)
- [7. Code — The Whole Thing in Five Lines, Then the Real One](#7-code--the-whole-thing-in-five-lines-then-the-real-one)
- [8. When It Breaks](#8-when-it-breaks)
- [9. Production and MLOps Notes — The Study Is an Artifact Too](#9-production-and-mlops-notes--the-study-is-an-artifact-too)
- [10. Interview Lens](#10-interview-lens)
- [11. Alternatives and How to Choose](#11-alternatives-and-how-to-choose)
- [🧠 Self-Test](#-self-test)

---

## 1. Intuition — The Loop Everyone Writes First

First, the distinction the whole topic rests on:

> A **parameter** is something the model *learns from data* — the split thresholds inside each tree, the weights in a network. A **hyperparameter** is something you hand the model **before** it learns anything — how many trees, how deep, how fast. The model cannot learn those, because they are *the rules of learning itself*.

And nobody chose yours. A baseline `XGBRegressor(random_state=42)` leaves `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda` at whatever the library ships — which for XGBoost is literally `None`, meaning "whatever the C++ library decides at fit time." 🎯 *Library defaults are not recommendations for your data — nobody at XGBoost has seen your data. They are the values least likely to embarrass the library across every dataset in the world at once.*

So: given a budget of a few hundred model fits, **how do you spend it?** Everyone writes the same first answer, and it is a nested loop:

```python
best = None
for n in [200, 400, 800]:
    for d in [4, 6, 8]:
        for lr in [0.03, 0.1, 0.3]:
            score = cv_mae(XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr))
            if best is None or score < best[0]:
                best = (score, n, d, lr)
```

It works. Three things break it, in this order:

1. **It multiplies.** Three values for three parameters is 27 fits. A fourth parameter is 81. A fifth, 243. You did not make the search smarter, you made it *longer*.
2. **Everything has to be a list.** `learning_rate` is a positive real number. You are pretending it has three legal values because a `for` loop needs something to iterate over.
3. **It learns nothing.** Combination 27 is chosen with exactly as much information as combination 1 — even though 26 results are sitting right there.

Optuna is the same loop with those three fixed. And the reason #2 matters more than it looks is that grid points cluster where you happened to guess:

![Two panels of nine trials each on the same two-dimensional space: the grid places them on only three distinct values of the important hyperparameter and misses the good range entirely, while random sampling tries nine distinct values and lands two trials inside it.](attachments/grid-vs-random-search.png)

Both panels spend **nine trials**. The grid spends them on three distinct values of the axis that matters (and three of the axis that doesn't); random spends nine distinct values on each. This is Bergstra and Bengio's 2012 argument for random search, and it is the floor Optuna builds on: even before any adaptive sampling, *not* discretising already buys you coverage. `(certain)`

---

## 2. Before You Tune — Can Your Metric Tell Two Models Apart?

The step everyone skips, and skipping it makes every tuning run that follows a waste of electricity. You need **a number that moves when the model gets better and stays put when it does not.**

The dataset for the whole lecture: **19,980 used cars**, nine features, target `selling_price` in lakhs. The distribution is the problem — **skew 9.27**, median 5.2, 99th percentile 45, **max 395**, with 150 of 19,980 cars above 50. A skew of 9.3 is not a detail; it decides which metric you are allowed to trust.

Now change *nothing* but `random_state` in the train/test split — which car lands in which half — and refit the same model six times:

![Three panels using the lecture's six-split experiment: R squared swinging 0.18 across split seeds on a model that did not change, MAE on the same six splits moving only 5.4 percent, and a scatter showing R squared falling as the test set's own variance rises with correlation minus 0.78.](attachments/metric-instability-r2-vs-mae.png)

| split seed | R² | MAE | test variance | priciest car in test |
|---|---|---|---|---|
| 0 | 0.930 | 0.979 | 70.2 | 132 |
| 1 | 0.812 | 1.056 | 69.1 | 145 |
| 2 | 0.890 | 1.073 | 63.6 | 111 |
| 3 | 0.902 | 1.031 | 64.8 | 111 |
| 4 | 0.925 | 0.979 | 69.3 | 111 |
| **5** | **0.752** | **1.123** | **102.5** | **395** |

**R² swings 0.1775 — a relative spread of 8.2% — for a model that did not change.** Look at seed 5: it is the split that happened to put the ₹395 lakh car in the test set. Test variance jumps from ~65 to over 100, and R² collapses from ~0.90 to 0.75. **One row out of 3,996 moved the headline score by fifteen points.**

That is not bad luck; it is *what R² is*. `R² = 1 − MSE/Var(y_test)` — and the denominator is **the test set's own variance**, which the table shows changing by 60% between splits. `corr(R², test variance) = −0.78`. So 0.93 and 0.75 are not two disagreeing measurements of the same quantity; they are measurements of **two different quantities**, and neither of them is "how good is the model."

**MAE has no denominator.** It is in lakhs, it means the same thing in every split, and it is the number the business can act on: *we are off by about a lakh*. It still moves with the split — 5.4% against R²'s 8.2% — but it moves for an honest reason and never stops being comparable.

> **Two rules come out of this, and the rest of the note runs on both.**
> **1. Optimise a metric in the units of the problem, not a normalised one.** → MAE, minimised.
> **2. Score every candidate on identical folds**, so the split cannot vary between them at all. → a fixed 3-fold CV on the *training* set, and the test set is not touched again until §9.

```python
CV = KFold(n_splits=3, shuffle=True, random_state=0)      # fixed folds for EVERY candidate

def cv_mae(model):
    """MAE across 3 folds of the training set. Lower is better."""
    return -cross_val_score(model, X_train, y_train, cv=CV,
                            scoring="neg_mean_absolute_error").mean()

baseline_cv   = cv_mae(XGBRegressor(random_state=42))      # 1.1470
baseline_test = mean_absolute_error(y_test, baseline.predict(X_test))  # 1.0244 — set aside
```

**Why the CV number (1.1470) is *worse* than the test number (1.0244), and why that is fine.** Each CV fold trains on only 2/3 of the training data, so each fold's model is weaker than the final one fitted on all of it. CV is the **selection** instrument — its job is to rank candidates consistently, not to estimate final performance. The test set does that, once, at the end. Confusing the two is how people report an optimistically biased number as if it were held-out. 🎯

*(The same trap appears from the other direction in [Introduction to MLOps §6.2](Introduction%20to%20MLOps.md#6-worked-example--two-drift-labs): a bike-share model whose MAE degraded 42% while R² barely moved, because February's target variance was 1.7× January's.)*

---

## 3. The Formal Core — Trial, Study, Sampler, Pruner

Four objects, and the whole API is them:

| Term | What it is |
|---|---|
| **Trial** | One execution of the objective function with one specific set of hyperparameters |
| **Study** | The collection of trials, plus the machinery that decides what to try next and keeps the history |
| **Sampler** | Decides **which configuration to try next**. Runs *once*, before each trial starts |
| **Pruner** | Decides whether the trial **now running** is worth finishing. Runs at *every step you report* |

![A two-panel diagram contrasting the sampler, which decides which configuration to try next and runs once before each trial, with the pruner, which decides whether the running trial is worth finishing and runs at every reported step, together with the API calls each one answers.](attachments/optuna-sampler-vs-pruner.png)

Keeping those last two apart matters, because a tuning run has one of each and changing one tells you nothing about the other. §5 changes the **sampler** and leaves the pruner alone; §6 does exactly the opposite.

### 3.1 The objective

A plain Python function. It takes a `trial` and returns **one number**. Inside, `trial.suggest_*` asks for a value and Optuna decides what to hand back:

```python
def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    return cv_mae(XGBRegressor(random_state=42, **params))     # ONE number to minimise
```

Read what those six lines say against the nested loop:

| | Nested loop | `trial.suggest_*` |
|---|---|---|
| `n_estimators` | three values you picked | **any integer** in 200–600 |
| `learning_rate` | three values you picked | **any real** in 0.01–0.3 |
| Cost of a 4th parameter | ×3 the combinations | **no extra trials** — though a bigger space may need more of them |
| Who picks the next combination | your `for` loop, in order | the sampler, from what it has seen |

| Call | Gives you |
|---|---|
| `trial.suggest_int("n", lo, hi)` | an integer; optional `step=` or `log=True` |
| `trial.suggest_float("lr", lo, hi)` | a float; optional `step=` or `log=True` |
| `trial.suggest_categorical("kind", ["a", "b"])` | one of a fixed list — for things with **no order** |

**`log=True` is doing real work, not cosmetics.** `learning_rate` and `reg_lambda` matter on a *multiplicative* scale: the gap 0.01→0.02 is the same kind of change as 0.1→0.2. Sampling uniformly from 0.01–0.3 would put **two thirds of your trials above 0.1**. `log=True` samples the exponent instead, so every order of magnitude gets equal attention. 🎯 *Use a log scale for any parameter you would naturally describe in powers of ten — learning rates, regularisation strengths, layer widths.*

### 3.2 The study, and how TPE actually decides

```python
study = optuna.create_study(
    direction="minimize",                          # MAE: lower is better
    sampler=optuna.samplers.TPESampler(seed=7),    # seeded, so the run reproduces
    study_name="milepost-price",
)
study.optimize(objective, n_trials=40)
study.best_params      # a dict you can splat straight into a model
```

**TPE — Tree-structured Parzen Estimator**, Optuna's default sampler. Bayesian optimisation, but it models the problem backwards from what you might expect. Instead of fitting `P(score | params)` like a Gaussian-process approach, it splits the finished trials at a quantile `γ` into a *good* group and the *rest*, fits a density over each, and samples candidates that maximise the ratio:

```
l(x) = density of params among the GOOD trials
g(x) = density of params among the REST
pick x maximising   l(x) / g(x)      ← proportional to Expected Improvement
```

Concretely it draws `n_ei_candidates` (default **24**) samples from `l(x)`, scores each by that ratio, and returns the winner. "Where have good scores been dense, and bad scores sparse?" — that is the whole intuition. `(certain)`

> ⚠️ **The trap that makes people dismiss Optuna, stated correctly first.** TPE **cannot model anything until it has results to model from**, so `TPESampler` draws its first `n_startup_trials` — **default 10** — purely at **random**. So a **12-trial study is a 10-trial random search with two informed guesses stapled on**, and a 10-trial study is random search with extra steps. Run something small, see no magic, and you will wrongly conclude the tool does not help. It genuinely is not doing the thing it is famous for at that size. Budget **≥ 30–50 trials** before you judge a sampler — and in §5, even at 27 it has barely got going.

---

## 4. How It Works — Define-by-Run, and Why a Grid Cannot

Everything so far could have been a `RandomizedSearchCV`. This cannot:

```python
def objective(trial):
    kind = trial.suggest_categorical("model", ["xgb", "ridge"])
    if kind == "xgb":
        params = {"max_depth": trial.suggest_int("max_depth", 3, 10)}   # only exists for xgb
    else:
        params = {"alpha": trial.suggest_float("alpha", 1e-3, 10, log=True)}
    ...
```

The search space is **built while the trial runs**, so it can contain `if`. Optuna calls this **define-by-run**, and it is why `max_depth` can exist only on the branches where it means something.

**Be fair to `GridSearchCV`, because the honest version of this claim is narrower than the marketing one.** It is *not* true that a grid cannot express branching — hand it a **list of dictionaries** and each becomes its own subspace with its own keys:

```python
GridSearchCV(est, [
    {"model": ["xgb"],   "max_depth": [3, 6, 10]},
    {"model": ["ridge"], "alpha": [1e-3, 1e-1, 10]},
])
```

That works, and `RandomizedSearchCV` takes the same form. **The difference is not capability, it is what you have to write.** Every branch enumerated by hand, every value discrete, and the list growing with the product of options inside each subspace.

Where that stops being a style preference is a **search space whose shape changes**:

```python
def build_model(trial):
    layers, n_in = [], n_features
    n_layers = trial.suggest_int("n_layers", 1, 3)          # a hyperparameter...
    for i in range(n_layers):
        # ...that decides WHICH OTHER HYPERPARAMETERS EXIST.
        n_out   = trial.suggest_int(f"units_l{i}", 16, 128, log=True)
        dropout = trial.suggest_float(f"dropout_l{i}", 0.0, 0.4)
        layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout)]
        n_in = n_out
    layers.append(nn.Linear(n_in, 1))
    return nn.Sequential(*layers)
```

A 1-layer trial has 3 hyperparameters; a 3-layer trial has 7. 🎯 *Which hyperparameters exist depends on the value of another hyperparameter* — as a list of dictionaries that is one subspace per depth, each holding the full cross-product of its own layer widths; as define-by-run it is a `for` loop.

---

## 5. Worked Example — Against Grid Search, Measured

Nothing here is asserted; it was run in front of the class. But there are **two different questions** hiding in this comparison, and running them together is how people draw the wrong conclusion:

| | Question | What changes | What stays fixed |
|---|---|---|---|
| **A** | Does a richer, continuous search space beat a hand-written grid? | the space (3 discrete params → 6 continuous) **and** the algorithm | metric, data, folds, ~candidate count |
| **B** | Does adaptive sampling beat random sampling? | **only the sampler** | metric, data, folds, objective, search space, trial count |

**B is the clean experiment** — `RandomSampler` and `TPESampler` run the identical objective over the identical space for the identical trial count, so any difference is attributable to the sampler and nothing else. **A is deliberately confounded**, and has to be: the comparison you actually face is *"the grid I would have written"* vs *"the Optuna objective I would have written"*, and nobody writes a six-dimensional continuous grid. So read A as a comparison of **workflows**, and let B carry the claim about sampling.

### 5.1 The grid, doing its best

To keep it fair, the grid gets the three hyperparameters that matter most and three sensible values each — the grid a careful person would actually write. **27 combinations × 3 folds = 81 model fits, 79 seconds.**

```
best cv MAE : 1.0953   best params: {learning_rate: 0.1, max_depth: 6, n_estimators: 600}
```

**Why grids stop at three or four parameters.** Not laziness — multiplication. At ~0.97s per fit:

| parameters | combinations | fits | wall clock |
|---|---|---|---|
| 2 | 9 | 27 | 26 s |
| 3 | **27** | **81** | **79 s** |
| 4 | 81 | 243 | 4 min |
| 5 | 243 | 729 | 12 min |
| 6 | 729 | 2,187 | 35 min |
| 7 | 2,187 | 6,561 | 106 min |

Three values per parameter is **already a coarse grid** — and the cost of adding the fourth parameter is *the whole search over again, three times*. This is why grid searches in the wild have three or four parameters and why the values are always round numbers: **the method forces it.** 🎯 Optuna's cost of a fourth parameter is **zero** — the trial count is whatever you set, no matter how many dimensions it searches.

### 5.2 The comparison

| search | params searched | trials | cv MAE | wall | vs baseline |
|---|---|---|---|---|---|
| baseline (no tuning) | 0 | 0 | 1.1470 | 0 s | — |
| **GridSearchCV** | 3 | 27 | **1.0953** | 79 s | 0.0518 |
| Optuna random | 6 | 27 | 1.0956 | 79 s | 0.0514 |
| Optuna TPE | 6 | 27 | 1.0995 | 73 s | 0.0475 |
| Optuna random | 6 | 40 | 1.0956 | 117 s | 0.0514 |
| **Optuna TPE** | 6 | 40 | **1.0872** | 131 s | **0.0598** |

**Read that honestly: at 27 trials the grid *wins*, and TPE is the worst of the three.** That is the `n_startup_trials=10` effect — at 27 trials TPE has had 17 informed guesses and has barely started. Only at 40 does it pull ahead, and even then the best-value gap (1.0872 vs 1.0956) is small.

**The best value is one number; the distribution is the story:**

| | best | median trial | median of last 20 | trials under 1.10 |
|---|---|---|---|---|
| random | 1.0956 | 1.1767 | 1.1622 | **1** |
| **TPE** | **1.0872** | **1.1288** | **1.1135** | **6** |

TPE put **six** trials under 1.10 where random put one, and its *last twenty* trials are centred half a step lower. That is what adaptive sampling buys: not a miracle best-value, but a search that **stops wasting trials on regions it has already learned are bad**. Reproduced locally on a different dataset, the same signature appears — near-identical best values, completely different clouds:

![Two optimization-history panels from a real forty-trial run: the random sampler keeps scattering trials across the whole score range for all forty trials, while TPE's cloud collapses into a tight low band after its first ten random startup trials, with the two best values almost identical.](attachments/optuna-random-vs-tpe-history.png)
*Reproduced locally (scikit-learn diabetes, HistGradientBoostingRegressor, 3-fold CV MAE, seed 7) so the behaviour is shown on a run you can rerun, not just described. The lecture's cars24 numbers are in the tables above.*

### 5.3 A trial budget is not a time budget

Trials are not all the same price. More trees and deeper ones cost more to fit — **and on this data they also tend to score better.** So a sampler that converges on the good region is also converging on the *expensive* region, and its trials get slower as it succeeds:

| | first 10 trials | last 10 trials | ratio |
|---|---|---|---|
| random | 2.09 s/trial | 3.08 s/trial | 1.5× |
| **TPE** | 2.15 s/trial | **4.65 s/trial** | **2.2×** |

🎯 *Worth knowing before promising anyone that "500 trials" will be done by lunchtime.* If a search has to finish inside a fixed window, the lever is **not** the trial count — it is the **range of the expensive hyperparameters**. `n_estimators` capped at 600 rather than 2000 sets the price of every trial in the search, including the ones you have not run yet. (Optuna also takes `timeout=` on `optimize()`, which bounds wall clock directly — use it when the deadline is the real constraint.)

### 5.4 Reading the plots — and their limits

`plot_optimization_history` draws every trial as a dot and the running best as a line; `plot_param_importances` estimates which hyperparameters were most associated with a good score; `plot_slice` shows, per parameter, what the score did. All three are one call on a study that kept its whole history.

Two caveats that keep you honest:

- **Importance is "within the region this study explored,"** not "which parameters matter." The estimate comes from 40 trials the sampler deliberately **clustered in one part of the space** — not a sweep of the whole thing. Read it as exploratory, for deciding where to look next.
- **The slice plot flattens five other dimensions onto one axis**, so an interaction between two parameters can hide completely. A visible trough suggests a promising range; a flat smear suggests low sensitivity *here*. Both readings are provisional.

And a fairness note: none of this is unique to Optuna — `GridSearchCV` keeps every configuration in `cv_results_` and you can plot that too. **What Optuna gives you is that the history, the plots, the adaptive sampling, the pruning, and the persistence are all the same object**, so none of it has to be assembled by hand.

---

## 6. Pruning — Stop Training the Ones Already Losing

Most trials are not close. With XGBoost at a second a fit, finishing the hopeless ones is annoying but survivable. Now make each fit take four minutes — which is what happens the moment a neural network is involved. A 60-trial search becomes four hours, and three and a half of those hours are spent finishing models that were visibly bad after thirty seconds.

**Pruning** is the fix: report your score as training goes, and let Optuna kill the trial when it is clearly behind. It needs a model that trains in **visible steps** — a network over epochs, a booster over rounds. Everything about it is two lines at the bottom of the loop:

```python
for epoch in range(EPOCHS):
    ...                                   # train one epoch
    val_mae = evaluate(model, X_val, y_val)

    trial.report(val_mae, epoch)          # "here is how I am doing at step N"
    if trial.should_prune():              # "...should I stop?"
        raise optuna.TrialPruned()
```

`report` hands Optuna an intermediate value. `should_prune()` asks the study's **pruner** whether this trial is worth continuing. `MedianPruner`, the default, answers *yes, stop* when this trial's best score so far is worse than the **median of what previous trials had reached by the same step**. Raising `TrialPruned` ends the trial — recorded as *pruned*, not *failed*, and **the sampler still learns from the epochs it did see.**

### 6.1 The bet, and its price

Run the identical sampler, identical seed, identical 25 trials, changing only the pruner (`NopPruner` never prunes — the control):

| | wall clock | pruned | epochs trained | best val MAE |
|---|---|---|---|---|
| no pruner | 19 s | 0/25 | 1000/1000 | **1.2587** |
| MedianPruner(5, 5) | 16 s | 9/25 | **720/1000** | 1.2610 |

**Read the two numbers together, because either one alone misleads.** *Epochs trained* is what the pruner did — 280 epochs of training that never happened, 17.4% wall clock saved. *Best val MAE* is what it cost: **+0.0023, slightly worse.** Pruning is a bet that a trial behind at epoch 5 will still be behind at epoch 40. That bet is usually right and occasionally wrong — **and this run is one of the times it was wrong**: the pruner killed a trial that would have come back. That is the price, and it is normally worth paying, because the epochs you save buy you *more trials*. 🎯

Here is the same experiment run locally, where the pruner was more aggressive and cost nothing at all:

![Two panels of validation MAE per epoch across twenty-five trials: without a pruner every trial runs all forty epochs, while with MedianPruner fourteen trials are killed as red stubs right at the warmup boundary, cutting total epochs from one thousand to five hundred and thirty-five for an identical best score.](attachments/optuna-pruning-intermediate-values.png)
*Reproduced locally (scikit-learn diabetes, MLPRegressor over 40 epochs, TPESampler seed 7) — 14/25 pruned, 535/1000 epochs, and the best score identical at 44.84.*

Notice **every red stub terminates at the same epoch**. That is `n_warmup_steps=5` doing its job: no trial can be judged before epoch 5, and the moment it can be, the hopeless ones all die at once.

### 6.2 The knobs that keep the bet sane

`MedianPruner(n_startup_trials=5, n_warmup_steps=5)`:

- **`n_startup_trials=5`** (the library default) — no pruning at all until 5 trials have *finished*, because there is no median to compare against yet.
- **`n_warmup_steps=5`** — no pruning inside the first 5 steps of any trial, because **everything looks bad at epoch 1**. The library default here is **`0`**, so set it explicitly; leaving it at 0 is how you kill a slow-starting configuration that would have won. `(certain)`

**Other pruners worth knowing by name:**

| Pruner | How it allocates |
|---|---|
| **MedianPruner** | Each trial vs the median of previous trials at the same step. The default because it is nearly always good enough |
| **SuccessiveHalvingPruner** | Give every trial a small budget, keep the top `1/reduction_factor`, multiply their budget, repeat. Budget-aware rather than pairwise |
| **HyperbandPruner** | Runs several successive-halving brackets with different aggressiveness, so it hedges the "how early is too early" question instead of guessing it |
| **PatientPruner** | Wraps another pruner and adds early-stopping-style patience — good when your metric is noisy epoch to epoch |
| **NopPruner** | Never prunes. The control for any pruning experiment |

**Trap:** pruning and **early stopping are different things** and you often want both. Early stopping ends training when *this* model stops improving; pruning ends it when this model is losing *to other trials*. A trial can be still improving (so early stopping keeps going) and hopelessly behind (so the pruner should kill it).

### 6.3 Did the neural network win?

The honest question, asked out loud — a notebook that introduces an MLP and then never mentions it again has taught the wrong lesson.

First, how **not** to answer it. The obvious move is to print the MLP's best validation MAE next to the XGBoost study's best CV MAE. **Those two numbers are not comparable:** the XGBoost number is a **mean over 3 CV folds**, each fitted from scratch; the MLP number is from **one fixed validation split**, taken as the *best epoch*, and that same split steered all 25 trials. The second is optimistically biased in a way the first is not. Score both the same way — fit on `X_fit`, evaluate on `X_val`:

```
tuned XGBoost : 0.9691
tuned MLP     : 1.2610
```

On tabular data of this size and shape, **gradient-boosted trees are very hard to beat**, and a 25-trial budget on an MLP was never going to do it. That is the expected result, not a disappointment — and it is why the model that ships is the XGBoost one. What the MLP earned its place for is the **mechanism**: a search space whose shape changes per trial, and a pruner that can watch training happen. Both transfer directly to the model where they pay for themselves — the one that takes four minutes a fit. 🎯

---

## 7. Code — The Whole Thing in Five Lines, Then the Real One

```python
def objective(trial):
    params = {"max_depth": trial.suggest_int("max_depth", 3, 10), ...}
    return cv_mae(XGBRegressor(**params))                 # ONE number, the thing to minimise

study = optuna.create_study(direction="minimize", storage="sqlite:///study.db")
study.optimize(objective, n_trials=60)
```

Everything else is a consequence:

| If you need | Add |
|---|---|
| A search space with `if` in it | Nothing — it is already a function |
| To stop bad trials early | `trial.report(v, step)` + `if trial.should_prune(): raise optuna.TrialPruned()` |
| To survive a kernel restart | `storage=` and `load_if_exists=True` |
| To know which knob mattered | `plot_param_importances(study)` |
| To resume tomorrow | `optuna.load_study(study_name=..., storage=...)` |
| To start from a config you already like | `study.enqueue_trial({...})` |
| To bound wall clock, not trial count | `study.optimize(objective, timeout=3600)` |
| To run trials on several machines | point them all at the same `storage=` URL |

And the rule that comes before all of it, from §2: **check that your metric can tell two models apart before you spend an hour optimising it.**

*(Versions the lecture ran on: `optuna 4.9.0 · scikit-learn 1.7.2 · xgboost 3.2.0 · torch 2.10.0`. The defaults quoted in this note — `TPESampler(n_startup_trials=10, n_ei_candidates=24)`, `MedianPruner(n_startup_trials=5, n_warmup_steps=0)` — were verified against Optuna **5.0.0** and are unchanged.)*

---

## 8. When It Breaks

- **You tuned a metric that cannot tell two models apart.** §2. The whole run is noise. Check metric stability across splits *first*, and always score candidates on identical folds.
- **The budget was too small for the sampler you chose.** Under ~30 trials, TPE is mostly its random startup phase. Either raise the budget or admit you are running a random search — both are fine; pretending otherwise is not.
- **You overfitted the validation set.** Running 500 trials against one CV split *selects for* configurations that happen to suit those folds. The tuned CV score is **optimistically biased** because you minimised it directly. Quote the held-out test number, and if the search was large, consider nested CV.
- **One split is not evidence of an improvement.** §9.2 — measure the *difference* across several splits, not once.
- **A trial crashed and took the study with it.** `study.optimize(..., catch=(ValueError,))` records the trial as FAILED and continues. Without it, one out-of-memory configuration ends a six-hour search.
- **The search space was wrong, not the sampler.** If every good trial sits at the edge of a range, your range is the constraint — widen it and rerun. `plot_slice` is how you notice.
- **Randomness inside the objective.** If `cv_mae` is itself noisy (unseeded shuffling, non-deterministic GPU kernels), the sampler is partly modelling noise. Seed everything the objective touches; the study's `sampler` seed alone is not enough.
- **You compared two models scored by different protocols.** §6.3 — CV mean vs best-epoch-on-one-split is not a comparison.
- **The pruner killed a late bloomer.** Raise `n_warmup_steps`, or switch to `PatientPruner` / `HyperbandPruner`. Pruning is a bet; know you are making it.

---

## 9. Production and MLOps Notes — The Study Is an Artifact Too

### 9.1 A study that survives the kernel

Everything in §5 and §6 lives in memory. Restart and several minutes of compute are gone, **along with every trial that explained why the winner won.** One argument fixes it:

```python
STUDY_DB = "sqlite:///artifacts/study.db"

persisted = optuna.create_study(
    study_name="milepost-price", storage=STUDY_DB,
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=7),
    load_if_exists=True,                 # re-running continues instead of crashing on the name
)
persisted.enqueue_trial(good_params)     # start from a config you already trust
persisted.optimize(objective, n_trials=20)

reloaded = optuna.load_study(study_name="milepost-price", storage=STUDY_DB)
reloaded.optimize(objective, n_trials=10)   # picks up at trial 21, not trial 1
```

The lecture's run: 20 trials → best 1.0872, a **128 KB** file; reload, ten more trials → 30 trials, best **1.0842**. Three things that buys you, and the middle one is the MLOps point:

- A six-hour search can be killed at hour three and **resumed**.
- The study is a **record**. Six months later, *"why is `max_depth` 9?"* has an answer with 90 trials behind it instead of a shrug. This is Principle 1 (version everything) applied to the search itself — the artifact that explains the artifact.
- **Several machines can run trials into the same database at once**, because the storage *is* the coordination point. (For real parallelism use a server-backed store — Postgres or MySQL — not SQLite on a network share; SQLite's locking will bite you.) `(certain)`

### 9.2 Ship the model, and prove the improvement

`best_params` came from cross-validation on the *training* set. The model that ships is **refit on all of the training data** with those parameters — the CV was for choosing, not for the final fit.

```
baseline test MAE : 1.0244 lakhs
tuned    test MAE : 0.9687 lakhs
improvement       : 0.0557 lakhs (5.4%)
```

**But that came from a single train/test split — and §2 is the reason to distrust exactly that.** The same model scored anywhere from 0.98 to 1.12 MAE depending only on which cars landed in the test set, and the improvement just measured is 0.05. So measure the **difference** the way §2 measured the metric — across several splits, baseline and tuned trained and scored on identical data each time:

```
mean improvement : 0.0434 lakhs
worst split      : +0.0094      best split: +0.0731
splits improved  : 8 of 8
```

🎯 **This is the number to quote:** not *"tuning gained 5%"* from one lucky split, but an improvement that holds on **8 of 8 splits the search never saw**. And note what makes it a real test — *if it had held on only four of eight, the right conclusion would have been that 40 trials bought nothing*, and that is a conclusion worth being able to reach.

### 9.3 What goes on disk

```python
joblib.dump(tuned_model,    "artifacts/price_model.joblib")
joblib.dump(baseline_model, "artifacts/baseline_model.joblib")

model_card = {
    "model": "XGBRegressor (no preprocessing — trees are scale-invariant)",
    "target": "selling_price (lakhs)", "features": FEATURES,
    "study_name": STUDY_NAME, "storage": "artifacts/study.db",
    "n_trials": len(persisted.trials), "best_trial": persisted.best_trial.number,
    "best_params": best_params,
    "cv_mae": 1.0872, "test_mae": 0.9687, "baseline_test_mae": 1.0244,
    "test_mae_mean_over_8_splits": 1.0033,
    "metric_note": "MAE, not R² — R² moves 0.18 on this target from the split seed alone",
}
```

| file | size | why it exists |
|---|---|---|
| `price_model.joblib` | 4,185 KB | the artifact that serves |
| `baseline_model.joblib` | 393 KB | the incumbent, so the comparison survives the notebook |
| `study.db` | 136 KB | **the reason the model has those parameters** |
| `model_card.json` | 0.9 KB | what it is, what it scored, which trial produced it |

Four production notes that fall out of this table:

- **Tuning made the artifact 10× bigger** (393 KB → 4,185 KB), because the search settled on hundreds of deeper trees. Load time went with it; prediction latency did not. On a Streamlit app that's a few milliseconds; swap in a 2 GB checkpoint and the ratio goes to hundreds. **Artifact size is a tuning outcome, and it belongs in your search space as a constraint** — if inference latency has a budget, cap `n_estimators` rather than discovering the problem at deploy time. 🎯
- **There is no preprocessing to carry here** — nothing in the XGBoost path was ever scaled, because a tree splits on thresholds and a monotonic rescaling cannot move where those thresholds fall. **The model is the whole artifact.** The moment a scaler *is* involved (the MLP path), it must be fitted on training data only and persisted *with* the model — a `sklearn.Pipeline` is the way to make that impossible to forget, and it is the subject of the next class.
- **Pin `requirements.txt` with `==`, never `>=`.** A pickle carries state, not code; the class it expects to be poured back into has to still exist with the same shape. An unpinned `scikit-learn` is how an artifact that loaded fine in March stops loading in June.
- **The model card is what makes the artifact explainable six months out**, and it should carry the *data* fingerprint too — a hash of the training rows and the date window, exactly as in [Introduction to MLOps §7](Introduction%20to%20MLOps.md#7-code--from-trust-me-to-verify-me). Which trial produced it is half of provenance; which rows trained it is the other half.

### 9.4 Retuning is not free, and it is not automatic

When [drift](Introduction%20to%20MLOps.md#8-drift-in-depth--kinds-shapes-and-detection-without-labels) triggers a retrain, the honest default is **refit with the existing `best_params`**, not re-run the search. Hyperparameters are far more stable than weights — the best `max_depth` for used-car pricing does not change because last month's inventory shifted. Re-search on a schedule (quarterly), on a large data-volume change, or when a refit-only retrain stops recovering performance. `enqueue_trial(current_params)` makes the re-search start from the incumbent instead of rediscovering it, which is both faster and a fair champion/challenger comparison. `(likely)`

---

## 10. Interview Lens

**What the question is really testing:** whether you understand that tuning is a *budget allocation* problem, and whether you know what your metric is actually measuring. Candidates who only say "I use Optuna instead of GridSearchCV" have named a tool; candidates who say "grid search's cost multiplies and it learns nothing between trials" have named the reason.

🎯 **Kill-shots:**

- *"Grid search's cost multiplies with every parameter you add and it learns nothing from trial 26 when choosing trial 27. Optuna's cost is the trial count you set regardless of dimensionality, and it samples the next configuration from a model of which regions have been scoring well."*
- *"Check that your metric can tell two models apart before you spend an hour optimising it — R² has a denominator that changes with the test set, so it moved 0.18 on a model I hadn't touched."*
- *"The sampler decides which configuration to try next; the pruner decides whether the trial now running is worth finishing. Different jobs, different objects, and a study has one of each."*
- *"Cross-validation is the selection instrument, not the estimate — the tuned CV score is optimistically biased because you minimised it directly. The test set gets looked at once."*

**Likely follow-ups:**

| Question | Answer |
|---|---|
| "Why is Optuna better than grid search?" | Three reasons, in order: the space is continuous rather than a hand-picked list; the cost of an extra parameter is zero trials instead of ×3 combinations; and it uses previous results to choose the next configuration. Plus pruning and persistence in the same object. Be fair though — a grid *can* express branching via a list of dicts; the difference is what you have to write, not raw capability. |
| "How does TPE work?" | It splits finished trials at a quantile into good and rest, fits a density over each (`l(x)` and `g(x)`), and samples candidates maximising `l(x)/g(x)`, which is proportional to Expected Improvement. It draws `n_ei_candidates` (24 by default) and picks the best. Crucially it needs data first — the first `n_startup_trials` (10) are pure random. |
| "When would you use random search instead?" | When the budget is under ~30 trials (TPE is still in startup anyway), when you want a clean baseline to prove adaptive sampling is earning its keep, or when you're running massively parallel — random parallelises perfectly, while TPE's advantage comes from *sequential* feedback it can't get if 200 workers all sample at once. |
| "Your tuned model is 5% better on the test set. Convince me." | I wouldn't, from one split — the same model moves 0.14 MAE on this data depending only on the split seed, and the improvement is 0.05. I'd retrain baseline and tuned on 8 different splits, score both on identical data each time, and report the mean delta plus how many splits improved. Ours was 8/8, mean 0.043. If it had been 4/8, the honest conclusion is that tuning bought nothing. |
| "What's the difference between pruning and early stopping?" | Early stopping ends training when *this model* stops improving. Pruning ends it when this trial is losing *to other trials* at the same step. A trial can be still improving and hopelessly behind — early stopping keeps going, the pruner should kill it. You usually want both. |
| "You have 8 hours and each fit takes 4 minutes. How do you spend it?" | First check the metric is stable. Then: a pruner, because that's where the leverage is on expensive fits — cap the budget per trial and kill the losers early. Persist the study to SQLite so a crash doesn't cost the whole 8 hours. Constrain the ranges of the *expensive* parameters, since a trial budget isn't a time budget and good configs tend to be slow ones. And use `timeout=` if the deadline is hard. |

---

## 11. Alternatives and How to Choose

| Method | How it picks | Use it when |
|---|---|---|
| **Manual / intuition** | You | You have strong priors and 3 parameters. Genuinely fine as a first pass — and a good hand-picked starting point is worth `enqueue_trial`-ing into a real search |
| **GridSearchCV** | Exhaustive cross-product | ≤3 parameters with genuinely discrete, meaningful values; you need the guarantee of full coverage; or reviewers want an auditable exhaustive sweep |
| **RandomizedSearchCV** | i.i.d. draws from distributions | A strong, embarrassingly-parallel baseline. Beats grid at equal budget because it doesn't waste trials on unimportant axes (Bergstra & Bengio 2012) |
| **HalvingGridSearchCV / HalvingRandomSearchCV** | Successive halving on a budget resource (`n_samples` or `n_estimators`) | You want pruning-like savings **without leaving scikit-learn**. The closest sklearn gets to Optuna's pruner |
| **Optuna — TPE** | Density ratio `l(x)/g(x)` over past trials | The default. 30+ trials, mixed/conditional spaces, define-by-run, pruning, persistence, easy parallelism |
| **Optuna — GPSampler** | Gaussian-process Bayesian optimisation | Few, *very* expensive trials over a small continuous space — GPs model smooth landscapes well but scale badly in dimension and trial count |
| **Optuna — CmaEsSampler** | Evolution strategy on the covariance | Continuous, moderately high-dimensional spaces with a large trial budget |
| **Hyperband / BOHB** | Bandit budget allocation (+ Bayesian modelling in BOHB) | Deep learning, where each trial's cost is dominated by epochs and the real question is *how much budget to give whom* |
| **Ray Tune** | Orchestration layer (can drive Optuna) | You need distributed trials across a cluster with fault tolerance — it complements Optuna rather than replacing it |

**The decision, in one rule:** 🎯 *fewer than ~30 trials → random search and be honest about it; 30+ trials on a mixed or conditional space → Optuna with TPE; expensive per-trial training that happens in steps → add a pruner, because that is where the money is.*

And the meta-rule that outranks all of them: **tuning is usually not where your gains are.** A 5.4% MAE improvement from 40 trials is real and worth having, but it is smaller than what you would get from a better feature, more data, or fixing a leak. Tune *after* the data work, not instead of it.

---

## 🧠 Self-Test

1. **Before tuning anything, what do you check about your metric — and what did the lecture find?**
   <details><summary>answer</summary>That it can <b>tell two models apart</b>: it must move when the model improves and stay put when it doesn't. Changing only the train/test <code>random_state</code> six times on an <i>unchanged</i> model, <b>R² swung 0.1775</b> (8.2% relative) while MAE moved 5.4%. The cause: <code>R² = 1 − MSE/Var(y_test)</code> has a <b>denominator that is the test set's own variance</b>, which changed by 60% across splits (<code>corr(R², test variance) = −0.78</code>). Seed 5 happened to put the ₹395 lakh car in the test set — one row of 3,996 moved the headline score fifteen points. MAE has no denominator, is in the units of the problem, and stays comparable. Two rules follow: optimise a metric in the problem's units, and score every candidate on <b>identical folds</b>.</details>

2. **Explain the difference between a sampler and a pruner, and name the API calls each answers.**
   <details><summary>answer</summary>The <b>sampler</b> decides <b>which configuration to try next</b> and runs <b>once, before each trial starts</b> — it answers the <code>trial.suggest_int/float/categorical(...)</code> calls. TPESampler, RandomSampler, GPSampler, GridSampler. The <b>pruner</b> decides whether the trial <b>now running</b> is worth finishing and runs at <b>every step you report</b> — it answers <code>trial.should_prune()</code> after <code>trial.report(value, step)</code>. MedianPruner, SuccessiveHalvingPruner, HyperbandPruner, NopPruner. A study has one of each, and changing one tells you nothing about the other — which is why a clean experiment varies exactly one of them.</details>

3. **You run a 12-trial TPE study, see no improvement over random, and conclude Optuna doesn't help. What did you get wrong?**
   <details><summary>answer</summary><b>TPE cannot model anything until it has results to model from</b>, so <code>TPESampler</code> draws its first <code>n_startup_trials</code> — <b>default 10</b> — purely at random. A 12-trial study is therefore a 10-trial random search with two informed guesses stapled on. The lecture's own 27-trial comparison shows the same effect: at 27 trials TPE (1.0995) was <i>worse</i> than both random (1.0956) and the grid (1.0953); only at 40 did it pull ahead (1.0872). Budget at least 30–50 trials before judging a sampler. And when you do judge it, look at the <b>distribution</b>, not just the best value: at 40 trials TPE put <b>6</b> trials under 1.10 where random put <b>1</b>.</details>

4. **What does `log=True` do in `suggest_float`, and when do you need it?**
   <details><summary>answer</summary>It samples the <b>exponent</b> rather than the value, so every order of magnitude gets equal attention. Needed for any parameter that matters on a <b>multiplicative</b> scale — learning rates, regularisation strengths, layer widths — where the gap 0.01→0.02 is the same <i>kind</i> of change as 0.1→0.2. Without it, uniform sampling from 0.01–0.3 puts <b>two thirds of your trials above 0.1</b>, so the small-learning-rate region that might contain the optimum is barely explored. Rule of thumb: if you'd naturally describe the range in powers of ten, use a log scale.</details>

5. **Pruning saved 280 of 1000 epochs but the best validation MAE got slightly *worse* (1.2587 → 1.2610). Was pruning a mistake?**
   <details><summary>answer</summary>No — that is the <b>known price of the bet</b>, and you read the two numbers together. Pruning bets that a trial behind at epoch 5 will still be behind at epoch 40; usually right, occasionally wrong, and this run is one of the times it was wrong — the pruner killed a trial that would have come back. What you bought is 17.4% wall clock and 280 epochs of training that never happened, and on an expensive model those saved epochs buy you <b>more trials</b>, which is worth far more than 0.0023 MAE. The knobs that keep the bet sane: <code>n_startup_trials=5</code> (no median to compare against before then) and <code>n_warmup_steps=5</code> (everything looks bad at epoch 1 — and note the library default is <b>0</b>, so set it explicitly).</details>

6. **The tuned model beats the baseline by 0.0557 MAE on the test set. Is that an improvement?**
   <details><summary>answer</summary><b>Not demonstrated by that number alone.</b> It came from one split, and §2 showed the same model moving 0.14 MAE across splits — the claimed gain is smaller than the measurement's own noise. Measure the <b>difference</b>, not the level: retrain baseline and tuned on several splits, score both on <i>identical</i> data each time, and report the distribution of the delta. Here: mean improvement <b>0.0434 lakhs</b>, worst split still +0.0094, and <b>8 of 8 splits improved</b>. That is what makes it real. Also note the number you quote is the <b>test</b> MAE, not the CV MAE — the CV score is optimistically biased because you minimised it directly.</details>

7. **Why persist a study to SQLite, and what does that unlock?**
   <details><summary>answer</summary><code>storage="sqlite:///study.db"</code> + <code>load_if_exists=True</code> writes every trial to disk as it goes. Three things: ① a six-hour search can be <b>killed at hour three and resumed</b> with <code>optuna.load_study</code>, which picks up where it stopped; ② the study becomes a <b>record</b> — six months later "why is max_depth 9?" has 90 trials behind it instead of a shrug, which is Principle 1 (version everything) applied to the search itself; ③ <b>several machines can run trials into the same database</b>, because the storage is the coordination point. For real parallelism use Postgres/MySQL rather than SQLite on a network share — SQLite's locking will bite you. Also useful: <code>enqueue_trial(params)</code> to start from a configuration you already trust instead of rediscovering it.</details>

8. **Your search must finish in two hours. Which lever do you pull?**
   <details><summary>answer</summary><b>Not the trial count</b> — a trial budget is not a time budget. Trials get <i>slower as the search succeeds</i>, because good configurations tend to be expensive ones: in the lecture's run TPE's last ten trials averaged 4.65 s against 2.15 s for its first ten, a <b>2.2×</b> drift (random drifted only 1.5×, precisely because it never converged on the expensive region). The real lever is the <b>range of the expensive hyperparameters</b> — capping <code>n_estimators</code> at 600 rather than 2000 sets the price of every trial in the search, including ones not yet run. Add a <b>pruner</b> if the model trains in steps, and pass <code>timeout=7200</code> to <code>optimize()</code> if the deadline is hard rather than approximate.</details>

---

*Covers: parameter vs hyperparameter · the nested loop's three failures · grid vs random coverage · metric stability as a precondition for tuning (R² instability, corr with test variance, why the CV score is worse than the test score) · trial/study/sampler/pruner · the suggest API and log-scale sampling · how TPE actually decides and the n_startup_trials trap · define-by-run and the honest limits of the grid comparison · the measured grid-vs-Optuna comparison and why the distribution matters more than the best value · trial budget vs time budget · pruning mechanics, the MedianPruner knobs, other pruners, and the price of the bet · XGBoost vs MLP scored honestly · study persistence, resumption and distribution · proving an improvement across splits · the model card and artifact-size fallout · retuning policy. Sourced from Scaler MLOps Session 2 (2026-09-21) and its live notebook; the random-vs-TPE and pruning figures are locally reproduced runs.*
