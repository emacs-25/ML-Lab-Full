# Experiment 7 — Observations
### Bagging, Boosting, and Stacked Ensemble Models (WDBC Dataset)

Fill each blank cell (`—`) using the corresponding output in `experiment.ipynb`, as noted under each table.
Run the notebook top to bottom first — all values below come directly from cells already in it.

---

## Table 1: Bagging Hyperparameter Evaluation

*Source: `bagging_results` DataFrame — section "4. Bagging Classifier". Sort by `Avg CV Accuracy (%)` descending and copy the top rows (add/remove rows as needed).*

| n_estimators | max_samples | max_features | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |

**Best Bagging parameters:** `n_estimators = —`, `max_samples = —`, `max_features = —` — *(copy from the `bagging_grid.best_params_` print output)*
**Best Bagging CV accuracy:** — *(from `bagging_grid.best_score_`)*

---

## Table 2: Boosting Hyperparameter Evaluation

### 2a. AdaBoost
*Source: `ada_results` DataFrame — section "5. Boosting Classifiers".*

| n_estimators | learning_rate | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |

**Best AdaBoost parameters:** `n_estimators = —`, `learning_rate = —`
**Best AdaBoost CV accuracy:** —

### 2b. Gradient Boosting
*Source: `gb_results` DataFrame.*

| n_estimators | learning_rate | max_depth | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |

**Best Gradient Boosting parameters:** `n_estimators = —`, `learning_rate = —`, `max_depth = —`
**Best Gradient Boosting CV accuracy:** —

**Boosting algorithm selected for final comparison:** — *(printed as "Selected boosting model: ..." — whichever of AdaBoost / Gradient Boosting scored higher)*

---

## Table 3: Stacked Ensemble Evaluation

*Source: `stack_results` DataFrame — section "6. Stacked Ensemble". Base models are fixed (SVM, Naive Bayes, Decision Tree); the columns below vary their sub-parameters and the meta-learner's regularisation.*

| SVM C | SVM kernel | DT max_depth | LogReg C (meta-learner) | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
| — | — | — | — | — | — |
| — | — | — | — | — | — |
| — | — | — | — | — | — |
| — | — | — | — | — | — |
| — | — | — | — | — | — |

**Best Stacking parameters:** — *(from `stack_grid.best_params_`)*
**Best Stacking CV accuracy:** — *(from `stack_grid.best_score_`)*

---

## Table 4: Performance Comparison of Ensemble Models

*Source: `performance_df` DataFrame — section "7. Model Evaluation on the Test Set". One row per model, computed on the held-out 20% test set.*

| Model | Accuracy (%) | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Bagging | — | — | — | — | — |
| Boosting (— ) | — | — | — | — | — |
| Stacked Ensemble | — | — | — | — | — |

---

## Additional Observations

- **Confusion matrices** (section 7, second plot): note how many malignant cases were misclassified as benign (false negatives) for each model — this is the clinically important error to minimize.
- **ROC curves** (section 7, third plot): compare the AUC values above — the model with AUC closest to 1.0 has the best overall class-separation ability.
- **Learning curves** (section 8): note whether the training and cross-validation curves converge (low variance) or stay far apart (high variance/bias) for each ensemble type.

## Observation Questions (from the manual)

- How does Bagging reduce variance? —
- How does Boosting address model bias? —
- Why does stacking benefit from heterogeneous models? —
- Which ensemble method performed best, and why? —
