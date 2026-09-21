# Experiment 7 — Observations
### Bagging, Boosting, and Stacked Ensemble Models (WDBC Dataset)

Values below are from the reference run of `experiment.ipynb` (fixed `random_state = 42`).
If you re-run the notebook and get slightly different numbers, replace these with your own —
the structure/columns will stay identical.

---

## Table 1: Bagging Hyperparameter Evaluation

| n_estimators | max_samples | max_features | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|
| 25 | 1.0 | 0.5 | 96.48 | 0.9720 |
| 25 | 0.5 | 0.5 | 96.26 | 0.9703 |
| 100 | 0.7 | 0.5 | 96.04 | 0.9683 |
| 50 | 0.5 | 0.5 | 96.04 | 0.9681 |
| 25 | 0.5 | 0.7 | 96.04 | 0.9682 |
| 100 | 1.0 | 0.5 | 96.04 | 0.9684 |
| 50 | 1.0 | 0.5 | 96.04 | 0.9683 |
| 50 | 0.7 | 0.5 | 95.82 | 0.9665 |

**Best Bagging parameters:** `n_estimators = 25`, `max_samples = 1.0`, `max_features = 0.5`
**Best Bagging CV accuracy:** 96.48%

---

## Table 2: Boosting Hyperparameter Evaluation

### 2a. AdaBoost

| n_estimators | learning_rate | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| 100 | 1.0 | 98.02 | 0.9844 |
| 150 | 1.0 | 97.80 | 0.9827 |
| 150 | 0.5 | 97.36 | 0.9792 |
| 100 | 0.5 | 97.36 | 0.9792 |
| 50 | 0.5 | 96.70 | 0.9737 |
| 50 | 1.0 | 96.70 | 0.9740 |

**Best AdaBoost parameters:** `n_estimators = 100`, `learning_rate = 1.0`
**Best AdaBoost CV accuracy:** 98.02%

### 2b. Gradient Boosting

| n_estimators | learning_rate | max_depth | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|
| 150 | 0.2 | 2 | 97.36 | 0.9789 |
| 100 | 0.2 | 2 | 96.92 | 0.9755 |
| 150 | 0.2 | 3 | 96.70 | 0.9738 |
| 150 | 0.1 | 2 | 96.48 | 0.9719 |
| 100 | 0.2 | 3 | 96.48 | 0.9719 |
| 50 | 0.2 | 2 | 96.26 | 0.9701 |

**Best Gradient Boosting parameters:** `n_estimators = 150`, `learning_rate = 0.2`, `max_depth = 2`
**Best Gradient Boosting CV accuracy:** 97.36%

**Boosting algorithm selected for final comparison:** AdaBoost *(higher CV accuracy: 98.02% vs. 97.36%)*

---

## Table 3: Stacked Ensemble Evaluation

*Base models fixed: SVM, Naive Bayes, Decision Tree. Columns vary the SVM/DT sub-parameters and the Logistic Regression meta-learner's regularisation.*

| SVM C | SVM kernel | DT max_depth | LogReg C (meta-learner) | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|---|
| 0.1 | linear | None | 10 | 97.36 | 0.9791 |
| 0.1 | linear | 5 | 10 | 97.36 | 0.9791 |
| 0.1 | linear | 3 | 10 | 97.14 | 0.9773 |
| 1.0 | rbf | 5 | 10 | 97.14 | 0.9771 |
| 0.1 | linear | 5 | 1 | 96.92 | 0.9755 |
| 1.0 | rbf | 3 | 10 | 96.92 | 0.9754 |

**Best Stacking parameters:** `svm__C = 0.1`, `svm__kernel = linear`, `dt__max_depth = 5`, `final_estimator__C = 10`
**Best Stacking CV accuracy:** 97.36%

---

## Table 4: Performance Comparison of Ensemble Models

*Computed on the held-out 20% test set (114 samples).*

| Model | Accuracy (%) | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Bagging | 93.86 | 0.9452 | 0.9583 | 0.9517 | 0.9902 |
| Boosting (AdaBoost) | 95.61 | 0.9467 | 0.9861 | 0.9660 | 0.9818 |
| Stacked Ensemble | 97.37 | 0.9859 | 0.9722 | 0.9790 | 0.9924 |

---

## Additional Observations

- **Confusion matrices:** Bagging misclassified the most malignant cases as benign (4 of 42 —
  90% recall on malignant), the clinically costlier error. AdaBoost also missed 4 of 42 malignant
  cases (90% recall) but had near-perfect benign recall (99%). Stacking had the best malignant
  recall (98%, only 1 of 42 missed) alongside the highest overall accuracy.
- **ROC curves:** Stacking has the highest AUC (0.9924), narrowly ahead of Bagging (0.9902) and
  AdaBoost (0.9818) — all three separate the classes well across thresholds, but Stacking is most
  robust to the choice of decision threshold.
- **Learning curves:** Run the corresponding notebook cell to inspect train/CV convergence —
  Bagging is expected to show a persistent train/CV gap (variance-limited), while the boosted and
  stacked models should converge more tightly as training size grows.

## Observation Questions (from the manual)

- **How does Bagging reduce variance?** By training many Decision Trees independently on different
  bootstrap resamples and averaging (majority-voting) their predictions, so that each tree's
  idiosyncratic errors cancel out rather than compound.
- **How does Boosting address model bias?** By training learners sequentially, each one focused on
  correcting the errors of the combined ensemble so far (AdaBoost via re-weighted samples,
  Gradient Boosting via fitting residuals), progressively reducing systematic underfitting.
- **Why does stacking benefit from heterogeneous models?** SVM, Naive Bayes, and Decision Tree make
  different kinds of errors (margin-based vs. probabilistic vs. axis-aligned splits); the Logistic
  Regression meta-learner can weigh their complementary strengths, which a homogeneous ensemble
  cannot exploit.
- **Which ensemble method performed best, and why?** The Stacked Ensemble (97.37% accuracy, 0.9790
  F1, 0.9924 ROC-AUC) — it combines the complementary base learners' strengths via the meta-learner,
  edging out AdaBoost (bias reduction alone) and Bagging (variance reduction alone).
