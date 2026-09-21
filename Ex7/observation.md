# Experiment 6 — Observations
### Dimensionality Reduction and Model Evaluation (With and Without PCA), WDBC Dataset

Values below are from the reference run of `experiment.ipynb` (fixed `random_state = 42`). If you re-run and get slightly different numbers, replace these with your own — the structure/columns stay identical.

---

## Dataset

- **Source:** UCI ML Repository, via `sklearn.datasets.load_breast_cancer` (Wisconsin Diagnostic Breast Cancer / WDBC).
- **Samples / Features:** 569 samples, 30 numeric features.
- **Target classes and distribution:** binary — malignant (0) vs benign (1); 212 malignant, 357 benign (62.7% benign).
- **Preprocessing:** no missing values; target already numeric; features standardized with `StandardScaler` (fit on train only); 80/20 stratified train-test split (455 train / 114 test).

---

## Table 1: PCA Variance Explained

| Setting | Chosen Components / Variance Target | Explained Variance (%) | Justification |
|---|---|---|---|
| With-PCA | 95% variance target | 95.27% | Reduces 30 features to 10 components while retaining ~95% of the dataset's variance — cuts dimensionality roughly 3.0x, lowering overfitting risk and training cost for distance/margin-based models with minimal information loss. |

---

## Table 2: SVM — Hyperparameter Tuning Results

*Top 8 of 12 combinations tried, sorted by No-PCA accuracy.*

| kernel | C | gamma | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| linear | 0.1 | scale | 97.58 | 97.58 |
| linear | 0.1 | auto | 97.58 | 97.58 |
| rbf | 10.0 | scale | 96.92 | 97.14 |
| rbf | 10.0 | auto | 96.92 | 95.16 |
| rbf | 1.0 | scale | 96.7 | 96.92 |
| rbf | 1.0 | auto | 96.7 | 95.6 |
| linear | 1.0 | auto | 96.48 | 97.8 |
| linear | 1.0 | scale | 96.48 | 97.8 |

**Best No-PCA parameters:** `{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}` — CV accuracy **97.58%**  
**Best With-PCA parameters:** `{'C': 1, 'gamma': 'scale', 'kernel': 'linear'}` — CV accuracy **97.80%**

---

## Table 3: Naive Bayes — Hyperparameter Tuning Results

*Top 5 of 5 combinations tried, sorted by No-PCA accuracy.*

| var_smoothing | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|
| 1e-11 | 93.41 | 91.65 |
| 1e-10 | 93.41 | 91.65 |
| 1e-09 | 93.41 | 91.65 |
| 1e-08 | 93.41 | 91.65 |
| 1e-07 | 93.41 | 91.65 |

**Best No-PCA parameters:** `{'var_smoothing': np.float64(1e-11)}` — CV accuracy **93.41%**  
**Best With-PCA parameters:** `{'var_smoothing': np.float64(1e-11)}` — CV accuracy **91.65%**

---

## Table 4: KNN — Hyperparameter Tuning Results

*Top 8 of 20 combinations tried, sorted by No-PCA accuracy.*

| n_neighbors | weights | metric | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| 3 | distance | manhattan | 97.14 | 95.6 |
| 3 | uniform | manhattan | 97.14 | 95.6 |
| 9 | uniform | euclidean | 96.7 | 96.26 |
| 9 | distance | euclidean | 96.7 | 96.26 |
| 7 | uniform | manhattan | 96.7 | 95.38 |
| 7 | distance | manhattan | 96.7 | 95.6 |
| 3 | distance | euclidean | 96.26 | 96.04 |
| 3 | uniform | euclidean | 96.26 | 96.04 |

**Best No-PCA parameters:** `{'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}` — CV accuracy **97.14%**  
**Best With-PCA parameters:** `{'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}` — CV accuracy **96.48%**

---

## Table 5: Logistic Regression — Hyperparameter Tuning Results

*Top 5 of 5 combinations tried, sorted by No-PCA accuracy.*

| C | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|
| 0.1 | 98.24 | 98.24 |
| 1.0 | 97.8 | 97.8 |
| 10.0 | 96.26 | 97.58 |
| 100.0 | 95.6 | 97.58 |
| 0.01 | 95.16 | 95.16 |

**Best No-PCA parameters:** `{'C': 0.1}` — CV accuracy **98.24%**  
**Best With-PCA parameters:** `{'C': 0.1}` — CV accuracy **98.24%**

---

## Table 6: Decision Tree — Hyperparameter Tuning Results

*Top 8 of 24 combinations tried, sorted by No-PCA accuracy.*

| max_depth | min_samples_split | criterion | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| 3 | 10 | entropy | 93.19 | 92.97 |
| 3 | 2 | entropy | 93.19 | 92.97 |
| 3 | 5 | entropy | 93.19 | 92.97 |
| 3 | 5 | gini | 92.75 | 91.65 |
| 5 | 2 | entropy | 92.75 | 92.09 |
| 3 | 10 | gini | 92.75 | 90.99 |
| 7 | 2 | entropy | 92.75 | 92.97 |
| None | 2 | entropy | 92.75 | 92.97 |

**Best No-PCA parameters:** `{'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 2}` — CV accuracy **93.19%**  
**Best With-PCA parameters:** `{'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 10}` — CV accuracy **93.85%**

---

## Table 7: Random Forest — Hyperparameter Tuning Results

*Top 8 of 18 combinations tried, sorted by No-PCA accuracy.*

| n_estimators | max_depth | max_features | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| 100 | 10 | log2 | 96.48 | 95.6 |
| 100 | None | log2 | 96.48 | 95.6 |
| 200 | 10 | sqrt | 96.26 | 95.82 |
| 100 | None | sqrt | 96.26 | 95.6 |
| 100 | 10 | sqrt | 96.26 | 95.6 |
| 200 | None | sqrt | 96.26 | 96.04 |
| 200 | 10 | log2 | 96.04 | 95.82 |
| 100 | 5 | sqrt | 96.04 | 95.16 |

**Best No-PCA parameters:** `{'max_depth': None, 'max_features': 'log2', 'n_estimators': 100}` — CV accuracy **96.48%**  
**Best With-PCA parameters:** `{'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 200}` — CV accuracy **96.04%**

---

## Table 8: AdaBoost — Hyperparameter Tuning Results

*Top 8 of 12 combinations tried, sorted by No-PCA accuracy.*

| n_estimators | learning_rate | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|
| 100.0 | 1.0 | 98.02 | 95.16 |
| 150.0 | 1.0 | 97.8 | 95.6 |
| 150.0 | 0.5 | 97.36 | 95.38 |
| 100.0 | 0.5 | 97.36 | 96.48 |
| 50.0 | 0.5 | 96.7 | 95.6 |
| 50.0 | 1.0 | 96.7 | 95.16 |
| 150.0 | 0.1 | 96.48 | 94.29 |
| 100.0 | 0.1 | 96.04 | 92.75 |

**Best No-PCA parameters:** `{'learning_rate': 1.0, 'n_estimators': 100}` — CV accuracy **98.02%**  
**Best With-PCA parameters:** `{'learning_rate': 0.5, 'n_estimators': 100}` — CV accuracy **96.48%**

---

## Table 9: Gradient Boosting — Hyperparameter Tuning Results

*Top 8 of 27 combinations tried, sorted by No-PCA accuracy.*

| n_estimators | learning_rate | max_depth | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| 150.0 | 0.2 | 2.0 | 97.36 | 96.48 |
| 100.0 | 0.2 | 2.0 | 96.92 | 96.04 |
| 150.0 | 0.2 | 3.0 | 96.7 | 95.82 |
| 150.0 | 0.1 | 2.0 | 96.48 | 96.04 |
| 100.0 | 0.2 | 3.0 | 96.48 | 96.48 |
| 50.0 | 0.2 | 2.0 | 96.26 | 95.6 |
| 100.0 | 0.1 | 2.0 | 96.26 | 95.38 |
| 100.0 | 0.1 | 4.0 | 96.04 | 94.95 |

**Best No-PCA parameters:** `{'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 150}` — CV accuracy **97.36%**  
**Best With-PCA parameters:** `{'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 150}` — CV accuracy **96.48%**

---

## Table 10: XGBoost — Hyperparameter Tuning Results

*Top 8 of 27 combinations tried, sorted by No-PCA accuracy.*

| n_estimators | learning_rate | max_depth | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|---|---|
| 50.0 | 0.2 | 3.0 | 97.14 | 96.48 |
| 100.0 | 0.1 | 3.0 | 97.14 | 96.04 |
| 150.0 | 0.1 | 3.0 | 97.14 | 95.82 |
| 100.0 | 0.2 | 3.0 | 97.14 | 96.7 |
| 100.0 | 0.1 | 4.0 | 96.92 | 96.48 |
| 100.0 | 0.2 | 2.0 | 96.92 | 95.6 |
| 50.0 | 0.2 | 4.0 | 96.92 | 96.7 |
| 150.0 | 0.1 | 2.0 | 96.7 | 95.82 |

**Best No-PCA parameters:** `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}` — CV accuracy **97.14%**  
**Best With-PCA parameters:** `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}` — CV accuracy **96.70%**

---

## Table 11: Stacking — Hyperparameter Tuning Results

*Top 3 of 3 combinations tried, sorted by No-PCA accuracy.*

| final_estimator__C | Avg CV Accuracy (No-PCA) % | Avg CV Accuracy (With-PCA) % |
|---|---|---|
| 10.0 | 97.14 | 96.92 |
| 1.0 | 96.7 | 96.48 |
| 0.1 | 96.48 | 94.73 |

**Best No-PCA parameters:** `{'final_estimator__C': 10}` — CV accuracy **97.14%**  
**Best With-PCA parameters:** `{'final_estimator__C': 10}` — CV accuracy **96.92%**

---

## Table 12: 5-Fold Cross-Validation Results — No-PCA

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg |
|---|---|---|---|---|---|---|
| SVM | 95.60 | 97.80 | 97.80 | 97.80 | 98.90 | 97.58 |
| Naive Bayes | 91.21 | 96.70 | 89.01 | 95.60 | 94.51 | 93.41 |
| KNN | 97.80 | 98.90 | 95.60 | 95.60 | 97.80 | 97.14 |
| Logistic Regression | 97.80 | 97.80 | 98.90 | 97.80 | 98.90 | 98.24 |
| Decision Tree | 90.11 | 94.51 | 91.21 | 93.41 | 96.70 | 93.19 |
| Random Forest | 96.70 | 97.80 | 93.41 | 95.60 | 98.90 | 96.48 |
| AdaBoost | 98.90 | 100.00 | 94.51 | 97.80 | 98.90 | 98.02 |
| Gradient Boosting | 95.60 | 100.00 | 94.51 | 97.80 | 98.90 | 97.36 |
| XGBoost | 98.90 | 97.80 | 94.51 | 95.60 | 98.90 | 97.14 |
| Stacking | 96.70 | 97.80 | 95.60 | 96.70 | 98.90 | 97.14 |

---

## Table 13: 5-Fold Cross-Validation Results — With-PCA

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg |
|---|---|---|---|---|---|---|
| SVM | 95.60 | 98.90 | 98.90 | 97.80 | 97.80 | 97.80 |
| Naive Bayes | 87.91 | 93.41 | 91.21 | 93.41 | 92.31 | 91.65 |
| KNN | 95.60 | 98.90 | 94.51 | 95.60 | 97.80 | 96.48 |
| Logistic Regression | 97.80 | 97.80 | 98.90 | 97.80 | 98.90 | 98.24 |
| Decision Tree | 87.91 | 93.41 | 93.41 | 98.90 | 95.60 | 93.85 |
| Random Forest | 94.51 | 96.70 | 93.41 | 97.80 | 97.80 | 96.04 |
| AdaBoost | 94.51 | 97.80 | 94.51 | 97.80 | 97.80 | 96.48 |
| Gradient Boosting | 94.51 | 97.80 | 93.41 | 97.80 | 98.90 | 96.48 |
| XGBoost | 95.60 | 96.70 | 96.70 | 98.90 | 95.60 | 96.70 |
| Stacking | 95.60 | 97.80 | 95.60 | 96.70 | 98.90 | 96.92 |

---

## Table 14: Summary — No-PCA vs With-PCA (sorted by With-PCA accuracy)

| Model | Avg Accuracy % (No-PCA) | Avg Accuracy % (With-PCA) | Delta (PCA - NoPCA) | Std Fold Acc % (No-PCA) | Std Fold Acc % (With-PCA) |
|---|---|---|---|---|---|
| Logistic Regression | 98.24 | 98.24 | 0.0 | 0.54 | 0.54 |
| SVM | 97.58 | 97.8 | 0.22 | 1.08 | 1.2 |
| Stacking | 97.14 | 96.92 | -0.22 | 1.12 | 1.28 |
| XGBoost | 97.14 | 96.7 | -0.44 | 1.79 | 1.2 |
| Gradient Boosting | 97.36 | 96.48 | -0.88 | 2.04 | 2.13 |
| KNN | 97.14 | 96.48 | -0.66 | 1.32 | 1.62 |
| AdaBoost | 98.02 | 96.48 | -1.54 | 1.89 | 1.62 |
| Random Forest | 96.48 | 96.04 | -0.44 | 1.89 | 1.79 |
| Decision Tree | 93.19 | 93.85 | 0.66 | 2.35 | 3.58 |
| Naive Bayes | 93.41 | 91.65 | -1.76 | 2.87 | 2.04 |

---

## Table 15: Test-Set Performance Comparison (All Models)

*Held-out 20% test set (114 samples), using each model's best hyperparameters.*

| Model | Setting | Accuracy (%) | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| SVM | No-PCA | 98.25 | 0.9861 | 0.9861 | 0.9861 | 0.9937 |
| SVM | With-PCA | 96.49 | 0.9857 | 0.9583 | 0.9718 | 0.995 |
| Naive Bayes | No-PCA | 92.98 | 0.9444 | 0.9444 | 0.9444 | 0.9868 |
| Naive Bayes | With-PCA | 92.11 | 0.9315 | 0.9444 | 0.9379 | 0.9709 |
| KNN | No-PCA | 96.49 | 0.9595 | 0.9861 | 0.9726 | 0.9714 |
| KNN | With-PCA | 95.61 | 0.9589 | 0.9722 | 0.9655 | 0.9788 |
| Logistic Regression | No-PCA | 97.37 | 0.9726 | 0.9861 | 0.9793 | 0.9957 |
| Logistic Regression | With-PCA | 97.37 | 0.9726 | 0.9861 | 0.9793 | 0.9954 |
| Decision Tree | No-PCA | 94.74 | 0.9459 | 0.9722 | 0.9589 | 0.9448 |
| Decision Tree | With-PCA | 92.98 | 0.9444 | 0.9444 | 0.9444 | 0.9426 |
| Random Forest | No-PCA | 95.61 | 0.9589 | 0.9722 | 0.9655 | 0.9924 |
| Random Forest | With-PCA | 93.86 | 0.9577 | 0.9444 | 0.951 | 0.9864 |
| AdaBoost | No-PCA | 95.61 | 0.9467 | 0.9861 | 0.966 | 0.9818 |
| AdaBoost | With-PCA | 93.86 | 0.9577 | 0.9444 | 0.951 | 0.9854 |
| Gradient Boosting | No-PCA | 95.61 | 0.9467 | 0.9861 | 0.966 | 0.9914 |
| Gradient Boosting | With-PCA | 94.74 | 0.9583 | 0.9583 | 0.9583 | 0.9841 |
| XGBoost | No-PCA | 94.74 | 0.9459 | 0.9722 | 0.9589 | 0.9934 |
| XGBoost | With-PCA | 93.86 | 0.9577 | 0.9444 | 0.951 | 0.9894 |
| Stacking | No-PCA | 97.37 | 0.9859 | 0.9722 | 0.979 | 0.9947 |
| Stacking | With-PCA | 96.49 | 0.9857 | 0.9583 | 0.9718 | 0.9934 |

---

## Observation Questions (from the manual)

- **Which models improved most with PCA? Which did not? Why?** SVM (+0.22 pts) and Decision Tree (+0.66 pts) improved slightly under PCA — removing correlated, redundant directions helped a margin-based model and reduced a single tree's tendency to split on noisy correlated features. Naive Bayes dropped the most (-1.76 pts) since PCA components are linear combinations of the original features and no longer approximately independent, violating Naive Bayes' conditional-independence assumption. AdaBoost also dropped notably (-1.54 pts), likely because its weak decision-stump learners lose some of the clean, axis-aligned separability present in the original feature space.
- **Did PCA reduce variance across folds (more stable results)?** Mixed — Logistic Regression's fold std stayed identical (0.54% both settings), Random Forest and XGBoost saw a small reduction in fold variance under PCA, but Decision Tree's fold variance actually increased (2.35% → 3.58%), so PCA does not uniformly stabilize every model on this dataset.
- **For high-dimensional data, was PCA beneficial in reducing overfitting?** WDBC's 30 features aren't very high-dimensional relative to the 455 training samples, so the benefit here was small; PCA's main effect was a modest dimensionality cut (30 → 10 features) rather than a large overfitting fix. On datasets with far more features than samples, the effect would likely be more pronounced.
- **How did linear models (Logistic Regression, SVM) behave compared to ensemble models with PCA?** Both linear models were flat-to-slightly-positive under PCA (Logistic Regression unchanged at 98.24%, SVM +0.22 pts), while most ensemble/tree-based models (Random Forest, AdaBoost, Gradient Boosting, XGBoost) lost 0.4–1.5 points — linear decision boundaries transfer cleanly onto a rotated/reduced axis system, while trees' axis-aligned splits are more disrupted by PCA's rotation.
- **Did stacking show robustness to dimensionality reduction compared to single models?** Yes, relatively — Stacking only dropped 0.22 pts under PCA, less than AdaBoost or Naive Bayes individually, consistent with an ensemble of heterogeneous base learners (SVM, Naive Bayes, Decision Tree) averaging out each member's individual sensitivity to the PCA transform.

---

## Final Conclusion

Best No-PCA model: **Logistic Regression**. Best With-PCA model: **Logistic Regression**. On this dataset, PCA (95% variance, 10 components) gave a small net benefit for linear/margin-based models (Logistic Regression, SVM) and Stacking, but a mild cost for most tree-based ensembles and Naive Bayes. Since WDBC's 30 features are only mildly redundant and the sample size is comfortable relative to feature count, PCA is optional here — worth using mainly for training-time/inference-time savings or when deploying distance-based models (KNN, SVM), rather than for a meaningful accuracy gain.