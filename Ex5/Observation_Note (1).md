# Observation Note — Experiment 5

**Subject:** ICS1512 — Machine Learning Algorithms Laboratory
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai
**Degree & Branch:** M.Tech (Integrated) Computer Science & Engineering, Semester V
**Academic Year:** 2026–2027 (Odd) · **Batch:** 2024–2029

## Objective
- Implement a Decision Tree classifier.
- Extend the Decision Tree into a Random Forest ensemble model.
- Study the impact of hyperparameters on overfitting and generalization.
- Select optimal hyperparameters using 5-Fold Cross-Validation.
- Compare single-tree and ensemble-tree models.

## Dataset
Wisconsin Diagnostic Breast Cancer Dataset — 569 samples, 30 numerical features, target classes
Malignant (M) and Benign (B). Loaded via `sklearn.datasets.load_breast_cancer`, which provides the
exact UCI dataset (https://archive.ics.uci.edu) bundled with scikit-learn.

## Summary of Tasks Performed

| # | Task | Library / Tool Used |
|---|------|----------------------|
| 1 | Loaded dataset, encoded class labels (0 = Malignant, 1 = Benign) | Scikit-learn (`load_breast_cancer`) |
| 2 | EDA: class distribution, correlation heatmap, feature boxplots by class | Pandas, Matplotlib, Seaborn |
| 3 | Train/test split (80–20, stratified) | Scikit-learn (`train_test_split`) |
| 4 | Trained baseline Decision Tree | Scikit-learn (`DecisionTreeClassifier`) |
| 5 | Defined hyperparameter search space (criterion, max_depth, min_samples_split, min_samples_leaf) | — |
| 6 | 5-Fold Cross-Validation over 90 hyperparameter combinations | Scikit-learn (`GridSearchCV`, `StratifiedKFold`) |
| 7 | Selected best Decision Tree hyperparameters, retrained | — |
| 8 | Trained baseline Random Forest | Scikit-learn (`RandomForestClassifier`) |
| 9 | Defined hyperparameter search space (n_estimators, max_depth, max_features, bootstrap); 5-Fold CV over 36 combinations | Scikit-learn (`GridSearchCV`) |
| 10 | Selected best Random Forest hyperparameters, retrained | — |
| 11 | Compared both tuned models: accuracy, precision, recall, F1, confusion matrix, ROC/AUC, feature importances | Scikit-learn, Matplotlib, Seaborn |
| 12 | 5-fold CV fold-wise performance comparison of tuned models | Scikit-learn (`cross_val_score`) |

## Hyperparameter Tuning Results

### Decision Tree Cross-Fold Results (top combinations)

| Criterion | Max Depth | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| gini | 10 | 93.85 | 0.9506 |
| gini | 5 | 93.85 | 0.9506 |
| gini | 7 | 93.85 | 0.9506 |
| gini | None | 93.85 | 0.9506 |
| entropy | 3 | 93.41 | 0.9480 |
| gini | 3 | 93.19 | 0.9455 |

**Best Decision Tree hyperparameters:** `criterion=gini`, `max_depth=5`, `min_samples_leaf=4`,
`min_samples_split=2` — Best avg CV accuracy: **93.85%**

### Random Forest Cross-Fold Results (top combinations)

| n_estimators | Max Depth | Max Features | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|
| 100 | 10 | log2 | 96.48 | 0.9718 |
| 100 | None | log2 | 96.48 | 0.9718 |
| 200 | 10 | log2 | 96.26 | 0.9703 |
| 50 | None | log2 | 96.26 | 0.9702 |
| 100 | None | sqrt | 96.26 | 0.9699 |

**Best Random Forest hyperparameters:** `n_estimators=100`, `max_depth=10`, `max_features=log2`,
`bootstrap=True` — Best avg CV accuracy: **96.48%**

## 5-Fold Cross-Validation Performance Comparison

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---|---|---|---|---|---|---|
| Decision Tree | 0.9451 | 0.9451 | 0.9121 | 0.9231 | 0.9670 | 0.9385 |
| Random Forest | 0.9670 | 0.9780 | 0.9341 | 0.9560 | 0.9890 | 0.9648 |

## Evaluation Metrics on Held-out Test Set (Tuned Models)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree (tuned) | 0.9035 | 0.9420 | 0.9028 | 0.9220 |
| Random Forest (tuned) | 0.9561 | 0.9589 | 0.9722 | 0.9655 |

Baseline (untuned) comparison: the unconstrained Decision Tree reached **100% train / 91.23% test**
accuracy (depth 7, 19 leaves) — a clear overfitting signature — while the default Random Forest reached
**100% train / 95.61% test** accuracy, already showing a smaller generalization gap purely from
ensembling.

## Observation Questions

**1. How does tree depth affect overfitting in Decision Trees?**
As `max_depth` increases, the tree splits on progressively smaller subsets of training data and
eventually memorizes noise. This was directly visible in the baseline tree (100% train vs. 91.23% test
accuracy). Cross-validation selected a moderate depth (5), trading a little training accuracy for
better generalization.

**2. Which hyperparameter had the greatest impact on performance?**
For the Decision Tree, `max_depth` had the largest effect on the overfitting/underfitting trade-off.
For the Random Forest, `n_estimators` together with `max_depth` mattered most — more trees reduce
variance through averaging — while `max_features` had a smaller marginal effect once enough trees were
combined.

**3. How does Random Forest improve generalization?**
Random Forest trains many trees on bootstrapped samples (bagging) and considers a random subset of
features at each split. Averaging (majority voting) across many decorrelated trees cancels out
individual trees' variance, giving more stable, better-generalizing predictions than a single deep
tree.

**4. Did ensemble learning always improve performance? Why or why not?**
Yes, here the tuned Random Forest outperformed the tuned Decision Tree on every fold and every test
metric. This is expected since the Decision Tree is a high-variance base learner on this dataset —
exactly the regime where ensembling helps most. In general, though, ensembling isn't guaranteed to
always win: if the base tree is already well-regularized and close to optimal, or the dataset is very
small/simple, the marginal accuracy gain shrinks while computational cost and interpretability get
worse.

## Conclusion
Decision Tree and Random Forest models were implemented and evaluated on the Wisconsin Diagnostic
Breast Cancer dataset using 5-fold cross-validation. Hyperparameters were selected based on average
cross-validation performance, ensuring robust generalization rather than overfitting to a single
train/test split. Random Forest reduced variance and improved stability compared to a single Decision
Tree: higher average CV accuracy (96.48% vs. 93.85%), higher test accuracy (95.61% vs. 90.35%) and
F1-score (0.9655 vs. 0.9220), a smaller train/test gap, and more consistent fold-wise performance.

## Key Learning Outcomes
- Understood how impurity measures (Gini index, entropy) drive Decision Tree splits, and how
  unconstrained tree depth leads to overfitting.
- Learned to define and search a hyperparameter grid, and to use `GridSearchCV` with stratified 5-fold
  cross-validation to select hyperparameters based on generalization performance rather than a single
  train/test split.
- Understood bagging and random feature selection as the two core mechanisms by which Random Forest
  reduces variance relative to a single tree.
- Practiced comparing models using multiple complementary metrics (accuracy, precision, recall,
  F1-score, confusion matrix, ROC-AUC) rather than relying on a single number.
- Observed empirically that ensemble methods generalize more consistently across folds than a single
  decision tree, while also being able to reason about when ensembling might *not* help.

## Files Submitted
- `Experiment_5.ipynb` — executed Jupyter notebook with well-commented code and captured outputs
- `Experiment_5_Report.pdf` / `.tex` — formatted LaTeX report with figures and result tables
- `Observation_Note.md` — this observation note
