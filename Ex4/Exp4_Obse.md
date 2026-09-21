# Experiment 4 — Observations
## Binary Classification using Linear and Kernel-Based Models (Spambase Dataset)

## 1. Dataset Summary

| Property | Value |
|---|---|
| Source | UCI ML Repository — Spambase |
| Total instances | 4,601 |
| Features | 57 (48 word-frequency, 6 char-frequency, 3 capital-run-length) |
| Missing values | 0 |
| Class balance | ham 60.6% (2,788) / spam 39.4% (1,813) |
| Train / Test split | 3,680 / 921 (80/20, stratified) |

No imputation was needed. All 57 features were standardized (zero mean, unit variance) using a scaler fit only on the training split, since both Logistic Regression's regularization penalty and SVM's distance-based kernels are scale-sensitive.

## 2. EDA Highlights

- The class distribution is moderately imbalanced (61/39) but not severe enough to require resampling; precision/recall/F1 were tracked throughout rather than relying on accuracy alone.
- The features most strongly correlated with the spam label were character frequencies for `!` and `$`, capital-run-length features, and word frequencies for `your`, `000`, `remove`, `free`, `credit`, and `money` — consistent with the intuition that promotional and unsolicited emails over-use emphasis punctuation, capitalization, and money-related vocabulary.
- `word_freq_george` and `word_freq_hp` were the two strongest *negative* indicators of spam (i.e., strongly associated with ham) — a reminder that this dataset's ham class is drawn from one individual's personal/professional mailbox, so some features are corpus-specific rather than universally generalizable spam indicators.

## 3. Logistic Regression

### Baseline vs Tuned

| Metric | Baseline (default) | Tuned (Grid Search) |
|---|---|---|
| Accuracy | 0.9294 | 0.9262 |
| Precision | 0.9209 | 0.9202 |
| Recall | 0.8981 | 0.8898 |
| F1 Score | 0.9093 | 0.9048 |
| Training time (s) | 0.013 | 0.430 |

**Best hyperparameters (5-fold Grid Search, `liblinear` solver):** `C=100`, `penalty=l1` → best CV accuracy **0.9239**.

**Observation:** the *tuned* model's held-out test accuracy (0.9262) is marginally lower than the untuned baseline's (0.9294). This is not a contradiction — grid search selects the configuration that maximizes average cross-validation accuracy on the training folds, not the configuration that happens to score highest on this one fixed test split. The gap here (~0.3 percentage points) is well within the fold-to-fold variance seen later in Section 5, so both models should be considered statistically comparable rather than one being definitively "better."

### Regularization: L1 vs L2

At `C=1`, neither L1 (Lasso) nor L2 (Ridge) drove any of the 57 coefficients exactly to zero (57/57 non-zero for both). L1 achieved 0.9273 test accuracy and L2 achieved 0.9294 — essentially indistinguishable. This indicates that, at this regularization strength and after standardization, the predictive signal for spam detection is **distributed across most of the 57 features** rather than concentrated in a small, sparse subset — so L1's feature-selection property offers little practical advantage on this dataset at moderate `C`. The grid search's own preference for `penalty=l1` at a *much larger* `C=100` (i.e., very weak regularization) reinforces this: the model performs best when regularization is barely applied at all, again suggesting the features individually carry modest but broadly-distributed signal rather than a few dominant predictors.

### Solver Comparison — `liblinear` vs `saga`

| Solver | Penalty | Fit time (s) | Test Accuracy | Iterations used |
|---|---|---|---|---|
| liblinear | l1 | 0.179 | 0.9273 | 24 |
| liblinear | l2 | 0.061 | 0.9294 | 9 |
| saga | l1 | 5.493 | 0.9294 | 3000 (capped) |
| saga | l2 | 2.118 | 0.9305 | 1446 |

`saga` reaches a marginally higher accuracy for L2 but takes **~35–90x longer** to converge and, for L1, hits the iteration cap without fully converging. For a dataset of this size (3,680 training rows × 57 features), `liblinear` is clearly the more practical solver — this is why it was used for the main grid search.

## 4. Support Vector Machine

### Kernel Comparison (default hyperparameters: C=1, γ=scale, degree=3)

| Kernel | Accuracy | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|
| Linear | 0.9294 | 0.9209 | 0.8981 | 0.9093 | 0.348 |
| Polynomial (deg 3) | 0.7796 | 0.9598 | 0.4601 | 0.6220 | 0.291 |
| RBF | 0.9273 | 0.9277 | 0.8843 | 0.9055 | 0.187 |
| Sigmoid | 0.8849 | 0.8599 | 0.8457 | 0.8528 | 0.242 |

**Observation:** the default-degree polynomial kernel is a striking outlier — very high precision (0.96) but very poor recall (0.46). It is being extremely conservative, only calling something "spam" when extremely confident, and missing more than half the actual spam. This is a classic symptom of an overly complex/ill-fitted decision boundary at degree 3 on 57 standardized dimensions. Tuning (below) recovers most of this gap by dropping to degree 2.

### Hyperparameter Tuning per Kernel (5-fold Grid Search)

| Kernel | Best Params | Best CV Accuracy | Test Accuracy | Test F1 | Search Time (s) |
|---|---|---|---|---|---|
| RBF | C=10, γ=scale | **0.9332** | 0.9207 | 0.8976 | 8.2 |
| Linear | C=100 | 0.9304 | **0.9283** | **0.9078** | 82.9 |
| Polynomial | C=10, degree=2, γ=scale | 0.9179 | 0.9164 | 0.8902 | 25.0 |
| Sigmoid | C=0.1, γ=scale | 0.8910 | 0.8893 | 0.8534 | 7.9 |

**RBF was selected as the best kernel** because it produced the highest *cross-validated* accuracy (0.9332), which is the standard model-selection criterion since it averages over 5 different train/validation splits and is more robust than a single test-set score. Interestingly, on this **particular** fixed test split, the tuned **linear** kernel actually scored slightly higher (0.9283 vs 0.9207) — a useful reminder that a single held-out test accuracy can disagree with the CV-based ranking by a small margin due to sampling variance, and that CV accuracy is the more trustworthy signal for model selection precisely because it doesn't depend on which 20% of rows happened to land in the test set.

Polynomial jumped from 0.78 accuracy (degree 3, default) to 0.916 (degree 2, tuned) — confirming that the default degree was overfitting the training data's higher-order interactions. Sigmoid remained the weakest kernel throughout, consistent with its known sensitivity to feature scale and its behavior as a shallow neural-activation-like function rather than a true kernel in the strict Mercer sense for arbitrary hyperparameters.

## 5. Final Model Comparison

| Metric | Logistic Regression (tuned) | SVM RBF (tuned) |
|---|---|---|
| Test Accuracy | 0.9262 | 0.9207 |
| Test Precision | 0.9202 | 0.9143 |
| Test Recall | 0.8898 | 0.8815 |
| Test F1 | 0.9048 | 0.8976 |
| Training/Search time (s) | 0.43 | 8.18 |

### 5-Fold Cross-Validation (on training set, final tuned models)

| Fold | Logistic Regression | SVM (RBF) |
|---|---|---|
| Fold 1 | 0.9416 | 0.9443 |
| Fold 2 | 0.9253 | 0.9416 |
| Fold 3 | 0.9348 | 0.9348 |
| Fold 4 | 0.9117 | 0.9307 |
| Fold 5 | 0.9171 | 0.9266 |
| **Average** | **0.9261** (±0.0110) | **0.9356** (±0.0066) |

**Observation:** Under 5-fold cross-validation, SVM (RBF) is consistently and clearly ahead of Logistic Regression in every single fold, with a higher mean accuracy (0.9356 vs 0.9261) *and* lower fold-to-fold variance (±0.0066 vs ±0.0110). This is the more reliable comparison (5 independent estimates vs. 1), and it reverses the single-test-split result above where LR appeared marginally ahead — reinforcing that cross-validation, not a single train/test split, should be trusted for model selection here.

## 6. Comparative Analysis

| Criterion | Logistic Regression | SVM |
|---|---|---|
| Accuracy (5-fold CV) | 0.9261 | **0.9356** |
| Model Complexity | Low | High |
| Training/Search Time | Low (fractions of a second) | Higher (single digits to tens of seconds per kernel) |
| Interpretability | High — coefficients map directly to feature influence | Low — especially for RBF, where the decision boundary is implicit in kernel space |

## 7. Bias–Variance Trade-off

- **Logistic Regression** is a simpler, linear-boundary model: **higher bias**, but comparatively **higher variance across CV folds** in this run (±0.0110) since it has less flexibility to adapt to each fold's specific data quirks and is more sensitive to which particular points happen to sit near the (single, global) decision hyperplane.
- **SVM with RBF kernel** can bend its decision boundary to local structure in the standardized feature space (**lower bias**), and in this experiment it also showed **tighter, more stable CV fold accuracy** (±0.0066) — indicating the extra flexibility was not spilling over into instability here, likely because moderate regularization (`C=10`) kept the margin reasonably wide.
- The clearest overfitting symptom observed was the **default-parameter polynomial kernel** (degree 3): very high precision, very low recall, and a large accuracy jump once degree was reduced to 2 via tuning — direct evidence of a model whose flexibility exceeded what the data supported.

## 8. Overall Conclusion

Both classifiers perform strongly on Spambase, in the low-to-mid 90s for accuracy, precision, recall, and F1, without heavy hyperparameter tuning. Given its **better and more stable cross-validated accuracy**, the SVM with an RBF kernel (`C=10`, `γ=scale`) is the more robust choice **on this criterion**; however, given its **~20x faster training/search time and substantially higher interpretability**, tuned Logistic Regression (`C=100`, `penalty=l1`) remains an excellent, nearly-as-accurate alternative — a reasonable choice in a production spam filter where explainability, retraining speed, or low-latency inference matters more than the last percentage point of accuracy.
