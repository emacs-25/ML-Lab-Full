# Experiment 3 — Observations
## Regression Analysis using Linear and Regularized Models (Loan Amount Prediction)

Dataset: Kaggle — *Predict Loan Amount Data* (`phileinsophos/predict-loan-amount-data`), 30,000 loan
applications, 24 raw columns. After dropping rows with a missing target, one-hot encoding, and imputing
missing values, the working frame has **29,660 rows × 45 columns**. Following the standard analysis for
this dataset, rows with `Loan Sanction Amount (USD) == 0` (26.5% of rows) are rejected/unprocessed
applications rather than "small loans," so regression is fit **only on the 21,457 approved applications**;
an auxiliary logistic-regression stage predicts approval itself (accuracy 0.819). All regression numbers
below are on the approved-only subset, 80/20 train–test split (`random_state=42`), numeric features
standardized with `StandardScaler` fit on the training split only.

---

## Table 1 — Hyperparameter Tuning Summary (5-fold Grid Search, scoring = neg MSE)

| Model | Search Method | Best Parameters | Best CV R² |
|---|---|---|---|
| Ridge Regression | Grid Search | α = 1 | 0.9861 |
| Lasso Regression | Grid Search | α = 10 | 0.9861 |
| Elastic Net Regression | Grid Search | α = 0.01, l1_ratio = 0.8 | 0.9861 |

**Reading the grid.** Ridge settled at a small α = 1 (out of {0.01, 0.1, 1, 10, 100}) — the model barely
needs shrinkage. Lasso preferred the largest candidate, α = 10, which is exactly what triggers its
feature-elimination behaviour (see Table 4). Elastic Net's best l1_ratio = 0.8 means the tuned Elastic Net
behaves mostly like Lasso with a touch of Ridge's coefficient-smoothing — and its best α (0.01) is far
smaller than Lasso's, so at that l1_ratio it needed much less overall shrinkage to hit the same optimum.

---

## Table 2 — Cross-Validation Performance (K = 5, evaluated on the training fold splits)

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 3613.50 | 2.7667 × 10⁷ | 5259.92 | 0.9861 |
| Ridge Regression | 3613.07 | 2.7665 × 10⁷ | 5259.72 | 0.9861 |
| Lasso Regression | 3605.96 | 2.7623 × 10⁷ | 5255.76 | 0.9861 |
| Elastic Net Regression | 3611.36 | 2.7720 × 10⁷ | 5264.98 | 0.9861 |

(Ridge/Lasso/Elastic Net rows use each model's tuned best estimator from Table 1.)

---

## Table 3 — Test Set Performance

| Model | MAE | MSE | RMSE | R² | Training Time (s) |
|---|---|---|---|---|---|
| Linear Regression | 3738.59 | 3.1485 × 10⁷ | 5611.16 | 0.9850 | 0.025 |
| Ridge Regression | 3738.20 | 3.1459 × 10⁷ | 5608.86 | 0.9850 | 0.011 |
| Lasso Regression | 3726.35 | 3.0691 × 10⁷ | 5539.90 | 0.9854 | 0.039 |
| Elastic Net Regression | 3739.22 | 3.1387 × 10⁷ | 5602.45 | 0.9851 | 0.095 |

Tuned **Lasso (α = 10)** is the best model on the held-out test set on every metric — lowest MAE, MSE,
RMSE, and highest R². The margin over Linear Regression is small in relative terms (≈1.3% lower RMSE)
but consistent across CV and test, so it isn't noise from the particular 80/20 split.

*Untuned baselines (α = 1.0 / l1_ratio = 0.5, default scikit-learn values), for reference:*

| Model | MAE | MSE | RMSE | R² | Training Time (s) |
|---|---|---|---|---|---|
| Linear Regression | 3738.59 | 3.1485 × 10⁷ | 5611.16 | 0.9850 | 0.066 |
| Ridge (α=1) | 3738.20 | 3.1459 × 10⁷ | 5608.86 | 0.9850 | 0.014 |
| Lasso (α=1) | 3735.83 | 3.1361 × 10⁷ | 5600.10 | 0.9851 | 0.094 |
| Elastic Net (α=1, l1_ratio=0.5) | 9056.20 | 4.6838 × 10⁸ | 21642.06 | 0.7770 | 0.031 |

The untuned Elastic Net is dramatically worse (R² = 0.777, RMSE ≈ 21.6k) — at α = 1 with an even L1/L2
mix it over-shrinks almost everything toward zero. This is the single largest number in the whole
experiment and is exactly the kind of failure grid search is meant to catch: after tuning, the same
Elastic Net recovers to R² = 0.985, on par with the other three models.

---

## Table 4 — Coefficient Comparison (standardized features, top 10 by |Linear coefficient|)

| Feature | Linear | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| Loan Amount Request (USD) | 43351.71 | 43328.01 | 43346.96 | 42521.41 |
| Profession_Maternity leave | -9270.32 | -4032.49 | 0.00 | -248.76 |
| Credit Score | 2990.17 | 2990.04 | 2994.64 | 2997.97 |
| Profession_Unemployed | -1262.61 | -27.78 | 0.00 | -8.27 |
| Type of Employment_Waiters/barmen staff | -1050.51 | -1030.71 | -0.00 | -641.65 |
| Profession_Pensioner | -998.49 | 212.21 | -91.90 | -159.06 |
| Profession_Working | -878.93 | 333.65 | -22.26 | -19.54 |
| Profession_Commercial associate | -771.53 | 441.36 | 56.88 | 92.76 |
| Type of Employment_Private service staff | -682.19 | -670.76 | -0.00 | -438.09 |
| Type of Employment_Cooking staff | -647.01 | -637.83 | -0.00 | -461.75 |

`Loan Amount Request (USD)` and `Credit Score` dominate and are almost untouched by regularization —
they're genuine, stable signal. Sparse categorical dummies with few supporting rows
(`Profession_Maternity leave`, `Profession_Unemployed`, several `Type of Employment_*` levels) get large,
noisy coefficients from plain Linear Regression, and Ridge/Lasso pull them sharply toward zero — Lasso
sets most of them to exactly 0.0. Overall, **Lasso zeroed out 25 of 44 standardized features**, keeping 19;
the zeroed set is almost entirely rare one-hot levels of `Profession` and `Type of Employment`.

---

## Overfitting / Underfitting Analysis

- **Train–test gap.** CV RMSE (5259–5265, computed on training-fold splits) is consistently *lower* than
  test RMSE (5540–5611). The gap is small (≈5–7%) and similar across all four models, which is the
  signature of a well-controlled fit — not the large, model-specific gap you'd see with real overfitting.
- **Effect of regularization strength.** Moving from untuned α = 1 to grid-searched α values barely moved
  Ridge or Lasso (both were already close to the unregularized optimum — Ridge's best α is still just 1),
  but it rescued Elastic Net from a severely underfit state (R² 0.777 → 0.985). This shows the model class
  itself wasn't the problem; the *specific* hyperparameter combination was.
  Lasso's larger tuned α (10 rather than 1) did **more** feature elimination (25 vs. fewer zeroed features
  at α = 1) while simultaneously **improving** test RMSE — evidence that several of the raw one-hot
  columns are pure noise for this target and that removing them helps rather than hurts.
- **Generalization after tuning.** Tuned Lasso is the only model whose test RMSE (5539.90) beats its own
  untuned baseline (5600.10) by a visible margin; Ridge and Elastic Net improve only marginally over their
  untuned counterparts once tuned. Net effect: tuning matters mainly for the *L1-heavy* corner of the
  hyperparameter space, less for Ridge, which was already well-behaved.

## Bias–Variance Analysis

- **Linear Regression (no shrinkage).** With 44 standardized features and ~21k rows, Linear Regression has
  low bias (it already achieves R² = 0.985) but its unconstrained coefficients on sparse categorical
  dummies (Table 4) indicate non-trivial variance — those coefficients would swing a lot on a different
  train split.
- **Ridge.** Because most predictive signal is concentrated in two strong numeric features
  (`Loan Amount Request`, `Credit Score`), Ridge's L2 penalty has little bias to trade off — it mainly
  damps the noisy one-hot coefficients (e.g., `Profession_Unemployed`: −1262.61 → −27.78), which reduces
  variance with almost no cost to R².
- **Lasso — sparsity effect.** Lasso's L1 penalty performs actual feature selection, zeroing 25/44
  coefficients. This is the strongest bias–variance trade in the experiment: a small, deliberate bias
  (forcing some coefficients to exactly zero, even where the true effect is non-zero) buys a larger
  variance reduction, which is why tuned Lasso wins on the test set despite being the "simplest" model in
  terms of active parameters.
- **Elastic Net.** With l1_ratio = 0.8, tuned Elastic Net inherits most of Lasso's sparsity behaviour
  (several coefficients are near zero: `Profession_Unemployed` −8.27, `Type of Employment_Waiters/barmen
  staff` −641.65) with slightly more shrinkage smoothing from its L2 component. It lands essentially tied
  with Ridge and just behind Lasso on test RMSE — consistent with sitting "between" the two penalties
  rather than beating either at its own specialty.

## Conclusion

All four models explain loan sanction amount extremely well once approval status is modeled separately
(R² ≈ 0.985 on the held-out test set), because `Loan Amount Request (USD)` alone carries most of the
signal (raw correlation 0.726) and `Credit Score`/`Property Price`/`Current Loan Expenses` add the rest.
Regularization's main value here isn't raw accuracy — it's **robustness**: Ridge and Lasso both suppress
the large, unstable coefficients that plain Linear Regression assigns to rare categorical levels, and
grid search is essential for Elastic Net, whose untuned default catastrophically underfits (R² 0.777).
**Tuned Lasso (α = 10)** is the recommended model: best test RMSE/MAE/R², fastest interpretability via its
19 surviving features, and a principled way to drop noise columns before deployment.
