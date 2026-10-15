# Experiment 4 — Observations
## Binary Classification using Linear and Kernel-Based Models (Spambase)

All numbers are from the executed `experiment.org` / `experiment.ipynb` (random_state = 42, stratified 80/20 split: 3,680 train / 921 test; 4,601 emails, 57 features, 60.6% ham / 39.4% spam, no missing values; features standardised with a scaler fitted on the training split).

## 1. EDA
- Strongest spam correlations: `your` 0.383, `000` 0.335, `remove` 0.332, `$` 0.324, `you` 0.274, `free` 0.263. Strongest ham correlations: `hp` -0.257, `hpl` -0.233, `george` -0.183, `1999`, `labs`, `650`. All are weak to moderate (max 0.38), so no single feature separates the classes.
- The ham indicators come from the dataset donor's own mailbox (HP Labs, George), so the models partly learn one inbox.

## 2. Logistic Regression
| Metric | Baseline | Tuned (C=1, L2, saga) |
|---|---|---|
| Accuracy | 0.9294 | 0.9305 |
| Precision | 0.9209 | 0.9211 |
| Recall | 0.8981 | 0.9008 |
| F1 | 0.9093 | 0.9109 |

- Grid (C x {l1,l2} x {liblinear,saga}, 5-fold): best CV 0.9242. Accuracy only drops at C = 0.01 (0.875 L1/liblinear, 0.856 L1/saga); from C = 1 all combinations sit at 0.922-0.924.
- Tuning changed one test email (326 -> 327 spam caught); the default was already near-optimal.
- L1 vs L2 (liblinear): L1 keeps 26/57 coefficients at C = 0.01 (test 0.870 vs 0.908 for L2) and 41/57 at C = 0.05 (0.923 vs 0.921); at C = 1 both keep 57/57.
- Solvers (C = 1): same accuracy (0.927-0.931); liblinear needs 9-24 iterations and a fraction of a second, saga 1446 (L2) and 3000 = cap, not converged (L1) and several seconds.
- Largest coefficients: george -3.71, hp -2.40, capital_run_length_longest +1.59, edu -1.56, meeting -1.43, `$` +1.38.

## 3. SVM
| Kernel (default C=1) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| linear | 0.9294 | 0.9209 | 0.8981 | 0.9093 |
| poly (deg 3) | 0.7796 | 0.9598 | 0.4601 | 0.6220 |
| rbf | 0.9273 | 0.9277 | 0.8843 | 0.9055 |
| sigmoid | 0.8849 | 0.8599 | 0.8457 | 0.8528 |

| Kernel | Best parameters | CV acc. | Test acc. | Test F1 |
|---|---|---|---|---|
| linear | C=100 | 0.9304 | 0.9273 | 0.9066 |
| rbf | C=10, gamma=scale | 0.9332 | 0.9207 | 0.8976 |
| poly | C=10, degree=2, gamma=scale | 0.9179 | 0.9164 | 0.8902 |
| sigmoid | C=0.1, gamma=scale | 0.8910 | 0.8893 | 0.8534 |

RBF selected by CV. Linear with C = 100 was the slowest to search (nearly hard margin).

## 4. Final comparison (test set)
| | Logistic Regression | SVM RBF |
|---|---|---|
| Accuracy | 0.9305 | 0.9207 |
| Precision | 0.9211 | 0.9143 |
| Recall | 0.9008 | 0.8815 |
| F1 | 0.9109 | 0.8976 |
| ROC-AUC | 0.9704 | 0.9702 |
| Confusion [[TN,FP],[FN,TP]] | [[530,28],[36,327]] | [[528,30],[43,320]] |

5-fold CV on the training set (tuned models): LR 0.9236 +/- 0.0117, SVM RBF 0.9356 +/- 0.0074 (SVM higher in all 5 folds).

Train / CV / test accuracy: LR 0.930 / 0.924 / 0.931; SVM linear 0.935 / 0.930 / 0.927; SVM RBF 0.970 / 0.933 / 0.921; SVM poly 0.953 / 0.918 / 0.916; SVM sigmoid 0.896 / 0.891 / 0.889. RBF and poly show the larger train-CV gap (about 0.035-0.037), i.e. higher variance.

## 5. Takeaways
- No clear winner: CV favours the RBF SVM by about 1.2 points, the test set favours logistic regression by about 1 point, ROC-AUC is equal. The test-set standard error is about 0.85 points, so the difference is within noise.
- Regularisation strength mattered more than penalty type; L1 gives sparsity only at strong regularisation.
- liblinear is as accurate as saga and much faster on this data.
- Kernel choice mattered mostly for poly (0.780 -> 0.916 after tuning) and sigmoid (weakest).
- Both models flag about 5% of legitimate emails as spam at the default threshold; a deployed filter would need a higher threshold.
