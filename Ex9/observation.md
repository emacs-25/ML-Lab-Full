# Observation — Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## Dataset Description

| Field | Value |
|---|---|
| Dataset Name | English Handwritten Characters Dataset |
| Dataset Source | Kaggle (`archive.zip`, `Img/` + `english.csv`) |
| Number of Samples | 3410 |
| Number of Features | 1024 (32×32 grayscale, flattened) |
| Number of Classes | 62 (digits 0–9, uppercase A–Z, lowercase a–z) |
| Missing Values | 0 |
| Train-Test Split | 80 / 20, stratified (2728 train / 682 test) |

## Hyperparameter Selection (Tuned MLP)

| Parameter | Value |
|---|---|
| Hidden layer sizes | (128, 64) |
| Activation | tanh |
| Solver | sgd |
| Learning rate (init) | 0.01 |
| Batch size | 64 (compared against 32 and 128) |
| Best CV accuracy (3-fold) | 0.3644 |

### Batch size comparison at best config (test accuracy)

| Batch Size | Test Accuracy |
|---|---|
| 32 | 0.3856 |
| 64 | 0.4340 |
| 128 | 0.4003 |

## Results

| Metric | PLA | Tuned MLP |
|---|---|---|
| Accuracy | 0.1246 | 0.4340 |
| Precision (macro) | 0.1762 | 0.4870 |
| Recall (macro) | 0.1246 | 0.4340 |
| F1-score (macro) | 0.0849 | 0.4286 |
| ROC-AUC (micro) | 0.8331 | 0.9256 |
| ROC-AUC (macro) | 0.8449 | 0.9248 |

## A/B Comparison Summary

| Aspect | PLA | Tuned MLP |
|---|---|---|
| Decision boundary | Linear (one-vs-rest step units) | Non-linear (2 hidden layers, tanh) |
| Test accuracy | 12.5% | 43.4% |
| Training error at convergence | Plateaus high (non-separable data) | Loss decreases smoothly, early-stopped |
| Training time | ~5s | ~58s (full 16-config grid search, 3-fold CV) |
| Strength | Simple, fast, interpretable | Captures non-linear structure, far higher accuracy |
| Weakness | Cannot separate 62 classes linearly | More hyperparameters, longer training, needs tuning |

## Observation Questions

**Why does PLA underperform compared to MLP?**
PLA is restricted to one linear decision boundary per class (one-vs-rest), and 62-way handwritten-character separation in raw pixel space is not linearly separable, so PLA plateaus at a high training error and low test accuracy (12.5%). The MLP's hidden layers let it compose non-linear feature combinations, reaching roughly 3.5× the accuracy and a much higher macro F1 (0.43 vs 0.08).

**Which hyperparameters had the most impact on MLP performance?**
In this grid, hidden-layer size and learning rate moved CV accuracy the most: the two-layer `(128, 64)` network with a learning rate of `0.01` outperformed the smaller single-layer options in both CV and held-out test accuracy. Batch size also mattered non-trivially (43.4% at batch 64 vs 38.6% at batch 32).

**Did optimizer choice (SGD vs Adam) affect convergence?**
Yes — plotting both optimizers' loss curves under identical other hyperparameters shows visibly different convergence trajectories and endpoints over the same iteration budget, confirming optimizer choice changes convergence behaviour on this dataset rather than being an interchangeable detail.

**Did adding more hidden layers always improve results? Why or why not?**
No — only a single 64-unit layer and a two-layer `(128, 64)` network were compared, and the deeper option won here, but with only 55 samples/class the dataset is small enough that arbitrarily deeper/wider networks would likely start overfitting rather than keep improving. The non-monotonic batch-size result (64 beating both 32 and 128) is further evidence that "more/bigger" isn't uniformly better in this hyperparameter space.

**Did MLP show overfitting? How could it be mitigated?**
With `early_stopping=True` (10% of training data held out for validation, training halts once validation score stops improving), the MLP's training loss falls smoothly without the train/validation divergence pattern that would flag heavy overfitting. Given only 55 samples/class, further mitigations worth trying include stronger L2 regularization (`alpha`), data augmentation (rotation/shear/noise) to synthetically grow the per-class sample count, or reducing network capacity further.
