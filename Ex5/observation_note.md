# Experiment 5 — Observations
## Decision Tree and Random Forest on the Wisconsin Diagnostic Breast Cancer data

Executed numbers (random_state = 42). 569 patients, 30 features, 357 benign / 212 malignant (malignant encoded as 1), no missing values. Stratified 80/20 split: 455 train / 114 test (72 benign, 42 malignant). All searches use the same stratified 5-fold splits. Trees need no feature scaling.

## 1. EDA
- 15 of 435 feature pairs have |r| > 0.95 (mean radius-perimeter 0.998, worst radius-perimeter 0.994, radius-area 0.987).
- Strongest correlation with malignancy: worst concave points 0.794, worst perimeter 0.783, mean concave points 0.777, worst radius 0.776. Weakest: symmetry error 0.007, texture error 0.008, mean fractal dimension 0.013.

## 2. Decision Tree
- Unrestricted tree: depth 8, 24 leaves, train 1.000, test 0.930.
- Depth curve: train 0.930 (depth 1) -> 1.000 (depth 8); CV 0.901 (depth 1), about 0.921 from depth 2, 0.923 from depth 7. Train-CV gap grows from 0.033 (depth 2) to 0.077 (depth 8).
- Grid (90 combinations, 450 fits): best = entropy, max_depth 5, min_samples_leaf 2, min_samples_split 10, CV 0.9363. Tuned tree: depth 5, 12 leaves, train 0.9758, test 0.9649.

## 3. Random Forest
- Default (100 trees): train 1.000, test 0.9737.
- Grid (36 combinations, 180 fits): best = 200 trees, max_depth 5, max_features sqrt, bootstrap False, CV 0.9714; four configurations tie at 97.14%, all configurations 96.3-97.1%. Tuned forest: train 0.9912, test 0.9649.

## 4. Comparison (test set, malignant = positive)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9744 |
| Random Forest (tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9937 |

Both: 72 TN, 0 FP, 4 FN, 38 TP.

5-fold CV: Decision Tree 0.9363 +/- 0.0274, Random Forest 0.9714 +/- 0.0228; forest better in all 5 folds.
30 random splits (tuned settings): forest 0.9585 vs tree 0.9357; forest wins 24, ties 3, loses 3 (settings were chosen on the seed-42 training data, so absolute values are slightly optimistic).

Feature importance: tree uses 6 of 30 features, worst perimeter 65.7%; forest spreads importance over worst perimeter 16.7%, worst area 13.2%, worst concave points 11.0%, worst radius 8.9%, mean concave points 8.5%.

## 5. Answers to the observation questions
1. Depth: deeper trees fit the training set (100% at depth 8) while CV accuracy stays near 0.92, so extra depth only widens the train-CV gap; a stump underfits.
2. Largest single-hyperparameter effects: min_samples_leaf 0.6 points and criterion 0.5 points (tree), bootstrap 0.5 points (forest). The model family (tree -> forest, +3.5 CV points) mattered far more.
3. The forest averages decorrelated trees (bootstrap samples + random feature subsets), reducing variance: higher CV accuracy, higher AUC, much lower sensitivity to hyperparameters.
4. Not always: tied on the tuned test split, and the untuned forest (0.974) beat the tuned forest by one patient; over 30 splits the forest lost 3 times. On a small, clean dataset a regularised tree is already strong; the forest costs more compute and loses single-tree interpretability.
