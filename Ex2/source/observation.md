# Observation — Experiment 2

**Title:** Email Spam/Ham Classification using Naïve Bayes and KNN
**Subject:** ICS1512 — Machine Learning Algorithms Laboratory
**Dataset:** Spambase Dataset, UCI Machine Learning Repository (id = 94) — 4601 emails, 57 numeric features, binary class label (1 = spam, 0 = ham)

---

## 1. Dataset Summary

| Item | Value |
|---|---|
| Total samples | 4601 |
| Features | 57 (numeric, word/char frequency + capital-run-length statistics) |
| Missing values | 0 |
| Duplicate rows | 391 (retained, as duplicates are a natural feature of the raw email corpus) |
| Class balance | Ham (0): 2788 (60.6%) &nbsp;&nbsp; Spam (1): 1813 (39.4%) |
| Train / Test split | 3450 / 1151 (75% / 25%, stratified) |

**Top correlated features with the class label:** `word_freq_your`, `word_freq_000`, `word_freq_remove`, `char_freq_$`, `word_freq_you`, `word_freq_free`, `word_freq_business`, `word_freq_hp`, `capital_run_length_total`, `word_freq_our`.

---

## 2. Naïve Bayes Comparison

| Metric | Gaussian NB | Multinomial NB | Bernoulli NB |
|---|---|---|---|
| Accuracy | 0.8254 | **0.9001** | 0.8818 |
| Precision | 0.7077 | **0.9472** | 0.8768 |
| Recall | **0.9493** | 0.7907 | 0.8150 |
| F1 | 0.8109 | **0.8619** | 0.8447 |
| ROC-AUC | 0.9368 | **0.9614** | 0.9503 |

**Observation:** Multinomial NB gave the best overall accuracy, precision, F1, and ROC-AUC, since Spambase features are word/character frequency counts that fit the multinomial event model well. Gaussian NB had the highest recall (catches the most spam) but the lowest precision — it over-predicts spam because the features are not actually Gaussian-distributed (they are sparse, right-skewed counts). Bernoulli NB, which only looks at presence/absence of a term, sits between the two.

---

## 3. KNN — Effect of Varying k (Euclidean distance, unweighted)

| k | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 1 | 0.8975 | 0.8733 | 0.8656 | 0.8695 |
| 3 | 0.9018 | 0.8814 | 0.8678 | 0.8746 |
| 5 | 0.8992 | 0.8789 | 0.8634 | 0.8711 |
| 7 | 0.9053 | 0.8929 | 0.8634 | 0.8779 |
| 9 | **0.9062** | 0.8932 | 0.8656 | 0.8792 |
| 11 | 0.9027 | 0.8904 | 0.8590 | 0.8744 |

**Observation:** Accuracy rises from k=1 to k=9 and then dips slightly at k=11, consistent with the classic bias-variance pattern: very small k overfits to local noise, while larger k starts to oversmooth the boundary. Best plain (untuned) KNN: **k = 9, accuracy = 0.9062**.

---

## 4. GridSearchCV vs RandomizedSearchCV (KNN tuning)

| Parameter | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Best k | 13 | 6 |
| Metric | manhattan | manhattan |
| Weights | distance | distance |
| Algorithm | ball_tree | kd_tree |
| CV Accuracy | 0.9241 | **0.9249** |
| Execution Time (s) | 41.19 | **16.82** |

**Observation:** Both searches converge on the **Manhattan distance** and **distance-weighted** voting as the strongest configuration. RandomizedSearchCV found a comparable (marginally higher) CV accuracy in roughly **40% of the time** taken by the exhaustive GridSearchCV, by sampling 25 random configurations instead of evaluating the full 8×2×2×2 = 64-combination grid. GridSearchCV's advantage is that it is exhaustive and guaranteed to find the best combination *within the specified grid*; RandomizedSearchCV trades that guarantee for speed and scales far better to larger search spaces.

---

## 5. Best Tuned KNN — Full Evaluation

Best parameters (GridSearchCV): `algorithm=ball_tree, metric=manhattan, n_neighbors=13, weights=distance`

| Metric | Value |
|---|---|
| Accuracy | 0.9209 |
| Precision | 0.9504 |
| Recall | 0.8436 |
| F1 | 0.8938 |
| ROC-AUC | 0.9754 |
| Train Time (s) | 0.0213 |
| Predict Time (s) | 0.2531 |

**Observation:** Tuning lifted KNN's accuracy from 0.9062 (best untried k) to **0.9209**, and its ROC-AUC to **0.9754** — the highest ROC-AUC of any model in this experiment.

---

## 6. KDTree vs BallTree

| Metric | KDTree | BallTree |
|---|---|---|
| Accuracy | 0.9209 | 0.9209 |
| Training Time (s) | 0.0167 | **0.0091** |
| Prediction Time (s) | 0.2864 | **0.2404** |

**Observation:** Both tree structures return **identical accuracy** since they compute exact nearest neighbors — they differ only in the internal search structure, not the result. On this 57-dimensional dataset, **BallTree was faster** for both training and prediction, likely because Ball trees tend to handle moderately high-dimensional, non-uniformly distributed data more efficiently than KD-trees, whose partitioning advantage degrades as dimensionality grows.

---

## 7. 5-Fold Cross-Validation (on training data)

| Fold | Naïve Bayes (Multinomial) | Best KNN |
|---|---|---|
| 1 | 0.8855 | 0.9130 |
| 2 | 0.8971 | 0.9232 |
| 3 | 0.8942 | 0.9420 |
| 4 | 0.8826 | 0.9261 |
| 5 | 0.8812 | 0.9159 |
| **Average** | **0.8881** | **0.9240** |

**Observation:** The tuned KNN model consistently outperforms the best Naïve Bayes variant across every fold, with a fairly tight spread (0.9130–0.9420), confirming that its 0.9209 test accuracy is a stable estimate rather than a lucky split.

---

## 8. Theoretical vs Experimental Time Complexity

**Theoretical:**

| Algorithm | Training | Prediction |
|---|---|---|
| Naïve Bayes | O(nd) | O(d) |
| KNN (Brute) | O(1) | O(nd) |
| KDTree | O(n log n) | O(log n) avg |
| BallTree | O(n log n) | O(log n) avg |

**Experimental (n≈3450 train, n≈1151 test, d=57):**

| Algorithm | Training (s) | Prediction (s) |
|---|---|---|
| Gaussian NB | 0.0144 | 0.0007 |
| Multinomial NB | **0.0021** | **0.0002** |
| Bernoulli NB | 0.0036 | 0.0011 |
| Best KNN | 0.0092 | **0.2358** |

**Observation:** The measured times track the theory closely — all three Naïve Bayes variants train and predict in a few milliseconds (linear/constant cost), while KNN's *prediction* time is two to three orders of magnitude larger than any Naïve Bayes variant, even with a tree-based search structure, because it must traverse the tree and compute distances for every query point at inference time. KNN's training time remains negligible (it is a lazy learner — "training" is just storing the data).

---

## 9. Additional Experiments

**Effect of train-test split ratio (tuned KNN):**

| Test Size | Train Size | Accuracy |
|---|---|---|
| 0.1 | 0.9 | 0.9046 |
| 0.2 | 0.8 | 0.9164 |
| 0.3 | 0.7 | **0.9247** |
| 0.4 | 0.6 | 0.9158 |

Accuracy is fairly stable across split ratios (0.90–0.92), peaking around a 70/30 split; very small test sets (10%) give a noisier, slightly lower estimate.

**Euclidean vs Manhattan distance:**

| Distance | Accuracy | F1 |
|---|---|---|
| Euclidean | 0.9140 | 0.8894 |
| **Manhattan** | **0.9209** | **0.8938** |

Manhattan distance outperformed Euclidean, consistent with the GridSearchCV result — likely because it is less sensitive to the heavy skew/outliers present in the frequency-count features.

**Weighted vs uniform KNN:**

| Weighting | Accuracy | F1 |
|---|---|---|
| Uniform | 0.9027 | 0.8670 |
| **Distance** | **0.9209** | **0.8938** |

Distance-weighted voting (closer neighbors count more) clearly improves both accuracy and F1 over uniform voting.

**Overfitting vs underfitting:** Small k (e.g. k=1) makes KNN sensitive to individual noisy points and tends to overfit; large k oversmooths the boundary and can underfit, losing local structure. The accuracy-vs-k curve in Section 3 reflects this — it rises then plateaus/dips, and the tuned search settles on a moderate k=13 with distance weighting as the best generalizing configuration.

---

## 10. Overall Classifier Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gaussian NB | 0.8254 | 0.7077 | **0.9493** | 0.8109 | 0.9368 |
| Multinomial NB | 0.9001 | **0.9472** | 0.7907 | 0.8619 | 0.9614 |
| Bernoulli NB | 0.8818 | 0.8768 | 0.8150 | 0.8447 | 0.9503 |
| **Best KNN (tuned)** | **0.9209** | 0.9504 | 0.8436 | **0.8938** | **0.9754** |

---

## 11. Answers to Analysis Questions

1. **Which Naïve Bayes variant performed best?** Multinomial NB (accuracy 0.9001, F1 0.8619, ROC-AUC 0.9614) — its count-based event model is the best match for word/character frequency features.
2. **Optimal value of k?** k = 13 with distance weighting and Manhattan distance (per GridSearchCV); the best plain (unweighted, Euclidean) k was 9.
3. **GridSearchCV vs RandomizedSearchCV?** RandomizedSearchCV reached a slightly higher CV accuracy (0.9249 vs 0.9241) in 16.82s vs 41.19s — about 2.4× faster — by sampling only 25 of the 64 possible grid combinations.
4. **KDTree vs BallTree?** Identical accuracy (both exact nearest-neighbor methods); BallTree was faster here for both training (0.0091s vs 0.0167s) and prediction (0.2404s vs 0.2864s).
5. **Theoretical vs practical complexity?** Practical timings matched theory: Naïve Bayes trains/predicts in milliseconds; KNN prediction is by far the most expensive step, consistent with its O(nd) / O(log n)-average complexity versus Naïve Bayes' O(d) prediction cost.
6. **Preferred classifier for large datasets?** Naïve Bayes — its linear-in-n training and constant/near-constant prediction cost per query scale far better than KNN, whose prediction cost grows with the size of the reference dataset even with a tree index.

---

## 12. Conclusion

Across all metrics, the **tuned KNN model (k=13, Manhattan distance, distance-weighted, BallTree/ball_tree algorithm)** was the strongest overall classifier (Accuracy 0.9209, F1 0.8938, ROC-AUC 0.9754), narrowly ahead of **Multinomial NB** (Accuracy 0.9001, ROC-AUC 0.9614). However, KNN's prediction time (~0.24s for 1151 test points) is roughly 100–1000× slower than any Naïve Bayes variant, so for very large-scale or latency-sensitive spam filtering, Multinomial Naïve Bayes remains the more practical choice, while tuned KNN is preferable when maximum accuracy is the priority and the dataset size is moderate.
