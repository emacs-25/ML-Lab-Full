# Observation — Experiment 8: Clustering Human Activity Recognition Data

## Dataset Description

| Field | Value |
|---|---|
| Dataset Name | Human Activity Recognition Using Smartphones (UCI HAR Dataset) |
| Dataset Source | UCI Machine Learning Repository |
| Number of Samples | 10299 (7352 train + 2947 test, combined for clustering) |
| Number of Features | 561 |
| Number of Classes | 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) |
| Missing Values | 0 |
| Train-Test Split | Original 70/30 (by subject); recombined for unsupervised clustering, ground-truth labels retained only for external evaluation |

## Preprocessing Summary

| Step | Detail |
|---|---|
| Missing value handling | None required (0 missing values found) |
| Duplicate feature names | 42 raw feature names in `features.txt` were duplicated; suffixed to make them unique (`_1`, `_2`, …) |
| Feature scaling | `StandardScaler` (zero mean, unit variance) applied to all 561 features |
| Dimensionality reduction | PCA fit on standardized features: 65 components retain ≥90% variance, 104 components retain ≥95% variance |

## Table 1: K-Means Elbow Method Results

(computed on the 65-component PCA-reduced, standardized feature space)

| Number of Clusters (k) | WCSS (Inertia) | Silhouette Score |
|---|---|---|
| 2 | 2697926.76 | 0.4386 |
| 3 | 2346425.10 | 0.3525 |
| 4 | 2207133.31 | 0.1814 |
| 5 | 2081104.64 | 0.1590 |
| 6 | 2003581.42 | 0.1400 |
| 7 | 1947312.31 | 0.1188 |
| 8 | 1892509.02 | 0.1021 |

**Selected k (by internal metrics):** k = 2. Both WCSS and Silhouette Score show their sharpest change between k=2 and k=3, and Silhouette peaks at k=2 (0.4386), then drops off steeply. This indicates the data is internally most separable into two broad clusters (dynamic vs. static activities) rather than the six semantic activity classes. **k = 6 is also fitted separately** below to allow direct, fair comparison against the six ground-truth activity labels.

## Table 2: DBSCAN ε (eps) Tuning (2D PCA space, minPts = 10)

| eps | Clusters Found | Noise Points | Noise % |
|---|---|---|---|
| 0.3 | 56 | 4789 | 46.5% |
| 0.4 | 66 | 2758 | 26.8% |
| 0.5 | 24 | 1696 | 16.5% |
| 0.6 | 9 | 1247 | 12.1% |
| 0.7 | 8 | 948 | 9.2% |
| 0.8 | 8 | 769 | 7.5% |
| **1.0 (chosen)** | **6** | **521** | **5.1%** |
| 1.2 | 4 | 367 | 3.6% |
| 1.5 | 4 | 248 | 2.4% |

**Chosen parameters:** eps = 1.0, minPts = 10 — this is the smallest eps that yields exactly 6 clusters (matching the number of ground-truth activities) while keeping noise under 10%, and sits just above the knee of the k-distance graph (90th percentile distance ≈ 0.82).

**Curse-of-dimensionality check:** the same tuning attempted directly on the 65-component PCA space (minPts = 2×65 = 130, the standard heuristic) could not find a usable eps — at eps = 16 it produced only 1 cluster with 1060/10299 (10.3%) points as noise, and smaller eps values produced 0 clusters (100% noise). This is why the reported DBSCAN result uses the 2D PCA projection, where Euclidean distances remain meaningful.

## Table 3: Clustering Evaluation Metrics (final models, k/eps as selected above)

| Algorithm | Feature Space | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | ARI ↑ | NMI ↑ |
|---|---|---|---|---|---|---|
| K-Means (k=6) | PCA-65D | 0.1400 | 2.0675 | 3287.03 | 0.4201 | 0.5597 |
| DBSCAN (eps=1.0, minPts=10) | PCA-2D | 0.3825 | 0.5247 | 9308.63 | 0.3213 | 0.4968 |
| Hierarchical (Ward, k=6) | PCA-65D | 0.1372 | 2.0443 | 3021.27 | 0.4936 | 0.6218 |

*Internal metrics for DBSCAN exclude noise points (521 of 10299). External metrics (ARI, NMI) treat noise as its own label, which slightly penalizes DBSCAN relative to the other two algorithms.*

## Table 4: Majority-Vote Cluster → Activity Mapping

| Algorithm | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 | Cluster 5 |
|---|---|---|---|---|---|---|
| K-Means (k=6) | STANDING | STANDING | WALKING_DOWNSTAIRS | WALKING_UPSTAIRS | LAYING | WALKING_DOWNSTAIRS |
| DBSCAN | LAYING | WALKING | WALKING_DOWNSTAIRS | WALKING | STANDING | WALKING_DOWNSTAIRS |
| Hierarchical (Ward) | LAYING | LAYING | STANDING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | WALKING_DOWNSTAIRS |

Confusion matrices for each algorithm (against the 6 true activities, using the mapping above) are produced in `experiment.ipynb`/`experiment.org` — see `confusion_matrices.png`.

## Observation Questions

**Which algorithm produced the most meaningful clusters? Why?**
By external agreement with the true activity labels, **Hierarchical (Ward) clustering** performed best (ARI = 0.494, NMI = 0.622), narrowly ahead of K-Means (ARI = 0.420, NMI = 0.560). Both algorithms clearly separate the static activities (LAYING/STANDING/SITTING) from the dynamic ones (WALKING variants), but neither cleanly splits SITTING from STANDING or the three WALKING sub-types from each other — these activities have very similar sensor-feature signatures. DBSCAN, judged purely by internal cluster compactness (Silhouette = 0.383, Davies–Bouldin = 0.525 — both far better than the other two), produced the most *internally cohesive* clusters, but its lower ARI/NMI shows those dense regions don't align with activity boundaries as well as the partition-based methods do.

**How sensitive was K-Means to the choice of k?**
Very sensitive. The Elbow/Silhouette results (Table 1) show that the data is internally best explained by just **k=2** (Silhouette 0.44, essentially splitting "moving" from "not moving"), and Silhouette degrades sharply for every k beyond that. Forcing k=6 to match the known activity count roughly triples WCSS's rate of reduction into diminishing returns and drops Silhouette to 0.14 — the six real activity classes are not equally well-separated in feature space, so K-Means happily reports a "good" clustering at k=2 that hides most of the semantic activity structure.

**Did DBSCAN detect noise or small clusters effectively?**
Yes, and this was one of its most informative behaviours. Sweeping eps (Table 2) showed the noise fraction fall smoothly from 46.5% down to under 3% as eps grew, and at the chosen eps=1.0 it isolated 521 points (5.1%) as noise/outliers — plausibly transition frames between activities or subjects with atypical movement signatures. It also found several very small, tight clusters (Clusters 2 and 3, mapped to WALKING_DOWNSTAIRS and WALKING) alongside two large dominant clusters, showing DBSCAN's strength at picking out small, dense sub-populations that K-Means and Ward linkage — which favour roughly equal-sized, spherical clusters — tend to merge into their larger neighbours.

**How does linkage choice (single/complete/ward) affect hierarchical clustering?**
Only Ward's linkage was run per the manual's instructions, but its behaviour is informative on its own: Ward merges clusters to minimize the increase in total within-cluster variance, which is why it produced the best external agreement (ARI/NMI) of all three algorithms here — it is effectively optimizing a similar objective to K-Means' WCSS but building it up hierarchically rather than committing to k centroids up front, letting it recover a cleaner LAYING vs. STANDING split (Cluster 0/1 vs Cluster 2) than K-Means managed. Single and complete linkage were not evaluated, but are generally expected to perform worse on this kind of moderate-dimensional, continuously-varying sensor data: single linkage is prone to "chaining" through the many near-duplicate transitional samples between activities, and complete linkage tends to produce more uneven, distance-sensitive splits than Ward's variance-based criterion.

**Which internal metric best matched your visual intuition of cluster quality?**
**Silhouette Score** best matched the 2D PCA/t-SNE visualizations (`ground_truth_pca_tsne.png`, `kmeans_clusters.png`, `dbscan_clusters.png`, `hac_clusters.png`) — DBSCAN's much higher Silhouette (0.383) visibly corresponds to its two large, well-separated blobs, and K-Means/Hierarchical's much lower Silhouette (~0.14) visibly corresponds to the same overlapping SITTING/STANDING and WALKING-variant regions seen in the scatter plots. Davies–Bouldin agreed in direction (lower/better for DBSCAN) but was harder to relate intuitively to the plots since it is unbounded and scale-dependent; Calinski–Harabasz favoured DBSCAN even more strongly than Silhouette did, largely because it rewards DBSCAN's tight, well-separated 2D clusters more than it penalizes the excluded noise points.
