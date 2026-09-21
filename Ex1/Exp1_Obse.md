# Observation Note — Experiment 1

**Subject:** ICS1512 — Machine Learning Algorithms Laboratory
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai
**Degree & Branch:** M.Tech (Integrated) Computer Science & Engineering, Semester V
**Academic Year:** 2026–2027 (Odd) · **Batch:** 2024–2029

## Aim
To explore the core functions of NumPy, Pandas, SciPy, Scikit-learn, and Matplotlib; to download
public datasets and identify the appropriate ML model for each; and to carry out the full ML
workflow — loading, EDA, preprocessing, feature selection, splitting, and evaluation — on five
benchmark datasets.

## Summary of Tasks Performed

| # | Task | Library / Tool Used |
|---|------|----------------------|
| 1 | Array creation, indexing, reshaping, matrix ops, broadcasting | NumPy |
| 2 | DataFrame creation, inspection, missing-value handling, `groupby` | Pandas |
| 3 | Descriptive statistics, hypothesis testing (t-test), linear algebra | SciPy |
| 4 | Line/scatter/bar/histogram plots, heatmaps, boxplots | Matplotlib, Seaborn |
| 5 | Baseline `fit`/`predict` workflow on Iris | Scikit-learn |
| 6 | Loaded and explored 5 datasets; identified ML task type for each | Scikit-learn, Pandas |
| 7 | EDA (distributions, correlation heatmaps, class-wise comparisons) | Matplotlib, Seaborn |
| 8 | Preprocessing: imputation, one-hot encoding, scaling | Scikit-learn (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) |
| 9 | Feature selection: ANOVA F-test, Chi-square test | Scikit-learn (`SelectKBest`, `f_classif`, `chi2`) |
| 10 | Train/test splitting (80/20, stratified where applicable) | Scikit-learn (`train_test_split`) |
| 11 | Model training and evaluation (accuracy, MAE, R², confusion matrix) | Scikit-learn |

## Dataset → ML Task Summary Table

| Dataset | Type of ML Task | Feature Selection Technique | Suitable ML Algorithm |
|---|---|---|---|
| Iris Dataset | Classification (Supervised) | ANOVA F-test (SelectKBest) | Decision Tree / Logistic Regression |
| Loan Amount Prediction | Regression (Supervised) | Correlation / domain-based selection | Linear Regression |
| Predicting Diabetes | Classification (Supervised) | Chi-square test | Random Forest Classifier |
| Classification of Email Spam | Classification (Supervised) | ANOVA F-test (SelectKBest) | Naive Bayes / Logistic Regression |
| Handwritten Character Recognition / MNIST | Classification (Supervised) | Variance / PCA on pixel intensities | Support Vector Machine (SVM) / CNN |

> **Note:** Iris, Diabetes, and the Handwritten-digit dataset were loaded via Scikit-learn's built-in
> loaders (`load_iris`, `load_diabetes`, `load_digits`) since they replicate the corresponding
> UCI datasets exactly. Loan Amount Prediction and Email Spam are not bundled with Scikit-learn, so
> structurally faithful synthetic datasets (same feature schema as the Kaggle/UCI versions) were
> generated for demonstration — substitute the real downloaded CSVs where internet access to
> Kaggle/UCI is available.

## Key Learning Outcomes
- Understood how **NumPy** arrays and vectorized operations form the numerical foundation for all
  higher-level ML libraries.
- Learned to use **Pandas** for loading, inspecting, cleaning, and aggregating tabular data.
- Practiced **SciPy**'s statistical functions (descriptive stats, t-tests) and linear algebra tools.
- Built exploratory visualizations (histograms, scatter plots, boxplots, correlation heatmaps) with
  **Matplotlib/Seaborn** to understand feature distributions and class separability before modeling.
- Applied the standard **Scikit-learn** workflow — preprocessing → feature selection → train/test
  split → model fitting → evaluation — consistently across classification and regression problems.
- Understood how to distinguish **classification** vs. **regression** tasks based on the nature of the
  target variable, and how to choose an appropriate feature-selection technique (ANOVA for
  continuous-vs-categorical relationships, Chi-square for categorical-vs-categorical) accordingly.
- Learned to handle real-world data issues such as missing values and categorical variables through
  imputation, encoding, and scaling pipelines.

## Files Submitted
- `Experiment_1.ipynb` — executed Jupyter notebook with well-commented code and captured outputs
- `Experiment_1_Report.pdf` / `.tex` — formatted LaTeX report with figures and summary table
- `Observation_Note.md` — this observation note
