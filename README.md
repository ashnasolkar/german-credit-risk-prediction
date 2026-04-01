# German Credit Risk Prediction

A machine learning project that predicts loan default risk levels for German bank customers using a multiclass Support Vector Machine (SVM). Risk levels (Low, Medium, High) are defined using domain-driven rules and predicted using customer financial and demographic attributes.

The model was deployed to a Tableau dashboard via TabPy.

---

## Purpose

The German Credit dataset does not include a credit score variable. This project addresses that gap by building a multiclass SVM model to predict the risk level (Low / Medium / High) of customers who do not have a credit score, using available financial and demographic proxy indicators such as savings, checking account status, job type, housing, loan purpose, credit amount, and duration — to support informed loan approval decisions.

---

## Dataset

| Property | Details |
|---|---|
| Source | German Credit Data |
| Records | 1,000 |
| Variables | Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose |
| Target | Risk_Level (Low / Medium / High) — engineered feature |

---

## Workflow

### 1. Data Cleaning
- Imputed missing values in `Saving accounts` (183) and `Checking account` (394) with `"Unknown"`
- Removed serial number column

### 2. Risk Level Engineering
Designed a domain-driven rule-based function to classify each record:

- **Low Risk**: Skilled/highly skilled job (type 2 or 3) or owns house; moderate/rich savings or checking account; credit amount ≤ 7,883 (IQR upper bound)
- **High Risk**: Unskilled job (type 0 or 1) or renting/free housing; little/unknown savings and checking account; credit amount > 7,883
- **Medium Risk**: All remaining records not meeting Low or High criteria

Converted Risk_Level to an ordered categorical variable: `Low < Medium < High`

**Distribution:** Low — 398, Medium — 587, High — 15

### 3. Feature Encoding & Scaling
- **Ordinal encoding**: `Saving accounts` and `Checking account` mapped to numeric order using `.map()`
- **One-hot encoding**: `Sex`, `Housing`, `Purpose` encoded using `pd.get_dummies()`
- **StandardScaler**: `Age`, `Credit amount`, `Duration` standardized

### 4. Correlation Analysis
- Built interactive correlation heatmap using Plotly (Viridis palette)
- `Checking account` (r = -0.52) and `Saving accounts` (r = -0.44) showed strongest negative correlation with Risk_Level
- `Job` showed virtually no correlation with Risk_Level

### 5. Multicollinearity Check (VIF)
- Calculated VIF for all features using `variance_inflation_factor` from `statsmodels`
- All VIF values below 5 — no multicollinearity concern, all features retained

### 6. SVM Classification
- Multiclass SVM (linear kernel, C=1, gamma='scale')
- `class_weight='balanced'` to handle class imbalance (only 15 High risk records)
- 70/30 stratified train-test split
- Evaluated using confusion matrix and classification report
- Visualized class separation using PCA (2D projection of actual vs. predicted labels)

---

## Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Low (0) | 0.80 | 0.82 | 0.80 | 119 |
| Medium (1) | 0.86 | 0.77 | 0.81 | 176 |
| High (2) | 0.24 | 1.00 | 0.38 | 5 |
| **Weighted Avg** | 0.82 | 0.79 | **0.80** | 300 |

- **Overall Accuracy: 79%**
- Linear kernel significantly outperformed RBF kernel (49% accuracy)
- High recall (100%) for High risk class — model correctly identifies all high-risk customers despite class imbalance

---

## Business Interpretation

The model provides a practical framework to assess risk for customers with no formal credit history. By evaluating proxy indicators such as account status and employment type, lenders can:
- Minimize default rates
- Offer adjusted loan terms for Medium risk profiles
- Confidently approve Low risk applicants

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Seaborn` `Matplotlib` `Plotly` `Statsmodels`

---

## How to Run

1. Place `german_credit_data.csv` in the same directory as the notebook
2. Open `german_credit_risk_prediction.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially

---

## References

- Google Dataset Search. (n.d.). German Credit Data. Retrieved May 8, 2025, from https://datasetsearch.research.google.com/search?src=0&query=free%20company%20bank&docid=L2cvMTF4MGMwcThtag%3D%3D
- Patel, R., & Makodia, V. (2017). Data mining using Python: Data mining and machine learning with Python, pandas, and scikit-learn. O'Reilly Media.
