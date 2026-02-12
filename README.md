# Adult Income Prediction (UCI Dataset) - Data Exploration & Planning Report

## 1. Project Overview
This project aims to solve a binary classification problem: predicting whether an individual's annual income exceeds $50,000 based on census data.

### Dataset Characteristics
* **Total Rows:** 48,842
* **Total Features:** 14 Predictors + 1 Target (`income`)
* **Data Types:** Mixed (Categorical + Numerical)
* **Target Distribution:**
    * `<=50K`: ~76% (Majority Class)
    * `>50K`: ~24% (Minority Class)

---

## 2. Key Exploratory Data Analysis (EDA) Findings

### A. Redundancy Analysis (Education vs. Education-Num)
Statistical analysis confirms a **perfect linear correlation** between these two features. `educational-num` is a direct numerical mapping of the `education` labels.
* **Mapping:** `Preschool (1)` → `HS-grad (9)` → `Bachelors (13)` → `Doctorate (16)`.
* **Issue:** Keeping both causes **Multicollinearity**, which destabilizes model coefficients and inflates standard errors in linear models.
* **Decision:** Drop the categorical `education` column and retain the numerical `educational-num`.



### B. Demographic Disparities (Bias & Fairness)
The dataset exhibits significant income distribution gaps across sensitive attributes:
* **Gender Gap:** Only **10.93%** of females earn `>50K`, compared to **30.38%** of males.
* **Racial Gap:** 'Asian-Pac-Islander' (26.93%) and 'White' (25.40%) groups have significantly higher proportions in the high-income bracket compared to other groups (approx. 11-12%).

### C. Distribution & Skewness
* **Capital-Gain:** Extremely right-skewed (**Skewness: ~11.89**), with the majority of values at zero.
* **Outliers:** `fnlwgt` (final weight) and `hours-per-week` contain extreme values that require robust handling.

---

## 3. Data Handling & Preprocessing Strategy

### 1. Missing Value Imputation
| Feature | Missing Count | Strategy |
| :--- | :--- | :--- |
| `workclass` | 2,799 | Mode Imputation (Most Frequent) |
| `occupation` | 2,809 | Mode Imputation |
| `native-country` | 857 | Mode Imputation |

### 2. Feature Engineering & Transformation
* **Log Transformation:** Apply `np.log1p` to `capital-gain` to reduce extreme skewness and normalize the distribution.
* **Scaling:**
    * `fnlwgt`: Use **RobustScaler** (to mitigate the influence of extreme outliers).
    * `age`, `hours-per-week`: Use **StandardScaler**.

### 3. Adaptive Encoding Strategy
* **Low Cardinality (`gender`, `race`):** One-Hot Encoding.
* **Medium Cardinality (`marital-status`, `workclass`):** One-Hot Encoding.
* **High Cardinality (`occupation`, `native-country`):** * **Strategy:** Use **Target Encoding** (with K-Fold smoothing to prevent leakage) or **Binary Encoding**.
    * **Grouping:** Categorize countries with < 100 observations into an "Other" group to reduce dimensionality.

---

## 4. Modeling Challenges & Agentic Planning

### I. Class Imbalance
Accuracy is a deceptive metric here. 
* **Strategy:** Prioritize **F1-Score** and **AUPRC**. Utilize `class_weight='balanced'` in model parameters.

### II. Fairness Risk
Significant income gaps in demographic features may lead to algorithmic bias.
* **Strategy:** Monitor fairness metrics (e.g., Disparate Impact) and consider bias mitigation techniques.

### III. Dimensionality (Encoding Explosion)
One-Hot Encoding high-cardinality features can create a very sparse matrix.
* **Strategy:** Prefer **Tree-based models** (XGBoost, LightGBM, CatBoost) which handle high-dimensional categorical data efficiently.

---

## 5. Agent Behaviour Summary
This project follows an **Adaptive Agentic Strategy** rather than a static pipeline:
1.  **Signal Extraction:** Automatically detects skewness, imbalance ratios, and feature redundancy.
2.  **Dynamic Planning:** Selects preprocessing paths based on data signals (e.g., if skewness > 1.0, apply Log Transformation).
3.  **Reflection Mechanism:** If performance (F1-Score) falls below a threshold, the agent re-evaluates encoding strategies and feature selection.

