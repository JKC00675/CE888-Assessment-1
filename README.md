# Data Exploration & Planning Report
## Dataset: Adult Income (UCI)

### 1. Key Findings from EDA
The Adult Income dataset is a binary classification problem aiming to predict whether an individual's annual income exceeds $50,000.

**Dataset Characteristics**
* **Number of rows**: 48,842
* **Number of features**: 15 (14 predictors + 1 target)
* **Target variable**: `income` (Predicts `<=50K` or `>50K`)
* **Feature types**:
    * **Categorical**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`, `income`
    * **Numerical**: `age`, `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`
* **Presence of missing values**: Found in `workclass` (2,799), `occupation` (2,809), and `native-country` (857)
* **Class imbalance**: 
    * `<=50K`: **76.07%** (37,155 samples)
    * `>50K`: **23.93%** (11,687 samples)



**Observations**
* The dataset is moderately imbalanced, with the majority class being ≤50K.
* **Capital-gain/loss** exhibits extreme right skewness, suggesting the need for scaling or transformation.
* **Education** and **educational-num** show redundancy (Multicollinearity).
* **Occupation** and **workclass** have high cardinality and correlated missing patterns.
* Income distribution differs significantly across gender groups.

---

### 2. Identified Challenges
**1. Imbalance Risk**
Accuracy alone may overestimate model performance. Models might favor the majority class, making **F1-score** and **Recall** more informative metrics.

**2. Bias & Fairness Risk**
Sensitive attributes such as `gender` and `race` may influence predictions, requiring fairness auditing.

**3. Encoding Explosion**
High-cardinality categorical variables like `native-country` may significantly increase dimensionality if not handled via Target or Binary encoding.

**4. Skewed Features**
Highly skewed variables (e.g., `capital-gain`) may distort linear models and require log transformation.

**5. Multicollinearity**
Redundant features like `educational-num` and `education` may reduce model stability and interpretability.

---

### 3. Proposed Agentic Planning Strategy
This agent will dynamically adapt based on dataset signals rather than following a fixed pipeline.

**Step 1 — Signal Extraction**
The agent extracts structural signals:
* **Imbalance ratio**: 3.18 : 1
* **Nullity Patterns**: Correlated missingness in work/occupation
* **Cardinality**: Number of unique labels per category
* **Skewness**: Statistical distribution of numerical features

**Step 2 — Adaptive Planning Rules**
* **If imbalance > 60/40:**
    * Prioritize **F1-score** over Accuracy.
    * Apply **class_weight='balanced'** or resampling techniques (SMOTE).
* **If high skewness:**
    * Apply **Log transformation** or PowerTransform to numerical inputs.
* **If high-cardinality:**
    * Use **Target Encoding** or limit One-Hot strategy.
    * Prefer tree-based models (Random Forest/XGBoost) which handle categorical splits better.
* **If strong correlation (>0.9):**
    * Remove redundant features (e.g., drop `education`, keep `educational-num`).

**Step 3 — Reflection Triggers**
* **If F1 < threshold:** Re-plan with different model architectures or ensemble methods.
* **If overfitting detected:** Increase regularization or reduce model complexity.
* **If fairness gap detected:** Implement adversarial debiasing or remove sensitive attributes.

---

### 4. Agent Behaviour Summary
The agent:
1. **Profiles** the dataset to understand its unique constraints.
2. **Extracts** structural signals to define the preprocessing path.
3. **Selects** models dynamically based on cardinality and skewness.
4. **Reflects** on performance to iteratively improve the strategy.

This adaptive design ensures robust performance across diverse data distributions.
