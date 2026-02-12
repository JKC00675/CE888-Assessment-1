# CE888-Assessment-1　Data Exploration & Planning Report
Dataset: Adult Income (UCI)

1. Key Findings from EDA

The Adult Income dataset is a binary classification problem aiming to predict whether an individual's annual income exceeds $50,000.

Dataset Characteristics

Rows: [填入]

Features: 14 predictors + 1 target

Mixed data types (categorical + numerical)

Presence of missing values

Class imbalance: [填比例]

Observations

The dataset is moderately imbalanced, with the majority class being ≤50K.

Capital-gain exhibits extreme right skewness.

Education and education-num show redundancy.

Occupation and workclass have high cardinality.

Income distribution differs significantly across gender groups.

2. Identified Challenges
1. Imbalance Risk

Accuracy alone may overestimate model performance.

2. Bias & Fairness Risk

Sensitive attributes such as sex and race may influence predictions.

3. Encoding Explosion

High-cardinality categorical variables may increase dimensionality.

4. Skewed Features

Highly skewed variables may distort linear models.

5. Multicollinearity

Redundant features may reduce model stability.

3. Proposed Agentic Planning Strategy

This agent will dynamically adapt based on dataset signals.

Step 1 — Signal Extraction

The agent extracts:

Dataset size

Class imbalance ratio

Missing value percentage

Cardinality of categorical features

Skewness of numerical features

Correlation strength

Step 2 — Adaptive Planning Rules
If imbalance > 60/40:

Use F1-score

Apply class_weight

Consider resampling

If high skewness:

Apply log transformation

If high-cardinality:

Limit encoding strategy

Prefer tree-based models

If strong correlation:

Remove redundant features

Step 3 — Reflection Triggers

After evaluation:

If F1 < threshold:

Re-plan with different models

If overfitting detected:

Reduce model complexity

If fairness gap detected:

Remove sensitive attributes

4. Agent Behaviour Summary

The agent does not follow a static pipeline.
Instead, it:

Profiles dataset

Extracts structural signals

Selects preprocessing dynamically

Chooses models based on dataset traits

Reflects and re-plans when performance is suboptimal

This adaptive design allows robust handling of diverse datasets.
