# Bank Account Fraud Detection

A machine learning project to detect fraudulent bank account applications using the **Bank Account Fraud (BAF) Dataset – Base variant**.

## Overview

This project aims to identify fraudulent applications while limiting the number of legitimate applicants incorrectly flagged as fraud.

The primary evaluation objective is to maximize the **True Positive Rate (TPR)** while maintaining a **False Positive Rate (FPR) of no more than 5%**.

## Requirements

- Python 3.13 or later
- pip

The project dependencies are listed in `requirements.txt`.

## Dataset

- **Source:** Bank Account Fraud (BAF) Dataset
- **Variant:** Base
- **Records:** 547,975 applications
- **Period:** Month 0 to Month 3
- **Target:** `fraud_bool`
- **Fraud rate:** 0.97%

The dataset is highly imbalanced, with fraudulent applications representing a small proportion of the records.

## Methodology

The workflow includes:

1. Data inspection and quality checks.
2. Filtering records to months 0–3.
3. Exploratory data analysis and feature analysis.
4. Chronological data splitting:
   - Training: Month 0–1
   - Validation: Month 2
   - Test: Month 3
5. Feature preprocessing using `StandardScaler` for Logistic Regression and `OneHotEncoder` for categorical features.
6. Model training and evaluation using Logistic Regression, Random Forest, and XGBoost.
7. Hyperparameter tuning for Logistic Regression and XGBoost using a temporal validation split.
8. Threshold selection based on the TPR at a constrained FPR.

Random Forest was evaluated as a baseline model and was not hyperparameter-tuned.

## Model Results

The following results were obtained after selecting thresholds using the validation set and re-evaluating on Month 3.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.8926 | 0.9134 |
| PR-AUC | 0.1298 | 0.1601 |
| Precision | 10.10% | 10.59% |
| Recall (TPR) | 44.83% | 54.24% |
| F1-Score | 16.49% | 17.72% |
| FPR | 3.71% | 4.26% |
| Threshold | 0.4334 | 0.7485 |

**Selected model: XGBoost**

XGBoost achieved a higher TPR of 54.24% while keeping the FPR at 4.26%, meeting the maximum 5% FPR requirement.

**Evaluation note:** Month 3 was initially evaluated using thresholds selected for a 5% validation FPR. After the initial test results exceeded the requirement, the threshold strategy was revised using a more conservative 2% validation FPR target. Consequently, the Month 3 results above should be treated as a diagnostic re-evaluation rather than a fully independent final test.