# Credit Card Fraud Detection

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset, sourced from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), contains 284,807 transactions from European cardholders in September 2013, with only 492 labeled as fraudulent (0.172%). The features include 28 PCA-transformed variables, `Time`, and `Amount`, with the target variable `Class` indicating fraud (1) or legitimate (0). The dataset’s severe class imbalance is a central challenge in this analysis.

## Objective
The goal is to build a machine learning model that effectively identifies fraudulent transactions despite the imbalanced dataset. We prioritize the **Area Under the Precision-Recall Curve (AUPRC)** as the key evaluation metric to balance precision and recall.

## Methodology
- **Data Exploration**: Loaded the dataset, visualized features like `Amount` and `Time`, and confirmed no missing values.
- **Data Preprocessing**: Scaled `Time` and `Amount`, and split the data into training and testing sets with stratification to preserve the class distribution.
- **Model Training**: Tested models such as Logistic Regression, Random Forest, and XGBoost, adjusting for class imbalance.
- **Model Evaluation**: Assessed performance using AUPRC and selected the best-performing model.

## Key Findings
- The top model (e.g., Random Forest) demonstrated a strong AUPRC, indicating robust fraud detection capabilities.
- Feature importance analysis revealed that certain PCA features and `Amount` were significant predictors of fraud.

## How to Run the Code
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
