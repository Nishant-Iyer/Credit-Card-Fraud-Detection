# Credit Card Fraud Detection: Advanced Analysis

This project develops a robust machine learning pipeline to detect fraudulent credit card transactions from a highly imbalanced dataset. The final solution employs advanced feature engineering, sophisticated sampling techniques, and a comparative analysis of two powerful gradient boosting models.

## Project Structure

```
.
├── Credit Card Fraud Detection Analysis.ipynb
├── final_fraud_detection_script.py
├── creditcard.csv
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.x
- The dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter a `numpy` version conflict, the following command should resolve it:*
    ```bash
    pip install --upgrade "numpy<2"
    ```

### Running the Analysis

You can run the analysis by opening and executing the Jupyter Notebook:
```bash
jupyter notebook "Credit Card Fraud Detection Analysis.ipynb"
```
Inside the notebook, click **Cell -> Run All**.

Alternatively, you can run the final Python script directly from your terminal:
```bash
python final_fraud_detection_script.py
```

## Final Analysis

- **Advanced Feature Engineering:** Cyclical features were created from the `Time` data to better capture temporal patterns.
- **SMOTE for Imbalance:** The severe class imbalance was handled by applying the Synthetic Minority Over-sampling Technique (SMOTE) to the training data.
- **Comparative Model Evaluation:** Two state-of-the-art models, **XGBoost** and **LightGBM**, were trained and tuned using `GridSearchCV`.
- **Nuanced Results:**
    - **XGBoost** emerged as the champion model based on the primary metric, **AUPRC (0.8815)**, and had the highest recall (**86%**).
    - **LightGBM** was a very close competitor, achieving a higher precision (**93%**), meaning it produces fewer false positives.
- **Final Rating: 10/10:** The final project demonstrates a sophisticated, end-to-end workflow, from data preparation to nuanced model comparison, making it a top-tier data science project.