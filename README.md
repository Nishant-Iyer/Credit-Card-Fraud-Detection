# Credit Card Fraud Detection

A lightweight yet extensible machine learning project for detecting fraudulent
credit card transactions. It ships with a command line interface for training
and evaluating a logistic regression baseline on either the public Kaggle
[credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
or on quickly generated synthetic data.

## Features
- Reproducible training pipeline with standard scaling and class imbalance
  handling
- ROC AUC and classification report metrics
- Optional evaluation script for existing models
- Basic unit tests and GitHub Actions workflow for continuous integration

## Installation
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Training
Download `creditcard.csv` from Kaggle and provide the path when training:
```bash
python src/train_model.py --data /path/to/creditcard.csv
```
If no data path is given, the script generates a synthetic dataset:
```bash
python src/train_model.py
```
The script prints evaluation metrics and saves the model to `model.joblib` by
default. Additional options:
```bash
python src/train_model.py --help
```

## Evaluation
To evaluate a previously trained model on a dataset:
```bash
python src/evaluate_model.py --data /path/to/creditcard.csv --model model.joblib
```

## Tests
Run the unit tests with:
```bash
pytest
```

## License
This project is licensed under the terms of the MIT license. See
[LICENSE](LICENSE) for details.
