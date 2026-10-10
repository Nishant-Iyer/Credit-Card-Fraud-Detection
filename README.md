# Credit Card Fraud Detection

A fraud model tuned for money, not accuracy. Only 0.17% of transactions are fraud, so instead of the default 0.5 cut-off, the pipeline picks the alert threshold that minimizes the real cost: analyst time spent reviewing alerts plus fraud that slips through.

**[Live demo](https://www.konnectingnots.com/demos/fraud.html)** · **[Case study](https://www.konnectingnots.com/projects/credit-card-fraud.html)**

## Results

On a held-out test set of 56,962 transactions (98 frauds), at a threshold of 0.232 chosen on a separate validation split:

| Metric | Value |
|---|---|
| AUPRC | 0.863 |
| ROC-AUC | 0.981 |
| Recall (fraud caught) | 84.7% (83 of 98) |
| Precision | 70.3% (35 false alarms) |

| Strategy | Total cost |
|---|---|
| Do nothing | $10,645 |
| Review every transaction | $284,810 |
| **Model** | **$2,666 (−75.0%)** |

Costs assume $5 per manual review and the full amount lost on a missed fraud (`config/config.yaml`). Figures come from `artifacts/metadata.json`.

## How it works

1. **Data:** the public European cardholder dataset (284,807 transactions, OpenML ID 1597), downloaded automatically.
2. **Features:** the 28 anonymized card features plus a robust-scaled amount. (The OpenML copy of the data has no time column.)
3. **Anomaly signal:** a PyTorch autoencoder trained on transactions; its reconstruction error is added as a feature.
4. **Imbalance:** SMOTE oversampling, applied during training only.
5. **Model:** stacking ensemble, with XGBoost and LightGBM feeding a logistic-regression meta-learner.
6. **Threshold:** a sweep of 200 thresholds on a validation split (20% of the training data), each priced with the business cost function. The test set is scored once, at the chosen threshold.
7. **Explanations:** SHAP attributions in the dashboard show why a transaction was flagged.

## Project layout

```
src/
  data/         download and split the dataset
  features/     robust scaler (and a cyclical time encoder for data that has a Time column)
  models/       autoencoder, stacking ensemble, trainer, cost-based evaluator
  pipeline/     training and inference pipelines
  api/          FastAPI service (/health, /predict, /predict/batch)
  spark/        PySpark batch scoring
dashboard.py    Streamlit dashboard
scripts/        export_demo.py: retrain and export the model as JSON for the browser demo
demo/           exported model and data used by the live demo
config/         model and business-cost settings
artifacts/      trained pipeline, metrics and charts
tests/          unit tests for transformers, pipeline and API
k8s/            Kubernetes deployment, service and autoscaler
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make train       # download data, train, write artifacts/
make test        # unit tests
make serve       # API on http://localhost:8000/docs
make dashboard   # Streamlit dashboard
make docker-up   # API + MLflow + dashboard together
```

The **Train and export demo model** GitHub Action retrains on every change to `src/` or `config/`, checks that the exported JSON model scores exactly like the trained pipeline, and commits the results to `demo/` and `artifacts/`.

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · imbalanced-learn · PyTorch · SHAP · FastAPI · Streamlit · MLflow · PySpark · Docker · Kubernetes

## License

MIT. See [LICENSE](LICENSE).
