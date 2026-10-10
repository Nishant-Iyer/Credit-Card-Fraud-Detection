# Credit Card Fraud Detection

A fraud model tuned for money, not accuracy. Only 0.17% of transactions are fraud, so instead of the default 0.5 cut-off, the pipeline picks the alert threshold that minimizes the real cost: analyst time spent reviewing alerts plus fraud that slips through.

**[Live demo](https://credit-card-fraud-detection-rvtygaztzdva8ecohigihs.streamlit.app)** · **[Case study](https://www.konnectingnots.com/projects/credit-card-fraud.html)**

## Results

On a held-out set of 56,962 transactions (98 frauds), at the cost-optimal threshold of 0.116:

| Metric | Value |
|---|---|
| AUPRC | 0.858 |
| ROC-AUC | 0.977 |
| Recall (fraud caught) | 87.8% (86 of 98) |
| Precision | 66.2% (44 false alarms) |

| Strategy | Total cost |
|---|---|
| Do nothing | $10,645 |
| Review every transaction | $284,810 |
| **Model at optimal threshold** | **$2,574 (−75.8%)** |

Costs assume $5 per manual review and the full amount lost on a missed fraud (`config/config.yaml`). Figures come from `artifacts/metadata.json`.

> The threshold is currently chosen on the same held-out set used to report results, so the savings figure is somewhat optimistic. Choosing it on a separate validation split is the next fix.

## How it works

1. **Data:** the public European cardholder dataset (284,807 transactions, OpenML ID 1597), downloaded automatically.
2. **Features:** time of day encoded as a cycle, and amounts robust-scaled.
3. **Anomaly signal:** a PyTorch autoencoder trained on transactions; its reconstruction error is added as a feature.
4. **Imbalance:** SMOTE oversampling, applied during training only.
5. **Model:** stacking ensemble, with XGBoost and LightGBM feeding a logistic-regression meta-learner.
6. **Threshold:** a sweep of 200 thresholds, each priced with the business cost function.
7. **Explanations:** SHAP attributions in the dashboard show why a transaction was flagged.

## Project layout

```
src/
  data/         download and split the dataset
  features/     cyclical time encoder, robust scaler
  models/       autoencoder, stacking ensemble, trainer, cost-based evaluator
  pipeline/     training and inference pipelines
  api/          FastAPI service (/health, /predict, /predict/batch)
  spark/        PySpark batch scoring
dashboard.py    Streamlit dashboard
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

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · imbalanced-learn · PyTorch · SHAP · FastAPI · Streamlit · MLflow · PySpark · Docker · Kubernetes

## License

MIT. See [LICENSE](LICENSE).
