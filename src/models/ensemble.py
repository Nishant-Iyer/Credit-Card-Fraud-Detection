from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from joblib import parallel_backend
from typing import Dict, Any, List

class FraudStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Stacking Ensemble for Fraud Detection.
    Combines XGBoost and LightGBM using a LogisticRegression meta-classifier.
    """
    def __init__(
        self,
        xgb_params: Dict[str, Any],
        lgb_params: Dict[str, Any],
        meta_params: Dict[str, Any] = None,
        cv: int = 3
    ):
        self.xgb_params = xgb_params
        self.lgb_params = lgb_params
        self.meta_params = meta_params or {"class_weight": "balanced", "random_state": 42}
        self.cv = cv
        
        # Instantiate estimators
        self.xgb_clf = xgb.XGBClassifier(**self.xgb_params, random_state=42, eval_metric='aucpr')
        self.lgb_clf = lgb.LGBMClassifier(**self.lgb_params, random_state=42, verbose=-1)
        self.meta_clf = LogisticRegression(**self.meta_params)
        
        # Create stacking classifier
        self.stacking_clf = StackingClassifier(
            estimators=[
                ('xgb', self.xgb_clf),
                ('lgb', self.lgb_clf)
            ],
            final_estimator=self.meta_clf,
            cv=self.cv,
            n_jobs=1,
            passthrough=False # Set to True if we want the meta-classifier to also see raw features
        )
        self.classes_ = [0, 1]

    def fit(self, X, y):
        with parallel_backend('threading'):
            self.stacking_clf.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_clf.predict(X)

    def predict_proba(self, X):
        return self.stacking_clf.predict_proba(X)
