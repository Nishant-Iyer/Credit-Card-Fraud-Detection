# Final Script for Credit Card Fraud Detection
# This script encapsulates the complete, advanced analysis workflow.

# --- 1. Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report, confusion_matrix

# --- 2. Data Loading and Exploration ---
print("--- Loading and Exploring Data ---")
df = pd.read_csv('creditcard.csv')
print("Class Distribution:\n", df['Class'].value_counts(normalize=True))

# --- 3. Feature Engineering ---
print("\n--- Engineering Cyclical Time Features ---")
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)
df['Time_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Time_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df.drop(['Time', 'Hour'], axis=1, inplace=True)

# --- 4. Data Preprocessing and Splitting ---
print("\n--- Preprocessing and Splitting Data ---")
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 5. Applying SMOTE for Class Imbalance ---
print("\n--- Applying SMOTE to Training Data ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Shape of training data after SMOTE:", X_train_smote.shape)

# --- 6. Model Training and Hyperparameter Tuning ---
print("\n--- Training and Tuning Models ---")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# -- XGBoost --
print("\nTraining XGBoost...")
# Using the best parameters found during the analysis
param_grid_xgb = {'learning_rate': [0.2], 'max_depth': [7], 'n_estimators': [200]}
xgb_model = XGBClassifier(random_state=42, eval_metric='aucpr', use_label_encoder=False)
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='average_precision', cv=cv, n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_train_smote, y_train_smote)
best_model_xgb = grid_search_xgb.best_estimator_

# -- LightGBM --
print("\nTraining LightGBM...")
# Using the best parameters found during the analysis
param_grid_lgb = {'learning_rate': [0.1], 'n_estimators': [300], 'num_leaves': [50]}
lgb_model = lgb.LGBMClassifier(random_state=42)
grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid_lgb, scoring='average_precision', cv=cv, n_jobs=-1, verbose=1)
grid_search_lgb.fit(X_train_smote, y_train_smote)
best_model_lgb = grid_search_lgb.best_estimator_

# --- 7. Final Model Evaluation ---
print("\n--- Final Model Evaluation on Test Set ---")

# -- XGBoost Evaluation --
print("\n--- XGBoost Results ---")
y_pred_xgb = best_model_xgb.predict(X_test)
y_pred_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
auprc_xgb = average_precision_score(y_test, y_pred_proba_xgb)
print(f"XGBoost AUPRC: {auprc_xgb:.4f}")
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# -- LightGBM Evaluation --
print("\n--- LightGBM Results ---")
y_pred_lgb = best_model_lgb.predict(X_test)
y_pred_proba_lgb = best_model_lgb.predict_proba(X_test)[:, 1]
auprc_lgb = average_precision_score(y_test, y_pred_proba_lgb)
print(f"LightGBM AUPRC: {auprc_lgb:.4f}")
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgb))

# --- 8. Visualization ---
print("\n--- Generating and Saving Visualizations ---")

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
axes[0].set_title('XGBoost Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
cm_lgb = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
axes[1].set_title('LightGBM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('final_confusion_matrices.png')
print("Saved final_confusion_matrices.png")

# Precision-Recall Curves
plt.figure(figsize=(10, 7))
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUPRC = {auprc_xgb:.4f})')
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_pred_proba_lgb)
plt.plot(recall_lgb, precision_lgb, label=f'LightGBM (AUPRC = {auprc_lgb:.4f})', linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.savefig('final_pr_curves.png')
print("Saved final_pr_curves.png")

print("\n--- Analysis Complete ---")
