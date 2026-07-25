"""
Model Training and Registration with Experimentation Tracking
----------------------------------------------------------------
- Loads the train/test splits produced by the previous job (data-prep).
- Builds a preprocessing + model pipeline for two candidate algorithms
  (Random Forest and XGBoost), each tuned with GridSearchCV.
- Logs every hyperparameter combination and metric to MLflow.
- Compares the two tuned models on the held-out test set and keeps the
  better one.
- Saves the best pipeline to tourism_project/deployment/ so the workflow
  can commit it back into the repository.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
import joblib
import os
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# MLflow tracking configuration
# ---------------------------------------------------------------------------
# The GitHub Actions job starts a local MLflow server (mlflow ui) on
# localhost:5000 before this script runs (see pipeline.yml).
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-package-prediction")

# ---------------------------------------------------------------------------
# 1. Load the train / test splits (downloaded from the workflow artifact)
# ---------------------------------------------------------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")

# ---------------------------------------------------------------------------
# 2. Preprocessing: scale numeric features, one-hot-encode categoricals
# ---------------------------------------------------------------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Handle class imbalance (~19% positive class)
neg, pos = ytrain.value_counts()[0], ytrain.value_counts()[1]
scale_pos_weight = neg / pos
print(f"scale_pos_weight (XGBoost): {scale_pos_weight:.3f}")

mlflow.sklearn.autolog(log_models=False, silent=True)

results = {}

# ---------------------------------------------------------------------------
# 3a. Candidate 1: Random Forest
# ---------------------------------------------------------------------------
rf_pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(class_weight="balanced", random_state=42),
)
rf_param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [4, 6, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

with mlflow.start_run(run_name="RandomForest_GridSearch"):
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid, scoring="f1", cv=5, n_jobs=-1
    )
    rf_grid.fit(Xtrain, ytrain)

    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(Xtest)
    rf_proba = rf_best.predict_proba(Xtest)[:, 1]

    rf_test_f1 = f1_score(ytest, rf_pred)
    rf_test_auc = roc_auc_score(ytest, rf_proba)

    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metric("cv_best_f1", rf_grid.best_score_)
    mlflow.log_metric("test_f1", rf_test_f1)
    mlflow.log_metric("test_roc_auc", rf_test_auc)

    print("\n=== Random Forest ===")
    print("Best params:", rf_grid.best_params_)
    print(f"CV best F1: {rf_grid.best_score_:.4f}")
    print(f"Test F1: {rf_test_f1:.4f} | Test ROC-AUC: {rf_test_auc:.4f}")
    print(classification_report(ytest, rf_pred))

    results["random_forest"] = {
        "pipeline": rf_best,
        "test_f1": rf_test_f1,
        "test_auc": rf_test_auc,
    }

# ---------------------------------------------------------------------------
# 3b. Candidate 2: XGBoost
# ---------------------------------------------------------------------------
xgb_pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    ),
)
xgb_param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
}

with mlflow.start_run(run_name="XGBoost_GridSearch"):
    xgb_grid = GridSearchCV(
        xgb_pipeline, xgb_param_grid, scoring="f1", cv=5, n_jobs=-1
    )
    xgb_grid.fit(Xtrain, ytrain)

    xgb_best = xgb_grid.best_estimator_
    xgb_pred = xgb_best.predict(Xtest)
    xgb_proba = xgb_best.predict_proba(Xtest)[:, 1]

    xgb_test_f1 = f1_score(ytest, xgb_pred)
    xgb_test_auc = roc_auc_score(ytest, xgb_proba)

    mlflow.log_params(xgb_grid.best_params_)
    mlflow.log_metric("cv_best_f1", xgb_grid.best_score_)
    mlflow.log_metric("test_f1", xgb_test_f1)
    mlflow.log_metric("test_roc_auc", xgb_test_auc)

    print("\n=== XGBoost ===")
    print("Best params:", xgb_grid.best_params_)
    print(f"CV best F1: {xgb_grid.best_score_:.4f}")
    print(f"Test F1: {xgb_test_f1:.4f} | Test ROC-AUC: {xgb_test_auc:.4f}")
    print(classification_report(ytest, xgb_pred))

    results["xgboost"] = {
        "pipeline": xgb_best,
        "test_f1": xgb_test_f1,
        "test_auc": xgb_test_auc,
    }

# ---------------------------------------------------------------------------
# 4. Pick the best model (by test F1-score) and save it for deployment
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["test_f1"])
best_pipeline = results[best_name]["pipeline"]

print(f"\nSelected best model: {best_name} "
      f"(test F1 = {results[best_name]['test_f1']:.4f}, "
      f"test ROC-AUC = {results[best_name]['test_auc']:.4f})")

with mlflow.start_run(run_name="Best_Model_Selection"):
    mlflow.log_param("selected_model", best_name)
    mlflow.log_metric("selected_test_f1", results[best_name]["test_f1"])
    mlflow.log_metric("selected_test_auc", results[best_name]["test_auc"])

os.makedirs("tourism_project/deployment", exist_ok=True)
MODEL_PATH = "tourism_project/deployment/best_tourism_model_v1.joblib"
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Best model saved to {MODEL_PATH}")
