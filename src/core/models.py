"""Tasks, model factories and metrics (was pipeline/step5).

Every task is fit with four model families; the winner is the best on the
VALIDATION split (macro-F1 for the classifier, RMSE for the regressors).
The test split is never used to pick a model.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score,
                             roc_auc_score, root_mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

# sklearn 1.8+ warns about penalty="elasticnet"; the setting is still honoured.
warnings.filterwarnings("ignore", message=r".*penalty.*elasticnet.*", category=FutureWarning)

CLASS_TARGET = "risk_class"
COMPOSITE_TARGET = "health_impact_score"   # defines risk_class, so never a predictor
DISEASE_TARGETS = {
    "respiratory": "respiratory_disease_rate",
    "cardio": "cardio_mortality_rate",
    "vector": "vector_disease_risk_score",
    "waterborne": "waterborne_disease_incidents",
    "heat": "heat_related_admissions",
}
# (task, kind, target) in training order
TASKS = [("overall_classifier", "classification", CLASS_TARGET)] + \
        [(k, "regression", t) for k, t in DISEASE_TARGETS.items()]
ALL_TARGETS = [CLASS_TARGET, COMPOSITE_TARGET, *DISEASE_TARGETS.values()]

MODEL_LABELS = {"logreg": "Logistic Regression", "elasticnet": "ElasticNet", "dt": "Decision Tree",
                "rf": "Random Forest", "xgb": "XGBoost"}


def model_features(committee: list[str]) -> list[str]:
    """Committee minus every target and the date keys (same order as the committee)."""
    drop = set(ALL_TARGETS) | {"date", "year"}
    return [c for c in committee if c not in drop]


def classifiers(p: dict, seed: int, n_classes: int) -> dict:
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga", random_state=seed,
                                       **p["logreg"])),
        ]),
        "dt": DecisionTreeClassifier(random_state=seed, **p["dt"]),
        "rf": RandomForestClassifier(n_jobs=p["n_jobs"], random_state=seed, **p["rf"]),
        # XGBoost is single-threaded so runs are bit-reproducible.
        "xgb": XGBClassifier(objective="multi:softprob", num_class=n_classes, tree_method="hist",
                             n_jobs=1, random_state=seed, eval_metric="mlogloss", **p["xgb"]),
    }


def regressors(p: dict, seed: int) -> dict:
    return {
        "elasticnet": Pipeline([
            ("scale", StandardScaler()),
            ("reg", ElasticNet(random_state=seed, **p["elasticnet"])),
        ]),
        "dt": DecisionTreeRegressor(random_state=seed, **p["dt"]),
        "rf": RandomForestRegressor(n_jobs=p["n_jobs"], random_state=seed, **p["rf"]),
        "xgb": XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1,
                            random_state=seed, **p["xgb"]),
    }


class EncodedClassifier:
    """XGBoost needs integer labels; this wrapper keeps the string classes outside."""

    def __init__(self, model, classes: list[str]) -> None:
        self.model = model
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        lookup = {c: i for i, c in enumerate(self.classes_)}
        self.model.fit(X, np.array([lookup[v] for v in y]))
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.model.predict(X), dtype=int)]

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def classifier_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
    }


def regressor_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def score(kind: str, model, X, y) -> dict[str, float]:
    if kind == "classification":
        return classifier_metrics(y, model.predict(X), model.predict_proba(X))
    return regressor_metrics(y, model.predict(X))


def pick_winner(kind: str, val: dict[str, dict]) -> str:
    """Best on validation: highest macro-F1 (then accuracy) or lowest RMSE (then MAE)."""
    if kind == "classification":
        return sorted(val, key=lambda m: (-val[m]["macro_f1"], -val[m]["accuracy"]))[0]
    return sorted(val, key=lambda m: (val[m]["rmse"], val[m]["mae"]))[0]
