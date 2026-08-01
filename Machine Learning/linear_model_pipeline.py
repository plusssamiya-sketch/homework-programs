"""
Machine Learning Homework 1: Regularized Regression Pipeline

Task:
    Predict a continuous target from mixed numerical features, compare OLS-style
    linear regression with Ridge and Lasso, and summarize diagnostics.

The script uses a built-in synthetic dataset so it can run in a clean repository.
Replace `make_regression_dataset` with a CSV loader for a real assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class RegressionReport:
    model_name: str
    rmse: float
    mae: float
    r2: float
    cv_rmse: float


def make_regression_dataset(n_samples: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    """Create a realistic tabular regression dataset with a categorical feature."""
    x, y = make_regression(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        noise=18.0,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, 7)])
    df["region"] = pd.qcut(df["x1"], q=3, labels=["low", "middle", "high"])
    df["experience_level"] = pd.cut(
        df["x2"],
        bins=[-np.inf, -0.5, 0.75, np.inf],
        labels=["junior", "mid", "senior"],
    )
    target = pd.Series(y + 8 * df["x3"] ** 2, name="target")
    return df, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def evaluate_model(name: str, estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, cv_rmse: float) -> RegressionReport:
    predictions = estimator.predict(x_test)
    return RegressionReport(
        model_name=name,
        rmse=float(mean_squared_error(y_test, predictions, squared=False)),
        mae=float(mean_absolute_error(y_test, predictions)),
        r2=float(r2_score(y_test, predictions)),
        cv_rmse=float(cv_rmse),
    )


def main() -> None:
    features, target = make_regression_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(features)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(random_state=RANDOM_STATE), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}),
        "Lasso": (Lasso(max_iter=10_000, random_state=RANDOM_STATE), {"model__alpha": [0.001, 0.01, 0.1, 1.0]}),
    }

    reports: list[RegressionReport] = []
    fitted_models: dict[str, GridSearchCV] = {}

    for name, (model, parameter_grid) in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=None,
        )
        search.fit(x_train, y_train)
        fitted_models[name] = search
        reports.append(evaluate_model(name, search.best_estimator_, x_test, y_test, -search.best_score_))

    result_table = pd.DataFrame([report.__dict__ for report in reports]).sort_values("rmse")
    best_name = result_table.iloc[0]["model_name"]
    best_estimator = fitted_models[str(best_name)].best_estimator_
    residuals = y_test - best_estimator.predict(x_test)

    print("\nModel comparison")
    print(result_table.to_string(index=False, float_format=lambda value: f"{value:0.3f}"))
    print(f"\nBest model: {best_name}")
    print("Residual diagnostics")
    print(pd.Series(residuals).describe().to_string(float_format=lambda value: f"{value:0.3f}"))


if __name__ == "__main__":
    main()
