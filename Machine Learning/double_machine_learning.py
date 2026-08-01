"""
Machine Learning Homework 3: Double Machine Learning

Task:
    Estimate an average treatment effect with cross-fitting. The assignment
    connects prediction tools to causal inference by residualizing both the
    outcome and treatment with flexible nuisance models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


SEED = 2026


@dataclass(frozen=True)
class DMLResult:
    ate: float
    standard_error: float
    ci_low: float
    ci_high: float
    outcome_r2: float
    treatment_r2: float


def simulate_partially_linear_data(n: int = 1_200, p: int = 8) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(n, p))
    theta = 1.25
    treatment_signal = 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * np.sin(x[:, 2])
    treatment = treatment_signal + rng.normal(scale=1.0, size=n)
    outcome_signal = 1.5 * np.sin(x[:, 0]) + 0.7 * x[:, 3] ** 2 - 0.5 * x[:, 4]
    outcome = theta * treatment + outcome_signal + rng.normal(scale=1.0, size=n)
    features = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, p + 1)])
    return features, pd.Series(treatment, name="treatment"), pd.Series(outcome, name="outcome")


def cross_fitted_residuals(features: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> tuple[np.ndarray, float]:
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=8,
        random_state=SEED,
        n_jobs=None,
    )
    residuals = np.zeros(len(target))
    predictions = np.zeros(len(target))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, test_idx in cv.split(features):
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        fold_predictions = model.predict(features.iloc[test_idx])
        predictions[test_idx] = fold_predictions
        residuals[test_idx] = target.iloc[test_idx] - fold_predictions

    return residuals, r2_score(target, predictions)


def estimate_dml(features: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> DMLResult:
    outcome_residuals, outcome_r2 = cross_fitted_residuals(features, outcome)
    treatment_residuals, treatment_r2 = cross_fitted_residuals(features, treatment)

    final_stage = LinearRegression(fit_intercept=False)
    final_stage.fit(treatment_residuals.reshape(-1, 1), outcome_residuals)
    ate = float(final_stage.coef_[0])

    score = treatment_residuals * (outcome_residuals - ate * treatment_residuals)
    denominator = np.mean(treatment_residuals**2)
    standard_error = float(np.sqrt(np.mean(score**2) / len(score)) / denominator)
    ci_low = ate - 1.96 * standard_error
    ci_high = ate + 1.96 * standard_error

    return DMLResult(ate, standard_error, ci_low, ci_high, outcome_r2, treatment_r2)


def main() -> None:
    features, treatment, outcome = simulate_partially_linear_data()
    result = estimate_dml(features, treatment, outcome)

    print("\nDouble machine learning estimate")
    print(f"ATE estimate: {result.ate:.3f}")
    print(f"Standard error: {result.standard_error:.3f}")
    print(f"95% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}]")
    print("\nNuisance model diagnostics")
    print(f"Outcome model cross-fitted R^2: {result.outcome_r2:.3f}")
    print(f"Treatment model cross-fitted R^2: {result.treatment_r2:.3f}")


if __name__ == "__main__":
    main()
