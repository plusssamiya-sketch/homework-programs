"""
Machine Learning Homework 2: Classification Model Selection

Task:
    Compare logistic regression, random forest, and gradient boosting classifiers
    using stratified train/test splits and ROC-AUC based cross-validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 7


def make_dataset() -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(
        n_samples=900,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        class_sep=1.15,
        weights=[0.62, 0.38],
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame(x, columns=[f"feature_{i:02d}" for i in range(1, 13)]), pd.Series(y, name="label")


def main() -> None:
    features, target = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "LogisticRegression": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
                ]
            ),
            {"model__C": [0.1, 1.0, 5.0]},
        ),
        "RandomForest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=3,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            {"model__max_depth": [4, 6, None]},
        ),
        "GradientBoosting": (
            Pipeline([("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"model__learning_rate": [0.03, 0.07, 0.1], "model__max_depth": [2, 3]},
        ),
    }

    rows = []
    fitted = {}
    for name, (pipeline, params) in candidates.items():
        search = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv)
        search.fit(x_train, y_train)
        fitted[name] = search
        probabilities = search.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "cv_auc": search.best_score_,
                "test_auc": roc_auc_score(y_test, probabilities),
                "test_accuracy": accuracy_score(y_test, predictions),
                "best_params": search.best_params_,
            }
        )

    summary = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    winner = summary.iloc[0]["model"]
    winner_model = fitted[str(winner)]
    winner_predictions = winner_model.predict(x_test)

    print("\nModel selection summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.3f}"))
    print(f"\nSelected model: {winner}")
    print("\nConfusion matrix")
    print(confusion_matrix(y_test, winner_predictions))
    print("\nClassification report")
    print(classification_report(y_test, winner_predictions, digits=3))


if __name__ == "__main__":
    main()
