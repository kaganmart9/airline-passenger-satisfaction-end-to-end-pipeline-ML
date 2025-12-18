from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logistic_regression(preprocess):
    """
    Logistic Regression baseline model.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return model


def build_random_forest(preprocess, n_estimators=300):
    """
    Random Forest classifier.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    return model
