import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Any


def permutation_importance_rf(model, X, y, n_repeats=5):
    """
    Compute permutation importance for Random Forest.
    Importance is returned at ORIGINAL feature level.
    """

    perm: Any = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df
