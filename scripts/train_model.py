import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from airline_satisfaction.data_utils import load_data, basic_cleaning
from airline_satisfaction.preprocessing import build_preprocessor
from airline_satisfaction.modeling import build_random_forest


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TRAIN = PROJECT_ROOT / "data/raw/train.csv"
DATA_TEST = PROJECT_ROOT / "data/raw/test.csv"
MODEL_PATH = PROJECT_ROOT / "models/random_forest_final.pkl"


def main():
    # Load and clean data
    train_df, _ = load_data(DATA_TRAIN, DATA_TEST)
    train_df = basic_cleaning(train_df)

    X = train_df.drop(columns=["satisfaction"])
    y = train_df["satisfaction"]

    # Identify feature types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    preprocess = build_preprocessor(num_cols, cat_cols)
    model = build_random_forest(preprocess)

    # Train
    model.fit(X_train, y_train)

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
