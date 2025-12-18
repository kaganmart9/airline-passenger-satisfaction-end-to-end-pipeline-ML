import joblib
from pathlib import Path

from airline_satisfaction.data_utils import load_single_csv, basic_cleaning
from airline_satisfaction.evaluation import evaluate_classification


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = PROJECT_ROOT / "data/raw/test.csv"
MODEL_PATH = PROJECT_ROOT / "models/random_forest_final.pkl"


def main():
    # Load model
    model = joblib.load(MODEL_PATH)

    # Load and clean test data
    test_df = load_single_csv(DATA_TEST)
    test_df = basic_cleaning(test_df)

    X_test = test_df.drop(columns=["satisfaction"])
    y_test = test_df["satisfaction"]

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate_classification(y_test, y_pred)

    print("Accuracy:", results["accuracy"])
    print(results["classification_report"])
    print("Confusion Matrix:\n", results["confusion_matrix"])


if __name__ == "__main__":
    main()
