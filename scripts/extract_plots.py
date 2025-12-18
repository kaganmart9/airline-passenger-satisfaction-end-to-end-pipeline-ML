
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Load data
print("Loading data...")
try:
    train_df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    # Try searching recursively if not found directly
    import glob
    files = glob.glob('**/train.csv', recursive=True)
    if files:
        train_df = pd.read_csv(files[0])
        print(f"Loaded from {files[0]}")
    else:
        raise FileNotFoundError("Could not find train.csv")

# Preprocessing
# Drop ID columns
cols_to_drop = ['Unnamed: 0', 'id']
train_df = train_df.drop([c for c in cols_to_drop if c in train_df.columns], axis=1)

# Encode target
# Assuming 'satisfaction' is the target column
if train_df['satisfaction'].dtype == 'object':
    y_train = train_df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
else:
    y_train = train_df['satisfaction']

X_train = train_df.drop('satisfaction', axis=1)

print("Data loaded and preprocessed.")

# --- Plot 1: Age vs Satisfaction and Flight Distance ---
print("Generating Plot 1...")
basic_df = pd.concat([X_train[["Age", "Flight Distance"]], y_train], axis=1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
ax1 = sns.boxplot(data=basic_df, x="satisfaction", y="Age")
ax1.set_title("Age vs Satisfaction")
ax1.set_xlabel("Satisfaction (0=Not satisfied, 1=Satisfied)")

plt.subplot(1, 2, 2)
ax2 = sns.boxplot(data=basic_df, x="satisfaction", y="Flight Distance")
ax2.set_title("Flight Distance vs Satisfaction")
ax2.set_xlabel("Satisfaction (0=Not satisfied, 1=Satisfied)")

plt.tight_layout()
plt.savefig('figures/satisfaction_age_distance.png')
print("Saved figures/satisfaction_age_distance.png")
plt.close()

# --- Plot 2: Feature Importance (Coefficients) ---
print("Training model for Plot 2...")

# Define columns
num_cols = ['Age', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service',
            'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Define Pipeline
num_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ]
)

baseline_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Fit model
baseline_model.fit(X_train, y_train)
print("Model trained.")

# Extract coefficients
# get feature names after preprocessing
# Note: sklearn version might affect get_feature_names_out usage.
# If older sklearn, might need workarounds. Assuming recent.

try:
    num_features = num_cols
    cat_features = (
        baseline_model.named_steps['preprocess']
        .named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(cat_cols)
    )
    all_features = np.concatenate([num_features, cat_features])
except AttributeError:
    # Fallback for older sklearn
    try:
        cat_features = (
            baseline_model.named_steps['preprocess']
            .named_transformers_['cat']
            .named_steps['encoder']
            .get_feature_names(cat_cols)
        )
        all_features = np.concatenate([num_features, cat_features])
    except:
        print("Could not get feature names. Skipping coefficient plot.")
        all_features = None

if all_features is not None:
    coefficients = baseline_model.named_steps['model'].coef_[0]

    coef_df = pd.DataFrame(
        {"feature": all_features, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)

    # Visualization
    plt.figure(figsize=(10, 8))

    # Plot top positive and negative
    top_n = 15
    top_pos = coef_df.head(top_n)
    top_neg = coef_df.tail(top_n)
    plot_df = pd.concat([top_pos, top_neg])

    sns.barplot(data=plot_df, y="feature", x="coefficient", palette="viridis")
    plt.title("Most Influential Logistic Regression Coefficients (Top 15 + / Top 15 -)")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png')
    print("Saved figures/feature_importance.png")
    plt.close()
