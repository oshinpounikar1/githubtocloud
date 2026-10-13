from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "salary_prediction_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


# ============================================================
# 2. Load and validate the dataset
# ============================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = ["Experience", "Salary"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        "Dataset must contain these columns: Experience, Salary"
    )

# Keep only the intended feature and target columns.
df = df[required_columns].copy()

# Convert values to numeric; invalid values become NaN.
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Replace infinite values and remove incomplete rows.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns).copy()

# Validate the data.
if (df["Experience"] < 0).any():
    raise ValueError("Experience cannot be negative.")

if (df["Salary"] <= 0).any():
    raise ValueError("Salary must be positive.")

if df["Experience"].nunique() < 2:
    raise ValueError(
        "Experience must contain at least two distinct values."
    )

if df["Salary"].nunique() < 2:
    raise ValueError(
        "Salary must contain at least two distinct values."
    )

print("Duplicate rows:", int(df.duplicated().sum()))

# Remove exact duplicate records.
df = df.drop_duplicates().reset_index(drop=True)

# X is the input feature; y is the prediction target.
X = df[["Experience"]].copy()
y = df["Salary"].copy()

# This script uses a 5-fold cross-validation evaluation.
# Ensure enough records remain for the split and validation.
if len(df) < 7:
    raise ValueError(
        "At least 7 valid, unique rows are required "
        "for this train/test split and 5-fold evaluation."
    )

print("\nValidation complete.")
print("Rows available:", len(df))
print("Input feature:", X.columns.tolist())
print("Target:", y.name)


# ============================================================
# 3. Split the dataset: 80% training, 20% testing
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 4. Create a mean baseline
# ============================================================

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)


# ============================================================
# 5. Train Simple Linear Regression
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)


# ============================================================
# 6. Predict test salaries
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. Cross-validation on training data
# ============================================================

cv_scores = cross_val_score(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
)

cv_mae = -cv_scores.mean()


# ============================================================
# 8. Evaluate the baseline and trained model
# ============================================================

def evaluate(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


results = pd.DataFrame([
    evaluate(
        "Mean Baseline",
        y_test,
        baseline_predictions,
    ),
    evaluate(
        "Simple Linear Regression",
        y_test,
        y_pred,
    ),
])

print("\nModel Comparison:")
print(results.round(2).to_string(index=False))

print(
    "\n5-Fold Cross-Validation MAE:",
    round(cv_mae, 2),
)


# ============================================================
# 9. Display the learned equation
# ============================================================

print("\nIntercept:", round(model.intercept_, 2))
print(
    "Experience coefficient:",
    round(model.coef_[0], 2),
)

print(
    f"Salary = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} * Experience)"
)


# ============================================================
# 10. Compare actual and predicted salaries
# ============================================================

comparison = pd.DataFrame({
    "Experience": X_test["Experience"].to_numpy(),
    "Actual_Salary": y_test.to_numpy(),
    "Predicted_Salary": y_pred,
})

comparison["Absolute_Error"] = (
    comparison["Actual_Salary"]
    - comparison["Predicted_Salary"]
).abs()

print("\nActual vs Predicted Salaries:")
print(comparison.head(15).round(2).to_string(index=False))


# ============================================================
# 11. Save the trained model for Flask
# ============================================================

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
print("Training completed.")

