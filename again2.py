import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load the dataset
df = pd.read_csv("salary_prediction_dataset.csv")


# ============================================================
# DATA VALIDATION AND LEAKAGE-PREVENTION BLOCK
# Place immediately after pd.read_csv()
# ============================================================

import numpy as np
import pandas as pd

# 1. Allow only the intended input and target columns
required_columns = ["Experience", "Salary"]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset is missing required columns.")

df = df[required_columns].copy()

# 2. Ensure numeric values
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Remove missing and infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns).copy()

# 4. Reject invalid values
if (df["Experience"] < 0).any():
    raise ValueError("Experience cannot be negative.")

if (df["Salary"] <= 0).any():
    raise ValueError("Salary must be positive.")

# 5. Check whether there is enough variation
if df["Experience"].nunique() < 2:
    raise ValueError("Experience needs at least two distinct values.")

if df["Salary"].nunique() < 2:
    raise ValueError("Salary needs at least two distinct values.")

# 6. Report possible duplicate records
print("Duplicate rows:", df.duplicated().sum())

# 7. Define X and y only after validation
X = df[["Experience"]].copy()
y = df["Salary"].copy()

print("Validation complete.")
print("Rows available:", len(df))
print("Input feature:", X.columns.tolist())
print("Target:", y.name)

# Continue with train_test_split() below this block.
# ============================================================


# 2. Select X and y
X = df[["Experience"]]
y = df["Salary"]

# 3. Split the dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Create the mean baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# 5. Train Simple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict test salaries
y_pred = model.predict(X_test)

# 7. Cross-validation on training data
cv_scores = cross_val_score(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

cv_mae = -cv_scores.mean()

# 8. Evaluate baseline and real model
def evaluate(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }

results = pd.DataFrame([
    evaluate("Mean Baseline", y_test, baseline_predictions),
    evaluate("Simple Linear Regression", y_test, y_pred)
])

print("\nModel Comparison:")
print(results.round(2).to_string(index=False))

print("\n5-Fold Cross-Validation MAE:", round(cv_mae, 2))

# 9. Display learned equation
print("\nIntercept:", round(model.intercept_, 2))
print("Experience coefficient:", round(model.coef_[0], 2))
print(
    f"Salary = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} * Experience)"
)

# 10. Compare actual and predicted salaries
comparison = pd.DataFrame({
    "Experience": X_test["Experience"].to_numpy(),
    "Actual_Salary": y_test.to_numpy(),
    "Predicted_Salary": y_pred
})

comparison["Absolute_Error"] = (
    comparison["Actual_Salary"] - comparison["Predicted_Salary"]
).abs()


print("\nActual vs Predicted Salaries:")
print(comparison.head(15).round(2).to_string(index=False))

# 11. Save the trained model
joblib.dump(model,r"C:\Users\HP\Downloads\model.pkl")
print("\nModel saved as model.pkl")


