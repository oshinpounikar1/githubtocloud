from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)

# =====================================================
# 2. LOAD THE TRAINED MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Run 'python train.py' first to create model.pkl."
    )

model = joblib.load(MODEL_PATH)

# =====================================================
# 3. HOME PAGE AND SALARY PREDICTION
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    years = ""

    if request.method == "POST":
        years = request.form.get("years", "").strip()

        try:
            if not years:
                raise ValueError("Please enter your years of experience.")

            experience = float(years)

            if not np.isfinite(experience):
                raise ValueError("Enter a valid finite number.")

            if experience < 0 or experience > 100:
                raise ValueError(
                    "Experience must be between 0 and 100 years."
                )

            # Model expects a 2D array with one feature.
            features = np.array([[experience]], dtype=float)

            predicted_salary = float(model.predict(features)[0])

            if not np.isfinite(predicted_salary):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            prediction = round(predicted_salary, 2)

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Salary prediction failed.")
            error = (
                "Prediction failed. Check that model.pkl was "
                "created by train.py using the same feature format."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        years=years
    )


# =====================================================
# 4. HEALTH-CHECK ENDPOINT
# =====================================================

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }, 200


# =====================================================
# 5. RUN THE FLASK APPLICATION
# =====================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

