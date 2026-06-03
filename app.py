from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Safe model loading (deployment friendly)
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "gaming_model.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ 13 features REQUIRED (1 dummy added)
        data = np.array([[
            float(request.form.get("age", 0)),
            float(request.form.get("gender", 0)),
            float(request.form.get("gaming_hours", 0)),
            float(request.form.get("study_hours", 0)),
            float(request.form.get("sleep_hours", 0)),
            float(request.form.get("attendance", 0)),
            float(request.form.get("gaming_genre", 0)),
            float(request.form.get("social_activity", 0)),
            float(request.form.get("device_usage", 0)),
            float(request.form.get("reaction_time_ms", 0)),
            float(request.form.get("addiction_score", 0)),
            float(request.form.get("stress_level", 0)),

            # 🔥 MISSING FEATURE FIX (dummy value)
            0
        ]])

        prediction = model.predict(data)

        return f"Predicted Academic Performance: {prediction[0]}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
