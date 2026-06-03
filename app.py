from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("gaming_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = [
        float(request.form["age"]),
        float(request.form["gender"]),
        float(request.form["gaming_hours"]),
        float(request.form["study_hours"]),
        float(request.form["sleep_hours"]),
        float(request.form["attendance"]),
        float(request.form["gaming_genre"]),
        float(request.form["social_activity"]),
        float(request.form["device_usage"]),
        float(request.form["reaction_time_ms"]),
        float(request.form["addiction_score"]),
        float(request.form["stress_level"])
    ]

    prediction = model.predict([data])

    return f"Predicted Academic Performance: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)
