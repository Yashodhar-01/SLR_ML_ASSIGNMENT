
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the available SLR model
if os.path.exists("SLR.pkl"):
    model_file = "SLR.pkl"
elif os.path.exists("SLR_MODEL_Assign.pkl"):
    model_file = "SLR_MODEL_Assign.pkl"
else:
    raise FileNotFoundError("SLR.pkl model file not found.")

with open(model_file, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        height = float(request.form["height"])

        if not np.isfinite(height) or height <= 0:
            return render_template(
                "index.html",
                prediction_text="Please enter a valid height"
            )

        # Predict weight using the trained SLR model
        input_data = np.array([[height]])
        prediction = model.predict(input_data)

        predicted_weight = round(float(prediction[0]), 2)

        return render_template(
            "index.html",
            prediction_text=predicted_weight
        )

    except (ValueError, TypeError, KeyError):
        return render_template(
            "index.html",
            prediction_text="Invalid height"
        )


if __name__ == "__main__":
    app.run(debug=True)


