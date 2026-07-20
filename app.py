import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained SVC model
MODEL_PATH = "Passed_or_Failed.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Ensure it is uploaded to your repository.")

@app.route("/", methods=["GET"])
def home():
    # This loads your web interface
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", prediction_text="Error: Model not loaded properly on the server.")

    try:
        # Extract features from the HTML form and convert to floats
        features = [
            float(request.form["gender"]),
            float(request.form["age"]),
            float(request.form["study_hours_per_week"]),
            float(request.form["attendance_rate"]),
            float(request.form["parent_education"]),
            float(request.form["internet_access"]),
            float(request.form["extracurricular"]),
            float(request.form["previous_score"]),
            float(request.form["final_score"])
        ]
        
        # Convert to a 2D array for scikit-learn prediction
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Determine the text to show based on the model's output (assuming 1=Pass, 0=Fail)
        result_text = "Pass" if prediction[0] == 1 else "Fail"
        
        return render_template("index.html", prediction_text=f"Prediction: The student will {result_text}!")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing input: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
