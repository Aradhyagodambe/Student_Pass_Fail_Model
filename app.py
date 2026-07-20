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
        # 1. Define mappings to convert words back to numbers
        # IMPORTANT: Ensure these numbers match how you originally trained your model!
        gender_map = {"Female": 0.0, "Male": 1.0}
        yes_no_map = {"No": 0.0, "Yes": 1.0}
        education_map = {
            "High School": 0.0,
            "Some College": 1.0,
            "Bachelors": 2.0,
            "Masters": 3.0,
            "PhD": 4.0
        }

        # 2. Extract and map string inputs from the dropdowns
        gender_val = gender_map[request.form["gender"]]
        internet_val = yes_no_map[request.form["internet_access"]]
        extra_val = yes_no_map[request.form["extracurricular"]]
        edu_val = education_map[request.form["parent_education"]]

        # 3. Extract the standard numerical inputs
        age_val = float(request.form["age"])
        study_hours_val = float(request.form["study_hours_per_week"])
        attendance_val = float(request.form["attendance_rate"])
        prev_score_val = float(request.form["previous_score"])
        final_score_val = float(request.form["final_score"])

        # 4. Combine into features list in the EXACT original order
        features = [
            gender_val,
            age_val,
            study_hours_val,
            attendance_val,
            edu_val,
            internet_val,
            extra_val,
            prev_score_val,
            final_score_val
        ]
        
        # Convert to a 2D array for scikit-learn prediction
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Determine the text to show based on the model's output (assuming 1=Pass, 0=Fail)
        result_text = "Pass" if prediction[0] == 1 else "Fail"
        
        return render_template("index.html", prediction_text=f"Prediction: The student will {result_text}!")

    except KeyError as e:
        # Catches errors if a dropdown was left blank
        return render_template("index.html", prediction_text=f"Error: Please ensure all dropdowns are selected. Missing: {str(e)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing input: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
