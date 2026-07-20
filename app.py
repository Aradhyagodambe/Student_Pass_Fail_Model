import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

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
    return jsonify({"message": "Flask ML API is running! Send a POST request to /predict."})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded properly on the server."}), 500

    try:
        data = request.get_json(force=True)
        
        # Expected feature order based on your model metadata:
        feature_order = [
            "gender", "age", "study_hours_per_week", "attendance_rate",
            "parent_education", "internet_access", "extracurricular",
            "previous_score", "final_score"
        ]
        
        # Extract features in the correct order
        features = [data[feat] for feat in feature_order]
        
        # Convert to a 2D array for scikit-learn prediction
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction[0])
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required feature field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render assigns a dynamic port via the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
