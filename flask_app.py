from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define the absolute path to the model file
# This prevents errors if you run the script from a different directory
MODEL_PATH = r"C:\Users\DELL\Study\Semester 5\Machine Learning\Project\Flask\stress_model.pkl"

# Load the trained model when the app starts
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print("ERROR: stress_model.pkl not found. Ensure it is in the same directory.")
    exit()

@app.route('/')
def home():
    # This route serves the initial input form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # This route processes the form submission
    try:
        # 1. Get and convert inputs from the HTML form (must match <input name="...">)
        sleep_hours = float(request.form['sleep_duration'])
        activity_minutes = float(request.form['activity_minutes'])
        heart_rate = float(request.form['heart_rate'])
        daily_steps = float(request.form['daily_steps'])

        # 2. Prepare features array (Order MUST match training: Sleep, Activity, HR, Steps)
        features = np.array([[sleep_hours, activity_minutes, heart_rate, daily_steps]])

        # 3. Make Prediction
        prediction = model.predict(features)[0]
        result = round(np.clip(prediction, 1.0, 10.0), 1) # Clip result to the 1-10 range

        # 4. Render the page again, passing the result to be displayed
        return render_template('index.html', 
                               prediction_text=f'Predicted Stress Level: {result} / 10',
                               input_data=request.form) # Pass inputs back to keep form filled

    except Exception as e:
        # Handle errors (e.g., user entered text instead of a number)
        error_message = f"Error processing input. Please check all values. ({e})"
        return render_template('index.html', prediction_text=error_message)

if __name__ == "__main__":
    # Flask runs on port 5000 by default
    app.run(debug=True)