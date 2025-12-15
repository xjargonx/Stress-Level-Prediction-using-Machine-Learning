import gradio as gr
import joblib as jb
import numpy as np

# 1. Load the trained model
model = jb.load(r'C:\Users\DELL\Study\Semester 5\Machine Learning\Project\stress_model.pkl')

# 2. Define the Prediction Function
# This function takes inputs from the website, runs the model, and returns the result
def predict_stress(sleep_hours, activity_minutes, heart_rate, daily_steps):
    # Prepare the input array (must match the order of training features)
    features = np.array([[ sleep_hours, activity_minutes, heart_rate, daily_steps]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the result (rounded to 1 decimal place)
    return round(prediction[0], 1)

# 3. Create the Gradio Interface
# We map the function inputs to Sliders for a nice UI
interface = gr.Interface(
    fn=predict_stress,
    inputs=[
        gr.Number(minimum=0, maximum=12, label="Sleep Duration (Hours)"),
        gr.Number(minimum=0, maximum=120, label="Physical Activity (Minutes/Day)"),
        gr.Number(minimum=40, maximum=120, label="Heart Rate (bpm)"),
        gr.Number(minimum=0, maximum=30000, label="Daily Steps")
    ],
    outputs=gr.Number(label="Predicted Stress Level (1-10)"),
    title="Stress Level Predictor",
    description="Enter your daily health metrics to predict your stress level based on the Sleep Health Dataset."
)

# 4. Launch the App
if __name__ == "__main__":
    interface.launch(share=True)