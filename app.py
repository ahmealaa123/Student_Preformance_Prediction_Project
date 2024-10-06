import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import joblib
import gradio as gr

# Load the data
df = pd.read_csv('Student Performance Prediction-B.csv')

# Convert X and Y to numpy arrays
X = df.drop(columns=['Student ID','Class']).to_numpy()
Y = df['Class'].to_numpy()

# Create an imputer to replace NaN values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the data and transform it
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# Save the trained model
model_file = 'KNeighborsClassifier.pkl'
joblib.dump(model, model_file)
print(f"model saved as {model_file}.")

# Create a function to make predictions
def predict(Quiz01, Assignment01, Midterm_Exam, Assignment02, Assignment03, Final_Exam, Total, Course_Grade):
    input_values = np.array([[Quiz01, Assignment01, Midterm_Exam, Assignment02, Assignment03, Final_Exam, Total, Course_Grade]])
    input_values = imputer.transform(input_values)
    prediction = model.predict(input_values)
    return prediction[0]

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 100, label="Quiz01"),
        gr.Slider(0, 100, label="Assignment01"),
        gr.Slider(0, 100, label="Midterm Exam"),
        gr.Slider(0, 100, label="Assignment02"),
        gr.Slider(0, 100, label="Assignment03"),
        gr.Slider(0, 100, label="Final Exam"),
        gr.Slider(0, 100, label="Total"),
        gr.Slider(0, 100, label="Course Grade")
    ],
    outputs=gr.Label(label="Predicted Class"),
    title="Student Performance Prediction",
    description="Enter the student's scores to predict their class."
)

# Launch the Gradio interface
demo.launch()