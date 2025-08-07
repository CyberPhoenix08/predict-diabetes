import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()

# Features and labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient data to predict diabetes.")

# User input
preg = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 200, 120)
bp = st.slider("Blood Pressure", 0, 122, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 1, 100, 33)

input_data = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [glucose],
    "BloodPressure": [bp],
    "SkinThickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success("Diabetic" if prediction == 1 else "Not Diabetic")
