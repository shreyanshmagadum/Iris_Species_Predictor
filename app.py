import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("iris_dataset.csv")

# Load model
model = joblib.load("iris_model.pkl")

# Title
st.title("Iris Species Predictor")

# Show dataset
st.write("### Dataset Sample:")
st.dataframe(df.head())

# User inputs
st.sidebar.header("Enter Flower Features")
sepal_length = st.sidebar.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.sidebar.number_input("Petal Width", min_value=0.0, step=0.1)

# Prediction
if st.sidebar.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[int(prediction[0])]
    
    st.write(f"### Predicted Species: **{result}**")
