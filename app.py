import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Streamlit app title
st.title("Iris Flower Prediction App 🌸")

st.write("Enter the measurements of the iris flower below to predict its species.")

# Input fields for user
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert prediction number to flower name
    species_dict = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_dict[prediction]

    # Display prediction
    st.success(f"The predicted species is: **{predicted_species}** ")
