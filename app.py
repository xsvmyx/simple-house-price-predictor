import streamlit as st
import joblib
import numpy as np


model = joblib.load("models/linear_model2.pkl")  

st.title("House price prediction")

square_meters = st.number_input("Surface (m²)", min_value=10.0, max_value=1000.0, step=1.0)
age = st.number_input("Age (years)", min_value=0, max_value=200, step=1)
distance = st.number_input("Distance to city (km)", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict the price"):
    
    X = np.array([[square_meters, age, distance]])
    
   
    y_pred = model.predict(X)[0]
    
    st.success(f"Price estimated : {y_pred:,.2f}")
