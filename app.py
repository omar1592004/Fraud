import streamlit as st
import numpy as np
import joblib

model = joblib.load(open("ModuleFraud.pkl", 'rb'))

st.title("Fraud Credit")

amount = int(st.number_input(label="Enter the number of Pregnancies if you are male type 0"))

oldbalanceOrig = int(st.number_input(label="Enter your Glucose level"))

newbalanceOrig = float(st.number_input(label="Enter your BloodPressure"))

ratio_to_median_purchase_price = float(st.number_input(label="Enter your SkinThickness"))

used_chip = float(st.number_input(label="Enter your Insulin level"))

used_pin_number = float(st.number_input(label="Enter your BMI"))

##online_order = float(st.number_input(label="Enter your Diabetes Pedigree Function "))

distance_from_home = int(st.number_input(label="Enter the distance_from_home "))

distance_from_last_transaction = int(st.number_input(label="Enter the distance from last transaction"))

fraud = int(st.number_input(label="Enter the fraud 0 or 1"))

data_array = np.array([[amount, oldbalanceOrig, newbalanceOrig, ratio_to_median_purchase_price, used_chip, used_pin_number, distance_from_home,distance_from_last_transaction, fraud]])

predict = st.button("Predict")

if (predict):
    
    y_hat = model.predict(data_array)
    
    if y_hat == 1:
        y_hat = "You maybe get diabetes"
    else:
        y_hat = "You will not get diabetes"
    
    st.success(y_hat)
    
# streamlit run app.py
