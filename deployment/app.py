import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os
import numpy as np

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Abhinavbhatnagar/mlops-tourism-project", filename="best_travel_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Travel Product Prediction App")
st.write("The Tourism product ")
st.write("Kindly enter the required detailsto check whether they will purchase it or not")


# Collect user input
Age = st.number_input("Age (Age of the customer)", min_value=18, max_value=65, value=25)
TypeofContact = st.selectbox("TypeofContact (How contact is made)", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("CityTier (City Type)", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("DurationOfPitch (For how long pitch is made)", value=12)
Occupation = st.selectbox("Occupation (Occupation of customer)",["Salaried", "Free Lancer","Small Business","Large Business"])
Gender = st.selectbox("Gender (Gender of customer)",["Male", "Female"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting  (total number of customer visitng)", min_value=1, max_value=5, value=2)
NumberOfFollowups = st.number_input("NumberOfFollowups  (total number of followups)", min_value=1, max_value=6, value=2)
ProductPitched = st.selectbox("ProductPitched (Category of product pitched)",["Basic", "Deluxe","Standard","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar  (Star of Property)", min_value=3, max_value=5, value=4)
MaritalStatus = st.selectbox("MaritalStatus (MaritalStatus of customer)",["Unmarried", "Divorced","Married"])
NumberOfTrips = st.number_input("NumberOfTrips (total number of trips", min_value=1, value=2)
Passport = st.selectbox("Passport (Passport yes or no. for Yes 1 for No its 0)",["Yes", "No"])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore  (Rating)", min_value=1, max_value=5, value=2)
OwnCar = st.selectbox("OwnCar (OwnCar yes or no. for Yes 1 for No its 0)",["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting  (No. of children travelling ", min_value=0, max_value=5, value=2)
Designation = st.selectbox("Designation (Designation in office)",["Executive", "Manager","Senior Manager","AVP","VP"])
MonthlyIncome = st.number_input("MonthlyIncome  (Monthly income of traveller )", min_value=1000, value=20237)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation':Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus':MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "travel" if prediction == 1 else "not travel"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
