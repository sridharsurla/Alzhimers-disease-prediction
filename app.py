#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Define the Streamlit app
st.title("Alzheimer's Early Prediction")

# Input form for user data
st.header("Enter Cognitive Assessment and Demographic Details:")

# Input fields for the required features
birth_year = st.number_input("Birth Year", min_value=1900, max_value=2023, value=1980)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
years_of_education = st.number_input("Years of Education", min_value=0, value=12)
cardio_decline = st.number_input("Cognitive Decline due to Cardiovascular Issues", min_value=-10, max_value=10, value=0)
stroke_decline = st.number_input("Cognitive Decline due to Stroke", min_value=-10, max_value=10, value=0)
decision_decline = st.number_input("Decision-Making Cognitive Decline", min_value=-10, max_value=10, value=0)
cognitive_memory_assessment = st.number_input("Cognitive Memory Assessment", min_value=0, max_value=10, value=0)
cognitive_orientation_assessment = st.number_input("Cognitive Orientation Assessment", min_value=0, max_value=10, value=0)
cognitive_judgment_assessment = st.number_input("Cognitive Judgment Assessment", min_value=0, max_value=10, value=0)
family_history = st.selectbox("Family History of Cognitive Decline", [0, 1])  # 0: No, 1: Yes
other_biomarkers = st.number_input("Other Biomarkers", min_value=0, value=0)
neuro_genetics = st.number_input("Neuropsychological Genetics", min_value=0.0, value=0.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Birth Year': [birth_year],
    'Sex': [sex],
    'Years of Education': [years_of_education],
    'Cognitive Decline due to Cardiovascular Issues': [cardio_decline],
    'Cognitive Decline due to Stroke': [stroke_decline],
    'Decision-Making Cognitive Decline': [decision_decline],
    'Cognitive Memory Assessment': [cognitive_memory_assessment],
    'Cognitive Orientation Assessment': [cognitive_orientation_assessment],
    'Cognitive Judgment Assessment': [cognitive_judgment_assessment],
    'Family History of Cognitive Decline': [family_history],
    'Other Biomarkers': [other_biomarkers],
    'Neuropsychological Genetics': [neuro_genetics]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.error(f"The individual is at risk with a probability of {prob[1]:.2f}")
    else:
        st.success(f"The individual is not at risk with a probability of {prob[0]:.2f}")

