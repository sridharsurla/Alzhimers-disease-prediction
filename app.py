#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Define the Streamlit app
st.title("Cognitive Assessment and Memory Prediction")

# Input form for user data
st.header("Enter Cognitive Assessment Details:")

# Input fields for all 18 features
memory_score = st.number_input("Memory Score", min_value=0.0, value=0.0)
cognitive_memory_assessment = st.number_input("Cognitive Memory Assessment", min_value=0.0, value=0.0)
cognitive_judgment_assessment = st.number_input("Cognitive Judgment Assessment", min_value=0.0, value=0.0)
stroke_to_decision_ratio = st.number_input("Stroke to Decision Ratio", min_value=0.0, value=0.0)
neuropsychological_genetics = st.number_input("Neuropsychological Genetics", min_value=0.0, value=0.0)
genetics_to_age_ratio = st.number_input("Genetics to Age Ratio", min_value=0.0, value=0.0)
cognitive_orientation_assessment = st.number_input("Cognitive Orientation Assessment", min_value=0.0, value=0.0)
family_history_decline = st.selectbox("Family History of Cognitive Decline", [0, 1])  # 0: No, 1: Yes
decision_decline = st.number_input("Decision-Making Cognitive Decline", min_value=0.0, value=0.0)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
biomarker_to_age_ratio = st.number_input("Biomarker to Age Ratio", min_value=0.0, value=0.0)
decline_due_to_stroke = st.number_input("Cognitive Decline due to Stroke", min_value=0.0, value=0.0)
other_biomarkers = st.number_input("Other Biomarkers", min_value=0.0, value=0.0)
years_of_education = st.number_input("Years of Education", min_value=0.0, value=0.0)
education_to_age_ratio = st.number_input("Education to Age Ratio", min_value=0.0, value=0.0)
birth_year = st.number_input("Birth Year", min_value=1900, max_value=2023, value=1980)
cardiovascular_to_age_ratio = st.number_input("Cardiovascular to Age Ratio", min_value=0.0, value=0.0)
decline_due_to_cardio = st.number_input("Cognitive Decline due to Cardiovascular Issues", min_value=0.0, value=0.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Memory Score': [memory_score],
    'Cognitive Memory Assessment': [cognitive_memory_assessment],
    'Cognitive Judgment Assessment': [cognitive_judgment_assessment],
    'Stroke_to_Decision_Ratio': [stroke_to_decision_ratio],
    'Neuropsychological Genetics': [neuropsychological_genetics],
    'Genetics_to_Age_Ratio': [genetics_to_age_ratio],
    'Cognitive Orientation Assessment': [cognitive_orientation_assessment],
    'Family History of Cognitive Decline': [family_history_decline],
    'Decision-Making Cognitive Decline': [decision_decline],
    'Sex': [sex],
    'Biomarker_to_Age_Ratio': [biomarker_to_age_ratio],
    'Cognitive Decline due to Stroke': [decline_due_to_stroke],
    'Other Biomarkers': [other_biomarkers],
    'Years of Education': [years_of_education],
    'Education_to_Age_Ratio': [education_to_age_ratio],
    'Birth Year': [birth_year],
    'Cardiovascular_to_Age_Ratio': [cardiovascular_to_age_ratio],
    'Cognitive Decline due to Cardiovascular Issues': [decline_due_to_cardio]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.error(f"The individual is at risk with a probability of {prob[1]:.2f}")
    else:
        st.success(f"The individual is not at risk with a probability of {prob[0]:.2f}")

