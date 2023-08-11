import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
stemmer = PorterStemmer()

# Load the trained pipeline
loaded_pipeline = joblib.load("C:\\Users\\USER\\Desktop\\salary\\trained_salary.pkl")

# Set the page title and icon
st.set_page_config(page_title="Salary Prediction App", page_icon="💰")

# Main title and description
st.title("Salary Prediction App")
st.write("Enter your information below to predict your potential salary.")

# User input section
st.sidebar.header("User Input")

# Get user input for years of experience
user_experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0)

# Education level options
education_options = ["master", "bachelor", "phd"]
user_education = st.sidebar.selectbox("Education Level", education_options)

# Job title input
job_title = st.sidebar.text_input("Job Title")

# Preprocess input
stemmed_job_title = ' '.join([stemmer.stem(word) for word in word_tokenize(job_title)])
stemmed_user_education = ' '.join([stemmer.stem(word) for word in word_tokenize(user_education)])
update_uder_education = stemmed_user_education.lower().rstrip("s")
update_job_title = stemmed_job_title.lower()

# Create a DataFrame for the new data
new_data = pd.DataFrame({
    'Years of Experience': [user_experience],
    'Education Level': [update_uder_education],
    'Job Title': [update_job_title],
})

# Predict and show result
if st.sidebar.button("Predict Salary"):
    predicted_salary = loaded_pipeline.predict(new_data)
    st.success(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Khizar Mehmood | Powered by NLTK and Streamlit")
