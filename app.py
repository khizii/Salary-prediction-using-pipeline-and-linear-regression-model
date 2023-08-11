import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

# Load the trained pipeline
loaded_pipeline = joblib.load("C:\\Users\\USER\\Desktop\\salary\\trained_salary.pkl")

st.title("Salary Prediction App")

# Get user input
user_experience = st.number_input("Enter years of experience:", min_value=0.0, max_value=50.0)
user_education = st.selectbox("Select education level:", ["master", "bachelor", "phd"])
job_title = st.text_input("Enter job title:")

stemmed_job_title= ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(job_title)])
stemmed_user_education= ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(user_education)])
update_uder_education=stemmed_user_education.lower().rstrip("s")
update_job_title=stemmed_job_title.lower()

# Create a DataFrame for the new data
new_data = pd.DataFrame({
    'Years of Experience': [user_experience],
    'Education Level': [update_uder_education],
    'Job Title': [update_job_title],
})

# Make predictions using the loaded pipeline
if st.button("Predict Salary"):
    predicted_salary = loaded_pipeline.predict(new_data)
    st.success(f"Predicted Salary: {predicted_salary[0]:.2f}")
