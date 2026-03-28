import streamlit as st
import requests


st.title("Predict Titanic")

api_url = "http://127.0.0.1:8001/predict"

p_class = st.number_input('Class:', value=0)
gender = st.selectbox('Gender:', ['Male', 'Female'])
age = st.number_input('Age:', value=0)
sib_sp = st.number_input('SibSp:', value=0)
parch = st.number_input('Parch:', value=0)
fare = st.number_input('Fare:')
embarked = st.selectbox('Embarked:', ['C', 'Q', 'S'])

person = {
    'p_class':p_class,
    'gender':gender,
    'age':age,
    'sib_sp':sib_sp,
    'parch':parch,
    'fare':fare,
    'embarked':embarked,
}

if st.button('Predict'):

    try:
        answer = requests.post(api_url, json=person, timeout=10)
        if answer.status_code == 200:
            result = answer.json()
            st.success(f'Result: {result.get('result')}')
            #st.json()
        else:
            st.error(f'Error: {answer.status_code}')
    except requests.exceptions.RequestException:
        st.error(f'Failed to connect to API')