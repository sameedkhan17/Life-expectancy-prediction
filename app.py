import streamlit as st
import pandas as pd
import pickle

# Title and description
st.title("Life Expectancy Predictor")
st.write(
    "This app predicts life expectancy based on multiple health and demographic indicators."
)

# Load the trained model
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model('linear_model.pkl')

# Sidebar inputs for features
def user_input_features():
    st.sidebar.header('Model Input Parameters')
    Adult_Mortality = st.sidebar.number_input(
        'Adult Mortality', min_value=0.0, value=172.0)
    infant_deaths = st.sidebar.number_input(
        'Infant Deaths', min_value=0.0, value=17.0)
    Alcohol = st.sidebar.number_input(
        'Alcohol (per capita)', min_value=0.0, value=7.2)
    percentage_expenditure = st.sidebar.number_input(
        'Percentage Expenditure', min_value=0.0, value=1150.0)
    BMI = st.sidebar.number_input(
        'BMI', min_value=0.0, value=44.1)
    under_five_deaths = st.sidebar.number_input(
        'Under-five Deaths', min_value=0.0, value=23.0)
    Polio = st.sidebar.number_input(
        'Polio Immunization (%)', min_value=0.0, max_value=100.0, value=81.0)
    Diphtheria = st.sidebar.number_input(
        'Diphtheria Immunization (%)', min_value=0.0, max_value=100.0, value=90.0)
    HIV_AIDS = st.sidebar.number_input(
        'HIV/AIDS Death Rate', min_value=0.0, value=1.06)
    GDP = st.sidebar.number_input(
        'GDP per Capita', min_value=0.0, value=9126.0)
    thinness_1_19 = st.sidebar.number_input(
        'Thinness 1-19 years', min_value=0.0, value=3.3)
    thinness_5_9 = st.sidebar.number_input(
        'Thinness 5-9 years', min_value=0.0, value=3.35)
    Income_composition = st.sidebar.number_input(
        'Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.64)
    Schooling = st.sidebar.number_input(
        'Schooling (years)', min_value=0.0, value=13.4)

    data = {
        'Adult Mortality': Adult_Mortality,
        'infant deaths': infant_deaths,
        'Alcohol': Alcohol,
        'percentage expenditure': percentage_expenditure,
        'BMI': BMI,
        'under-five deaths': under_five_deaths,
        'Polio': Polio,
        'Diphtheria': Diphtheria,
        'HIV/AIDS': HIV_AIDS,
        'GDP': GDP,
        'thinness  1-19 years': thinness_1_19,
        'thinness 5-9 years': thinness_5_9,
        'Income composition of resources': Income_composition,
        'Schooling': Schooling
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Prediction
if st.button('Predict Life Expectancy'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Life Expectancy')
    st.write(f"{prediction[0]:.2f} years")
