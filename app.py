import streamlit as st 
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit_option_menu as som

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'Sym_1': 0, 'Sym_2': 0, 'Sym_3': 0, 'Sym_4': 0, 'Sym_5': 0, 'Sym_6': 0,
        'Sym_7': 0, 'Sym_8': 0, 'Sym_9': 0, 'Sym_10': 0, 'Sym_11': 0, 'Sym_12': 0,
        'Sym_13': 0, 'Sym_14': 0, 'Sym_15': 0, 'Sym_16': 0, 'Sym_17': 0, 'Sym_18': 0,
        'Sym_19': 0, 'Sym_20': 0, 'Sym_21': 0
    }

if 'model' not in st.session_state:
    st.session_state.model = jb.load('models/model_DT.joblib')

# Sidebar menu
with st.sidebar:
    menu_option = ['Prediction', 'Select Model', 'Train Model']
    selected_option = som.option_menu('Diabetes Prediction System Based on Your Lifestyle', options=menu_option, icons=['hospital', 'database-fill-add', 'train-front'], menu_icon='bandaid')

# Prediction page
if selected_option == 'Prediction':
    st.header('Diabetes Prediction System Based on Your Lifestyle')
    
    with st.form(key='prediction_form'):
        st.subheader('Health Information')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.session_state.inputs['Sym_1'] = st.radio('High Blood Pressure (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_1'])
            st.session_state.inputs['Sym_2'] = st.radio('High Cholesterol (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_2'])
            st.session_state.inputs['Sym_3'] = st.radio('Cholesterol Check in 5 Years (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_3'])
            st.session_state.inputs['Sym_4'] = st.number_input('BMI Score:', value=st.session_state.inputs['Sym_4'])
            st.session_state.inputs['Sym_5'] = st.radio('Smoker (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_5'])
            st.session_state.inputs['Sym_6'] = st.radio('Stroke (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_6'])

        with col2:
            st.session_state.inputs['Sym_7'] = st.radio('Heart Disease or Attack (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_7'])
            st.session_state.inputs['Sym_8'] = st.radio('Physical Activity in the past 30 days (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_8'])
            st.session_state.inputs['Sym_9'] = st.radio('Fruits Consumption (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_9'])
            st.session_state.inputs['Sym_10'] = st.radio('Veggies Consumption (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_10'])
            st.session_state.inputs['Sym_11'] = st.radio('Heavy Alcohol Consumption (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_11'])
            st.session_state.inputs['Sym_12'] = st.radio('Health Care Coverage (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_12'])

        with col3:
            st.session_state.inputs['Sym_13'] = st.radio('No Doctor because of cost in the past 12 months (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_13'])
            st.session_state.inputs['Sym_14'] = st.number_input('General Health Score: scale of 1-5', value=st.session_state.inputs['Sym_14'])
            st.session_state.inputs['Sym_15'] = st.number_input('Mental Health Check: Days not good in the past 30 days', value=st.session_state.inputs['Sym_15'])
            st.session_state.inputs['Sym_16'] = st.number_input('Physical Health: Days not good in the past 30 days', value=st.session_state.inputs['Sym_16'])
            st.session_state.inputs['Sym_17'] = st.radio('Difficulty Walking (0: No, 1: Yes):', [0, 1], index=st.session_state.inputs['Sym_17'])

        with col4:
            st.session_state.inputs['Sym_18'] = st.radio('What is your Sex (0: Female, 1: Male):', [0, 1], index=st.session_state.inputs['Sym_18'])
            st.session_state.inputs['Sym_19'] = st.number_input('What is your Age: 1 = 18-24, 9 = 60-64, 13 = 80 or older', value=st.session_state.inputs['Sym_19'])
            st.session_state.inputs['Sym_20'] = st.number_input('What is your Level of Education: Scale 1-6', value=st.session_state.inputs['Sym_20'])
            st.session_state.inputs['Sym_21'] = st.number_input('What is your Income: Scale 1-8', value=st.session_state.inputs['Sym_21'])

        submit_button = st.form_submit_button(label='Make Prediction')

    if submit_button:
        def prediction(inputs):
            data = list(inputs.values())
            for i in range(len(data)):
                if data[i] != 0:
                    data[i] = str(data[i]).lower().strip()
                    data[i] = int(data[i])
            pred = st.session_state.model.predict([data])
            return pred[0]

        dia_prediction = prediction(st.session_state.inputs)
        if dia_prediction == 0:
            st.success('You are not Diabetic')
        elif dia_prediction == 1:
            st.error('You are Diabetic or Pre-Diabetes. Perhaps you should consult with your doctor')

# Select Model page  
elif selected_option == 'Select Model':
    st.title('Select the model for prediction')
    model_option = st.radio("Choose a model for prediction:", ('K-Nearest Neighbors','Decision Tree','Random Forest', 'Extremely Random Tree', 'Neural Networks'))
    st.write(f'You selected: {model_option}')

    if model_option == 'K-Nearest Neighbors':
        model_option = 'knn'
    elif model_option == 'Decision Tree':
        model_option = 'dt'
    elif model_option == 'Random Forest':
        model_option = 'RFC'
    elif model_option == 'Extremely Random Tree':
        model_option = 'ERT'
    elif model_option == 'Neural Networks':
        model_option = 'MLPC'

    if st.button("Load the Model"):
        st.session_state.model = jb.load(f'models/model_{model_option}.joblib')
        st.success("Model Loaded")

# Train model   
elif selected_option == 'Train Model':
    st.title('Model Training Page')
    st.header("Train the model")
    st.write("Click on the button to start training the model")
    if st.button("Start Training"):
        st.success("Future Work")
