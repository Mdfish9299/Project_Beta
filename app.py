
import streamlit as st 
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit_option_menu as som

# load the model 
model_RFC = jb.load('model_RFC.joblib')

# slider 
with st.sidebar:
    # Options
    menu_option = ['Prediction', 'Select Model','Train Model']
    
    # selecte Option
    selected_option = som.option_menu('Diabetes Prediction System Based on Lifestyle',options= menu_option , icons = ['hospital','database-fill-add','train-front'], menu_icon='bandaid')
 

# Prediction page
if selected_option == 'Prediction':
    
    # Header of Web page
    st.header(body='Diabetes Prediction System Based on Lifestyle')
    
    # devide the page into 4 col
    col1, col2, col3, col4 = st.columns(4)


    # in 1st column
    with col1:
        Sym_1 = st.text_input('High Blood Pressure: 0 = No , 1 = Yes',0) 
        Sym_2 = st.text_input('High Cholesterol: 0 = No , 1 = Yes',0)
        Sym_3 = st.text_input('Cholesterol Check in 5 Years: 0 = No , 1 = Yes',0)
        Sym_4 = st.text_input('BMI Score:',0)
        Sym_5 = st.text_input('Smoker: 0 = No , 1 = Yes',0)
        Sym_6 = st.text_input('Stroke: 0 = No , 1 = Yes',0)

    with col2:
        Sym_7 = st.text_input('Heart Disease or Attack: 0 = No , 1 = Yes',0) 
        Sym_8 = st.text_input('Physical Activity in the past 30 days: 0 = No , 1 = Yes',0)
        Sym_9 = st.text_input('Fruits Consumption: 0 = No , 1 = Yes',0)
        Sym_10 = st.text_input('Veggies Consumption: 0 = No , 1 = Yes',0)
        Sym_11 = st.text_input('Heavy Achcohol Consumption (> 14 drinks/Wk for men and > 7 for women): 0 = No , 1 = Yes',0)
        Sym_12 = st.text_input('Health Care Coverage: 0 = No , 1 = Yes',0)

    with col3:
        Sym_13 = st.text_input('No Doctor because of cost in the past 12 months: 0 = No , 1 = Yes',0) 
        Sym_14 = st.text_input('General Health Score: scale of 1-5 - 1- excellent, 5-poor',0)
        Sym_15 = st.text_input('Mental Health Check: Days not good in the past 30 days',0)
        Sym_16 = st.text_input('Physical Health: Days not good in the past 30 days',0)
        Sym_17 = st.text_input('Difficulty Walking: 0 = No , 1 = Yes',0)

    with col4:
        Sym_18 = st.text_input('What is your Sex:: 0 = Female , 1 = Male',0) 
        Sym_19 = st.text_input('What is your Age: 1 = 18-24 9 = 60-64 13 = 80 or older (Every 5 years increases 1) ',0)
        Sym_20 = st.text_input('What is your Level of Education: Scale 1-6 1 = Never attended, 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate)',0)
        Sym_21 = st.text_input('What is your Income: Scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more',0)
        

    # code for prediction
    def prediction(Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17,Sym_18,Sym_19,Sym_20,Sym_21):

        # input data
        data = [Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17,Sym_18,Sym_19,Sym_20,Sym_21]

        # removing white space if have and handling the case error
        for i in range(len(data)):
            if data[i]!=0:
                data[i] = str(data[i]).lower().strip()

        # make the prediction
        pred = model_RFC.predict([data])

        # return the prediction
        return pred[0]

    dia_prediction = ''

    # submit button
    if st.button('Make Prediction'):
        dia_prediction = prediction(Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17,Sym_18,Sym_19,Sym_20,Sym_21)

    if dia_prediction == 0:
        st.error('You are not Diabetic')
    elif dia_prediction == 1:
        st.success
        st.success('You are Diabetic or have a chance of Diabetes')

# Select Model page  
elif selected_option == 'Select Model':
    
    # Header 
    st.title('Select the model for prediction')
    
    # Radio box for model selection
    model_option = st.radio(
        "Choose a model for prediction:",
        ('K-Nearest Neighbors','Decision Tree','Random Forest', 'Extremely Random Tree', 'Neural Networks')
    )

    # Display the selected model
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
        model_option = 'NN'

    # Load the model
    model_selected = jb.load(f'model_{model_option}.joblib')
        
# Train model   
elif selected_option == 'Train Model':
    
    # Header
    st.title('Model Training Page')
    
    # Header
    st.header("Train the model")
    
    # Instruction
    st.write("Click on the button to start traning the model")
    
    if st.button("Start Training"):
            
        st.success("Future Work")
        
        