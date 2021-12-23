import streamlit as st
import numpy as np
import pandas as pd

import joblib

def run_ml_app():
    classifier = joblib.load('data/best_model.pkl')
    scaler_X = joblib.load('data/scaler_X.pkl')

    st.subheader('데이터를 입력하면 당뇨병을 예측!')

    # Pregnancies	Glucose	BloodPressure	SkinThickness	
    # Insulin	BMI	DiabetesPedigreeFunction	Age

    pregnancies = st.number_input('임신횟수', min_value=0)
    glucose = st.number_input('Glucose', min_value=0)
    pressure = st.number_input('BloodPressure', min_value=0)
    skinthickness = st.number_input('SkinThickness', min_value=0)
    insulin = st.number_input('Insulin', min_value=0)    
    bmi = st.number_input('BMI', min_value=0.0, format='%.1f')
    diabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, format='%.2f')
    age = st.number_input('AGE',min_value=0)

    if st.button('결과 보기') :
        new_data = np.array([pregnancies,glucose,pressure,
                    skinthickness,insulin,bmi,
                    diabetesPedigreeFunction, age ])

        new_data = new_data.reshape(1, 8)

        new_data = scaler_X.transform(new_data)

        y_pred = classifier.predict(new_data)

        print(y_pred[0])
        if y_pred[0] == 0 :
            st.write('예측 결과는, 당뇨병이 아닙니다.')
        else :
            st.write('예측 결과는, 당뇨병입니다.')


    





