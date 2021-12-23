import streamlit as streamlit
import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
import seaborn as sns
import joblib

def main() :
    menu = [' 홈', '데이터분석', '인공지능']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == '홈' :
        st.subheader('당뇨병 데이터 분석 및 예측')

    elif choice == '데이터분석' :
        run_ead_app()

    elif choice == '인공지능' :
        run_ml_app()

    
if __name__ == '__main__' :
    main()

    