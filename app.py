import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from eda_app import run_eda_app
from ml_app import run_ml_app

def main():
    menu = ['홈', '데이터분석', '인공지능']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == '홈' :
        st.subheader('당뇨병 데이터 분석 및 예측')

    elif choice == '데이터분석' :
        run_eda_app()

    elif choice == '인공지능' :
        run_ml_app()

if __name__ == '__main__' :
    main()

