import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from eda_app import run_eda_app
from ml_app import run_ml_app

def main():
    st.title('자동차 가격 예측')
    
    # 사이드바 메뉴
    menu = ['Home','EDA','ML']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home' :
        st.write('이 앱은 고객데이터와 자동차 구매 데이터에 대한 내용입니다. 해당 고객의 정보를 입력하면, 얼마정도의 차를 구매 할 수 있는지를 예측하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')

    elif choice == 'EDA' :
        run_eda_app()

    elif choice == 'ML' :
        run_ml_app()





if __name__=='__main__':
    main()