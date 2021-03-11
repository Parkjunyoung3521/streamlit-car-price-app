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

def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
   
    radio_menu = ['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)

    if selected_radio == '데이터프레임' :
        st.dataframe(car_df)

    elif selected_radio == '통계치' :
        st.dataframe(car_df.describe())

    columns = car_df.columns
    columns = list(columns)

    selected_columns = st.multiselect('컬럼을 선택하시오.',columns)

    if len(selected_columns) != 0 :
        st.dataframe(car_df[selected_columns])

    else :
        st.write('선택한 컬럼이 없습니다.')

    # 상관계수를 화면에 보여주도록 만듭니다.
    # 멀티셀렉트에 컬럼명을 보여주고,
    # 해당 컬럼들에 대한 상관계수를 보여주세요.
    # 단, 컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야 합니다.

    print(car_df.dtypes != object)

    corr_columns = car_df.columns[car_df.dtypes != object]
    selected_corr = st.multiselect('상관계수 컬럼 선택',corr_columns)

    if len(selected_corr) > 0 :
        st.dataframe(car_df[selected_corr].corr() )

        # 위에서 선택한 컬럼들을 이용해서, 시본의 페어플롯을 그린다.
        
        fig = sns.pairplot(data= car_df[selected_corr])
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')


    # 컬럼을 하나만 선택하면, 해당 컬럼의 min과 max에 해당하는
    # 사람의 데이터를 화면에 보여주는 기능

    number_columns = car_df.columns[car_df.dtypes != object]
    selected_col = st.selectbox('컬럼 선택',number_columns)

    min_data = car_df[selected_col] == car_df[selected_col].min()
    st.write('최소값 데이터')
    st.dataframe(car_df.loc[min_data],)

    max_data = car_df[selected_col] == car_df[selected_col].max()
    st.write('최대값 데이터')
    st.dataframe(car_df.loc[max_data],)

    # 고객 이름을 검색 할 수 있는 기능 개발

    # 1. 유저한테 검색어를 받자
    word = st.text_input('검색어를 입력하세요.')

    # 2. 검색어를 데이터프레임 커스터머 네임에서 검색하자.
    result = car_df.loc[car_df['Customer Name'].str.contains(word,case=False),]

    # 3. 화면에 결과를 보여주자
    st.dataframe(result)