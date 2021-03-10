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

import joblib

def run_ml_app():
    st.subheader('Machine Learning')

    model = tensorflow.keras.models.load_model('data/car_ai.h5')

    new_data  = np.array([0,  38, 90000,2000,500000])

    new_data = new_data.reshape(1,-1)

    sc_X = joblib.load('data/sc_X.pkl')

    new_data = sc_X.transform(new_data)

    y_pred = model.predict(new_data)

    st.write(y_pred[0][0])

    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform(y_pred)

    st.write(y_pred_original)