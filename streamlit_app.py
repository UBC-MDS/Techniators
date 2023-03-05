import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto
import altair as alt

st.title('Techniators 💻 Fake News Detector')

st.markdown("""
This app receives client input and detect whether the news is fake or not.
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
""")

st.sidebar.header('User Input Features')

def load_data():
    df = pd.read_csv('https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/True.csv')
    return df


def data_preprocessing(text):
    preprocessed_text = text
    return preprocessed_text

def read_model():

    return 0

def create_visualization():
    # word cloud

    # freq

    return 0

def model_predict():
    return 0

df = load_data()

st.dataframe(df.head())

