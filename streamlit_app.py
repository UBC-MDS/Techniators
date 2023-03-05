import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto
import altair as alt
from wordcloud import WordCloud
from PIL import Image

# ==============================
# Main functions
# ==============================
@st.cache
def load_data():
    df = pd.read_csv('https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/True.csv')
    return df


def data_preprocessing(text):
    preprocessed_text = text
    return preprocessed_text

def read_model(url):


    return 0

def create_wordCloud(text):
    # word cloud

    return plt

def create_distribution(text):
    # frequency plot

    return plt


def model_predict():
    return 0


# ==============================
# main page
# ==============================
st.set_page_config(page_title="Techniator - Fake News Detector",
                   page_icon="img/sticker_icon.png")

# Display the image in the middle of the page
image = Image.open("img/sticker_icon.png")
st.title('Techniators 💻 Fake News Detector')

st.markdown("""
This app receives client input and detect whether the news is fake or not.
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
""")



# ==============================
# sidebar settings
# ==============================
st.sidebar.image(
    image,
    caption="Techniator - Fake News Detector",
    use_column_width=True,
    output_format="JPEG" 
)
st.sidebar.title("Input News Article")
# input text area
max_chars = 6000  # limit text input char
input_height = 300  # height of the text area
text_input = st.sidebar.text_area("Enter News Article (max {} characters):".format(max_chars), max_chars=max_chars, height=input_height)

# ==============================
# submission trigger
# ==============================
if st.sidebar.button("Submit"):
    st.write("Input Text:")
    st.write(text_input)

    processed_text = data_preprocessing(text_input)

    
    # st.write("Word Cloud:")
    # wc_plot = create_visualization(text_input)
    # st.pyplot(wc_plot)
