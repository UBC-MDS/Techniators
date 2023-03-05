import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto
from wordcloud import WordCloud
from tqdm import tqdm
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
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

def preprocess_text(text_data):
    preprocessed_text = []
    # from each sentence
    for sentence in tqdm(text_data):
        # Converts sentence to lowercase
        sentence = sentence.lower()
        # Removes all non-word and non-space characters
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token 
                                            for token in str(sentence).split() 
                                                if token not in stopwords.words('english')))
                                                # Removes stopwords

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
    
    if text_input == '':
        st.title("👈 Please input some text")
    else:
        
        processed_text = preprocess_text([text_input])

        st.write("Processed Text:")
        st.write(processed_text[0])
        # st.write("Word Cloud:")
        # wc_plot = create_visualization(text_input)
        # st.pyplot(wc_plot)
