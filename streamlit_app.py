import string
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
nltk.download("vader_lexicon")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer

sid = SentimentIntensityAnalyzer()

# ==============================
# Main functions
# ==============================
@st.cache
def load_data():
    real = pd.read_csv('https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/True.csv')
    fake = pd.read_csv('https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/Fake.csv')
    return real, fake


def data_cleaning(real_df, fake_df):
    #Rename locally and add labels
    real = real_df
    fake = fake_df
    fake['label'] = 'fake'
    real['label'] = 'real'
    
    #Strip columns from whitespace
    cols = ['title', 'text', 'date']
    real[cols] = real[cols].apply(lambda x: x.str.strip())
    fake[cols] = fake[cols].apply(lambda x: x.str.strip())
    
    # Remove unecessary text from the beginning of the text column in real df.
    real['text'] = real['text'].str.replace('^(.*?)?\s[-]', '', regex=True)
    
    # Combining datasets, replace empty cells with "NA" (they exist only text column) & drop these columns
    all_news = pd.concat([real, fake])
    all_news = all_news.replace(r'^\s*$', np.nan, regex=True)
    all_news = all_news.dropna(subset=['text'])
    
    # Cleaning & Parse date column
    match_condn = r'\b([0-9])\b'
    replace_str = r'0\1'
    all_news['date'] = all_news['date'].str.replace(match_condn, replace_str, regex=True)

    replacement = {
    "^Jan\s": "January ",
    "^Feb\s": "February ",
    "^Mar\s": "March ",
    "^Apr\s": "April ",
    "^Jun\s": "June ",
    "^Jul\s": "July ",
    "^Aug\s": "August ",
    "^Sep\s": "September ",
    "^Oct\s": "October ",
    "^Nov\s": "November ",
    "^Dec\s": "December ",
    "14[-]Feb[-]18$": "February 14, 2018",
    "15[-]Feb[-]18$": "February 15, 2018",
    "16[-]Feb[-]18$": "February 16, 2018",
    "17[-]Feb[-]18$": "February 17, 2018",
    "18[-]Feb[-]18$": "February 18, 2018",
    "19[-]Feb[-]18$": "February 19, 2018",
    }

    all_news['date'] = all_news['date'].replace(replacement, regex=True)

    # Converting date column to datetime
    all_news['date'] = pd.to_datetime(all_news['date'], format='%B %d, %Y', errors='coerce')
    #This results in 10 empty date columns because they were not dates in the first place
    
    # Limiting categories of Subjects to only 2 (politicsNews and worldnews)
    all_news.query("subject == 'left-news'")

    subject_replace = {'News':'other', 
                   'politics':'politicsNews', 
                   'left-news' : 'other',
                   'Government News': 'politicsNews',
                   'US_News': 'politicsNews',
                   'Middle-east': 'worldnews', 
                      }

    all_news = all_news.replace({"subject": subject_replace})
    
    # Combine title and text columns into one column and drop the individual ones 
    all_news["title_text"] = all_news["title"] +" "+ all_news["text"]
    all_news = all_news.drop(['title', 'text'], axis=1)
    cols = all_news.columns.tolist()
    cols = cols[-1:] + cols[:-1] #reorder columns to "text_tile" comes first
    all_news = all_news[cols]
    
    return all_news


def get_sentiment(text): #Adapted from DSCI 574
    """
    Returns the compound score representing the sentiment: -1 (most extreme negative) and +1 (most extreme positive)
    The compound score is a normalized score calculated by summing the valence scores of each word in the lexicon.

    Parameters:
    ------
    text: (str)
    the input text

    Returns:
    -------
    sentiment of the text: (str)
    """
    scores = sid.polarity_scores(text)
    return scores["compound"]

def feature_engineering(full_df):
    #Special Character, Uppercase letters and Sentiment Score
    #Rename locally
    all_news = full_df
    
    full_list = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.whitespace
    all_news['special_char_count'] = all_news['title_text'].str.count(f'[^{re.escape(full_list)}]')
    all_news['uppercase_letter_count'] = all_news['title_text'].str.findall(r'[A-Z]').str.len()
    all_news = all_news.assign(sentiment_score=all_news["title_text"].apply(get_sentiment))
    
    return all_news


def read_model(url):


    return 0

def create_wordCloud(text):
    ''' 
    A function to display plots given an English text input. 
    '''

    #To generate word cloud
    words = " ".join([text])
    wordcloud = WordCloud(max_font_size=40, 
                          width=500, 
                          height=400, 
                          stopwords = STOPWORDS,
                          max_words=100,
                          min_word_length = 2,
                          collocations=False)

    return wordcloud.generate(words).to_image()

def create_distribution(text):
    ''' 
    A function to display frequency of top 20 words in text
    '''
    counts = {}
    words = text.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

        freq_df = pd.DataFrame(counts).sort_values()

        plt(freq_df)

    return counts   

    #Display highest frequency words in text
    
    
    return visualizer.show()


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
