# Techniators 💻 Fake News Detector 🗞️

This is a Streamlit web application that detects fake news articles using natural language processing (NLP) and machine learning (ML) algorithms.

## 📰 Demo

## 📊 Features

The Fake News Detector app can:

- ✅ Analyze the text of a news article and predict whether it is fake or real
- ✅ Show the client the probability of the article being fake or real
- ✅ Allow the client to input their own news article for analysis
- ✅ Display a word cloud and word frequency plot from the input text

## Technologies

This application was built using:

- Python 3.8
- Streamlit
- Scikit-learn
- Natural Language Toolkit (NLTK)
- AWS EC2
- AWS Sagemaker
- AWS S3

## Installation

To install and run the app locally, follow these steps:

Clone the repository: git clone https://github.com/UBC-MDS/Techniators.git

Install the required packages: ```pip install -r requirements.txt```

Start the app: ```streamlit run stream_app.py```

## Usage

Once the app is running, you can:

1. Enter the text of a news article in the input area
2. Click the "Submit" button to see whether the article is fake or real
3. View the probability score to see how confident the model is in its prediction
4. View the word cloud and frequency plot for more insights about your input news article

## Contributing

If you'd like to contribute to the Techniators - Fake News Detector application, follow these steps:

0. Read out [Contributing](CONTRIBUTING.md) document
1. Fork the repository
2. Create a new branch for your feature: ```git checkout -b my-feature-branch```
3. Make your changes and commit them: ```git commit -m "Add new feature"```
4. Push your changes to your forked repository: ```git push origin my-feature-branch```
5. Create a pull request from your branch to the main branch of the original repository

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
