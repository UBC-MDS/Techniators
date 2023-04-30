# Techniators 💻 Fake News Detector 🗞️

This is a Streamlit web application that detects fake news articles using natural language processing (NLP) and machine learning (ML) algorithms.

## 📰 Demo

![demo](img/demo.gif)

## 📊 Features

The Fake News Detector app can:

- ✅ Analyze the text of a news article and predict whether it is fake or real
- ✅ Show the client the prediction result of the article being fake or real
- ✅ Allow the client to input their own news article for analysis
- ✅ Display a word cloud and word frequency plot from the new article input

## Technologies

This application was built using:

- Python
- Streamlit
- Scikit-learn
- Natural Language Toolkit (NLTK)
- AWS EC2
- AWS Sagemaker
- AWS S3

## Installation

### Local

To install and run the app locally, follow these steps:

Clone the repository: git clone https://github.com/UBC-MDS/Techniators.git

Install the required packages: ```pip install -r requirements.txt```

Start the app: ```streamlit run streamlit_app.py```

### Docker

To install and deploy the app on docker, follow these steps:

Clone the repository: git clone https://github.com/UBC-MDS/Techniators.git

Run docker commands:

```{bash}
docker build -t <image-name> . 
docker run -p 8501:8501 <image-name>
```

*Note that the docker files are for reference, please input the corresponding docker image name.*

## Usage

Once the app is running, you can:

1. Enter the text of a news article in the input area
2. Click the "Submit" button to see whether the article is fake or real
3. View the probability score to see how confident the model is in its prediction
4. View the word cloud and frequency plot for more insights about your input news article

## Contributing

Team Members: [Sarah Abdelazim](https://github.com/missarah96), [Lisa Sequeira](https://github.com/LisaSeq), [Caesar Wong](https://github.com/caesarw0)

If you'd like to contribute to the Techniators - Fake News Detector application, follow these steps:

0. Read our [Contributing](CONTRIBUTING.md) document
1. Fork the repository
2. Create a new branch for your feature: ```git checkout -b my-feature-branch```
3. Make your changes and commit them: ```git commit -m "Add new feature"```
4. Push your changes to your forked repository: ```git push origin my-feature-branch```
5. Create a pull request from your branch to the main branch of the original repository

## Special Thanks

We would like to extend our most sincere thanks to the [HackHPC](http://hackhpc.org/) [Hack the Threat](https://hackhpc.github.io/HacktheThreat23/) hackathon organisers for providing us with the opportunity to participate part and showcase our project. We had a place to work together with people who shared our interests throughout the hackathon to accomplish a common objective. We appreciate the organisers' assistance and direction throughout the event.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
