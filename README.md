# Sentiment Analysis Web App using Flask

## Overview :
This project is a sentiment analysis web application built using Flask and a custom-trained machine learning model. 
The model was trained on an open dataset for Natural Language Processing (NLP) and is used to classify text as either positive or negative.
The model and TfidfVectorizer were saved after training, and the web app allows users to input text and receive real-time sentiment predictions based on the trained model.

## Features :
- **NLP Application**: This is a Natural Language Processing (NLP) application for analyzing text sentiment.
- **Sentiment Analysis**: Classifies text input as either positive or negative.
- **Flask Backend**: Powered by Flask, a lightweight Python web framework.
- **Custom Model**: The machine learning model was trained using an open dataset and saved for use in this application.

## How to Run the Project :
**Make sure the following libraries are installed:**
- Python 3.x
- Flask
- NLTK
- Scikit-learn
- Joblib

You can install the necessary libraries by running:
- `pip install flask nltk scikit-learn joblib`


#### Running the App : 
Install the dependencies: Ensure all required libraries are installed by running:
- `pip install -r requirements.txt`
- Run the Flask app: Start the app with the following command: `python app.py`
- Access the web app: Open your browser and go to `http://127.0.0.1:5000/`. Input text and analyze its sentiment in real time.

## Model Details :
The model was trained using a **Logistic Regression algorithm** for sentiment classification, with text data processed using **TfidfVectorizer**.
The dataset was cleaned and preprocessed using various NLP techniques such as tokenization, stopword removal, and stemming.

## Screenshots
Here’s how the Sentiment Analysis web app looks:

![Sentiment Analysis Screenshot](screen_sh/1.png)
\
![Sentiment Analysis Screenshot](screen_sh/2.png)




