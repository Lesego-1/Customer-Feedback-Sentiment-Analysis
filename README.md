# Customer Feedback Sentiment Analysis

## Problem Statement
Your company receives large amounts of customer feedback daily across various channels (emails, social media, surveys, product reviews, etc.). While some of this feedback is structured (e.g., ratings), most of it is unstructured (text data). Your task is to analyze this feedback and automatically classify the sentiment of each feedback message, while also extracting key insights to suggest potential product improvements based on frequent customer complaints or requests.

---
## File Structure
- **data_loader.py** : Loads data from a csv file and creates a DataFrame.
- **text_cleaning.py** : Contains functions that cleans text in the DataFrame, tokenizes and lemmatizes words, removes stop words and vectorizes the tokens.
- **absa_model.py** : Creates an Aspect Based Sentiment Analysis model and returns each aspect along with their sentiment.
- **sentiment_model.py** : Contains Naive Bayes and LSTM Classification models which classify the sentiment of the text.
- **summarization_model.py** : Creates a BART model to summarize the text.
- **topic_model.py** : Creates a topic model to identify topics in the text.
- **main.ipynb** - Main file where all the functions are executed and the comparison of sentiment models is done.

---
## Results

---
## Conclusions
