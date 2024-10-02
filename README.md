# Customer Feedback Sentiment Analysis

## Problem Statement
Your company receives large amounts of customer feedback daily across various channels (emails, social media, surveys, product reviews, etc.). While some of this feedback is structured (e.g., ratings), most of it is unstructured (text data). Your task is to analyze this feedback and automatically classify the sentiment of each feedback message.

## File Structure
- **data_loader.py** : Loads data from a csv file and creates a DataFrame.
- **text_cleaning.py** : Contains functions that cleans text in the DataFrame, tokenizes and lemmatizes words, removes stop words and vectorizes the tokens.
- **sentiment_model.py** : Contains Naive Bayes and LSTM Classification models which classify the sentiment of the text.
- **main.py** - Main file where all the functions are executed and the comparison of sentiment models is done.

## Results
The models both did a good job at classifying the sentiments of the texts. The Naive Bayes Model had a 89% accuracy score while, the LSTM Model had 100% accuracy score. These numbers could change slightly when run on larger datasets but the change should not be significant that what is observed.

## Conclusions
Overall, the LSTM model has done a better job at classifying the sentiment of user reviews which can be beneficial for trying to filter out negative reviews and finding out key themes and patterns and potentially identify areas of improvement for the products.