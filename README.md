# Customer Feedback Sentiment Analysis

## Problem Statement
Your company receives large amounts of customer feedback daily across various channels (emails, social media, surveys, product reviews, etc.). While some of this feedback is structured (e.g., ratings), most of it is unstructured (text data). Your task is to analyze this feedback and automatically classify the sentiment of each feedback message.

## File Structure
- **data_loader.py** : Loads data from a csv file and creates a DataFrame.
- **text_cleaning.py** : Contains functions that cleans text in the DataFrame, tokenizes and lemmatizes words, removes stop words and vectorizes the tokens.
- **sentiment_model.py** : Contains Naive Bayes and LSTM Classification models which classify the sentiment of the text.
- **absa_model.py** : Contains an function that extracts aspects and opinions and classifies the sentiment of the text.
- **summarization_model.py** : Performs text summarization using a hugging face model.
- **main.py** - Main file where all the functions are executed and the comparison of sentiment models is done.

## Results
#### Sentiment Analysis
The models both did a good job at classifying the sentiments of the texts. The Naive Bayes Model had a 89% accuracy score while, the LSTM Model had 100% accuracy score. These numbers could change slightly when run on larger datasets but the change should not be significant that what is currently observed.

### Aspect Based Sentiment Analysis
The model properly returns the result in the correct format. Returning the aspect, the opinion that is used to classify the sentiment and lastly, the sentiment of the opinion.

### Text Summarization
The summarization works decently well, which provides a good summary of the given text. The sentence structure of the result is not proper, but the text itself is the proper summarized text. Although the summarized text is correct, it takes a long time to generate an answer for longer text.

## Conclusions
### Sentiment Analysis
Overall, the LSTM model has done a better job at classifying the sentiment of user reviews which can be beneficial for trying to filter out negative reviews and finding out key themes and patterns and potentially identify areas of improvement for the products.

### Aspect Based Sentiment Analysis
It provides insights on which aspects have positive and negative sentiments, which clarifies the aspects of the business to improve. It also provides the opinion that was given for that aspect which shows the specific thing about the aspect to improve.

### Text Summarization
The summarized text that is returned provides a more concice version which still has all the relevant details. The sentence may not be ended properly, which is the only drawback.