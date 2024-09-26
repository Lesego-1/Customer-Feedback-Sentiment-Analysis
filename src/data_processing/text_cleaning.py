import pandas as pd
import re
from langdetect import detect, DetectorFactory
import spacy

# Filter only English Reviews

DetectorFactory.seed = 0 # Ensure consistent results

def is_english(text):
    # Checks if a text is English or not. Returns True if it is English, False if otherwise.
    try:
        return detect(text) == "en" # Checks if text is English
    except:
        return False # In case of error such as empty string

def filter_english(dataframe, text_col):
    """
    Filters out any rows that contain Non-English language.
    """
    dataframe['is_english'] = dataframe[text_col].apply(is_english) # Create new boolean column to classify if the text is english
    english_df = dataframe[dataframe['is_english']] # New DataFrame that only has rows with english values
    
    english_df = english_df.drop(columns=["is_english"]) # Drop is_english column
    
    return english_df

# Text Cleaning and Regular Expression
def regex(text):
    """
    Applies regular expression to a text to remove punctuation marks
    """
    text = re.sub(r'[^\w\s]', "", text) # Replace punctuation marks with empty string
    text = re.sub(r'[\s+]', " ", text) # Replace multiple spaces with one space
    
    return text.strip()
    
def clean_text(dataframe, text_col):
    """
    Ensures data is consistent and removes punctuation for better model performance.
    """
    dataframe = dataframe.dropna(subset=[text_col]) # Remove rows with missing values in text column
    
    dataframe[text_col] = dataframe[text_col].apply(regex) # Remove punctuation marks
    
    return dataframe

# Tokenizaton
def tokenize(text):
    # Tokenizes a text and returns the tokens
    
    nlp = spacy.load("en_core_web_sm") # Create NLP Pipeline
    
    doc = nlp(text) # Process the text
    tokens = [token.text for token in doc] # Stores the tokens
    
    return tokens

def tokenize_words(dataframe, text_col):
    """
    Tokenizes every row in the text column down. Creates a new column containing tokenized words.
    Returns the new DataFrame.
    """
    dataframe["tokenized_words"] = dataframe[text_col].apply(tokenize) # Tokenize words and add to new column
    
    return dataframe

# Stop Word Removal
def stop_word_filter(tokens):
    # Removes stop words from an array of tokens and returns the filtered tokens
    nlp = spacy.load("en_core_web_sm") # Create NLP Pipeline
    stop_words = nlp.Defaults.stop_words # Create a list of stop words
    
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words] # Remove stop words
    
    return filtered_tokens

def remove_stop_words(dataframe, token_col):
    """
    Removes stop words from tokens array and returns the DataFrame.
    """
    dataframe[token_col] = dataframe[token_col].apply(stop_word_filter)
    
    return dataframe

# Lemmatization
def lemmatize(tokens):
    """
    Lemmatizes the text and returns the lemmatized words.
    """
    nlp = spacy.load("en_core_web_sm") # Create NLP Pipeline
    
    text = " ".join(tokens) # Create a text version of the tokens
    doc = nlp(text) # Process the text
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return lemmatized_tokens

def lemmatize_tokens(dataframe, token_col):
    """
    Lemmatizes the tokens in the token column and returns the DataFrame.
    """
    dataframe[token_col] = dataframe[token_col].apply(lemmatize)
    
    return dataframe

# Vectorization
def vectorize_tokens(tokens):
    # Vectorizes the array of tokens and returns the array of vectors.
    nlp = spacy.load("en_core_web_lg") # Create NLP Pipeline
    
    text = " ".join(tokens) # Create a text version of the tokens
    doc = nlp(text) # Process the text
    
    vectorized_tokens = [doc.vector] # Vectorize tokens and insert into a list
    
    return vectorized_tokens
    
def vectorize(dataframe, token_col):
    """
    Takes tokens and provides a new column containing their vectors. 
    Returns the DataFrame.
    """
    nlp = spacy.load("en_core_web_lg") # Create NLP Pipeline
    
    dataframe["Vectors"] = dataframe[token_col].apply(vectorize_tokens) # Vectorize tokens and add to Vectors column
    
    return dataframe

def get_model_data(dataframe, cols):
    """
    Drops all columns that are not specified in the cols array and returns the DataFrame.
    """
    new_dataframe = dataframe[cols]
    
    return new_dataframe