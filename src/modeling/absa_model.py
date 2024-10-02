import spacy
import pandas as pd

nlp = spacy.load("en_core_web_lg") # Load NLP Pipeline

def extract_aspects(text):
    # Extracts the aspects(nouns) from a text
    doc = nlp(text) # Process text
    aspects = [word.text for word in doc.noun_chunks] # Store apects
    return aspects

def calculate_aspect(tokens, model):
    """
    Extracts aspects from the given text and predicts the sentiment with the model provided.
    The text predicted is the text with the aspect added on the end to reinforce the focus of the model on the aspect.
    Returns the aspect and sentiment.
    """
    text = " ".join(str(token) for token in tokens) # Create text format of tokens
    aspects = extract_aspects(text) # Extract aspects
    result = {} # Initialize result variable
    
    for aspect in aspects:
        full_text = text + " " + aspect # Create full text to predict
        doc = nlp(full_text) # Process text
        vectorized_text = [doc.vector] # Vectorize the text
        
        sentiment = model.predict(vectorized_text) # Classify sentiment
        result[aspect] = sentiment[0] # Add apsect and related sentiment to result dictionary
        
    return result

# def aspect_based_sentiment_analysis(dataframe, tokens_col):
#     # Adds aspects and their sentiments to the DataFrame
#     dataframe["Aspects"] = dataframe[tokens_col].apply(calculate_aspect)
#     return dataframe