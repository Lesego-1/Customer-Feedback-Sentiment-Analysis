import spacy
import pandas as pd
from textblob import TextBlob

nlp = spacy.load("en_core_web_lg") # Load NLP Pipeline

# def extract_aspects(text):
#     # Extracts the aspects(nouns) from a text
#     doc = nlp(text) # Process text
#     aspects = [word.text for word in doc.noun_chunks] # Store apects
#     return aspects

# def calculate_aspect(tokens, model):
#     """
#     Extracts aspects from the given text and predicts the sentiment with the model provided.
#     The text predicted is the text with the aspect added on the end to reinforce the focus of the model on the aspect.
#     Returns the aspect and sentiment.
#     """
#     text = " ".join(str(token) for token in tokens) # Create text format of tokens
#     aspects = extract_aspects(text) # Extract aspects
#     result = {} # Initialize result variable
    
#     for aspect in aspects:
#         full_text = text + " " + aspect # Create full text to predict
#         doc = nlp(full_text) # Process text
#         vectorized_text = [doc.vector] # Vectorize the text
        
#         sentiment = model.predict(vectorized_text) # Classify sentiment
#         result[aspect] = sentiment[0] # Add apsect and related sentiment to result dictionary
        
#     return result

def absa_model(text):
    """
    Takes in text and determines the text's aspects and the opinions associated with the aspects.
    Uses a list of words and sentiment polarity to classify the sentiment of the opinion.
    Determines opions if they are adjectives or adverbs to the parent token (noun).
    Returns the aspects, opinions and their sentiments.
    """
    # Define positive and negative words
    positive_words = ["amazing", "good", "great", "excellent", "fantastic", "positive", "happy"]
    negative_words = ["poor", "bad", "terrible", "horrible", "negative", "unhappy", "disappointing"]
    
    doc = nlp(text) # Process the text
    
    # Initialize lists for results
    aspects, opinions, sentiments = [], [], []
    
    # Iterate through tokens to find aspects (nouns) and opinions (adjectives)
    for token in doc:
        if token.pos_ == "NOUN":
            # Look for adjectives related to the aspect
            for child in token.children: # Loop through all dependants of the token
                if child.dep_ == "amod" and child.pos_ == "ADJ": # If child is an adjective
                    # Define aspect and its correlated opinion
                    aspect = token
                    opinion = child
                    # Add aspect and opinion to the lists
                    aspects.append(aspect)
                    opinions.append(opinion)
                    
                    # Determine sentiment
                    if opinion in positive_words:
                        sentiment = "positive"
                    elif opinion in negative_words:
                        sentiment = "negative"
                    else: # Use textblob for polarity
                        sentiment = "positive" if TextBlob(opinion).sentiment.polarity >= 0 else "negative"
                    
                    sentiments.append(sentiment) # Add sentiment to the list
                    
    # Iterate through tokens to find aspects (nouns) and opinions (adverbs)
    for token in doc:
        if token.pos_ == "NOUN":
            # Look for adverbs related to the aspect
            for child in token.children: # Loop through all dependants of the token
                if child.dep_ == "advmod" and child.pos_ == "ADV": # If child is an adverb
                    aspect = token
                    opinion = child
                    
                    aspects.append(aspect)
                    opinions.append(opinion)
                    
                    # Determine Sentiment
                    if opinion in positive_words:
                        sentiment = "positive"
                    elif opinion in negative_words:
                        sentiment = "negative"
                    else:
                        sentiment = "postive" if TextBlob(opinion).sentiment.polarity >= 0 else "negative"
                        
                    sentiments.append(sentiment)
    
    return list(zip(aspects, opinions, sentiments))