from src.data_processing.data_loader import load_dataframe
from src.data_processing.text_cleaning import filter_english, clean_text, tokenize_words, remove_stop_words, lemmatize_tokens, vectorize_tokens, get_model_data, classify_sentiment
from src.modeling.sentiment_model import split_data, naive_bayes_model, lstm_model
from src.modeling.absa_model import absa_model
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# df = load_dataframe("amazon_uk_shoes_products_dataset_2021_12.csv") # Initialize DataFrame
# df = df[:50] # Sample DataFrame because of computational expenses
# df = filter_english(df, "review_text") # Filter DataFrame to contain only english reviews

# # Clean the text
# df = clean_text(df, "review_text")
# df = tokenize_words(df, "review_text")
# df = remove_stop_words(df, "tokenized_words")
# df = lemmatize_tokens(df, "tokenized_words")
# df = vectorize_tokens(df, "tokenized_words")

# df_model = classify_sentiment(df, "review_rating") # Create DataFrame to use for model
# df_model = get_model_data(df, ["review_text", "tokenized_words", "Vectors", "sentiment"]) # Define DataFrame with relevant columns

# # Declare X and y variables
# X = df_model["Vectors"]
# y = df_model["sentiment"]

# X_train, X_test, y_train, y_test = split_data(X, y) # Split data into train and test

# # Reshape data into 2 Dimensions
# X_train_2d = np.stack(X_train)
# X_train_2d = np.concatenate(X_train_2d, axis=0)
# X_test_2d = np.stack(X_test)
# X_test_2d = np.concatenate(X_test_2d, axis=0)

# nb_pred = naive_bayes_model(X_train_2d, X_test_2d, y_train) # Train Naive Bayes Model
# lstm_pred = lstm_model(X_train, X_test, y_train, y_test) # Train LSTM Model

# # Compare performance of models with accuracy
# nb_accuracy = accuracy_score(y_test, nb_pred)
# lstm_accuracy = accuracy_score(y_test, lstm_pred)
# # Print Accuracy of the models
# print(f"Naive Bayes Accuracy Score: {nb_accuracy}")
# print(f"LSTM Model Accuracy Score: {lstm_accuracy}")

# Aspect Based Sentiment Analysis
text = "The battery life is amazing but the screen quality is poor." # Example text
absa_model_results = absa_model(text)
# Print absa results
# print(f"ABSA Model Results: {absa_model_results}")