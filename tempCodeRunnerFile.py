df = load_dataframe("amazon_uk_shoes_products_dataset_2021_12.csv") # Initialize DataFrame
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