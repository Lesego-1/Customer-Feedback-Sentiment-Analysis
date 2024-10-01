from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import  Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences

def split_data(X, y):
    """
    Takes X and y variables and splits into train and test data.
    Returns X and y train and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75, test_size=0.25) # Split data into train and test
    
    return X_train, X_test, y_train, y_test

def logistic_regression_model(X_train, X_test, y_train):
    # Fits train data into logistic regression model and returns the predicted test values.
    model = LogisticRegression(random_state=42) # Initialize model
    model.fit(X_train, y_train) # Fit train data into model
    
    y_pred = model.predict(X_test) # Classify text
    
    return y_pred

def naive_bayes_model(X_train, X_test, y_train):
    # Fits train data into model and returns predicted values.
    scaler = MinMaxScaler() # Initialize scaler
    # Normalize the train and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = MultinomialNB() # Intialize model
    model.fit(X_train, y_train) # Fit train data into model
    
    y_pred = model.predict(X_test) # Classify text
    
    return y_pred

def random_forest_model(X_train, X_test, y_train):
    # Fits train data into model and returns predicted y values.
    model = RandomForestClassifier(random_state=42) # Initialize model
    model.fit(X_train, y_train) # Fit train data into model
    
    y_pred = model.predict(X_test) # Classify text
    
    return y_pred

def lstm_model(X_train, X_test, y_train, y_test):
    """
    Creates an LSTM model and returns the predicted y values.
    Makes use of padding to ensure that the sequences are of equal length.
    """
    max_sequence_length = 20 # Define max length for padding
    # Pad the sequences of train and test data
    X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, padding="post", truncating="post", value=0.0)
    X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length, padding="post", truncating="post", value=0.0)
    
    model = Sequential() # Initialize model
    # Add LSTM model
    model.add(LSTM(128, input_shape=(X_train_padded.shape[1], X_train_padded.shape[2]), return_sequences=False))
    
    model.add(Dropout(0.5)) # Add dropout layer to prevent overfitting
    model.add(Dense(1, activation="sigmoid")) # Output layer
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acuracy']) # Compile model
    model.fit(X_train_padded, y_train, epochs=2, batch_size=64, validation_data=(X_test_padded, y_test)) # Fit the model
    
    y_pred_prob = model.predict(X_test) # Predict probability of positive classification
    y_pred = int((y_pred_prob >= 0.5)) # Get actual classification
    
    return y_pred