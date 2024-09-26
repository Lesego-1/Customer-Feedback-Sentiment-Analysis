import pandas as pd

def load_dataframe(filename):
    # Initialize DataFrame
    df = pd.read_csv(filename)
    
    return df