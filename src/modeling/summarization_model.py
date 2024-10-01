from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd

def bart_model(text):
    '''
    Uses the BART pre-trained model to summurize text.
    Returns the summurized text.'''
    # Load BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    
    inputs = tokenizer([text], max_length=75, return_tensors='tf', truncation=True) # Tokenize input text
    summary_ids = model.generate(inputs['input_ids'], max_length=20, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True) # Generate the summary id's
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) # Decode summary id's to get actual summary
    
    return summary

def summarize_text(dataframe, text_col):
    # Summurizes text and inserts summarizations into new column in the DataFrame.
    dataframe['Summary'] = dataframe[text_col].apply(bart_model)
    
    return dataframe