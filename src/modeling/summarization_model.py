# from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
from transformers import pipeline

def summarize_text(text):
    # Returns the summarizes text
    summarizer = pipeline("summarization") # Intialize summarizer
    
    summary = summarizer(text, max_length=20, min_length=1, do_sample=False) # Genrate summary
    
    return summary[0]['summary_text'] # Return summarized text