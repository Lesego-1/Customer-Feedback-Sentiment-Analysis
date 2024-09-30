from gensim.models import LdaModel
from gensim import corpora

def topic_distribution_to_vector(topic_dist, num_topics):
    """
    Converts topic distribution to a fixed sized vector.
    The vector will contain the topics represented by indexes and the values will be thier probabilities.
    Returns the vector.
    """
    vector = [0] * num_topics # Declare the vector to be the size of num_topics
    for topic, prob in topic_dist:
        vector[topic] = prob
    return vector

def LDA(tokens):
    """
    Trains an LDA Topic Model and returns a dictionary of the topic id's and the top 5 words per topic.
    """
    # Set dictionary and corpus for LDA
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens] # Convert tokens into bag of words format
    
    num_topics = 2
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42) # Initialize model
    topic_features = [model.get_document_topics(bow) for bow in corpus] # Get topic distributions
    
    top_words_per_topic = model.show_topics(num_topics=num_topics, num_words=5, formatted=False) # Get top 5 words per topic
    
    result = {} # Define dictionary to store result
    
    # Store topic id and words in result
    for topic_id, words in top_words_per_topic:
        result[topic_id] = words
        
    return result

def LDA_model(dataframe, tokens_col):
    """
    Implements LDA topic modeling on each column.
    Stores the topics and their words in a new column.
    Returns the dataframe.
    """
    dataframe["Topics"] = dataframe[tokens_col].apply(LDA) # Create new column
    
    return dataframe