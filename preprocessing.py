import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import FastText
import pandas as pd
import numpy as np
import nltk
from gensim.models import KeyedVectors as gensim_KeyedVectors
# model = api.load('fasttext-wiki-news-subwords-300')
model = gensim_KeyedVectors.load_word2vec_format('fasttext_model/wiki-news-300d-1M-subword.bin', binary=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text):
    if not isinstance(text, str):
        text = str(text)
        
        
    return nltk.sent_tokenize(text)

def preprocess_text(text):
    
    if not isinstance(text, str):
        text = str(text)
    return simple_preprocess(text)

def text_to_vector(paragraph,max_seq_length,max_sentences):
    
    sentences = split_into_sentences(paragraph)
    sentences = sentences[:max_sentences]
    sentence_vectors = []
    for sentence in sentences:
        words = preprocess_text(sentence)
        vectors = [model[word] for word in words if word in model]
        if len(vectors)>max_seq_length:
            vectors = vectors[:max_seq_length]
            
        else:
            vectors = [np.zeros(300)]*(max_seq_length-len(vectors))+vectors # do pre-padding
        sentence_vectors.append(vectors)
        
    while(len(sentence_vectors)<max_sentences):
        sentence_vectors.append(np.zeros((max_seq_length,model.vector_size)))
        
    return np.array(sentence_vectors)

def avg_sequence_length(data):
    
    return data['reviewText'].apply(lambda x: len(str(x).split())).mean()

def load_data(data_path,max_seq_length,maxsentences,label_shifting=0):
    
    print("Loading data from ",data_path)
    data = pd.read_csv(data_path)
    avg_length = avg_sequence_length(data)
    print("Average sequence length: ",avg_length)
    
    X = np.array([text_to_vector(text,max_seq_length,maxsentences) for text in data['reviewText']])
    if label_shifting:
        y = data['overall'].values-1
    else:
        y = data['overall'].values
        
    print("Data Loading completed")
    
    return X,y


# if __name__ == "__main__":
    
#     X,y = load_data('data.csv',max_seq_length=10,maxsentences=5)
#     print(X.shape)
#     print(y.shape)
   
            
    

