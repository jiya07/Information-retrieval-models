import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import string


# Extracting the passages from .txt file
def read_data(file):
    data = []
    with open(file, 'r') as f:
        data = [line.rstrip('\n') for line in f]
    return data


# Preprocessing the pasage test
def preprocess_text(data, remove_stopwords = False):
    passages = []
    for text in data:
        # Changing the text to lowercase
        text = text.lower()
        
        # Removing punctuations
        text = "".join([char for char in text if char not in string.punctuation])
        
        # Remove numbers
        text = re.sub(r'\d+','', text)
        
        # Tokenize words
        words = word_tokenize(text)
        
        # Remove stopwords if given
        if remove_stopwords:
            words = [word for word in words if word not in stopwords.words('english')]
        
        # Lemmatization and Stemming
        for w in words:
            w = WordNetLemmatizer().lemmatize(w)
            w = PorterStemmer().stem(w)
        passages.append(words)

    return passages


# Calculating the vocabulary of the passage texts
def passage_vocab(passages):
    all_passages = []

    for passage in passages:
        all_passages += passage

    # Generating vocabulary with number of occurences of each term
    vocab = Counter(all_passages)
    return vocab


def plot_freq_rank(vocabulary):
    
    # Extracting terms and their frequencies
    terms  = list(vocabulary.keys())
    freqs = list(vocabulary.values())
    
    # Probability of occurence of terms
    total_terms = np.sum(freqs)
    term_prob = {}
    for term, freq in vocabulary.items():
        term_prob[term] = freq / total_terms
    
    # Calculating the rank
    term_rank = {}
    for rank, (term, freq) in enumerate(sorted(vocabulary.items(), key = lambda x: x[1], reverse = True)):
        term_rank[term] = rank + 1
    
    freq_ranks = [term_rank[term] for term in terms]
    norm_freqs = [term_prob[term] for term in terms]
    
    # Vocabulary size
    N = len(terms)    
    
    # Zipf's law
    I = np.sum([1/i for i in range(1, N + 1)])
    k = np.arange(1, N + 1)
    zipf = 1/ (k * I)
    
    plt.scatter(freq_ranks, norm_freqs, color ='g', label = 'Data')
    plt.plot(k, zipf, '-.', color ='r', label = 'Zipf\'s law')
    plt.xlabel('Frequency ranking')
    plt.ylabel('Probability of term occurance') 
    plt.title('Zipf\'s law distribution')
    plt.legend()
    plt.show()
    plt.tight_layout()
    
    plt.scatter(freq_ranks, norm_freqs, color ='g', label = 'Data')
    plt.plot(k, zipf, '-.', color ='r', label = 'Zipf\'s law')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency ranking')
    plt.ylabel('Probability of term occurance')
    plt.title('Log-Log plot with Zipf\'s law distribution')
    plt.legend()
    plt.show()
    plt.tight_layout()


if __name__ == '__main__':

    passage_collection = read_data('passage-collection.txt')

    # Preprocessing text without removing stopwords
    passages = preprocess_text(passage_collection)
    vocabulary = passage_vocab(passages)

    plot_freq_rank(vocabulary)

    # Removing stopwords
    passages2 = preprocess_text(passage_collection, remove_stopwords = True)
    vocabulary2 = passage_vocab(passages2)
    plot_freq_rank(vocabulary2)