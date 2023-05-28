import pandas as pd
import numpy as np
from task1 import preprocess_text
from task2 import inverted_indx
import math
import warnings
warnings.filterwarnings("ignore")


# Import queries data and preprocess the queries
test_queries = pd.read_csv('test-queries.tsv', sep = '\t', header = None, names = ['qid', 'query'])
query_terms = preprocess_text(test_queries['query'], remove_stopwords = True)

# Import candidate data and preprocess the passages
candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep = '\t', header= None, names = ['qid', 'pid', 'query', 'passage'])
passages = preprocess_text(candidate_passages['passage'], remove_stopwords = True)

# Import inverted index from task 2
inverted_index = inverted_indx

# Laplace smoothing
def laplace_smoothing(query_data, candidate_passage, inverted_index):

    # Vocab size
    V = len(inverted_index.keys())

    laplace_df = pd.DataFrame()
    
    for i in range(len(query_data)):
        qid = query_data['qid'][i]
        query = query_data['query'][i]
        passages = candidate_passage[candidate_passage['qid'] == qid][['pid', 'passage']]
        #print(passages)
        data = []
        for j, (pid, passage) in passages.iterrows():
            score = 0
            D = len(passage)
            for term in query:
                m = 0
                if term in passage:
                    #print("term", term)
                    m += 1
                s = math.log((m + 1) / (D + V))
                score += s
            data.append([qid, pid, score])
            
        df1 = pd.DataFrame(data, columns = ['qid', 'pid', 'score']).sort_values(by = ['score'], ascending = False)
        laplace_df = laplace_df.append(df1.iloc[:100])
        
    return laplace_df.reset_index(drop = True)

test_queries['query'] = query_terms
candidate_passages['passage'] = passages

laplace_data = laplace_smoothing(test_queries, candidate_passages, inverted_index)
laplace_data.to_csv('laplace.csv', header = False, index = False)

# Lidstone correction
def lidstone_correction(query_data, candidate_passage, inverted_index):

    V = len(inverted_index.keys())
    lidstone_df = pd.DataFrame()
    epsilon = 0.1

    for i in range(len(query_data)):
        qid = query_data['qid'][i]
        query = query_data['query'][i]
        passages = candidate_passage[candidate_passage['qid'] == qid][['pid', 'passage']]
        #print(passages)
        data = []

        for j, (pid, passage) in passages.iterrows():
            score = 0
            D = len(passage)
            for term in query:
                m = 0
                if term in passage:
                    m = inverted_index[term][pid]
                s = math.log((m + epsilon)/(D + (epsilon * V)))
                score += s
            data.append([qid, pid, score])

        df1 = pd.DataFrame(data, columns = ['qid', 'pid', 'score']).sort_values(by = ['score'], ascending = False)
        lidstone_df = lidstone_df.append(df1.iloc[:100])
    
    return lidstone_df.reset_index(drop = True)

lidstone_data = lidstone_correction(test_queries, candidate_passages, inverted_index)
lidstone_data.to_csv('lidstone.csv', header = False, index = False)

# Dirichlet smoothing
def dirichlet_smoothing(query_data, candidate_passage, inverted_index):
    
    mu = 50
    dirichlet_df = pd.DataFrame()
    
    # Vocularity size
    V = len(inverted_index.keys())
    
    common_word_dict = {}
    for term, pid_count in inverted_index.items():
        term_count = sum(pid_count.values())
        common_word_dict[term] = term_count

    for i in range(len(query_data)):
        qid = query_data['qid'][i]
        query = query_data['query'][i]
        passages = candidate_passage[candidate_passage['qid'] == qid][['pid', 'passage']]
        data = []

        for j, p in passages.iterrows():
            pid = p['pid']
            passage = p['passage']
            
            # Passage length
            D = len(passage)
            score = 0
            
            for term in query:
                f = 0
                for t in passage:
                    if t == term:
                        f +=1
                        c = common_word_dict[t]
                    else: 
                        c = 0
                        
                # To avoid -inf values
                if f > 0:
                    s = math.log((D / (D + mu)) * (f / D) + (mu / (D + mu)) * (c / V))
                else:
                    s = math.log(mu / V)
                score += s
            data.append([qid, pid, score])
            
        df1 = pd.DataFrame(data, columns = ['qid', 'pid', 'score']).sort_values(by = ['score'], ascending = False)
        dirichlet_df = dirichlet_df.append(df1.iloc[:100])
       
    return dirichlet_df.reset_index(drop = True)

dirichlet_data = dirichlet_smoothing(test_queries, candidate_passages, inverted_index)
dirichlet_data.to_csv('dirichlet.csv', header = False, index = False)