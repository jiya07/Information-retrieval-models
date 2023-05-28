import pandas as pd
import numpy as np
import math
from task2 import inverted_indx
from task1 import preprocess_text
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# Importing queries data and preprocessing the queries
test_queries = pd.read_csv('test-queries.tsv', sep = '\t', header = None, names = ['qid', 'query'])

# Importing candidate passages and preprocessing the passages
candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep = '\t', header= None, names = ['qid', 'pid', 'query', 'passage'])
passage_terms = preprocess_text(candidate_passages['passage'], remove_stopwords = True)
candidate_passages['passage'] = passage_terms

# Importing inverted_index from task2
inverted_index = inverted_indx


# Calculating TF-IDF of passages
def tf_idf_passage(inverted_index):
    
    # Term frequency inverse document frequency dictionary
    tf_idf = {}
    
    # Term frequency dictionary 
    tf = {}
    
    # Inverse document frequency dictionary
    idf = {}
    
    # Total number of passages
    N = len(inverted_index.keys())
    
    for term, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            tf[pid] = {}
            idf[pid] = {}
            tf_idf[pid] = {}
    
    for term, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            
            # Passages in which the term occurs
            n = len(inverted_index[term])
            
            # Frequency of term in the passage
            tf[pid][term] = count
            
            # Inverse document frequency of the term 
            idf[pid][term] = math.log10(N / n)
            
            # TF-IDF 
            tf_idf[pid][term] = tf[pid][term] * idf[pid][term]

    return tf, idf, tf_idf

tf_passages, idf_passages, tf_idf_passages = tf_idf_passage(inverted_index)


# Calculating TF-IDF of queries using IDF of passages
def tf_idf_query(query_data, idf_passage):
    
    tf_queries = {}
    idf_queries = {}
    tf_idf_queries = {}
    
    qids = query_data['qid']
    queries = query_data['query']
    for qid in qids:
        tf_queries[qid] = {}
        idf_queries[qid] = {}
        tf_idf_queries[qid] = {}
        
    # Term frequency in each query
    for i in range(len(queries)):
        terms = dict(Counter(queries[i]))
        qid = qids[i]
        #print(terms)
        tf_queries[qid] = terms
        
    # Inverse document frequency of terms from passage
    idf_term_passage = {}
    
    for pid, term_idf in idf_passage.items():
        for term, idf in term_idf.items():
            if term not in idf_term_passage.keys():
                idf_term_passage[term] = idf
    
    for qid, term_count in tf_queries.items():
        for term in term_count.keys():
            if term in idf_term_passage.keys():
                idf_queries[qid][term] = idf_term_passage[term]
            else:
                idf_queries[qid][term] = 0
            # TF-IDF
            tf_idf_queries[qid][term] = tf_queries[qid][term] * idf_queries[qid][term]
    
    return tf_queries, idf_queries, tf_idf_queries

queries = preprocess_text(test_queries['query'], remove_stopwords = True)
test_queries['query'] = queries

tf_queries, idf_queries, tf_idf_queries = tf_idf_query(test_queries, idf_passages)

# Cosine similarity 
def calc_cosine_similarity(query_data, candidate_passage, tf_idf_queries, tf_idf_passages):
    
    df_cosine_sim = pd.DataFrame()
    qids = query_data['qid']
    for i in range(len(qids)):
        qid = qids[i]
        pids = candidate_passage[candidate_passage['qid'] == qid][['pid', 'passage']]
        
        data = []
        for i,p in pids.iterrows():
            pid = p['pid']
            query_vector = tf_idf_queries[qid]
            passage_vector = tf_idf_passages[pid]
            
            inner_product = 0
            common_terms = set(query_vector.keys()).intersection(set(passage_vector.keys()))
            for term in common_terms:
                inner_product += query_vector[term] * passage_vector[term]
        
            norm1 = np.linalg.norm(list(query_vector.values())) 
            norm2 = np.linalg.norm(list(passage_vector.values()))
            
            if norm1 == 0 and norm2 == 0:
                cosine_sim = 0
            else:
                cosine_sim = inner_product / (norm1*norm2)
                
            data.append([qid, pid, cosine_sim])
        
        df1 = pd.DataFrame(data, columns = ['qid', 'pid', 'score']).sort_values(by = ['score'], ascending = False)
        df_cosine_sim = df_cosine_sim.append(df1.iloc[:100])
    
    return df_cosine_sim.reset_index(drop = True)

df_cosine_sim = calc_cosine_similarity(test_queries, candidate_passages, tf_idf_queries, tf_idf_passages)
df_cosine_sim.to_csv('tfidf.csv', header = False, index = False)


# Calculating average passage length and number of passages
passage_lengths = [len(x) for x in candidate_passages['passage']]
avdl = sum(passage_lengths)/len(passage_lengths)
N = candidate_passages.shape[0]

# Implementing BM25 using inverted index
def calc_bm25(query_data, candidate_passage, inverted_index, avg_len, N):
    
    k1=1.2
    k2=100
    b=0.75
    df_bm25 = pd.DataFrame()
    qids = query_data['qid']
    queries = query_data['query']
    for i in range(len(qids)):
        qid = qids[i]
        passages = candidate_passage[candidate_passage['qid'] == qid][['pid', 'passage']]
        
        term_count = Counter(queries[i])
        terms = term_count.most_common(len(term_count))
        
        data = []
        for j, (pid, passage) in passages.iterrows():
            #print(pid)
            d_len = len(passage)
            K = k1*((1-b)+(b*d_len/avg_len))
            score = 0

            for i in range(len(terms)):
                term = terms[i][0]
                qf = terms[i][1]
                if term in passage:
                    f = inverted_index[term][pid]
                    ni = len(inverted_index[term])
                else:
                    f = 0
                    ni = 0
                score += math.log(((0 + 0.5)/(0 - 0 + 0.5)) / ((ni - 0 + 0.5)/(N - ni - 0))) * ( k1 + 1) * f * (k2 + 1) * qf / ((K + f) * (k2 + qf))
                
            data.append([qid, pid, score])

        df1 = pd.DataFrame(data, columns = ['qid', 'pid', 'score']).sort_values(by = ['score'], ascending = False)
        df_bm25 = df_bm25.append(df1.iloc[:100])
    
    return df_bm25.reset_index(drop = True)

df_bm25 = calc_bm25(test_queries, candidate_passages, inverted_index, avdl, N)
df_bm25.to_csv('bm25.csv', header = False, index = False)