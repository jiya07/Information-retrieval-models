import pandas as pd
from task1 import preprocess_text


candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep = '\t', header= None, names = ['qid', 'pid', 'query', 'passage'])

# Dropping duplicate values
candidate_passages_distinct = candidate_passages.drop_duplicates(subset = 'pid',keep = 'first',inplace = False).reset_index(drop=True)
passages = preprocess_text(candidate_passages_distinct['passage'], remove_stopwords = True)


def inverted_index(pids, passages):
    inv_indx = {}
    
    for i in range(len(passages)):
        p = pids[i]
        passage = passages[i]

        for term in passage:
            count = passage.count(term)

            if term not in inv_indx.keys():
                inv_indx[term] = {p : count}
            elif term in inv_indx.keys():
                inv_indx[term].update({p : count})
                
    return inv_indx

# Calculating inverted index of passages
inverted_indx = inverted_index(candidate_passages_distinct['pid'], passages)
    