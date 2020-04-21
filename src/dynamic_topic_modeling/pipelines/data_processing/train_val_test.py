from typing import Any, Dict

import os
import random
import pickle
import numpy as np
import torch 
import scipy.io
import pandas as pd 
import ast
from gensim.corpora import Dictionary,MmCorpus
def train_val_test(dataset : pd.DataFrame,dictionary : Dictionary , corpus : MmCorpus, 
                   test_size: float , val_size : float) -> Dict[str,Any] :
    
    # Make train val test index  

    num_docs = len(corpus)
    vaSize = int(np.floor(val_size*num_docs))
    tsSize = int(np.floor(test_size*num_docs))
    trSize = int(num_docs - vaSize - tsSize)
    idx_permute = np.random.permutation(num_docs).astype(int)
    print('Reading data....')
    # Make sure our text column is of type list 
    dataset['text']=dataset['text'].apply(lambda x: ast.literal_eval(x))

    word2id = dict([(w, j) for j, w in dictionary.items()])
    id2word = dict([(j, w) for j, w in dictionary.items()])

    # Remove words not in train_data
    print('Starting vocabulary : {}'.format(len(dictionary)))
    print('Removing words not in train data .....')
    vocab = list(set([w for idx_d in range(trSize) for w in dataset['text'][idx_permute[idx_d]] if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  New vocabulary after removing words not in train: {}'.format(len(vocab)))

    docs_tr = [[word2id[w] for w in dataset['text'][idx_permute[idx_d]] if w in word2id] for idx_d in range(trSize)]
    timestamps_tr = pd.DataFrame(dataset['timeslice'][idx_permute[range(trSize)]])
    idx_tr = idx_permute[range(trSize)]

    docs_ts = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize]] if w in word2id] for idx_d in range(tsSize)]
    timestamps_ts = pd.DataFrame(dataset['timeslice'][idx_permute[range(trSize,trSize+tsSize)]])
    idx_ts = idx_permute[range(trSize,trSize+tsSize)]

    docs_va = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize+tsSize]] if w in word2id] for idx_d in range(vaSize)]
    timestamps_va = pd.DataFrame(dataset['timeslice'][idx_permute[range(tsSize+trSize,num_docs)]])
    idx_va=idx_permute[range(tsSize+trSize,num_docs)]

    print('  Number of documents in train set : {} [this should be equal to {} and {}]'.format(len(docs_tr), trSize, len(timestamps_tr)))
    print('  Number of documents in test set : {} [this should be equal to {} and {}]'.format(len(docs_ts), tsSize, len(timestamps_ts)))
    print('  Number of documents in validation set: {} [this should be equal to {} and {}]'.format(len(docs_va), vaSize, len(timestamps_va)))

    
        # Split test set in 2 halves, the first containing the first half of the words in documents, and second part the second
        # half of words in documents. Will be use to gather test completion perplexity.

    print('Splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

    print('Creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    print('  Total number of words used in train set : ', len(words_tr))
    print('  Total number of words used in test set : ', len(words_ts))
    print('  Total number of words used in test firt set (first half of documents words): ', len(words_ts_h1))
    print('  Total number of words used in test firt set (first half of documents words): ', len(words_ts_h2))
    print('  Total number of words used in val set : ', len(words_va))

    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)


    # Get doc indices
    print('Getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)


    print('Creating bow representation...')

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return scipy.sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))



    print(' Train bag of words shape : {}'.format(bow_tr.shape))
    print(' Test bag of words shape : {}'.format(bow_ts.shape))
    print(' Test set 1 bag of words shape : {}'.format(bow_ts_h1.shape))
    print(' Test set 2 bag of words shape : {}'.format(bow_ts_h2.shape))
    print(' Val bag of words shape : {}'.format(bow_va.shape))




    print('Done splitting data.')

    return dict(
        BOW_train=bow_tr,
        BOW_test=bow_ts,
        BOW_test_h1=bow_ts_h1,
        BOW_test_h2=bow_ts_h2,
        BOW_val=bow_va,
        timestamps_train=timestamps_tr,
        timestamps_test=timestamps_ts,
        timestamps_val=timestamps_va,
        train_vocab_size=len(vocab),
        train_num_times=len(np.unique(timestamps_tr['timeslice'])),
        idx_train=idx_tr,
        idx_test=idx_ts,
        idx_val=idx_va
        )


