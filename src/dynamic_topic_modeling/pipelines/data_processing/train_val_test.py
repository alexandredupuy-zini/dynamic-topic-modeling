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
    docs_ts = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize]] if w in word2id] for idx_d in range(tsSize)]
    timestamps_ts = pd.DataFrame(dataset['timeslice'][idx_permute[range(trSize,trSize+tsSize)]])
    docs_va = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize+tsSize]] if w in word2id] for idx_d in range(vaSize)]
    timestamps_va = pd.DataFrame(dataset['timeslice'][idx_permute[range(tsSize+trSize,num_docs)]])

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

    ### Batch & RNN input


    def get_batch(tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """fetch input data by batch."""
        batch_size = len(ind)
        data_batch = np.zeros((batch_size, vocab_size))
        if temporal:
            times_batch = np.zeros((batch_size, ))
        for i, doc_id in enumerate(ind):
            doc = np.array(tokens[doc_id])
            count = np.array(counts[doc_id])
            if temporal:
                timestamp = np.array(times['timeslice'])[doc_id]
                times_batch[i] = timestamp
            if len(doc) == 1: 
                doc = [doc.squeeze()]
                count = [count.squeeze()]
            else:
                doc = doc.squeeze()
                count = count.squeeze()
            if doc_id != -1:
                for j, word in enumerate(doc):
                    data_batch[i, word] = count[j]
        data_batch = torch.from_numpy(data_batch).float().to(device)
        if temporal:
            times_batch = torch.from_numpy(times_batch).to(device)
            return data_batch, times_batch
        return data_batch

    def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        indices = torch.randperm(num_docs)
        indices = torch.split(indices, 1000) 
        rnn_input = torch.zeros(num_times, vocab_size).to(device)
        cnt = torch.zeros(num_times, ).to(device)

        ## Loop over each batch. For rnn inp, we set the number of batch to a fixed size of 1000 as the authors code. We set this to a pretty big number to be sure that
        ## we our batch contains all time slices
        for idx, ind in enumerate(indices): 
            data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)

            ## Loop over the number of time slice
            for t in range(num_times):

                ## tmp represents the data indices where the time slice is equal to t
                tmp = (times_batch == t).nonzero() 

                ## docs is a tensor of shape [1,n_words] where each element is the total number of times a word is occurring over time slice t
                ## For example, if tensor[0,0] = 5, it means that the word at index 0 of vocabulary apperead 5 times in the first time slice. 
                ## we just set this condition so that a tensor of shape[1,n_words] does not sum over all it's elements.

                if data_batch[tmp].size()[0] == 1 :
                    docs=data_batch[tmp].squeeze()
                else : 
                    docs = data_batch[tmp].squeeze().sum(0)

                rnn_input[t] += docs

                ## cnt[t] is the number of documents in time slice t
                cnt[t] += len(tmp)

            if idx % 20 == 0:
                print('idx: {}/{}'.format(idx, len(indices)))
        ## The final rnn input is a tensor of shape [n_time_slice,n_words] where each element [i,j] represents the mean number of time the word j in the total number of 
        ## documents in time slices i. If tensor[0,0]=0.2, it means that the word at index 0 of vocabulary appears 0.2 times / documents attached to time slice t.
        rnn_input = rnn_input / cnt.unsqueeze(1)
        return rnn_input

    def split_bow(bow_in, n_docs):
        indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
        return indices, counts

    num_times=len(np.unique(timestamps_tr))
    vocab_size=len(vocab)

    train_tokens,train_counts = split_bow(bow_tr,n_docs_tr)
    valid_tokens,valid_counts=split_bow(bow_va,n_docs_va)
    test_tokens,test_counts=split_bow(bow_ts,n_docs_ts)
    test_1_tokens,test_1_counts=split_bow(bow_ts_h1,n_docs_ts_h1)
    test_2_tokens,test_2_counts=split_bow(bow_ts_h2,n_docs_ts_h2)

    print('Getting train RNN input ....')
    train_rnn_inp = get_rnn_input(
        train_tokens, train_counts, timestamps_tr, num_times, vocab_size, n_docs_tr)

    print('Getting val RNN input ....')
    valid_rnn_inp = get_rnn_input(
        valid_tokens, valid_counts, timestamps_va, num_times, vocab_size, n_docs_va)

    print('Getting test RNN input ....')
    test_rnn_inp = get_rnn_input(
        test_tokens, test_counts, timestamps_ts, num_times, vocab_size, n_docs_ts)

    print('Getting test half 1 RNN input ....')
    test_1_rnn_inp = get_rnn_input(
        test_1_tokens, test_1_counts, timestamps_ts, num_times, vocab_size, n_docs_ts_h1)

    print('Getting test half 2 RNN input ....')
    test_2_rnn_inp = get_rnn_input(
        test_2_tokens, test_2_counts, timestamps_ts, num_times, vocab_size, n_docs_ts_h2)

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
        train_rnn_inp=train_rnn_inp,
        test_rnn_inp=test_rnn_inp,
        test_1_rnn_inp=test_1_rnn_inp,
        test_2_rnn_inp=test_2_rnn_inp,
        valid_rnn_inp=valid_rnn_inp
        )


