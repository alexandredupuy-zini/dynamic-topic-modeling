import os
import random
import pickle
import numpy as np
import torch
import scipy.io
import requests
from io import BytesIO
from zipfile import ZipFile

def get_batch(device,tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):

    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    if temporal:
        times_batch = np.zeros((batch_size, ))
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
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

def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs, GPU):

    if GPU and  torch.cuda.is_available():
        device = torch.device("cuda")
        print('CUDA GPU enabled')

    else :
        print('default to CPU')
        device=torch.device("cpu")

    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 200)
    rnn_input = torch.zeros(num_times, vocab_size)
    cnt = torch.zeros(num_times, )

    ## Loop over each batch. For rnn inp, we set the number of batch to a fixed size of 1000 as the authors code. We set this to a pretty big number to be sure that
    ## we our batch contains all time slices
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(torch.device('cpu'),tokens, counts, ind, vocab_size, temporal=True, times=times)

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

        ## The final rnn input is a tensor of shape [n_time_slice,n_words] where each element [i,j] represents the mean number of time the word j in the total number of
        ## documents in time slices i. If tensor[0,0]=0.2, it means that the word at index 0 of vocabulary appears in 20% of documents at time slice i.
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input.to(device)
