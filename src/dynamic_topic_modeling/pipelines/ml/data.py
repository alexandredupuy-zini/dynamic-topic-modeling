import os
import random
import pickle
import numpy as np
import torch 
import scipy.io
import requests 
from io import BytesIO
from zipfile import ZipFile

def get_batch(tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        L = count.shape[1]
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
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)
        for t in range(num_times):
            tmp = (times_batch == t).nonzero()
            docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp)
        if idx % 20 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input

