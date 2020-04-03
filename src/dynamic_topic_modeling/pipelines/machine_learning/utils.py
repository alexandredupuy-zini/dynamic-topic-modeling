import numpy as np

def split_bow(bow_in, n_docs):
    indices = [np.array([w for w in bow_in[doc,:].indices], ndmin=2) for doc in range(n_docs)]
    counts = [np.array([c for c in bow_in[doc,:].data], ndmin=2) for doc in range(n_docs)]
    return indices, counts
