from sklearn.manifold import TSNE
import torch
import numpy as np
import bokeh.plotting as bp
import scipy.io
import pandas as pd

from bokeh.plotting import save
from bokeh.models import HoverTool
import matplotlib.pyplot as plt
from collections import defaultdict
import fasttext
from sklearn.metrics.pairwise import cosine_similarity

def split_bow_2(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    indices_arr=[np.array(element) for element in indices]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    counts_arr=[np.array(element) for element in counts]
    return np.array(indices_arr), np.array(counts_arr)

def get_cos_sim_from_model(word,model,top_n=10) :
    cs=defaultdict()
    wv=model[word]
    if type(model)==fasttext.FastText._FastText :
        all_words=model.words
    else :
        all_words=list(model.wv.vocab.keys())
    for words in [i for i in all_words if i!=word] :
        curr_wv=model[words]
        cs[words]=cosine_similarity(wv.reshape(1,-1),curr_wv.reshape(1,-1)).flatten()[0]
    sorted_cs = dict(sorted(cs.items(), key=lambda kv: kv[1],reverse=True)[:top_n])
    return sorted_cs

def get_cos_sim_from_embedding(word,embedding,vocab,top_n=10) :
    cs=defaultdict()
    wv=embedding[vocab.token2id[word]]
    for words in [i for i in vocab.token2id if i!=word] :
        curr_wv=embedding[vocab.token2id[words]]
        cs[words]=cosine_similarity(wv.reshape(1,-1),curr_wv.reshape(1,-1)).flatten()[0]
    sorted_cs = dict(sorted(cs.items(), key=lambda kv: kv[1],reverse=True)[:top_n])
    return sorted_cs

def bow_to_dense_tensor(csr_bow) :
    csr_bow=csr_bow.tocoo()
    values = csr_bow.data
    indices = np.vstack((csr_bow.row, csr_bow.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = csr_bow.shape
    tensor=torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    return tensor

def nearest_neighbors(word, embeddings, vocab, num_words):
    vectors = embeddings.detach().numpy()
    index = vocab.index(word)
    query = embeddings[index].detach().numpy()
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def visualize(docs, _lda_keys, topics, theta):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    # project to 2D
    tsne_lda = tsne_model.fit_transform(theta)
    colormap = []
    for name, hex in matplotlib.colors.cnames.items():
        colormap.append(hex)

    colormap = colormap[:len(theta[0, :])]
    colormap = np.array(colormap)

    title = '20 newsgroups TE embedding V viz'
    num_example = len(docs)

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

    plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                 color=colormap[_lda_keys][:num_example])
    plt.show()

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

    if GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print('CUDA GPU enabled')

    else :
        print('default to CPU')
        device=torch.device("cpu")

    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 200)
    rnn_input = torch.zeros(num_times, vocab_size)
    cnt = torch.zeros(num_times, )

    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(torch.device('cpu'),tokens, counts, ind, vocab_size, temporal=True, times=times)
        ## Loop over the number of time slice
        for t in range(num_times):
            tmp = (times_batch == t).nonzero() ## tmp represents the data indices where the time slice is equal to t
            ## docs is a tensor of shape [1,n_words] where each element is the total number of times a word is occurring over time slice t
            ## For example, if tensor[0,0] = 5, it means that the word at index 0 of vocabulary apperead 5 times in the first time slice.
            ## we just set this condition so that a tensor of shape[1,n_words] does not sum over all it's elements.
            if data_batch[tmp].size()[0] == 1 :
                docs=data_batch[tmp].squeeze()
            else :
                docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp) ## cnt[t] is the number of documents in time slice t

    ## The final rnn input is a tensor of shape [n_time_slice,n_words] where each element [i,j] represents the mean number of time the word j in the total number of
    ## documents in time slices i. If tensor[0,0]=2, it means that the word at index 0 of vocabulary appears in 2 documents in time slice 0.
    for t in range(num_times) :
        if cnt[t] == 0 :
            cnt[t]=1
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input.to(device)

def predict(Topic_distribution : np.ndarray, dataset : pd.DataFrame, index_train : np.ndarray, index_test : np.ndarray, index_val : np.ndarray) :
    data=dataset.copy()
    indexes=np.hstack((index_train,index_test,index_val))
    data=data.iloc[indexes]
    topic_prediction=np.argmax(Topic_distribution,axis=1)
    data['predicted_topic']=topic_prediction
    data.reset_index(drop=True,inplace=True)
    n_topics=Topic_distribution.shape[1]
    proba_k=pd.DataFrame(Topic_distribution,columns=['proba_'+str(i) for i in range(n_topics)])
    final_data=pd.concat([data,proba_k], axis=1)
    final_data.sort_values('timestamp',inplace=True)
    return final_data
