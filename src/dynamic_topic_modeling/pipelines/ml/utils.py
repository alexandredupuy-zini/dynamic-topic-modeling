from sklearn.manifold import TSNE
import torch 
import numpy as np
import bokeh.plotting as bp

from bokeh.plotting import save
from bokeh.models import HoverTool
import matplotlib.pyplot as plt 
import matplotlib 

tiny = 1e-6

def split_bow_2(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    indices_arr=[np.array(element) for element in indices]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    counts_arr=[np.array(element) for element in counts]

    return np.array(indices_arr), np.array(counts_arr)

def bow_to_dense_tensor(csr_bow) :
    csr_bow=csr_bow.tocoo()
    values = csr_bow.data
    indices = np.vstack((csr_bow.row, csr_bow.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = csr_bow.shape

    tensor=torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    
    return tensor

def _reparameterize(mu, logvar, num_samples):
    """Applies the reparameterization trick to return samples from a given q"""
    std = torch.exp(0.5 * logvar) 
    bsz, zdim = logvar.size()
    eps = torch.randn(num_samples, bsz, zdim).to(mu.device)
    mu = mu.unsqueeze(0)
    std = std.unsqueeze(0)
    res = eps.mul_(std).add_(mu)
    return res

def get_document_frequency(data, wi, wj=None):

    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if wi in doc:
                D_wi += 1
        return D_wi

    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(data, beta, num_topics, num_coherence):

    D = len(data) ## number of docs...data is list of documents
    TC = []
   
    for k in range(num_topics):
        print('\tDone {}/{}'.format(k,num_topics))
        top_10 = list(beta[k].argsort()[-num_coherence:][::-1])
        #top_words = [vocab[a] for a in top_10]

            
        
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            p_wi=D_wi/D
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                p_wj=D_wj/D
                p_wi_wj=D_wi_wj/D

                if D_wi_wj == 0 :
                    tc_pairwise=-1
                elif D_wi_wj==D_wi and D_wi_wj==D_wj : 
                    tc_pairwise=1
                # get f(w_i, w_j)
                else : 
                    tc_pairwise = np.log(p_wi_wj/(p_wi*p_wj))/-np.log(p_wi_wj)

                # update tmp: 

                tmp += tc_pairwise
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC_k=TC_k/counter
        TC.append(TC_k)
    print(TC)    
    #TC = np.mean(TC) / counter
    return TC

def model_topic_coherence(data,beta,num_times,num_topics,num_coherence=6) : 
    

    tc=np.zeros((num_times,num_topics))
    
    for timestep in range(num_times): 
        print('-'*100)
        print('Timestep {}/{}'.format(timestep,num_times))
        print('-'*100)
        print('\n')
        tc[timestep,:]=get_topic_coherence(data,beta[:,timestep,:],num_topics,num_coherence)
        
    return tc

def log_gaussian(z, mu=None, logvar=None):
    sz = z.size()
    d = z.size(2)
    bsz = z.size(1)
    if mu is None or logvar is None:
        mu = torch.zeros(bsz, d).to(z.device)
        logvar = torch.zeros(bsz, d).to(z.device)
    mu = mu.unsqueeze(0)
    logvar = logvar.unsqueeze(0)
    var = logvar.exp()
    log_density = ((z - mu)**2 / (var+tiny)).sum(2) # b
    log_det = logvar.sum(2) # b
    log_density = log_density + log_det + d*np.log(2*np.pi)
    return -0.5*log_density

def logsumexp(x, dim=0):
    d = torch.max(x, dim)[0]   
    if x.dim() == 1:
        return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
        return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim) + tiny) + d

def flatten_docs(docs): #to get words and doc_indices
    words = [x for y in docs for x in y]
    doc_indices = [[j for _ in doc] for j, doc in enumerate(docs)]
    doc_indices = [x for y in doc_indices for x in y]
    return words, doc_indices
    
def onehot(data, min_length):
    return list(np.bincount(data, minlength=min_length))

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
