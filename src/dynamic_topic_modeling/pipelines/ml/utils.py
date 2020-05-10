from sklearn.manifold import TSNE
import torch 
import numpy as np
import bokeh.plotting as bp

from bokeh.plotting import save
from bokeh.models import HoverTool
import matplotlib.pyplot as plt 
import matplotlib 
from collections import defaultdict
import fasttext
from sklearn.metrics.pairwise import cosine_similarity

tiny = 1e-6

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

def _reparameterize(mu, logvar, num_samples):
    """Applies the reparameterization trick to return samples from a given q"""
    std = torch.exp(0.5 * logvar) 
    bsz, zdim = logvar.size()
    eps = torch.randn(num_samples, bsz, zdim).to(mu.device)
    mu = mu.unsqueeze(0)
    std = std.unsqueeze(0)
    res = eps.mul_(std).add_(mu)
    return res


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
