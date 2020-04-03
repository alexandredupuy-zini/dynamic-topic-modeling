import numpy as np
import matplotlib.pyplot as plt
from gensim import matutils
import os
from sklearn.manifold import MDS

from gensim.models import LdaModel

import torch
from torch import optim

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from gensim.models import Word2Vec
#import gensim.downloader as api

from .utils import split_bow
from .metrics import extract_top_n_words, calculate_topic_distances, show_most_m_represantative_docs
from .metrics import topic_coherence, topic_diversity
from .models.etm import ETM
from .models.etm_helpers import train, get_batch


def cluster_word_embeddings(vocab, train_docs, num_topics):
    # Import word2vec model
    #w2v = api.load("glove-twitter-25")
    w2v = Word2Vec.load('./data/04_features/word2vec_model.bin')

    # Get (normalized) word embeddings
    words = list(vocab.token2id)
    words = [w for w in words if w in w2v.wv.vocab]

    embeddings = np.array([w2v[w] for w in words])
    embeddings_norm = preprocessing.normalize(embeddings)

    dict = {w:emb for w,emb in zip(words, embeddings_norm)}

    # Get word weights
    count_dict = {w:0 for w in words}
    for doc in train_docs:
        for w in doc:
            count_dict[w] += 1
    weights = np.array([count_dict[w] for w in words])

    # Train model
    #model = KMeans(n_clusters=num_topics).fit(embeddings_norm)
    model = KMeans(n_clusters=num_topics).fit(embeddings_norm, sample_weight=weights)

    # Get beta
    vocab_size = len(vocab)
    beta = np.zeros((num_topics, vocab_size))

    #preds = model.predict(embeddings_norm)
    preds = model.predict(embeddings_norm, sample_weight=weights)
    preds = preds.reshape((-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore').fit(preds)

    similarities = np.zeros((num_topics, vocab_size))

    wvs = np.array([dict[w] for w in vocab.token2id])
    wvs = wvs.reshape((vocab_size, -1))
    for k in range(num_topics):
        cluster_center = model.cluster_centers_[k].reshape((1, -1))
        similarities[k] = cosine_similarity(cluster_center, wvs)

    beta = enc.transform(preds).toarray().T
    beta = beta * similarities
    beta = beta / beta.sum(axis=1, keepdims=True)

    # Get theta
    theta = []
    for doc in train_docs:
        word_vectors = np.array([dict[w] for w in doc])
        preds = model.predict(word_vectors)
        tmp = enc.transform(preds.reshape((-1, 1))).toarray().mean(axis=0)
        theta.append(tmp)
    theta = np.array(theta)

    return model, beta, theta


def cluster_doc_embeddings():
    pass


def train_model_lda(train_corpus, train_docs,
                    val_corpus, val_docs,
                    dictionary,
                    num_topics=10,
                    epochs=2):
    #filepath = './data/06_models/trained_model_lda.model'
    #if os.path.isfile(filepath):
    #    return LdaModel.load(filepath)

    # Set training parameters.
    chunksize = 2000
    passes = epochs
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    model = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    beta = model.get_topics()
    theta = matutils.corpus2dense(model.get_document_topics(train_corpus), num_topics).T

    return model, beta, theta


def train_model_etm(train_sparse, train_docs,
                    val_sparse, val_docs,
                    vocab, docs,
                    dict_embeddings=None, dict_embeddings_norm=None,
                    num_topics=10, epochs=100,
                    flag_pretrained_embeddings=True, flag_finetune_embeddings=False,
                    embedding_size=300,
                    batch_size=1000, lr=5e-3, wdecay=1.2e-6):

    # Get data
    vocab_size = len(vocab)
    num_docs_train = len(train_docs)
    num_docs_val = len(val_docs)
    train_tokens, train_counts = split_bow(train_sparse, num_docs_train)
    val_tokens, val_counts = split_bow(train_sparse, num_docs_val)

    del train_sparse
    del train_docs
    del val_sparse
    del val_docs

    # Get embeddings
    embeddings = None
    if flag_pretrained_embeddings:
        #embeddings = np.zeros((vocab_size, embedding_size))
        embeddings = []
        for i, word in enumerate(vocab.token2id):
            #embeddings[i] = dict_embeddings_norm[word]
            if word in dict_embeddings_norm:
                embeddings.append(dict_embeddings_norm[word])
            else:
                embeddings.append(np.random.normal(0., 0.01, (300,)))
        embeddings = np.array(embeddings)

    # Get model
    model = ETM(num_topics=num_topics, vocab_size=vocab_size, embedding_size=embedding_size,
                flag_finetune_embeddings=flag_finetune_embeddings, embeddings=embeddings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)

    # Train model
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    train(model, optimizer, num_docs_train, train_tokens, train_counts, vocab_size, device, epochs, batch_size)

    # Get results
    beta = model.get_beta().cpu().detach().numpy()

    data_batch = get_batch(train_tokens, train_counts, torch.randperm(10000), vocab_size, device)
    sums = data_batch.sum(1).unsqueeze(1)
    norm_corpus = data_batch / sums
    mu_theta, logsigma_theta = model.encode(norm_corpus)
    theta = mu_theta.cpu().detach().numpy()

    return model, beta, theta


def eval_model(topic_distributions, doc_topic_matrix, dictionary, train_docs,
               num_topics, top_n_show=10, m_most=1, top_n_coherence=10, top_n_diversity=25):

    # Metrics

    ## Qualitative Metrics

    ### Top-n words per topic
    for topic_id in range(num_topics):
        top_n_words, top_n_freqs = extract_top_n_words(topic_distributions, dictionary, topic_id, top_n_show)

        plt.figure(figsize=(10, 4))
        plt.title('Topic number ' + str(topic_id))
        plt.bar(top_n_words, top_n_freqs)
        plt.show()

    ### Topic sizes
    #

    ### Topic distances
    distances = calculate_topic_distances(topic_distributions)
    plt.figure(figsize=(6, 6))
    plt.imshow(distances, cmap='Greys')
    plt.show()

    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(distances).embedding_
    plt.figure(figsize=(6, 6))
    plt.scatter(pos[:, 0], pos[:, 1], s=100, lw=0)
    plt.show()

    ### Most m representative documents
    for topic_id in range(num_topics):
        print('Topic number ', topic_id, show_most_m_represantative_docs(doc_topic_matrix, m_most, train_docs, topic_id))


    ## Quantitative Metrics

    ### Topic Coherence
    mean_tc = 0.
    for topic_id in range(num_topics):
        tc = topic_coherence(topic_distributions, dictionary, topic_id, top_n_coherence, train_docs, method='npmi')
        print('Topic number ', topic_id, 'tc:', tc)
        mean_tc += tc
    print('Topic coherence:', mean_tc / num_topics)

    ### Topic Diversity
    print('Topic diversity', topic_diversity(topic_distributions, dictionary, top_n_diversity))

    ### Predictive Quality
    #
