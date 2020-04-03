from kedro.pipeline import Pipeline, node

from .embeddings import train_word_embeddings, train_doc_embeddings
from .nodes import cluster_word_embeddings, cluster_doc_embeddings
from .nodes import train_model_lda, train_model_etm
from .nodes import eval_model

# Word2Vec
train_word_embeddings_node = node(
    func=train_word_embeddings,
    inputs=['raw_dataset', 'dictionary'],
    outputs=['word2vec_model', 'dict_embeddings', 'dict_embeddings_norm'],
    name='Train word embeddings')

# Doc2Vec
train_doc_embeddings_node = node(
    func=train_doc_embeddings,
    inputs=['raw_dataset'],
    outputs=['doc2vec_model'],
    name='Train doc embeddings')

# Clustering word embeddings
cluster_word_embeddings_node = node(
    func=cluster_word_embeddings,
    inputs=['dictionary', 'train_docs', 'params:num_topics'],
    outputs=['trained_model_kmeans_words', 'beta', 'theta'],
    name='Cluster word embeddings')

# Clustering doc embeddings
#cluster_doc_embeddings_node = node(
#    func=cluster_doc_embeddings,
#    inputs='raw_dataset',
#    outputs='word2vec_model',
#    name='Cluster doc embeddings')

# LDA
train_model_lda_node = node(
    func=train_model_lda,
    inputs=['train_corpus', 'train_docs',
            'val_corpus', 'val_docs',
            'dictionary',
            'params:num_topics', 'params:epochs'],
    outputs=['trained_model_lda', 'beta', 'theta'],
    name='Train model')

# ETM
train_model_etm_node = node(
    func=train_model_etm,
    inputs=['train_sparse', 'train_docs',
            'val_sparse', 'val_docs',
            'dictionary', 'raw_dataset',
            'dict_embeddings', 'dict_embeddings_norm',
            'params:num_topics', 'params:epochs',
            'params:flag_pretrained_embeddings', 'params:flag_finetune_embeddings',
            'params:embedding_size'],
    outputs=['trained_model_etm', 'beta', 'theta'],
    name='Train model')

# Eval metrics
eval_model_node = node(
    func=eval_model,
    inputs=["beta", "theta", "dictionary", "train_docs",
            "params:num_topics", "params:top_n_show", "params:m_most",
            "params:top_n_coherence", "params:top_n_diversity"],
    outputs=None,
    name='Evaluate model')

def create_pipeline(**kwargs) :
    return Pipeline(
    [
        #train_word_embeddings_node,
        #train_doc_embeddings_node,

        cluster_word_embeddings_node,
        #cluster_doc_embeddings,
        #train_model_lda_node,
        #train_model_etm_node,

        eval_model_node
    ], tags="Machine Learning")


# TODO

# Pre-processing
## Stronger pre-processing: only keep nouns/adjs/verbs

# Word embeddings
## Test pre-trained word embeddings (en and fr)
## Better word embeddings: fastText, elmo, bert ?

# Models
## Improve LDA with HDP (no pre-defined nb of topics)
## Hierarchical Clustering (no pre-defined nb of topics)
## From word emb to doc dist: word movers' distance

# Eval
## Visualization/Eval tools for word embeddings

# Dynamic approaches
## d-LDA
## d-ETM
## sliding window
## dynamic word embeddings
## sequential k-means
## dynamic vis/eval tools
