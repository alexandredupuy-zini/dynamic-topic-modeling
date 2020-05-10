import numpy as np
from sklearn import preprocessing
import fasttext
import torch

from collections import defaultdict
from .utils import get_cos_sim_from_model


def get_word_embeddings(model,vocab) : 

    #war, cold = model['war'].reshape((1, -1)), model['cold'].reshape((1, -1))
    #print('Cosine distance between "cold" and "war" in embedding space (gensim metric):', model.similarity('cold', 'war'))
    #print('Cosine distance between "cold" and "war" in embedding space (sklearn metric):', cosine_similarity(cold, war))
    print('Most similar (in term of cosine similarity) words to "contrat" in word embedding :')
    for key,value in get_cos_sim_from_model('contrat',model).items() : 
        print(key, ' : ', value)
    
    words = list(vocab.token2id)
    #words = [w for w in words if w in model.words]

    embeddings = np.array([model[w] for w in words])
    embeddings_norm = preprocessing.normalize(embeddings)

    embeddings=torch.from_numpy(embeddings)
    embeddings_norm=torch.from_numpy(embeddings_norm)


    return embeddings, embeddings_norm

def train_fasttext_embeddings(path_to_text_data, vocab, dim : int, window:int, min_count :int, model : str , epoch :int, thread = 4):

    # Get data

    print('Number of unique words:', len(vocab))
    print('')
    print('Training embedding on {} epochs'.format(epoch))
    print('Current parameters : ')
    print('\twindow : {}'.format(window))
    print('\tembedding dim : {}'.format(dim))
    print('\tminimum word count : {}'.format(min_count))
    print('\ttype of model : {}'.format(model))
    # Train model


    model = fasttext.train_unsupervised(path_to_text_data ,model=model, dim=dim, ws=window, minCount=min_count, thread=thread, epoch=epoch, bucket=100000,verbose=1)

    # Get embedding dict
    embeddings, embeddings_norm = get_word_embeddings(model, vocab)

    return model, embeddings_norm