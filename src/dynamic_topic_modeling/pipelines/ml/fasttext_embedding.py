import numpy as np
from sklearn import preprocessing
import fasttext
import torch

from collections import defaultdict
from .utils import get_cos_sim_from_model
from sklearn.model_selection import ParameterGrid

def get_word_embeddings(model,vocab,word_to_check) :
    #war, cold = model['war'].reshape((1, -1)), model['cold'].reshape((1, -1))
    #print('Cosine distance between "cold" and "war" in embedding space (gensim metric):', model.similarity('cold', 'war'))
    #print('Cosine distance between "cold" and "war" in embedding space (sklearn metric):', cosine_similarity(cold, war))
    print('Most similar (in term of cosine similarity) words to {} in word embedding :'.format(word_to_check))
    for key,value in get_cos_sim_from_model(word_to_check,model).items() :
        print(key, ' : ', value)

    words = list(vocab.token2id)

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

def grid_search(path_to_texts, param_grid, word_to_check) :

    params=ParameterGrid(param_grid)
    results=defaultdict(list)
    for ind in range(params.__len__()) :
        print('-'*100)
        print('Done {}/{}'.format(ind,params.__len__()))
        curr_params=params.__getitem__(ind)
        print('\tCurrent set of params : {}'.format(curr_params))
        model=fasttext.train_unsupervised(path_to_texts, dim=300,model='skipgram',epoch=curr_params['epoch'],ws=curr_params['ws'],minCount=curr_params['minCount'],thread=4,bucket=100000,verbose=0)
        results['ws'].append(curr_params['ws'])
        results['epoch'].append(curr_params['epoch'])
        results['minCount'].append(curr_params['minCount'])
        z=get_cos_sim_from_model(word_to_check,model)

        results['cos_sim'].append(z)
        print('\tNumber of words in vocab : {}'.format(len(model.words)))
        print('\tMost similar to "{}" : '.format(word_to_check))
        for key,value in z.items() :
            print('\t',key,' : ', value)
        del model

    return results
