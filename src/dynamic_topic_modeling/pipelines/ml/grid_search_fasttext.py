
import pandas as pd
import fasttext
import re
import unidecode
from collections import defaultdict
from sklearn.model_selection import ParameterGrid
from .utils import get_cos_sim_from_model

def grid_search(path_to_texts,param_grid) : 

    params=ParameterGrid(param_grid)
    results=defaultdict(list)
    for ind in range(params.__len__()) :  
        print('-'*100)
        print('Done {}/{}'.format(ind,params.__len__()))
        curr_params=params.__getitem__(ind)   
        print('\tCurrent set of params : {}'.format(curr_params))
        model=fasttext.train_unsupervised('data\\05_model_input\\texts_used_for_embeddings.txt', dim=300,model='skipgram',epoch=curr_params['epoch'],ws=curr_params['ws'],minCount=curr_params['minCount'],thread=4,bucket=100000,verbose=0)
        results['ws'].append(curr_params['ws'])
        results['epoch'].append(curr_params['epoch'])
        results['minCount'].append(curr_params['minCount'])
        z=get_cos_sim_from_model('contrat',model) 
       
        results['cos_sim'].append(z)
        print('\tNumber of words in vocab : {}'.format(len(model.words)))
        print('\tMost similar to "contrat" : ')
        for key,value in z.items() : 
            print('\t',key,' : ', value)
        del model

    return results