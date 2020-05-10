
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io
from time import time
from .data import get_batch

from torch import nn, optim
from torch.nn import functional as F
from .detm_helpers import get_eta,get_beta,get_theta
from .detm import DETM
from .utils import nearest_neighbors



def get_val_completion_ppl(model, num_docs_valid, eval_batch_size, vocab_size, emb_size, 
                       valid_tokens, valid_counts, valid_times, valid_rnn_inp):
    """Returns val completion perplexity.
    """
    device=torch.device('cpu')

    model.eval()
    with torch.no_grad():

        alpha = model.mu_q_alpha
        indices = torch.split(torch.tensor(range(num_docs_valid)), eval_batch_size)
        tokens = valid_tokens
        counts = valid_counts
        times = valid_times
        eta = get_eta(model, valid_rnn_inp)
        acc_loss = 0
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch, times_batch = get_batch(device,tokens, counts, ind, vocab_size, emb_size, temporal=True, times=times)
            sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = data_batch / sums

            eta_td = eta[times_batch.type('torch.LongTensor')]
            theta = get_theta(model, eta_td, normalized_data_batch)

            alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]

            beta = get_beta(model,alpha_td)
            beta=beta.permute(1, 0, 2)
            loglik = theta.unsqueeze(2) * beta
            loglik = loglik.sum(1)
            loglik = torch.log(loglik)
            nll = -loglik * data_batch
            nll = nll.sum(-1)
            loss = nll / sums.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
            
        cur_loss = acc_loss / cnt
        ppl_all = round(math.exp(cur_loss))
        print('*'*100)
        print('VAL PPL: {}'.format(ppl_all))
        print('*'*100)
        return ppl_all


def get_test_completion_ppl(model, test_1_tokens, test_1_counts, test_2_tokens,test_2_counts, test_times, num_docs_test, eval_batch_size, vocab_size, emb_size,
                            test_1_rnn_inp, test_2_rnn_inp) : 

    device=torch.device('cpu')
    model.eval()
    with torch.no_grad() :

        alpha = model.mu_q_alpha
        indices = torch.split(torch.tensor(range(num_docs_test)), eval_batch_size)
        tokens_1 = test_1_tokens
        counts_1 = test_1_counts

        tokens_2 = test_2_tokens
        counts_2 = test_2_counts

        eta_1 = get_eta(model, test_1_rnn_inp)

        acc_loss = 0
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch_1, times_batch_1 = get_batch(
            device,tokens_1, counts_1, ind, vocab_size, emb_size, temporal=True, times=test_times)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            normalized_data_batch_1 = data_batch_1 / sums_1

            eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
            theta = get_theta(model, eta_td_1, normalized_data_batch_1)

            data_batch_2, times_batch_2 = get_batch(
                    device, tokens_2, counts_2, ind, vocab_size, emb_size, temporal=True, times=test_times)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)

            alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
            beta = get_beta(model,alpha_td).permute(1, 0, 2)

            loglik = theta.unsqueeze(2) * beta
            loglik = loglik.sum(1)
            loglik = torch.log(loglik)
            nll = -loglik * data_batch_2
            nll = nll.sum(-1)
            loss = nll / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('TEST Doc Completion PPL: {}'.format(ppl_dc))
        print('*'*100)
        return ppl_dc


def _diversity_helper(beta,num_topics, num_tops=25):
    list_w = np.zeros((num_topics, num_tops))
    for k in range(num_topics):
        top_words = (beta[k].argsort()[-num_tops:][::-1])
        list_w[k, :] = top_words[:num_tops]
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (num_topics * num_tops)
    return diversity

def diversity_by_topics(beta,num_times,num_tops=25) : 
    list_w = np.zeros((num_times, num_tops))
    for ts in range(num_times):
        top_words = (beta[ts].argsort()[-num_tops:][::-1])
        list_w[ts, :] = top_words[:num_tops]
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (num_times * num_tops)
    return diversity



def get_doc_freq(bow,wi,wj=None): 
    if wj is None : 
        return bow[:,wi].sum(axis=0)
    new=bow[:,wi]+bow[:,wj]
    return bow[:,wj].sum(axis=0),new[new==2].shape[0]

def get_one_hot_bow(bow) :
    """This function takes in input a basic BOW such as CountVecotrizer BOWs and return a one-hot BOW. This is a BOW where 
       each element (i,j) of the matrix is either a 0 if the word j is not in document i, and 1 if word j is in doc j.
       This differs from original BOW as in these standard BOW, each element (i,j) of the matrix is either 0 if the word j 
       is in document i or n_occu where n_occu is an integer that represents the number of times the word j appears in document i"""
    t0=time()
    
    bow_new=np.zeros((bow.shape[0],bow.shape[1]))
    for idx in range(bow.shape[0]) : 
        for i in np.argwhere(bow[idx]) :
            bow_new[idx,i[1]]=1
    return bow_new

def get_topic_coherence(data, beta, num_topics, num_coherence):

    D = len(data) ## number of docs...data is list of documents
    TC = []
   
    for k in range(num_topics):
        top_10 = list(beta[k].argsort()[-num_coherence:][::-1])
        #top_words = [vocab[a] for a in top_10]

            
        
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_doc_freq(data, word)
            p_wi=D_wi/D
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_doc_freq(data, word, top_10[j])
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

    #TC = np.mean(TC) / counter
    return TC

def model_topic_coherence(data,beta,num_times,num_topics,num_coherence=10) : 
    

    tc=np.zeros((num_times,num_topics))
    
    times=[]
    for timestep in range(num_times): 
        t0=time()

        if timestep%10 == 0 : 
            print('-'*100)
            print('Timestep {}/{}'.format(timestep,num_times))
            print('-'*100)
        tc[timestep,:]=get_topic_coherence(data,beta[:,timestep,:],num_topics,num_coherence)
        
        fitting_time=time()-t0                
        times.append(fitting_time)
    total_time=round(np.sum(times),2)
    print('Total fitting time for {} timesteps & {} max words is : {}s'.format(num_times,num_coherence,total_time))
    return tc

def get_topic_quality(model,beta,data,num_diversity=25,num_coherence=10):
    """Returns topic coherence and topic diversity.
    """

    num_times=model.num_times
    num_topics=model.num_topics

    print('#'*100)
    data=get_one_hot_bow(data)

    print('Getting topic diversity per times...')
    TD_all = np.zeros((num_times,))
    for tt in range(num_times):
        TD_all[tt] = _diversity_helper(beta[:, tt, :],num_topics, num_diversity)

    TD_times = np.mean(TD_all)
    print('Averaged Topic Diversity by times is : {}'.format(TD_times))

    print('Getting Topic Diversity by topics...')
    TD_topics= np.zeros((num_topics,))
    for k in range(num_topics) : 
        TD_topics[k]=diversity_by_topics(beta[k,:,:],num_times)

    TD_all_topics=np.mean(TD_topics)

    print('Averaged Topic Diversity by topic is : {}'.format(TD_all_topics))

    print('\n')
    print('#'*100)
    print('\n')

    print('Getting topic coherence...')
    tc=model_topic_coherence(data,beta,num_times,num_topics,num_coherence)
    overall_tc=0 
    for tt in range(num_times) : 
        overall_tc+=np.sum(tc[tt])/num_topics
    overall_tc=overall_tc/num_times

    print('Averaged Topic Coherence is : {}'.format(overall_tc))
    print('\n')
    quality = overall_tc * TD_times
    print('Topic Quality is: {}'.format(quality))
    print('#'*100)

    return TD_all,TD_times,TD_topics,TD_all_topics,tc,overall_tc,quality