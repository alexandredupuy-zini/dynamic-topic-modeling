from __future__ import print_function

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

from .data import get_batch

from torch import nn, optim
from torch.nn import functional as F

from .detm import DETM
from .utils import nearest_neighbors, model_topic_coherence





def train(model,epoch,optimizer,num_docs_train,batch_size,vocab_size,emb_size,log_interval, clip,
          train_rnn_inp,train_tokens,train_counts,train_times):
    """Train DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    cnt = 0
    device=model.device


    indices = torch.randperm(num_docs_train)
    indices = torch.split(indices, batch_size) 
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = get_batch(
            device, train_tokens, train_counts, ind, vocab_size, emb_size, temporal=True, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)
        normalized_data_batch = data_batch / sums

        loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, num_docs_train)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        cnt += 1

        if idx % log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_nll = round(acc_nll / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
    lr = optimizer.param_groups[0]['lr']
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    print('*'*100)
    
#def visualize():
#    """Visualizes topics and embeddings and word usage evolution.
#    """
#    model.eval()
#    with torch.no_grad():
#        alpha = model.mu_q_alpha
#        beta = model.get_beta(alpha) 
#        print('beta: ', beta.size())
#        print('\n')
#        print('#'*100)
#        print('Visualize topics...')
#        times = [0,int(beta.size()[1]/2),beta.size()[1]-1]
#        topics_words = []
#        for k in range(args.num_topics):
#            for t in times:
#                gamma = beta[k, t, :]
#                top_words = list(gamma.detach().numpy().argsort()[-args.num_words+1:][::-1])
#                topic_words = [vocab[a] for a in top_words]
#                topics_words.append(' '.join(topic_words))
#                print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 
#
#       print('\n')
#        print('Visualize word embeddings ...')
#        queries = ['economic', 'assembly', 'security', 'management', 'rights',  'africa']
#        try:
#            embeddings = model.rho.weight  # Vocab_size x E
#        except:
#            embeddings = model.rho         # Vocab_size x E
#        neighbors = []
#        for word in queries:
#            try : 
#                print('word: {} .. neighbors: {}'.format(
#                    word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
#            except : 
#                print('{} not found in dictionary'.format(word))
##        print('#'*100)
#

def get_eta(model, rnn_inp):
    device=torch.device('cpu')
    model.eval()
    with torch.no_grad() : 
        inp = model.q_eta_map.cpu()(rnn_inp.cpu()).unsqueeze(1)
        hidden = tuple([i.to(device) for i in model.init_hidden()])
        output, _ = model.q_eta.cpu()(inp, hidden)
        output = output.squeeze()
        etas = torch.zeros(model.num_times, model.num_topics).cpu()
        inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
        etas[0] = model.mu_q_eta.cpu()(inp_0)
        for t in range(1, model.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            etas[t] = model.mu_q_eta.cpu()(inp_t)
        return etas


def get_theta(model, eta, bows):
    device=torch.device('cpu')
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta.cpu()(inp)
        mu_theta = model.mu_q_theta.cpu()(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    

def get_beta(model, alpha):
    """Returns the topic matrix \beta of shape K x V
    """
    model.eval()
    with torch.no_grad() : 
        alphas=alpha.view(alpha.size(0)*alpha.size(1),model.rho_size).cpu()
        rho=model.rho.weight[:].T.cpu()
        logit = torch.mm(alphas,rho)
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta 

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
            print(theta.unsqueeze(2).size())
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

def get_topic_quality(model,beta,data,num_diversity=25,num_coherence=2):
    """Returns topic coherence and topic diversity.
    """

    num_times=model.num_times
    num_topics=model.num_topics


    print('#'*100)
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

    print('Getting topic coherence...')
    tc=model_topic_coherence(data,beta,num_times,num_topics,num_coherence)
    overall_tc=0 
    for tt in range(num_times) : 
        overall_tc+=np.sum(tc[tt])/num_topics
    overall_tc=overall_tc/num_times

    print('Averaged Topic Coherence is : {}'.format(overall_tc))
    print('\n')
    print('Getting topic quality...')
    quality = overall_tc * TD_times
    print('Topic Quality is: {}'.format(quality))
    print('#'*100)

    return TD_all,TD_times,TD_topics,TD_all_topics,tc,overall_tc,quality

