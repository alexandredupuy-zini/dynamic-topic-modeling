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
from .utils import nearest_neighbors





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


    indices = torch.from_numpy(np.array([i for i in range(num_docs_train)]))
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

