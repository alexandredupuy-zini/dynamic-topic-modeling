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

import data 

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence,


pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

def train(epoch):
    """Train DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size) 
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = data.get_batch(
            train_tokens, train_counts, ind, vocab_size, emb_size, temporal=True, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, num_docs_train)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
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
    
def visualize():
    """Visualizes topics and embeddings and word usage evolution.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())
        print('\n')
        print('#'*100)
        print('Visualize topics...')
        times = [0,int(beta.size()[1]/2),beta.size()[1]-1]
        topics_words = []
        for k in range(args.num_topics):
            for t in times:
                gamma = beta[k, t, :]
                top_words = list(gamma.detach().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

        print('\n')
        print('Visualize word embeddings ...')
        queries = ['economic', 'assembly', 'security', 'management', 'rights',  'africa']
        try:
            embeddings = model.rho.weight  # Vocab_size x E
        except:
            embeddings = model.rho         # Vocab_size x E
        neighbors = []
        for word in queries:
            try : 
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
            except : 
                print('{} not found in dictionary'.format(word))
        print('#'*100)

def _eta_helper(rnn_inp):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_eta(source):
    model.eval()
    with torch.no_grad():
        if source == 'val':
            rnn_inp = valid_rnn_inp
            return _eta_helper(rnn_inp)
        else:
            rnn_1_inp = test_1_rnn_inp
            return _eta_helper(rnn_1_inp)

def get_theta(eta, bows):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    

def get_completion_ppl(source):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
            times = valid_times

            eta = get_eta('val')

            acc_loss = 0
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, times_batch = data.get_batch(
                    tokens, counts, ind, args.vocab_size, args.emb_size, temporal=True, times=times)
                sums = data_batch.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch

                eta_td = eta[times_batch.type('torch.LongTensor')]
                theta = get_theta(eta_td, normalized_data_batch)
                alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
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
            ppl_all = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} PPL: {}'.format(source.upper(), ppl_all))
            print('*'*100)
            return ppl_all
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens_1 = test_1_tokens
            counts_1 = test_1_counts

            tokens_2 = test_2_tokens
            counts_2 = test_2_counts

            eta_1 = get_eta('test')

            acc_loss = 0
            cnt = 0
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            for idx, ind in enumerate(indices):
                data_batch_1, times_batch_1 = data.get_batch(
                    tokens_1, counts_1, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1

                eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
                theta = get_theta(eta_td_1, normalized_data_batch_1)

                data_batch_2, times_batch_2 = data.get_batch(
                    tokens_2, counts_2, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)

                alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
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
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            return ppl_dc

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.detach().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def get_topic_quality():
    """Returns topic coherence and topic diversity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())

        print('\n')
        print('#'*100)
        print('Get topic diversity...')
        num_tops = 25
        TD_all = np.zeros((args.num_times,))
        for tt in range(args.num_times):
            TD_all[tt] = _diversity_helper(beta[:, tt, :], num_tops)
        TD = np.mean(TD_all)
        print('Topic Diversity is: {}'.format(TD))

        print('\n')
        print('Get topic coherence...')
        print('train_tokens: ', train_tokens[0])
        TC_all = []
        cnt_all = []
        for tt in range(args.num_times):
            tc, cnt = get_topic_coherence(beta[:, tt, :].detach().numpy(), train_tokens, vocab)
            TC_all.append(tc)
            cnt_all.append(cnt)
        print('TC_all: ', TC_all)
        TC_all = torch.tensor(TC_all)
        print('TC_all: ', TC_all.size())
        print('\n')
        print('Get topic quality...')
        quality = tc * diversity
        print('Topic Quality is: {}'.format(quality))
        print('#'*100)