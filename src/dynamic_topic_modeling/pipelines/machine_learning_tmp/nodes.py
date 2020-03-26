import scipy
import pandas as pd
import gensim
import numpy as np

import torch
from torch import nn, optim

from .detm import DETM
from .detm_helpers import train, get_val_completion_ppl, get_test_completion_ppl, get_eta, get_theta, get_beta, get_topic_quality
from .utils import split_bow_2, bow_to_dense_tensor
from .data import get_rnn_input,get_batch

def get_model(num_topics: int, num_times: int, vocab_size: int,
              t_hidden_size: int, eta_hidden_size: int, rho_size: int,
              emb_size: int, enc_drop: float, eta_nlayers: int,
              eta_dropout: float, theta_act: str, delta: float, GPU:bool):

    model = DETM(num_topics=num_topics, num_times=num_times,
               vocab_size=vocab_size, t_hidden_size=t_hidden_size,
               eta_hidden_size=eta_hidden_size, rho_size=rho_size,
               emb_size=emb_size, enc_drop=enc_drop, eta_nlayers=eta_nlayers,
               eta_dropout=eta_dropout, theta_act=theta_act,
               delta=delta,GPU=GPU)

    return model


def train_model(model,
                bow_train,train_times,
                bow_test_1, bow_test_2, test_times,
                bow_val,val_times,
                log_interval: int, batch_size: int, eval_batch_size : int, n_epochs:int, optimizer:str, lr:float,
                wdecay:float, anneal_lr:bool, nonmono:int, lr_factor:float, clip_grad : float, seed : int
         ) :


    ## set seed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    vocab_size=model.vocab_size
    emb_size=model.emsize
    device = model.device
    GPU=model.train_on_gpu
    model.to(device)

    print('\nDETM architecture: {}'.format(model))
    ## train model on data by looping through multiple epochs

    num_docs_train=bow_train.shape[0]
    num_docs_test=bow_test_1.shape[0]
    num_docs_val=bow_val.shape[0]

    train_tokens, train_counts = split_bow_2(bow_train,num_docs_train)
    val_tokens, val_counts = split_bow_2(bow_val,num_docs_val)
    test_tokens_1, test_counts_1 = split_bow_2(bow_test_1,num_docs_test)
    test_tokens_2, test_counts_2 = split_bow_2(bow_test_2,num_docs_test)

    train_times=np.array(train_times).squeeze()
    val_times=np.array(val_times).squeeze()
    test_times=np.array(test_times).squeeze()

    num_times=len(np.unique(train_times))

    print('Getting train RNN input ....')
    train_rnn_inp = get_rnn_input(
        train_tokens, train_counts, train_times, num_times, vocab_size, num_docs_train,GPU)
    print('Train RNN shape : {}\n'.format(train_rnn_inp.size()))

    print('Getting val RNN input ....')
    val_rnn_inp = get_rnn_input(
        val_tokens, val_counts, val_times, num_times, vocab_size, num_docs_val,GPU)
    print('Val RNN shape : {}\n'.format(val_rnn_inp.size()))


    print('Getting test half 1 RNN input ....')
    test_1_rnn_inp = get_rnn_input(
        test_tokens_1, test_counts_1, test_times, num_times, vocab_size, num_docs_test,GPU)
    print('Test H1 RNN shape : {}\n'.format(test_1_rnn_inp.size()))

    print('Getting test half 2 RNN input ....')
    test_2_rnn_inp = get_rnn_input(
        test_tokens_2, test_counts_2, test_times, num_times, vocab_size, num_docs_test,GPU)
    print('Test H2 RNN shape : {}'.format(test_2_rnn_inp.size()))

    print('\n')
    print('*'*100)
    print('\n')

    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    bad_hit=0

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wdecay)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=wdecay)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wdecay)
    elif optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=wdecay)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs+1):

        train(model, epoch, optimizer, num_docs_train, batch_size, vocab_size, emb_size, log_interval, clip_grad, train_rnn_inp, train_tokens, train_counts, train_times)

        val_ppl = get_val_completion_ppl(model, num_docs_val, eval_batch_size, vocab_size, emb_size,
                       val_tokens, val_counts, val_times, val_rnn_inp)


        if val_ppl <best_val_ppl :
            bad_hit=0
            best_epoch = epoch
            best_val_ppl = val_ppl

        else :

            bad_hit+=1
            lr = optimizer.param_groups[0]['lr']

            ## check whether to anneal lr
            if anneal_lr and bad_hit == nonmono  and lr > 1e-5 :
                optimizer.param_groups[0]['lr'] /= lr_factor


        all_val_ppls.append(val_ppl)
        model.to(device)

    model.eval()
    with torch.no_grad():


        print('computing test perplexity...')
        test_ppl = get_test_completion_ppl(model, test_tokens_1, test_counts_1, test_tokens_2, test_counts_2, test_times,
                                                       num_docs_test, eval_batch_size, vocab_size, emb_size, test_1_rnn_inp, test_2_rnn_inp)
        device=torch.device('cpu')
        model.to(device)

        bows,times_batch=get_batch(device=device,tokens=train_tokens,counts=train_counts,ind=[i for i in range(len(train_tokens))],vocab_size=vocab_size,temporal=True,times=train_times)

        ##computing word distribution beta##
        alpha = model.mu_q_alpha.cpu()
        beta = get_beta(model,alpha).cpu().numpy()

        ##computing word embedding rho##
        rho = model.rho.weight.cpu().detach().numpy()

        ##computing topic distribution theta##

        eta=get_eta(model,train_rnn_inp).cpu()
        sums = bows.sum(1).unsqueeze(1)
        normalized_data_batch=bows/sums
        eta_td=eta[times_batch.type('torch.LongTensor')]
        theta = get_theta(model,eta_td, normalized_data_batch)
        theta=theta.cpu().numpy()

        ## computing topic embedding alpha
        alpha=model.mu_q_alpha.cpu().detach().numpy()


    return model,beta,rho,theta,alpha

def eval(trained_model, beta, bow_train,vocab,num_diversity: int,
         num_coherence: int ) :

    train_tokens,train_counts = split_bow_2(bow_train,bow_train.shape[0])

    TD_all,TD_times,TD_topics,TD_all_topics,tc,overall_tc,quality = get_topic_quality(trained_model,beta,train_tokens,num_diversity,num_coherence)

    return dict(
        TD_all=TD_all,
        TD_times=TD_times,
        TD_topics=TD_topics,
        TD_all_topics=TD_all_topics,
        tc=tc,
        overall_tc=overall_tc,
        quality=quality
        )
