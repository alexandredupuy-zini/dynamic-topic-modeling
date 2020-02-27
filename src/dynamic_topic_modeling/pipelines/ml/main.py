
from .detm import DETM
from .utils import split_bow
import scipy
import pandas as pd
import gensim
import numpy as np 
import torch




def get_model(BOW_train: scipy.sparse,BOW_test: scipy.sparse,BOW_test_h1: scipy.sparse,BOW_test_h2: scipy.sparse,
             BOW_val : scipy.sparse, timestamp_train : pd.DataFrame, timestamp_test : pd.DataFrame, timestamp_val : pd.DataFrame,
             vocab : gensim.corpora.Dictionary, num_times : int , vocab_size : int, t_hidden_size : int, 
              eta_hidden_size : int , rho_size : int , emb_size : int , enc_drop : float, eta_nlayers : int, 
              train_embeddings : bool, theta_act : str, delta : float, embeddings) :

    vocab_size=len(vocab)


    print('Getting training data ...')
    train_tokens,train_counts=split_bow(BOW_train,BOW_train.shape[0])
    train_times = timestamp_train
    num_times = len(np.unique(train_times))
    num_docs_train = len(train_tokens)
    train_rnn_inp = data.get_rnn_input(
        train_tokens, train_counts, train_times, num_times, vocab_size, num_docs_train)

    print('Getting validation data ...')
    valid_tokens,valid_counts = split_bow(BOW_val,BOW_val.shape[0])
    valid_times = timestamp_val
    num_docs_valid = len(valid_tokens)
    valid_rnn_inp = data.get_rnn_input(
        valid_tokens, valid_counts, valid_times, num_times, vocab_size, num_docs_valid)

    print('Getting testing data ...')
    test_tokens, test_counts = split_bow(BOW_test,BOW_test.shape[0])
    test_times = timestamp_test
    num_docs_test = len(test_tokens)
    test_rnn_inp = data.get_rnn_input(
        test_tokens, test_counts, test_times, args.num_times, args.vocab_size, num_docs_test)

    test_1_tokens = test['tokens_1']
    test_1_counts = test['counts_1']
    test_1_times = test_times
    num_docs_test_1 = len(test_1_tokens)
    test_1_rnn_inp = data.get_rnn_input(
        test_1_tokens, test_1_counts, test_1_times, args.num_times, args.vocab_size, num_docs_test)

    test_2_tokens = test['tokens_2']
    test_2_counts = test['counts_2']
    test_2_times = test_times
    num_docs_test_2 = len(test_2_tokens)
    test_2_rnn_inp = data.get_rnn_input(
        test_2_tokens, test_2_counts, test_2_times, args.num_times, args.vocab_size, num_docs_test)

    
    
    return model,train_rnn_inp







def get_model(num_topics : int , num_times : int , vocab_size : int, t_hidden_size : int, 
              eta_hidden_size : int , rho_size : int , emb_size : int , enc_drop : float, eta_nlayers : int, 
              train_embeddings : bool, theta_act : str, delta : float, embeddings) : 

    model=DETM(num_topics=num_topics,num_times=num_times,vocab_size=vocab_size,t_hidden_size=t_hidden_size,
               eta_hidden_size=eta_hidden_size,rho_size=rho_size,emb_size=emb_size,enc_drop=enc_drop,eta_nlayers=eta_nlayers,
               train_embeddings=train_embeddings, theta_act=theta_act,delta=delta,embeddings=embeddings)

    return model 

def main(model, train_rnn_inp,train_times,num_docs_train,
         test_rnn_inp,test_times,num_docs_test,
         valid_rnn_inp,valid_times,num_docs_valid,
         test_1_rnn_inp,test_2_rnn_inp,vocab_size,
         epochs:int, visualize_every: int,anneal_lr:int,nonmono:int,lr_fatcor:float,load_from: str,
         ) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print('\nDETM architecture: {}'.format(model))
    model.to(device)
    ## train model on data by looping through multiple epochs

    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    for epoch in range(1, epochs):
        detm_helper.train(epoch)
        if epoch % visualize_every == 0:
            detm_helper.visualize()
        val_ppl = detm_helper.get_completion_ppl('val')
        print('val_ppl: ', val_ppl)
        if val_ppl < best_val_ppl:
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if anneal_lr and (len(all_val_ppls) > nonmono and val_ppl > min(all_val_ppls[:-nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= lr_factor
        all_val_ppls.append(val_ppl)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        print('saving topic matrix beta...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).detach().numpy()
        if args.train_embeddings:
            print('saving word embedding matrix rho...')
            rho = model.rho.weight.detach().numpy()
        print('computing validation perplexity...')
        val_ppl = get_completion_ppl('val')
        print('computing test perplexity...')
        test_ppl = get_completion_ppl('test')

    return model,beta,rho 
