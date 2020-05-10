import matplotlib.pyplot as plt
import numpy as np 


def plot_word_evolution(query,topic_number ,beta ,vocab, mapper_date) : 
    """Plot the word evolution of words considered in list_of_words over each time slices. Basically plots beta values of 
    each words in list_of_words over time slices.
    Takes in input : 
        - list_of_words : a list of words you want to plot
        - topic_number : a topic number between [0,number_of_topics]
        - beta_values : the beta distribution over words of shape [n_topics,time_length,n_words_in_vocab]
        - vocab : a list of all words in the vocabulary
        - save_path : specify a path if you want to save the graph
        - mapper_topic : a dictionary that maps topic number to its name : {0 : "Nuclear Problems"} for example
        - timelist : a list containing each time slice you used when model was trained. e.g [1970, 1971, ...]"""
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), dpi=360, facecolor='w', edgecolor='k')
    topic_name=str(topic_number)
    
    tokens_5 = [vocab.token2id[w] for w in query]
    betas_5 = [beta[topic_number-1, :, x] for x in tokens_5]
    for i, comp in enumerate(betas_5):
        axes.plot([mapper_date[int(i)] for i in range(beta.shape[1])],comp, label=query[i], lw=2, linestyle='--', marker='o', markersize=4)
    x=[mapper_date[int(i)] for i in np.arange(0,beta.shape[1],3)]
    axes.set_xticks(x)
    axes.legend(frameon=False)
    #axes.set_xticklabels(timelist[0::10])
    axes.set_title('Topic '+topic_name, fontsize=12)
    axes.set_ylabel('Probability given the topic')
    axes.set_xlabel('Time')
    save='word_evolution_topic_'+str(topic_number)
    for i in query : 
        save+='_'+i
    plt.savefig('data\\08_reporting\\'+save)

def plot_words(query,beta,vocab,mapper_date) : 
    for topic_number,words in query.items() : 
        topic_number=int(topic_number)
        plot_word_evolution(query=words,topic_number=topic_number,beta=beta,vocab=vocab,mapper_date=mapper_date)


