from .metrics import get_topic_quality
from .utils import get_cos_sim_from_embedding
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords

def eval(trained_model, beta, rho, theta, dataset_preprocessed, raw_dataset, data_bow,vocab, mapper_date, idx_tr, idx_ts, idx_va, num_diversity : int ,num_coherence : int ) : 

    sw=stopwords.words('french')+['les','tres']
    print('Topic inspection ......')
    print('-'*100)
    print('Top words per topics when stop words included')

    df_with_sw=pd.DataFrame(index=[i for i in range(beta.shape[1])],columns=[i for i in range(beta.shape[0])])
    df_with_sw.columns = pd.MultiIndex.from_tuples(
        zip(['Topic']*beta.shape[0], 
            df_with_sw.columns))
    df_with_sw.index = pd.MultiIndex.from_tuples(
        zip(['Time']*beta.shape[1], 
            df_with_sw.index))
    df_without_sw=df_with_sw.copy()

    for k in range(beta.shape[0]):
        for t in range(beta.shape[1]) :
            gamma = beta[k, int(t), :]
            top_words = list(gamma.argsort()[::-1])
            topic_words = [vocab[a] for a in top_words]

            if t==0 or t==int((beta.shape[1]-1)/2) or t==beta.shape[1]-1 :   
                print('Topic {} .. Time: {} ===> {}'.format(k, mapper_date[int(t)], topic_words[:5])) 
            df_with_sw.iloc[t,k]=topic_words
        print('')

    print('Top words per topics when stop words are not included')

    for k in range(beta.shape[0]):
        for t in range(beta.shape[1]) :
            
            gamma = beta[k, int(t), :]
            top_words = list(gamma.argsort()[::-1])
            topic_words=[]               
            for word in top_words :            
                if vocab[word] not in sw : 
                    topic_words.append(vocab[word])
            if t==0 or t==int((beta.shape[1]-1)/2) or t==beta.shape[1]-1 :        
                print('Topic {} .. Time: {} ===> {}'.format(k, mapper_date[int(t)], topic_words[:5])) 
            df_without_sw.iloc[t,k]=topic_words
        print('')

    print('Most similar words to "contrat" in trained embeddings :')
    for key,value in get_cos_sim_from_embedding('contrat', rho,vocab).items() : 
        print(key,' : ',value)

    TD_all_times,overall_TD_times,TD_all_topics,overall_TD_topics,tc,overall_tc,quality = get_topic_quality(trained_model,beta,data_bow,num_diversity,num_coherence)

    n_topics=beta.shape[0]

    eval_df=pd.DataFrame(columns=['Topic','TD','TC','TQ','Overall_TD','Overall_TC','Overall_TQ','Max_proba','Most_representative'])
    eval_df['Topic']=[k for k in range(n_topics)]
    eval_df.loc[:,'TD']=TD_all_topics
    eval_df.loc[:,'TC']=tc.mean(axis=0)
    eval_df.loc[:,'TQ']=TD_all_topics*tc.mean(axis=0)
    eval_df.loc[:,'Overall_TD']=[overall_TD_times]*(n_topics)
    eval_df.loc[:,'Overall_TC']=[overall_tc]*(n_topics)
    eval_df.loc[:,'Overall_TQ']=[quality]*(n_topics)
    
    eval_df.loc[:,'Max_proba']=np.max(theta,axis=0)
    idx=np.hstack((idx_tr,idx_ts,idx_va))
    good_idx=idx[np.argmax(theta,axis=0)]
    raw_idx=dataset_preprocessed.iloc[good_idx]['raw_index'].values
    eval_df.loc[:,'Most_representative']=raw_dataset.loc[raw_idx]['text'].values
    
    return dict(
        eval_summary=eval_df,
        TD_all_times=TD_all_times,
        overall_TD_times=overall_TD_times,
        overall_TD_topics=overall_TD_topics,
        TD_all_topics=TD_all_topics,
        tc=tc,
        overall_tc=overall_tc,
        quality=quality,
        topic_description_sw=df_with_sw,
        topic_description_no_sw=df_without_sw
        )