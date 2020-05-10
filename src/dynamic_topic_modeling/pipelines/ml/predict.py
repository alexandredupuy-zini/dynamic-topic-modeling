import numpy as np 
import pandas as pd 

def predict(Topic_distribution : np.ndarray, dataset : pd.DataFrame, index_train : np.ndarray, index_test : np.ndarray, index_val : np.ndarray) : 
    
    data=dataset.copy()
    indexes=np.hstack((index_train,index_test,index_val))
    data=data.iloc[indexes]
    topic_prediction=np.argmax(Topic_distribution,axis=1)
    data['predicted_topic']=topic_prediction
    data.reset_index(drop=True,inplace=True)

    n_topics=Topic_distribution.shape[1]
    proba_k=pd.DataFrame(Topic_distribution,columns=['proba_'+str(i) for i in range(n_topics)])

    final_data=pd.concat([data,proba_k], axis=1)
    final_data.sort_values('timestamp',inplace=True)
    
    
    

    return final_data