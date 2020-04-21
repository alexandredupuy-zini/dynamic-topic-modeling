import numpy as np 
import pandas as pd 

def predict(Topic_distribution : np.ndarray,UN_dataset : pd.DataFrame, index_train : np.ndarray, index_test : np.ndarray, index_val : np.ndarray) : 
    
    data=UN_dataset.copy()
    indexes=np.hstack((index_train,index_test,index_val))
    data.sort_values('timestamp',inplace=True)
    data.reset_index(drop=True,inplace=True)
    topic_prediction=np.argmax(Topic_distribution,axis=1)
    topic_predictions=pd.DataFrame({'Topic':topic_prediction},index=indexes)
    prediction_df=data.merge(topic_predictions,left_on=data.index,right_on=topic_predictions.index,how='inner')

    return prediction_df