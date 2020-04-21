from .metrics import get_topic_quality
import pandas as pd
import numpy as np 

def eval(trained_model, beta, data_bow,vocab,num_diversity : int ,num_coherence : int ) : 


    TD_all,TD_times,TD_topics,TD_all_topics,tc,overall_tc,quality = get_topic_quality(trained_model,beta,data_bow,num_diversity,num_coherence)

    return dict(
        TD_all=TD_all,
        TD_times=TD_times,
        TD_topics=TD_topics,
        TD_all_topics=TD_all_topics,
        tc=tc,
        overall_tc=overall_tc,
        quality=quality
        )