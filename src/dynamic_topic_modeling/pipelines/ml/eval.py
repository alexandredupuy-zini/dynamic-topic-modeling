from .detm_helpers import get_topic_quality
from .utils import split_bow_2
import pandas as pd
import numpy as np 

def eval(trained_model, beta, bow_train,vocab,num_diversity : int ,num_coherence : int ) : 

    train_tokens,train_counts=split_bow_2(bow_train,bow_train.shape[0])

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