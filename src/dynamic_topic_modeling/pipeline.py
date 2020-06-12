"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline

from dynamic_topic_modeling.pipelines import data_processing as dp
from dynamic_topic_modeling.pipelines import ml

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    preprocess=dp.create_pipeline_1()
    train_val_test=dp.create_pipeline_2()
    get_embeddings=dp.create_pipeline_3()

    grid_search_fasttext=ml.grid_search_fasttext_embedding()
    get_fasttext_embedding=ml.create_pipeline_0()
    get_detm=ml.create_pipeline_1()
    train_model=ml.create_pipeline_2()
    eval=ml.create_pipeline_3()
    predict=ml.create_pipeline_4()
    plot_words=ml.plot_word_evolution()

    return {
        "preprocess" : preprocess,
        "train_val_test_split": train_val_test,
        "get_embeddings" : get_embeddings,
        "grid_search_fasttext" : grid_search_fasttext,
        "get_fasttext_embedding":get_fasttext_embedding,
        "get_model" : get_detm,
        "train_model" : train_model,
        "eval_model" : eval,
        "predict" : predict,
        "plot_word_evolution": plot_words,
        "__default__": preprocess+get_fasttext_embedding+train_val_test+get_detm+train_model+eval+predict,
    }
