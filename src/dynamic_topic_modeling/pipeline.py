"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline

from dynamic_topic_modeling.pipelines import data_processing as dp
from dynamic_topic_modeling.pipelines import ml

# Here you can define your data-driven pipeline by importing your functions
# and adding them to the pipeline as follows:
#
# from nodes.data_wrangling import clean_data, compute_features
#
# pipeline = Pipeline([
#     node(clean_data, 'customers', 'prepared_customers'),
#     node(compute_features, 'prepared_customers', ['X_train', 'Y_train'])
# ])
#
# Once you have your pipeline defined, you can run it from the root of your
# project by calling:
#
# $ kedro run


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    download_data = dp.create_pipeline_1()
    preprocess=dp.create_pipeline_2()
    train_val_test=dp.create_pipeline_3()
    get_embeddings=dp.create_pipeline_4()

    grid_search_fasttext=ml.grid_search_fasttext_embedding()
    get_fasttext_embedding=ml.create_pipeline_0()
    get_detm=ml.create_pipeline_1()
    train_model=ml.create_pipeline_2()
    eval=ml.create_pipeline_3()
    predict=ml.create_pipeline_4()
    plot_words=ml.plot_word_evolution()

    return {
        'download_data' : download_data,
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
        "__default__": download_data+preprocess+get_fasttext_embedding+train_val_test+get_detm+train_model+eval+predict,
    }

