"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline

from dynamic_topic_modeling.pipelines import data_processing as dp

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
    return {
        'download_data' : download_data,
        "preprocess" : preprocess,
        "train_val_test_split": train_val_test,
        "__default__": download_data+preprocess+train_val_test,
    }

