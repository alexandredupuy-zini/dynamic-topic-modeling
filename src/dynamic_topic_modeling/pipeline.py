"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from dynamic_topic_modeling.pipelines import data_processing as dp
from dynamic_topic_modeling.pipelines import machine_learning as ml


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.
    Args:
        kwargs: Ignore any additional arguments added in the future.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    #data_processing_pipeline = dp.create_pipeline()
    machine_learning_pipeline = ml.create_pipeline()

    return {
        #"data processing": data_processing_pipeline,
        "machine learning": machine_learning_pipeline,
        "__default__": machine_learning_pipeline
                    #data_processing_pipeline
                     #+ machine_learning_pipeline,
    }
