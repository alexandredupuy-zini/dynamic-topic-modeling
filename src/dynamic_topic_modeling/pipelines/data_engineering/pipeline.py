from kedro.pipeline import Pipeline, node
from dynamic_topic_modeling.pipelines.data_engineering.nodes import (
    preprocess_UNGD,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_UNGD,
                inputs=["UN_dataset", "params:extreme_no_below", "params:extreme_no_above", "params:enable_bigram", "params:min_bigram_count"],
                outputs=dict(
                    UNGD_preprocessed="UNGD",
                    corpus="corpus",
                    dictionnary="dictionnary",
                ),
                name="UNGD_preprocessing",
            )
        ]
    )
