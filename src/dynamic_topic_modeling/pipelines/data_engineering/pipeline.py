from kedro.pipeline import Pipeline, node
from dynamic_topic_modeling.pipelines.data_engineering.nodes import (
	preprocess_UNGD,
)
def create_pipeline(**kwargs):

		return Pipeline(
			[
				node(
					func=preprocess_UNGD,
					inputs=["params:extreme_no_below", "params:extreme_no_above", "params:enable_bigram", "params:min_bigram_count","params:download_dataset",'params:basic_word_analysis'],
					outputs=dict(
						UNGD_preprocessed="UN_preprocessed",
						corpus="UN_corpus",
						dictionnary="UN_dictionary",
					),
					name="UNGD_preprocessing",
				)
			]
		)
		
