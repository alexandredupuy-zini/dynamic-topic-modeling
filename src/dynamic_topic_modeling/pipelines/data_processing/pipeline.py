

from kedro.pipeline import Pipeline, node
from .download_data import download_file_from_google_drive
from .preprocess import preprocess_dataset

def create_pipeline_1(**kwargs):
    return Pipeline(
        [
            node(
               func=download_file_from_google_drive,
               inputs=['params:dataset_drive_id',
                        'params:data_save_path'],
               outputs="UN_dataset"
               )
            ]
        )
def create_pipeline_2(**kwargs):
    return Pipeline(
        [
            node(
					func=preprocess_dataset,
					inputs=["UN_dataset","params:extreme_no_below", "params:extreme_no_above", 
			 "params:enable_bigram", "params:min_bigram_count",'params:basic_word_analysis'],
					outputs=dict(
						dataset_preprocessed="UN_preprocessed",
						corpus="UN_corpus",
						dictionnary="UN_dictionary")
					)
        ]
    )
