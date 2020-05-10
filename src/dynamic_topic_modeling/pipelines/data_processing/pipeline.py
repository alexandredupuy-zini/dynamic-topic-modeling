

from kedro.pipeline import Pipeline, node
from .download_data import download_file_from_google_drive
from .preprocess import preprocess_dataset
from .train_val_test import train_val_test
from .embeddings import get_embeddings
from .handle_verbatim_errors import handle_errors

def create_pipeline_1(**kwargs):
    return Pipeline(
        [
            node(
               func=download_file_from_google_drive,
               inputs=['params:dataset_drive_id',
                        'params:data_save_path'],
               outputs="UN_dataset"
               )
            ],tags="Download data"
        )
def create_pipeline_2(**kwargs):
    return Pipeline(
        [
            node(
					func=preprocess_dataset,
					inputs=["Verbatim_soge_raw","params:extreme_no_below", "params:extreme_no_above", 
			 "params:enable_bigram", "params:min_bigram_count",'params:basic_word_analysis',"params:lemmatize","params:temporality","params:language"],
					outputs=dict(
						dataset_preprocessed="Verbatim_soge_preprocessed",
						dictionary="Verbatim_dictionary",
                        vocab_size="Verbatim_vocab_size",
                        date_range='mapper_date')
					)
        ],tags="Preprocessing"
    )
def create_pipeline_3(**kwargs):
    return Pipeline(
        [
            node( func=train_val_test,
		    inputs=["Verbatim_soge_preprocessed","Verbatim_dictionary", "Verbatim_corpus", 
			         "params:test_size", "params:val_size"],
				  outputs=dict(
                        BOW_train="BOW_train",
                        BOW_test="BOW_test",
                        BOW_test_h1="BOW_test_h1",
                        BOW_test_h2="BOW_test_h2",
                        BOW_val="BOW_val",
                        timestamps_train="timestamp_train",
                        timestamps_test="timestamp_test",
                        timestamps_val="timestamp_val",
                        train_vocab_size="train_vocab_size",
                        train_num_times='train_num_times',
                        idx_train='index_train_set',
                        idx_test='index_test_set',
                        idx_val='index_val_set'
                        )
					)
        ], tags="Split data into train val test"
    )
def create_pipeline_4(**kwargs) : 
    return Pipeline(
        [
            node(func=get_embeddings,
                 inputs=["params:emb_filepath","UN_dictionary"],
                 outputs="pretrained_embedding"
                 )
            ]
        )
    