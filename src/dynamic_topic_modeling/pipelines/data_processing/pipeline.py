

from kedro.pipeline import Pipeline, node
from .download_data import download_file_from_google_drive
from .preprocess import preprocess_dataset
from .train_val_test import train_val_test
from .embeddings import download_embeddings,get_embeddings

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
					inputs=["UN_dataset","params:extreme_no_below", "params:extreme_no_above", 
			 "params:enable_bigram", "params:min_bigram_count",'params:basic_word_analysis'],
					outputs=dict(
						dataset_preprocessed="UN_preprocessed",
						corpus="UN_corpus",
						dictionary="UN_dictionary")
					)
        ],tags="Preprocessing"
    )
def create_pipeline_3(**kwargs):
    return Pipeline(
        [
            node( func=train_val_test,
		    inputs=["UN_preprocessed","UN_dictionary", "UN_corpus", 
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
                        train_rnn_inp="train_rnn_inp",
                        test_rnn_inp='test_rnn_inp',
                        test_1_rnn_inp="test_1_rnn_inp",
                        test_2_rnn_inp="test_2_rnn_inp",
                        valid_rnn_inp='val_rnn_inp'
                        )
					)
        ], tags="Split data into train val test"
    )
def create_pipeline_4(**kwargs):
    return Pipeline(
        [
            node(   
                    inputs=[],
					func=download_embeddings,
					outputs="Glove_embeddings"
                    )
					
        ],tags="Download Glove embedding from Stanford website"
    )
def create_pipeline_5(**kwargs):
    return Pipeline(
        [
            node(   
                    inputs=["Glove_embeddings","params:emb_size","UN_dictionary","params:fill_emb_with"],
					func=get_embeddings,
					outputs="UN_embeddings"
                    )
					
        ],tags="Merge embeddings with corpus"
    )