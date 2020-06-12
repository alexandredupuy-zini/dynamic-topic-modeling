from kedro.pipeline import Pipeline, node

from .nodes import get_embeddings, preprocess_dataset, train_val_test

def create_pipeline_1(**kwargs):
    return Pipeline(
        [
            node(
					func=preprocess_dataset,
					inputs=["DataSet","params:extreme_no_below", "params:extreme_no_above",
			 "params:enable_bigram", "params:min_bigram_count",'params:basic_word_analysis',
             "params:lemmatize","params:temporality","params:language",
             "params:path_to_texts_for_embedding", "params:split_by_paragraph"],
					outputs=dict(
						dataset_preprocessed="DataSet_preprocessed",
						dictionary="Dictionary",
                        vocab_size="Vocab_size",
                        date_range='Mapper_date')
					)
        ], tags="Preprocessing"
    )

def create_pipeline_2(**kwargs):
    return Pipeline(
        [
            node( func=train_val_test,
		    inputs=["DataSet_preprocessed","Dictionary",
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

def create_pipeline_3(**kwargs) :
    return Pipeline(
        [
            node(func=get_embeddings,
                 inputs=["params:emb_filepath","Dictionary"],
                 outputs="pretrained_embedding"
                 )
            ]
        )
