from kedro.pipeline import Pipeline, node

from .nodes import download_data_UN, preprocess_dataset, split_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
               func=download_data_UN,
               inputs=["params:dataset_drive_id",
                       "params:data_save_path"],
               outputs="raw_dataset",
               name='Download data'
            ),
            node(
                func=preprocess_dataset,
                inputs=["raw_dataset",
                        "params:flag_split_by_paragraph",
                        "params:flag_lemmatize",
                        "params:flag_bigram",
                        "params:min_bigram_count",
                        "params:flag_word_analysis"],
                outputs=dict(
                    docs="docs",
                    #corpus="corpus",
                    timestamps="timestamps",
                    #dictionary="dictionary"
            	),
                name='Pre-process data'
            ),
            node(
                func=split_data,
    		    inputs=["docs",
                        #"corpus",
                        "timestamps",
                        #"dictionary",
                        "params:extreme_no_below",
                        "params:extreme_no_above",
    			        "params:test_size",
                        "params:val_size"],
                outputs=dict(
                    train_docs="train_docs", val_docs="val_docs", test_docs="test_docs",
                    train_corpus="train_corpus", train_timestamps="train_timestamps",
                    val_corpus="val_corpus", val_timestamps="val_timestamps",
                    test_corpus="test_corpus", test_timestamps="test_timestamps",
                    test_corpus_h1="test_corpus_h1",
                    test_corpus_h2="test_corpus_h2",
                    dictionary="dictionary"),
                name='Split data'
    		)
        ], tags="Process data"
    )
