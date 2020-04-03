from kedro.pipeline import Pipeline, node

from .nodes import get_data, preprocess_dataset, split_data

get_data_node = node(
   func=get_data,
   inputs=["params:dataset"],
   outputs="raw_dataset",
   name='Get data'
)

preprocess_data_node = node(
    func=preprocess_dataset,
    inputs=["raw_dataset",
            "params:flag_split_by_paragraph",
            "params:flag_lemmatize",
            "params:flag_bigram",
            "params:min_bigram_count",
            "params:flag_word_analysis"],
    outputs=dict(
        docs="docs",
        timestamps="timestamps"
    ),
    name='Pre-process data'
)

split_data_node = node(
    func=split_data,
    inputs=["docs",
            "timestamps",
            "params:extreme_no_below",
            "params:extreme_no_above",
            "params:test_size",
            "params:val_size"],
    outputs=dict(
        # docs, corpus, sparse matrix, timestamps x3 (train/val/test) = ~12 datasets
        train_docs="train_docs", train_corpus="train_corpus", train_sparse="train_sparse",
        val_docs="val_docs", val_corpus="val_corpus", val_sparse="val_sparse",
        test_docs="test_docs", test_corpus="test_corpus", test_sparse="test_sparse",
        test_docs_h1="test_docs_h1", test_corpus_h1="test_corpus_h1", test_sparse_h1="test_sparse_h1",
        test_docs_h2="test_docs_h2", test_corpus_h2="test_corpus_h2", test_sparse_h2="test_sparse_h2",
        train_timestamps="train_timestamps", val_timestamps="val_timestamps", test_timestamps="test_timestamps",
        dictionary="dictionary"),
    name='Split data'
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            get_data_node,
            preprocess_data_node,
            split_data_node,
        ], tags="Process data"
    )
