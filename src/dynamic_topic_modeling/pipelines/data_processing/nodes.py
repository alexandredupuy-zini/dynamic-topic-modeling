from typing import Any, Dict
from kedro.context import load_context

from time import time
import pandas as pd
import numpy as np
from scipy import sparse

from gensim.corpora import Dictionary, MmCorpus

from .get_datasets import get_data_20NG, get_data_UNGD, get_data_SOGE
from .utils import split_by_paragraph
from .utils import lowerize, tokenize, remove_stop_words, remove_numbers
from .utils import remove_word_with_length, lemmatize, add_bigram
from .utils import remove_vocab, remove_empty, remove_by_threshold
from .utils import convert_to_bow, create_sparse_matrix


def get_data(dataset='20NG'):
    if dataset == '20NG':
        df = get_data_20NG()
    elif dataset == 'UNGD':
        df = get_data_UNGD()
    elif dataset == 'SOGE':
        df = get_data_SOGE()
    return df


def preprocess_dataset(dataset:pd.DataFrame,
                       flag_split_by_paragraph:bool,
                       flag_lemmatize:bool,
                       flag_bigram:bool,
                       min_bigram_count:int,
                       flag_word_analysis:bool) -> Dict[str, Any]:
    t0 = time()

    print('\nCurrent set of parameters :\n')
    print('\tflag_split_by_paragraph : {}'.format(flag_split_by_paragraph))
    print('\tflag_lemmatize : {}'.format(flag_lemmatize))
    print('\tflag_bigram : {}'.format(flag_bigram))
    print('\tmin_bigram_count : {}'.format(min_bigram_count))
    #print('\textreme_no_below : {}'.format(extreme_no_below))
    #print('\textreme_no_above : {}'.format(extreme_no_above))
    print('\tflag_word_analysis : {}\n'.format(flag_word_analysis))

    print('\nStart preprocessing on dataset')

    if "text" not in dataset.columns:
        raise ValueError('Dataset does not have a column named "text". You must rename the your text column to "text".')
    if "timestamp" not in dataset.columns:
        raise ValueError('Dataset does not have a column named "timestamp". You must rename your time column to "timestamp".')

    dataset.sort_values('timestamp', inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    docs = dataset['text'].values
    timestamps = dataset['timestamp'].values

    if flag_split_by_paragraph:
        print('\nSplitting by paragraph...')
        docs, timestamps = split_by_paragraph(docs, timestamps)

    print('\nLowerizing...')
    docs = lowerize(docs)

    print('\nTokenizing...')
    docs = tokenize(docs)

    if flag_bigram:
        print('\nAdding bigrams...')
        docs = add_bigram(docs, min_bigram_count)

    if flag_word_analysis:

        print('\nBasic word analysis enabled. It will take more time to compute......')

        len_starting_vocab = len(Dictionary(docs))
        print('\nBeginning dictionary contains : {} words'.format(len_starting_vocab))

        print('\nRemoving stop words...')
        docs = remove_stop_words(docs)
        curr_len_vocab = len(Dictionary(docs))
        len_rm_words = len_starting_vocab - curr_len_vocab
        len_vocab = curr_len_vocab
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} stopwords from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))
        print('\tCurrent length of the vocabulary:', len_vocab)

        print('\nRemoving unique numbers (not words that contain numbers)...')
        docs = remove_numbers(docs)
        curr_len_vocab = len(Dictionary(docs))
        len_rm_words = len_vocab - curr_len_vocab
        len_vocab = curr_len_vocab
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} numeric words from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))
        print('\tCurrent length of the vocabulary:', len_vocab)

        print('\nRemoving words that contain only one character...')
        docs = remove_word_with_length(docs, length=1)
        curr_len_vocab = len(Dictionary(docs))
        len_rm_words = len_vocab - curr_len_vocab
        len_vocab = curr_len_vocab
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} one length characters from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))
        print('\tCurrent length of the vocabulary:', len_vocab)

        if flag_lemmatize:
            print('\nLemmatizing...')
            docs = lemmatize(docs)
            curr_len_vocab = len(Dictionary(docs))
            len_rm_words = len_vocab - curr_len_vocab
            len_vocab = curr_len_vocab
            freq = round(len_rm_words / len_starting_vocab, 3) * 100
            print('\tRemoved {} words from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))
            print('\tCurrent length of the vocabulary:', len_vocab)

    else:

        print('\nBasic word analysis disabled.')

        print('\nRemoving stop words...')
        docs = remove_stop_words(docs)

        print('\nRemoving unique numbers (not words that contain numbers)...')
        docs = remove_numbers(docs)

        print('\nRemoving words that contain only one character...')
        docs = remove_word_with_length(docs, length=1)

        if flag_lemmatize:
            print('\nLemmatizing...')
            docs = lemmatize(docs)


    # Create a dictionary representation of the documents.
    #print('\nCreating dictionary...')
    #dictionary = Dictionary(docs)

    #print('\nFiltering extremes...')
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    #dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
    #extreme_no_below_str = str(extreme_no_below) if extreme_no_below > 1 else str(extreme_no_below*100)+'%'
    #extreme_no_above_str = str(extreme_no_above) if extreme_no_above > 1 else str(extreme_no_above*100)+'%'
    #print('\tKeeping words in no less than {} documents & in no more than {} documents'.format(extreme_no_below_str, extreme_no_above_str))

    # Bag-of-words representation of the documents.
    #corpus = [dictionary.doc2bow(doc) for doc in docs]

    #print('Number of unique tokens: %d' % len(dictionary))
    #print('Number of documents: %d' % len(corpus))


    print('\nTimestamps processing...')
    unique_time = np.unique(timestamps)
    mapper_time = dict(zip(unique_time, range(len(unique_time))))
    timestamps = np.array([mapper_time[timestamps[i]] for i in range(len(timestamps))])


    print('\nDone in {} minutes'.format(int((time()-t0)/60)))


    return dict(
        docs=docs,
        #corpus=corpus,
        timestamps=timestamps,
        #dictionary=dictionary
    )


def split_data(docs:np.array,
               #corpus:MmCorpus,
               timestamps:np.array,
               #dictionary:Dictionary,
               extreme_no_below,
               extreme_no_above,
               test_size:float,
               val_size:float) -> Dict[str,Any]:

    # Split indexes into train/val/test sets
    print('\nSplitting indexes into train/val/test')
    num_docs = len(docs)

    val_len = int(val_size * num_docs)
    test_len = int(test_size * num_docs)
    train_len = int(num_docs - val_len - test_len)

    idx_permute = np.random.permutation(num_docs).astype(int)

    # Split docs and timestamps into train/val/test sets
    print('\nSpliiting docs and timestamps into train/val/test')
    train_docs = [docs[idx_permute[i]] for i in range(train_len)]
    val_docs = [docs[idx_permute[train_len+i]] for i in range(val_len)]
    test_docs = [docs[idx_permute[train_len+val_len+i]] for i in range(test_len)]

    train_timestamps = [timestamps[idx_permute[i]] for i in range(train_len)]
    val_timestamps = [timestamps[idx_permute[train_len+i]] for i in range(val_len)]
    test_timestamps = [timestamps[idx_permute[train_len+val_len+i]] for i in range(test_len)]

    print('\tNumber of documents in train set : {} [this should be equal to {} and {}]'.format(len(train_docs), train_len, len(train_timestamps)))
    print('\tNumber of documents in test set : {} [this should be equal to {} and {}]'.format(len(test_docs), test_len, len(test_timestamps)))
    print('\tNumber of documents in validation set: {} [this should be equal to {} and {}]'.format(len(val_docs), val_len, len(val_timestamps)))

    # Create a dictionary representation of the documents.
    print('\nCreating dictionary...')
    train_dictionary = Dictionary(train_docs)

    print('\tFiltering extremes...')
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    train_dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
    extreme_no_below_str = str(extreme_no_below) if extreme_no_below > 1 else str(extreme_no_below*100)+'%'
    extreme_no_above_str = str(extreme_no_above) if extreme_no_above > 1 else str(extreme_no_above*100)+'%'
    print('\tKeeping words in no less than {} documents & in no more than {} documents'.format(extreme_no_below_str, extreme_no_above_str))
    print('\tNumber of unique tokens: %d' % len(train_dictionary))

    # Remove words not in train_data
    print('\nRemoving words not in train data .....')
    train_vocab = train_dictionary.token2id
    train_docs = remove_vocab(train_docs, train_vocab)
    val_docs = remove_vocab(val_docs, train_vocab)
    test_docs = remove_vocab(test_docs, train_vocab)

    # Remove empty documents
    print('\nRemoving empty documents')
    train_docs, train_timestamps = remove_empty(train_docs, train_timestamps)
    test_docs, test_timestamps = remove_empty(test_docs, test_timestamps)
    val_docs, val_timestamps = remove_empty(val_docs, val_timestamps)

    # Remove test documents with length=1
    print('\nRemoving test documents with length 1')
    test_docs, test_timestamps = remove_by_threshold(test_docs, test_timestamps, 1)

    # Split test set in 2 halves
    print('\nSplitting test set in 2 halves')
    test_docs_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in test_docs]
    test_docs_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in test_docs]

    # Convert to Bag-of-Words representation
    print('\nCreating bow representation...')
    train_corpus = convert_to_bow(train_docs, train_dictionary)
    val_corpus = convert_to_bow(val_docs, train_dictionary)
    test_corpus = convert_to_bow(test_docs, train_dictionary)

    test_corpus_h1 = convert_to_bow(test_docs_h1, train_dictionary)
    test_corpus_h2 = convert_to_bow(test_docs_h2, train_dictionary)

    print('\tTrain bag of words shape : {}'.format(len(train_corpus)))
    print('\tVal bag of words shape : {}'.format(len(val_corpus)))
    print('\tTest bag of words shape : {}'.format(len(test_corpus)))
    print('\tTest set 1 bag of words shape : {}'.format(len(test_corpus_h1)))
    print('\tTest set 2 bag of words shape : {}'.format(len(test_corpus_h2)))

    # Convert to sparse matrices (scipy COO sparse matrix)
    print('\nCreating sparse matrices')
    train_sparse = create_sparse_matrix(train_docs, train_dictionary)
    test_sparse = create_sparse_matrix(test_docs, train_dictionary)
    test_sparse_h1 = create_sparse_matrix(test_docs_h1, train_dictionary)
    test_sparse_h2 = create_sparse_matrix(test_docs_h2, train_dictionary)
    val_sparse = create_sparse_matrix(val_docs, train_dictionary)

    print('\nDone splitting data.')

    return dict(
        train_docs=train_docs, train_corpus=train_corpus, train_sparse=train_sparse,
        val_docs=val_docs, val_corpus=val_corpus, val_sparse=val_sparse,
        test_docs=test_docs, test_corpus=test_corpus, test_sparse=test_sparse,
        test_docs_h1=test_docs_h1, test_corpus_h1=test_corpus_h1, test_sparse_h1=test_sparse_h1,
        test_docs_h2=test_docs_h2, test_corpus_h2=test_corpus_h2, test_sparse_h2=test_sparse_h2,
        train_timestamps=train_timestamps, val_timestamps=val_timestamps, test_timestamps=test_timestamps,
        dictionary=train_dictionary
    )
