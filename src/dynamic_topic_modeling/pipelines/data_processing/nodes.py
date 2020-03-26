from typing import Any, Dict
from kedro.context import load_context

import os
import requests
from time import time
import pandas as pd
import numpy as np
from scipy import sparse

from gensim.corpora import Dictionary, MmCorpus
from gensim import matutils

from .dataset_preprocessing import preprocess_UN
from .utils import split_by_paragraph
from .utils import lowerize, tokenize, remove_stop_words, remove_numbers
from .utils import remove_word_with_length, lemmatize, add_bigram
from .utils import remove_vocab, remove_empty, remove_by_threshold
from .utils import convert_to_bow


def download_data_UN(id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    if not os.path.isfile(destination):

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        print('Downloading dataset...')
        save_response_content(response, destination)
        print('Done downloading')

    df = pd.read_csv(destination)
    df = preprocess_UN(df) # apply specific pre-processing specific to UN dataset
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

    docs = dataset['text']
    timestamps = dataset['timestamp']

    if flag_split_by_paragraph:
        print('\nSplitting by paragraph...')
        docs, timestamps = split_by_paragraph(docs, timestamps)

    print('\nLowerizing...')
    docs = lowerize(docs)

    print('\nTokenizing...')
    docs = tokenize(docs)

    if flag_word_analysis:

        print('Basic word analysis enabled. It will take more time to compute......\n')

        len_starting_vocab = len(Dictionary(docs))
        print('\nBeginning dictionary contains : {} words\n'.format(len_starting_vocab))

        print('\nRemoving stop words...')
        docs = remove_stop_words(docs)
        len_vocab = len(Dictionary(docs))
        len_rm_words = len_starting_vocab - len_vocab
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} stopwords from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))

        print('\nRemoving unique numbers (not words that contain numbers)...')
        docs = remove_numbers(docs)
        len_rm_words = len_vocab - len(Dictionary(docs))
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} numeric words from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))

        print('\nRemoving words that contain only one character...')
        remove_word_with_length(docs, length=1)
        len_rm_words = len_vocab - len(Dictionary(docs))
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\tRemoved {} one length characters from dictionary. It represents {}% of total words in starting vocabulary'.format(len_rm_words, freq))

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

    if flag_bigram:
        print('\nAdding bigrams...')
        docs = add_bigram(docs)

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

    # Split indexes into train/test/valid sets
    num_docs = len(docs)

    val_len = int(val_size * num_docs)
    test_len = int(test_size * num_docs)
    train_len = int(num_docs - val_len - test_len)

    idx_permute = np.random.permutation(num_docs).astype(int)

    # Split docs and timestamps into train/val/test sets
    train_docs = [docs[idx_permute[i]] for i in range(train_len)]
    val_docs = [docs[idx_permute[train_len+i]] for i in range(val_len)]
    test_docs = [docs[idx_permute[train_len+val_len+i]] for i in range(test_len)]

    train_timestamps = [timestamps[idx_permute[i]] for i in range(train_len)]
    val_timestamps = [timestamps[idx_permute[train_len+i]] for i in range(val_len)]
    test_timestamps = [timestamps[idx_permute[train_len+val_len+i]] for i in range(test_len)]

    print('  Number of documents in train set : {} [this should be equal to {} and {}]'.format(len(train_docs), train_len, len(train_timestamps)))
    print('  Number of documents in test set : {} [this should be equal to {} and {}]'.format(len(test_docs), test_len, len(test_timestamps)))
    print('  Number of documents in validation set: {} [this should be equal to {} and {}]'.format(len(val_docs), val_len, len(val_timestamps)))

    # Create a dictionary representation of the documents.
    print('Creating dictionary...')
    train_dictionary = Dictionary(train_docs)

    print('\nFiltering extremes...')
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    train_dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
    extreme_no_below_str = str(extreme_no_below) if extreme_no_below > 1 else str(extreme_no_below*100)+'%'
    extreme_no_above_str = str(extreme_no_above) if extreme_no_above > 1 else str(extreme_no_above*100)+'%'
    print('\tKeeping words in no less than {} documents & in no more than {} documents'.format(extreme_no_below_str, extreme_no_above_str))

    print('Number of unique tokens: %d' % len(train_dictionary))
    
    # Remove words not in train_data
    print('Removing words not in train data .....')
    train_vocab = train_dictionary.token2id
    val_docs = remove_vocab(val_docs, train_vocab)
    test_docs = remove_vocab(test_docs, train_vocab)
    print('  New vocabulary after removing words not in train: {}'.format(len(train_dictionary)))

    # Remove empty documents
    train_docs, train_timestamps = remove_empty(train_docs, train_timestamps)
    test_docs, test_timestamps = remove_empty(test_docs, test_timestamps)
    val_docs, val_timestamps = remove_empty(val_docs, val_timestamps)

    # Remove test documents with length=1
    test_docs, test_timestamps = remove_by_threshold(test_docs, test_timestamps, 1)

    # Split test set in 2 halves
    test_docs_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in test_docs]
    test_docs_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in test_docs]

    # Convert to Bag-of-Words representation
    print('Creating bow representation...')
    train_corpus = convert_to_bow(train_docs, train_dictionary)
    val_corpus = convert_to_bow(val_docs, train_dictionary)
    test_corpus = convert_to_bow(test_docs, train_dictionary)

    test_corpus_h1 = convert_to_bow(test_docs_h1, train_dictionary)
    test_corpus_h2 = convert_to_bow(test_docs_h2, train_dictionary)

    # Convert to sparse matrices (scipy COO sparse matrix)
    #sparse_train_corpus = matutils.corpus2csc(train_corpus).tocoo()
    #sparse_val_corpus = matutils.corpus2csc(val_corpus).tocoo()
    #sparse_test_corpus = matutils.corpus2csc(test_corpus).tocoo()

    #sparse_test_corpus_h1 = matutils.corpus2csc(test_corpus_h1).tocoo()
    #sparse_test_corpus_h2 = matutils.corpus2csc(test_corpus_h2).tocoo()

    print(' Train bag of words shape : {}'.format(len(train_corpus)))
    print(' Val bag of words shape : {}'.format(len(val_corpus)))
    print(' Test bag of words shape : {}'.format(len(test_corpus)))
    print(' Test set 1 bag of words shape : {}'.format(len(test_corpus_h1)))
    print(' Test set 2 bag of words shape : {}'.format(len(test_corpus_h2)))

    print('Done splitting data.')

    return dict(
        train_docs=train_docs, val_docs=val_docs, test_docs=test_docs,
        train_corpus=train_corpus, train_timestamps=train_timestamps,
        val_corpus=val_corpus, val_timestamps=val_timestamps,
        test_corpus=test_corpus, test_timestamps=test_timestamps,
        test_corpus_h1=test_corpus_h1,
        test_corpus_h2=test_corpus_h2,
        dictionary=train_dictionary
    )
