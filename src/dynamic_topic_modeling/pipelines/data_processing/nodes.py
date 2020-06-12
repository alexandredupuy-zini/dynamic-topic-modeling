from typing import Any, Dict
from kedro.context import load_context

from time import time
import pandas as pd
import numpy as np
import unidecode
from scipy import sparse
import torch
import scipy.io

from gensim.corpora import Dictionary

from .utils import split_by_paragraph, handle_errors, date_conversion, tokenize, add_bigram, remove_stop_words, remove_numbers, remove_word_with_length, lemmatize, remove_empty_docs, timestamps_preprocessing, get_most_important_words,create_bow, create_doc_indices, create_list_words

def preprocess_dataset(dataset : pd.DataFrame , extreme_no_below: int, extreme_no_above: float, enable_bigram: bool,
        min_bigram_count: int, basic_word_analysis : bool, lemmatizing : bool, temporality : str,
        language: str, path_to_texts_for_embedding : str, split_by_paragraph : bool) -> Dict[str, Any]:
    """Node for preprocessing the UN General Debates dataset.
        Parameters are taken from conf/base/parameters.yml.
        The data and the parameters will be loaded and provided to this function
        automatically when the pipeline is executed and it is time to run this node.

        Args:
            dataset: Source data. Must have a column named "text" to be processed. Dataset must be in catalog.yml
        Returns:
            Preprocessed dataset,
            vocabulary size,
            dictionnary,
            date range
        Parameters :
            extreme_no_below : if >1 : for a word w, delete this word from vocabulary if w in less than extreme_no_below documents. if in [0,1], for a word w, delete this word from vocabulary if w in less than extreme_no_below% documents
            extreme_no_above : in [0,1], for a word w, delete this word from vocabulary if w in more than extreme_no_below% documents
            enable_bigram : Boolean, decide if you want bigrams or not in the dictionary
            min_bigram_count : Int, threshold for bigrams :  Bigram will be added to the dictionary if in more than min_bigram_count documents
            basic_word_analysis : Boolean, set to True if you want to print some basic word anaylis (basically the number of words removed from each preprocces steps.)
            lemmatizing : Boolean, set to True if lemmatizing is wanted
            temporality : 'year', 'month' or 'week' according to desired time slices
            language : source language for the corpus
            path_to_texts_for_embedding : txt file containting materials for fasttext training
            split_by_paragraph : boolean set to True if documents need to be split by paragraphs
        """
    t0=time()

    print('\n\nCurrent set of parameters :\n')
    print('\textreme_no_below : {}'.format(extreme_no_below))
    print('\textreme_no_above : {}'.format(extreme_no_above))
    print('\tenable_bigram : {}'.format(enable_bigram))
    print('\tmin_bigram_count : {}'.format(min_bigram_count))
    print('\tlemmatizing : {}'.format(lemmatizing))
    print('\ttemporality : {}'.format(temporality))
    print('\tlanguage : {}\n'.format(language))
    print('\nStart preprocessing of dataset')

    if "text" not in dataset.columns :
            raise ValueError('Dataset does not have a column named "text". You must rename the your text column to "text".')
    if "timestamp" not in dataset.columns :
            raise ValueError('Dataset does not have a column named "timestamp". You must rename your time column to "timestamp".')

    if split_by_paragraph:
        print('\nSplitting by paragraphs...')
        dataset['text'], dataset['timestamp'] = split_by_paragraph(dataset['text'].values, dataset['timestamp'].values)

    dataset['raw_index']=dataset.index.values
    init_n_obs=dataset.shape[0]
    print('Starting number of observations : {}'.format(init_n_obs))

    ##Dropping NAN
    dataset.dropna(subset=['text','timestamp'],inplace=True)
    no_na_n_obs=dataset.shape[0]
    print('Number of observations after deleting missing values : {} = {} missing values'.format(no_na_n_obs,init_n_obs-no_na_n_obs))

    #Dropping errors on date
    dataset=handle_errors(dataset, no_na_n_obs)
    final_n_obs=dataset.shape[0]
    print('Final number of observations after handling errors on date : {} = {} errors on date'.format(final_n_obs,no_na_n_obs-final_n_obs))
    print('Deleted a total of {} observations.'.format(init_n_obs-final_n_obs))

    dataset['timestamp'] = date_conversion(dataset)

    dataset.sort_values('timestamp',inplace=True)
    dataset.reset_index(drop=True,inplace=True)
    dataset['index'] = dataset.index.values

    docs = dataset['text']

    docs = docs.str.lower()
    docs = docs.apply(lambda x: unidecode.unidecode(x))

    print('\nTokenizing...')
    docs = tokenize(docs)

    if basic_word_analysis :
        print('\nBasic word analysis enabled. It will take more time to compute...\n')

        if enable_bigram:
            print('\nAdding bigrams...')
            before_vocab=len(Dictionary(docs))
            docs = add_bigram(docs, min_bigram_count)
            bigram_vocab=len(Dictionary(docs))
            print('\nFound {} bigrams in text\n'.format(bigram_vocab-before_vocab))

        len_starting_vocab=len(Dictionary(docs))
        print('\nBeginning dictionary contains : {} words\n'.format(len_starting_vocab))

        print('\nRemoving stopwords...')
        docs = remove_stop_words(docs, language)
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

        print('-'*100)
        len_rm_words = len_starting_vocab-len_vocab
        freq = round(len_rm_words / len_starting_vocab, 3) * 100
        print('\nRemoved {} total words from beginning dictionary. It represents {}% of total words in starting vocabulary\n'.format(len_rm_words, freq))
        print('-'*100)

    else:
        print('\nWord analysis disabled')

        if enable_bigram:
            docs = add_bigram(docs, min_bigram_count)

        print('\nRemoving stopwords...')
        docs = remove_stop_words(docs, language)

        print('\nRemoving unique numbers (not words that contain numbers)...')
        docs = remove_numbers(docs)

        print('\nRemoving words that contain only one character...')
        docs = remove_word_with_length(docs, length=1)

    if lemmatizing :
        print('\nLemmatizing...')
        docs = lemmatize(docs)

    dataset['text'] = docs

    dictionary = Dictionary(dataset['text'])

    bef = len(dictionary)
    print('\nFiltering extremes...')
    dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
    if basic_word_analysis:
        print('\n')
        print('-'*100)
        if (extreme_no_above!=1) or (extreme_no_below!=1) :
            if extreme_no_below>1 :
                extreme_no_below_str=str(extreme_no_below)+' '+'documents'
            else :
                extreme_no_below_str = str(extreme_no_below*100)+'%'+' '+'documents'
            if extreme_no_above>1 :
                extreme_no_above_str=str(extreme_no_above)+' '+'documents'
            else :
                extreme_no_above_str = str(extreme_no_above*100)+'%'+' '+'documents'
        print('\nKeeping words in no less than {} & in no more than {}:'.format(extreme_no_below_str, extreme_no_above_str))
        print('Number of unique tokens reduced from {} to {}, representing {} % of total vocabulary.'.format(bef, len(dictionary),np.round(((bef-len(dictionary))/bef)*100,3)))

    dataset['text']=dataset['text'].apply(lambda x: [w for w in x if w in list(dictionary.token2id)])

    print('\nRemoving words that contain only one character...')
    dataset['text'] = remove_word_with_length(dataset['text'], length=1)

    print('\nDeleting rows that do not contain any text...')
    dataset = remove_empty_docs(dataset)
    print('\tDeleted {} rows because of no text'.format(final_n_obs-dataset.shape[0]))

    print('\nNumber of unique tokens: %d' % len(dictionary))
    print('\nNumber of documents: %d \n' % len(dataset))


    print('\nPreprocessing timestamps...')
    n_years=int(str(dataset['timestamp'].iloc[-1]).split('-')[0])-int(str(dataset['timestamp'].iloc[0]).split('-')[0])
    n_months=int(str(dataset['timestamp'].iloc[-1]).split('-')[1])-int(str(dataset['timestamp'].iloc[0]).split('-')[1])

    dataset, date_range = timestamps_preprocessing(dataset, n_years, n_months, temporality)

    date_range=[str(i).split(' ')[0] for i in date_range]

    for ind in range(len(date_range)-1) :
        print('Timeslice {} date range : from {} to {}'.format(ind,date_range[ind],date_range[ind+1]))
    for subsample in dataset.groupby('timeslice') :
        print('Number of observations for timeslice {} : {}'.format(subsample[0],subsample[1].shape[0]))
    print('-'*100)

    mapper_date=dict(zip([i for i in range(len(date_range))],date_range))

    dataset['text']=dataset['text'].apply(lambda x: ' '.join(x))
    good_idx=[]
    for idx in range(dataset.shape[0]) :
        if dataset['text'].iloc[idx] != '' :
            good_idx.append(idx)
    dataset=dataset.iloc[good_idx]

    print('\nBuilding file for fasttext training....')
    text_for_embeddings=list(dataset['text'])
    with open(path_to_texts_for_embedding,'w') as f :
        for text in text_for_embeddings :
            f.write(text+'\n')

    print('Final data shape : {}'.format(dataset.shape))
    print('\nDone in {} minutes'.format(int((time()-t0)/60)))

    return dict(
        dataset_preprocessed=dataset,
        dictionary=dictionary,
        vocab_size=len(dictionary),
        date_range=date_range
    )

def get_embeddings(filepath,vocab) :
    vectors = {}
    c=0
    with open(filepath, 'rb') as f:
        for l in f:
            if c%1000 == 0 :
                print(c)
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
            c+=1
    embeddings = np.zeros((len(vocab),300))
    for i, word in enumerate(vocab):
        try:
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(300, ))
    embeddings = torch.from_numpy(embeddings)
    return embeddings

def train_val_test(dataset : pd.DataFrame, dictionary : Dictionary ,
                   test_size: float , val_size : float) -> Dict[str,Any] :

    # Make train val test index
    num_docs = len(dataset)
    vaSize = int(np.floor(val_size*num_docs))
    tsSize = int(np.floor(test_size*num_docs))
    trSize = int(num_docs - vaSize - tsSize)
    idx_permute = np.random.permutation(num_docs).astype(int)
    print('Reading data....')

    # Make sure our text column is of type list
    dataset['text']=dataset['text'].apply(lambda x: x.split(' '))
    word2id = dict([(w, j) for j, w in dictionary.items()])
    id2word = dict([(j, w) for j, w in dictionary.items()])

    # Remove words not in train_data
    print('Starting vocabulary : {}'.format(len(dictionary)))

    vocab=list(dictionary)

    docs_tr = [[word2id[w] for w in dataset['text'][idx_permute[idx_d]] if w in word2id] for idx_d in range(trSize)]
    timestamps_tr = pd.DataFrame(dataset['timeslice'][idx_permute[range(trSize)]])
    idx_tr = idx_permute[range(trSize)]

    docs_ts = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize]] if w in word2id] for idx_d in range(tsSize)]
    timestamps_ts = pd.DataFrame(dataset['timeslice'][idx_permute[range(trSize,trSize+tsSize)]])
    idx_ts = idx_permute[range(trSize,trSize+tsSize)]

    docs_va = [[word2id[w] for w in dataset['text'][idx_permute[idx_d+trSize+tsSize]] if w in word2id] for idx_d in range(vaSize)]
    timestamps_va = pd.DataFrame(dataset['timeslice'][idx_permute[range(tsSize+trSize,num_docs)]])
    idx_va=idx_permute[range(tsSize+trSize,num_docs)]

    print('  Number of documents in train set : {} [this should be equal to {} and {}]'.format(len(docs_tr), trSize, len(timestamps_tr)))
    print('  Number of documents in test set : {} [this should be equal to {} and {}]'.format(len(docs_ts), tsSize, len(timestamps_ts)))
    print('  Number of documents in validation set: {} [this should be equal to {} and {}]'.format(len(docs_va), vaSize, len(timestamps_va)))

    # Split test set in 2 halves, the first containing the first half of the words in documents, and second part the second
    # half of words in documents. Will be use to gather test completion perplexity.

    print('Splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

    print('Creating lists of words...')

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    print('  Total number of words used in train set : ', len(words_tr))
    print('  Total number of words used in test set : ', len(words_ts))
    print('  Total number of words used in test firt set (first half of documents words): ', len(words_ts_h1))
    print('  Total number of words used in test firt set (first half of documents words): ', len(words_ts_h2))
    print('  Total number of words used in val set : ', len(words_va))

    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)

    # Get doc indices
    print('Getting doc indices...')

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)

    print('Creating bow representation...')

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

    print(' Train bag of words shape : {}'.format(bow_tr.shape))
    print(' Test bag of words shape : {}'.format(bow_ts.shape))
    print(' Test set 1 bag of words shape : {}'.format(bow_ts_h1.shape))
    print(' Test set 2 bag of words shape : {}'.format(bow_ts_h2.shape))
    print(' Val bag of words shape : {}'.format(bow_va.shape))

    print('\nMost import words in train BOW : \n')
    print(get_most_important_words(bow_tr,id2word))
    print('\nMost import words in val BOW : \n')
    print(get_most_important_words(bow_va,id2word))
    print('\nMost import words in test BOW : \n')
    print(get_most_important_words(bow_ts,id2word))
    print('\nDone splitting data.')

    return dict(
        BOW_train=bow_tr,
        BOW_test=bow_ts,
        BOW_test_h1=bow_ts_h1,
        BOW_test_h2=bow_ts_h2,
        BOW_val=bow_va,
        timestamps_train=timestamps_tr,
        timestamps_test=timestamps_ts,
        timestamps_val=timestamps_va,
        train_vocab_size=len(vocab),
        train_num_times=len(np.unique(timestamps_tr['timeslice'])),
        idx_train=idx_tr,
        idx_test=idx_ts,
        idx_val=idx_va
        )
