import numpy as np
from scipy import sparse
import scipy.io
import datefinder
import pandas as pd
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases

def split_by_paragraph(docs, timestamps):
    tmp_docs, tmp_timestamps = [], []
    for i, doc in enumerate(docs):
        splitted_doc = doc.split('.\n')
        for sd in splitted_doc:
            tmp_docs.append(sd)
            tmp_timestamps.append(timestamps[i])
    return tmp_docs, tmp_timestamps

def handle_errors(dataset, no_na_n_obs):
    no_err=[]
    for i in range(no_na_n_obs) :
        try :
            pd.to_datetime(dataset['timestamp'].iloc[i])
            no_err.append(i)
        except :
            pass
    return dataset.iloc[no_err].copy()

def date_conversion(dataset):
    return dataset['timestamp'].apply(lambda x: [pd.to_datetime(str(i).split(' ')[0]) for i in datefinder.find_dates(str(x))][0])

def tokenize(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])
    return docs

def add_bigram(docs, min_bigram_count=20):
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=min_bigram_count, delimiter=b' ')
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs

def remove_stop_words(docs, language):
    if language=='en' :
        stop_words = set(stopwords.words('english'))
    elif language=='fr' :
        stop_words=set(stopwords.words('french'))
    for idx in range(len(docs)):
        docs[idx] = [w for w in docs[idx] if not w in stop_words]
    return docs

def remove_numbers(docs):
    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    return docs

def remove_word_with_length(docs, length=1):
    # Remove words that are only (length=1) character.
    docs = [[token for token in doc if len(token) > length] for doc in docs]
    return docs

def lemmatize(docs):
    # Lemmatize the documents
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    return docs

def remove_empty_docs(dataset):
    no_errors=[]
    for idx in range(len(dataset)) :
        if len(dataset['text'].iloc[idx])>1 :
            no_errors.append(idx)
    return dataset.iloc[no_errors]

def timestamps_preprocessing(dataset, n_years, n_months, temporality):
    if temporality=='year'  :
        n_periods = n_years

    elif temporality=='month' :
        n_periods=n_years*12+n_months

    elif temporality == "week" :
        n_periods=n_years*52+n_months*4

    date_range=pd.date_range(start=str(dataset['timestamp'].iloc[0]),end=str(dataset['timestamp'].iloc[-1]),periods=n_periods)
    if dataset['timestamp'].iloc[0] not in date_range :
        date_range=sorted(date_range.tolist()+[dataset['timestamp'].iloc[0]])
    mapper=dict(zip(date_range,[i for i in range(len(date_range))]))

    new_dic=defaultdict()
    for date,timeslice in mapper.items() :
        z=pd.date_range(start=pd.to_datetime(str(date).split(' ')[0]), periods=367)
        for dates in z :
            new_dic[dates]=timeslice
    dataset['timeslice']=dataset['timestamp'].apply(lambda x: new_dic[x])
    mapper_timeslices=dict(zip(np.unique(dataset['timeslice']),[i for i in range(len(np.unique(dataset['timeslice'])))]))
    dataset['timeslice']=dataset['timeslice'].apply(lambda x: mapper_timeslices[x])

    return dataset, date_range

def get_most_important_words(bow, vocab, top_n=10):
    dic=defaultdict()
    for idx,cnt in enumerate(np.array(bow.toarray().sum(axis=0)).flatten()) :
        dic[vocab[idx]]=cnt
    sorted_dic=dict(sorted(dic.items(), key=lambda kv: kv[1],reverse=True)[:top_n])
    return sorted_dic

def create_bow(doc_indices, words, n_docs, vocab_size):
    return scipy.sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]
