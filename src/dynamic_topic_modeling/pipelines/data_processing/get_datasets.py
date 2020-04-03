import os
import requests
import pandas as pd

from gensim import matutils
from sklearn.datasets import fetch_20newsgroups

def preprocess_20NG(df):
    return df

def get_data_20NG():
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    tmp_train = [train_data.data[doc] for doc in range(len(train_data.data))]
    tmp_test = [test_data.data[doc] for doc in range(len(test_data.data))]
    data = tmp_train + tmp_test
    timestamps = [0] * len(data)
    df = pd.DataFrame({'text':data, 'timestamp':timestamps})
    df = preprocess_20NG(df)
    return df


def preprocess_UNGD(df):
    df.rename({'year':'timestamp'}, axis=1, inplace=True)
    df = df[['timestamp', 'text']].copy()
    return df

def get_data_UNGD(id='1Gx1oBjcsJOgklxLGZ8iEoNJ5XHnCEcsm',
                  destination='data/01_raw/un-general-debates.csv'):

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
    df = preprocess_UNGD(df) # apply specific pre-processing specific to UN dataset
    return df


def preprocess_SOGE(df):
    df.rename({'Date':'timestamp'}, axis=1, inplace=True)
    df.rename({'raisons_recommandation':'text'}, axis=1, inplace=True)
    df = df[['timestamp', 'text']].copy()
    return df

def get_data_SOGE(data_path='data/01_raw/verbatims_soge.csv'):
    df = pd.read_csv(data_path, sep=';', encoding='latin-1')
    df = preprocess_SOGE(df)
    return df
