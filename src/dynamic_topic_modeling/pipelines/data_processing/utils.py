import numpy as np

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

def lowerize(docs):
    # Convert to lowercase.
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
    return docs

def tokenize(docs):
    # Split into words.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])
    return docs

def remove_stop_words(docs):
    stop_words = set(stopwords.words('english'))
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

def add_bigram(docs, min_bigram_count):
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs

def remove_vocab(docs, vocab):
    docs = np.array([[w for w in doc if w in vocab] for doc in docs])
    return docs

def remove_empty(docs, timestamps):
    tmp_docs = []
    tmp_timestamps = []
    for i, doc in enumerate(docs):
        if(doc != []):
            tmp_docs.append(doc)
            tmp_timestamps.append(timestamps[i])
    return tmp_docs, tmp_timestamps

def remove_by_threshold(docs, timestamps, threshold):
    tmp_docs = []
    tmp_timestamps = []
    for i, doc in enumerate(docs):
        if(len(doc) > threshold):
            tmp_docs.append(doc)
            tmp_timestamps.append(timestamps[i])
    return tmp_docs, tmp_timestamps

def convert_to_bow(docs, dictionary):
    return [dictionary.doc2bow(doc) for doc in docs]
