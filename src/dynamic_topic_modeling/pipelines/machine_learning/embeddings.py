import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def split_by_sentence(docs):
    tmp = []
    for i, doc in enumerate(docs):
        splitted_doc = doc.split('.\n')
        for sd in splitted_doc:
            sentences = sd.split('. ')
            for s in sentences:
                tmp.append(s)
    return tmp

def lowerize(docs):
    # Convert to lowercase.
    for idx in range(len(docs)):
        docs[idx] = str(docs[idx]).lower()
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

def add_bigram(docs, min_bigram_count=20):
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=min_bigram_count)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs

def remove_vocab(docs, vocab):
    docs = np.array([[w for w in doc if w in vocab] for doc in docs])
    return docs

class MyDocuments(object):
    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        for line in self.docs:
            yield line

def get_word_embeddings(model, vocab):
    # Check embeddings
    print('Vocab length:', len(model.wv.vocab))
    print('Embedding size:', model.vector_size)
    print('Most similar to "war" in embedding space:', model.most_similar('war'))
    war, cold = model['war'].reshape((1, -1)), model['cold'].reshape((1, -1))
    print('Cosine distance between "cold" and "war" in embedding space (gensim metric):', model.similarity('cold', 'war'))
    print('Cosine distance between "cold" and "war" in embedding space (sklearn metric):', cosine_similarity(cold, war))

    words = list(vocab.token2id)
    words = [w for w in words if w in model.wv.vocab]

    embeddings = np.array([model[w] for w in words])
    embeddings_norm = preprocessing.normalize(embeddings)

    dict_embeddings = {w:emb for w,emb in zip(words, embeddings)}
    dict_embeddings_norm = {w:emb for w,emb in zip(words, embeddings_norm)}

    return dict_embeddings, dict_embeddings_norm


def train_word_embeddings(raw, vocab):
    # Get data
    docs = raw['text'].values


    # Pre-process data
    min_bigram_count = 20
    length = 1
    no_below = 100
    no_above = 0.70

    docs = dataset['text'].values

    print('\nSplitting by sentence...')
    docs = split_by_sentence(docs)

    print('\nLowerizing...')
    docs = lowerize(docs)

    print('\nTokenizing...')
    docs = tokenize(docs)

    #print('\nAdding bigrams...')
    #docs = add_bigram(docs, min_bigram_count=min_bigram_count)

    print('\nRemoving stop words...')
    docs = remove_stop_words(docs)

    print('\nRemoving unique numbers (not words that contain numbers)...')
    docs = remove_numbers(docs)

    print('\nRemoving words that contain only one character...')
    docs = remove_word_with_length(docs, length=length)

    print('\nLemmatizing...')
    docs = lemmatize(docs)

    vocab = Dictionary(docs)
    vocab.filter_extremes(no_below=no_below, no_above=no_above)

    docs = remove_vocab(docs, list(vocab.token2id))

    print('Number of sentences:', len(docs))
    print('Number of unique words:', len(vocab))


    # Prepare data
    sentences = MyDocuments(docs)

    # Train model
    size = 300
    window = 5
    min_count = 1
    workers = 8
    sg = 1
    iter = 50

    model = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)

    # Get embedding dict
    dict_embeddings, dict_embeddings_norm = get_word_embeddings(model, vocab)

    return model, dict_embeddings, dict_embeddings_norm


def read_corpus(docs):
    for i, text in enumerate(docs):
        yield TaggedDocument(text, [i])


def train_doc_embeddings(raw):
    # Get data
    docs = raw['text'].values


    # Pre-process data
    min_bigram_count = 20
    length = 1
    no_below = 100
    no_above = 0.70

    docs = dataset['text'].values

    print('\nSplitting by sentence...')
    docs = split_by_sentence(docs)

    print('\nLowerizing...')
    docs = lowerize(docs)

    print('\nTokenizing...')
    docs = tokenize(docs)

    #print('\nAdding bigrams...')
    #docs = add_bigram(docs, min_bigram_count=min_bigram_count)

    print('\nRemoving stop words...')
    docs = remove_stop_words(docs)

    print('\nRemoving unique numbers (not words that contain numbers)...')
    docs = remove_numbers(docs)

    print('\nRemoving words that contain only one character...')
    docs = remove_word_with_length(docs, length=length)

    print('\nLemmatizing...')
    docs = lemmatize(docs)

    vocab = Dictionary(docs)
    vocab.filter_extremes(no_below=no_below, no_above=no_above)

    docs = remove_vocab(docs, list(vocab.token2id))

    print('Number of sentences:', len(docs))
    print('Number of unique words:', len(vocab))


    # Prepare data
    sentences = list(read_corpus(docs))

    # Train model
    size = 300
    min_count = 1
    iter = 50
    model = Doc2Vec(vector_size=size, min_count=min_count, epochs=iter)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    return model
