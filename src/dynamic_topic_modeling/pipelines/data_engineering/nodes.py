
from typing import Any, Dict

import pandas as pd

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary

def preprocess_UNGD(UNGD: pd.DataFrame, extreme_no_below: int, extreme_no_above: float, enable_bigram: bool, min_bigram_count: int) -> Dict[str, Any]:
    """Node for preprocessing the UN General Debates dataset.
    Parameters are taken from conf/base/parameters.yml.
    The data and the parameters will be loaded and provided to this function
    automatically when the pipeline is executed and it is time to run this node.

        Args:
            UNGD: Source data.
        Returns:
            Preprocessed dataset,
            corpus,
            dictionnary

    """

    UNGD = UNGD.reset_index().drop(columns=['session', 'country'])
    
    UNGD['text'] = UNGD['text'].str.lower()
    
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(UNGD)):
        UNGD['text'][idx] = tokenizer.tokenize(UNGD['text'][idx])  # Split into words.
        UNGD['text'][idx] = [w for w in UNGD['text'][idx] if not w in stop_words]

    # Remove numbers, but not words that contain numbers.
    UNGD['text'] = [[token for token in doc if not token.isnumeric()] for doc in UNGD['text']]

    # Remove words that are only one character.
    UNGD['text'] = [[token for token in doc if len(token) > 1] for doc in UNGD['text']]
    
    lemmatizer = WordNetLemmatizer()
    UNGD['text'] = [[lemmatizer.lemmatize(token) for token in doc] for doc in UNGD['text']]
    
    if enable_bigram:
        # Add bigrams and trigrams to docs (only ones that appear ... times or more).
        bigram = Phrases(UNGD['text'], min_count=min_bigram_count)
        for idx in range(len(UNGD['text'])):
            for token in bigram[UNGD['text'][idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    UNGD['text'][idx].append(token)
    
    dictionary = Dictionary(UNGD['text'])
    bef = len(dictionary)
    
    # Filter out words that occur less than ... documents, or more than ...% of the documents.
    dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
    print('####################')
    print('Number of unique tokens reduced from %d to %d' % (bef, len(dictionary)))

    corpus = [dictionary.doc2bow(doc) for doc in UNGD['text']]
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print('####################')
    
    return dict(
        UNGD_preprocessed=UNGD,
        corpus=corpus,
        dictionnary=dictionary,
    )