# Data Engineering pipeline


## Overview

This pipeline preprocesses the UN General Debates dataset into the corpus and dictionary associated (`preprocess_UNGD` node)

## Pipeline inputs


### `params:extreme_no_below`

|      |                    |
| ---- | ------------------ |
| Type | `int` |
| Description | Filter out words that occur less than ... documents |

### `params:extreme_no_above`

|      |                    |
| ---- | ------------------ |
| Type | `float` |
| Description | Filter out words that occur more than ...% of the documents |

### `params:enable_bigram`

|      |                    |
| ---- | ------------------ |
| Type | `bool` |
| Description | Add bigrams and trigrams |

### `params:min_bigram_count`

|      |                    |
| ---- | ------------------ |
| Type | `int` |
| Description | Add bigrams and trigrams that appear ... times or more |

### `params:download_dataset`

|      |                    |
| ---- | ------------------ |
| Type | `bool` |
| Description | Download the UNGD dataset from google drive when set to True |

### `params:basic_word_analysis`

|      |                    |
| ---- | ------------------ |
| Type | `bool` |
| Description | Performs basic word analysis at each preprocess stage when set to True |

## Pipeline outputs

### `UNGD_preprocessed`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing preprocessed text |

### `UN_corpus`

|      |                    |
| ---- | ------------------ |
| Type | `gensim.corpora.MmCorpus` |
| Description | doc2bow |

### `UN_dictionary`

|      |                    |
| ---- | ------------------ |
| Type | `gensim.corpora.dictionary` |
| Description | Mapping between words and their integer ids |
