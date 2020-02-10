# Data Engineering pipeline


## Overview

This pipeline preprocesses the UN General Debates dataset into the corpus and dictionary associated (`preprocess_UNGD` node)

## Pipeline inputs

### `UNGD`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Input data to preprocess |

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

## Pipeline outputs

### `UNGD_preprocessed`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set features |

### `corpus`

|      |                    |
| ---- | ------------------ |
| Type | `gensim.corpora.MmCorpus` |
| Description | doc2bow |

### `dictionary`

|      |                    |
| ---- | ------------------ |
| Type | `gensim.corpora.dictionary` |
| Description | Mapping between words and their integer ids |
