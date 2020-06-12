# Dynamic Topic Modeling

## Overview

This project combines the DETM model and fastText embeddings.

This is project was generated using `Kedro 0.15.4`, which document is available [here](https://kedro.readthedocs.io).

## Data

Data needs to be added in the following file : `data/01_raw/`.

Outputs are saved in the `data` file, except for the `plot_word_evolution` graphs that are saved in `results`.

## Parameters

To specify the input dataset, you need to adapt the `DataSet` parameter in `conf/base/catalog.yml` with the correct filepath and arguments.

You might want to custom some parameters, which are located in `conf/base/parameters.yml`, such as :
* enable_bigram
* basic_word_analysis
* lemmatize
* temporality
* language
* additionnal_stop_words
* window
* iterations
* min_count
* param_grid
* word_to_check
* num_topics
* n_epochs
* query

## Commands

To install dependencies, run:

```
kedro install
```

To run the project, run:

```
kedro run
```

To run a grid search for the embeddings, run:

```
kedro run --pipeline grid_search_fasttext
```

To visualize the results of the model with specific words, run:

```
kedro run --pipeline plot_word_evolution
```
