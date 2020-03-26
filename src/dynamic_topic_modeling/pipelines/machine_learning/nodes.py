from gensim.models import LdaModel

def train_model_lda(corpus, dictionary, num_topics=10):
    # Set training parameters.
    chunksize = 2000
    passes = 1
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    return model

#def eval_model_lda(model):
