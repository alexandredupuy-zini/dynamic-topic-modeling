from kedro.pipeline import Pipeline, node

from .nodes import train_model_lda
#from .nodes import eval_model_lda

train_model_lda_node = node(
    func=train_model_lda,
    inputs=['train_corpus',
            'dictionary',
            'params:num_topics'],
    outputs='trained_model_lda',
    name='Train model')

#eval_model_lda_node = node(
#    func=eval_model_lda,
#    inputs=["trained_model_lda"],
#    outputs=dict(),
#    name='Evaluate model')

def create_pipeline(**kwargs) :
    return Pipeline(
    [
        train_model_lda_node,
        #eval_model_lda_node
    ], tags="Machine Learning")
