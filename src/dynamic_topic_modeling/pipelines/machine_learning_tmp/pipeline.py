from kedro.pipeline import Pipeline, node

from .nodes import get_model, train_model

def create_pipeline(**kwargs) :
    return Pipeline(
        [
            node(
                func=get_model,
                inputs=["params:num_topics",
                        "train_num_times",
                        "train_vocab_size",
                        "params:t_hidden_size",
                        "params:eta_hidden_size",
                        "params:rho_size",
                        "params:emb_size",
                        "params:enc_drop",
                        "params:eta_nlayers",
                        "params:eta_dropout",
                        "params:theta_act",
                        "params:delta",
                        "params:GPU"],
                outputs="DETM_model",
                name='Get model'
                ),
            node(
                func=train_model,
                inputs=["DETM_model",
                        "train_corpus", 'train_timestamps',
                        "test_corpus_h1", 'test_corpus_h2', 'test_timestamps',
                        "val_corpus", "val_timestamps",
                        "params:log_interval", "params:batch_size",
                        "params:eval_batch_size", "params:n_epochs",
                        "params:optimizer", "params:learning_rate",
                        "params:wdecay", "params:anneal_lr", "params:nonmono",
                        "params:lr_factor", "params:clip_grad", "params:seed"],
                outputs=["Trained_DETM_model",
                         "Word_distribution",
                         "Word_embedding",
                         "Topic_distribution",
                         "Topic_embedding"],
                name='Train model'
                ),
            node(
                func=eval,
                inputs=["Trained_DETM_model",
                        "Word_distribution",
                        "BOW_train",
                        "UN_dictionary",
                        "params:num_diversity",
                        "params:num_coherence"
                        ]
                ,
                outputs=dict(
                    TD_all="TD_by_times",
                    TD_times="Averaged_TD_by_times",
                    TD_topics="TD_by_topics",
                    TD_all_topics="Averaged_TD_by_topics",
                    tc="Topic_coherence",
                    overall_tc="Averaged_TC",
                    quality="Topic_Quality"
                    ),
                name='Evaluate model'
                )
            ], tags="Machine Learning"
        )
