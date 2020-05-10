from kedro.pipeline import Pipeline, node 
from .train import get_model,train_model
from .eval import eval
from .predict import predict
from .fasttext_embedding import train_fasttext_embeddings
from .grid_search_fasttext import grid_search
from .plot_evolution import plot_words


def grid_search_fasttext_embedding(**kwargs) : 
    return Pipeline(
        [
            node(
                func=grid_search,
                inputs=[
                        "params:path_to_texts_for_embedding",
                        "params:param_grid"
                        ],
                outputs="Grid_search_fasttext_results"
                    
                )
            ]
        )
def create_pipeline_0(**kwargs) : 
    return Pipeline(
        [
            node(
                func=train_fasttext_embeddings,
                inputs=["params:path_to_texts_for_embedding","Verbatim_dictionary","params:dim","params:window","params:min_count","params:model","params:iterations"],
                outputs=["fasttext_model","fasttext_embeddings"]
                )
            ]
        )

def create_pipeline_1(**kwargs) : 

    return Pipeline(
        [
            node(
                func=get_model,
                inputs=["params:num_topics","train_num_times","train_vocab_size","params:t_hidden_size",
               "params:eta_hidden_size","params:rho_size","params:emb_size","params:enc_drop","params:eta_nlayers","params:eta_dropout",
               "params:theta_act","params:delta","params:gamma2","params:GPU","params:train_embeddings","params:seed","params:pretrained_embeddings",
               "fasttext_embeddings"],
                outputs="DETM_model"
                )
            ]
        )

#pipeline.only_nodes_with_tags(di)
#dir(Pipeline)



def create_pipeline_2(**kwargs) : 
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=[
                        "DETM_model",
                        "BOW_train",'timestamp_train',
                        "BOW_test","BOW_test_h1",'BOW_test_h2','timestamp_test',
                        "BOW_val","timestamp_val",
                        "params:eval_metric",
                        "params:log_interval", "params:batch_size","params:eval_batch_size","params:n_epochs","params:optimizer","params:learning_rate",
                        "params:wdecay","params:anneal_lr","params:nonmono","params:lr_factor", "params:clip_grad","params:seed","params:early_stopping",
                        "params:early_stopping_rounds"
                        ]
                ,
                outputs=["Trained_DETM_model","Word_distribution","Word_embedding","Topic_distribution","Topic_embedding"]
                )
            ]
        )

def create_pipeline_3(**kwargs) : 
    return Pipeline(
        [
            node(
                func=eval,
                inputs=[
                        "Trained_DETM_model",
                        "Word_distribution",
                        "Word_embedding",
                        "Topic_distribution",
                        'Verbatim_soge_preprocessed',
                        "Verbatim_soge_raw",
                        "BOW_train",
                        "Verbatim_dictionary",
                        "mapper_date",
                        "index_train_set",
                        "index_test_set",
                        "index_val_set",
                        "params:num_diversity",
                        "params:num_coherence"
                        ]
                ,
                outputs=dict(
                    eval_summary='eval_summary',
                    TD_all_times="TD_by_times",
                    overall_TD_times="Averaged_TD_by_times",
                    TD_all_topics="TD_by_topics",
                    overall_TD_topics="Averaged_TD_by_topics",
                    tc="Topic_coherence",
                    overall_tc="Averaged_TC",
                    quality="Topic_Quality",
                    topic_description_sw="Topic_description_with_sw",
                    topic_description_no_sw='Topic_description_without_sw'
                    )
                )
            ]
        )

def create_pipeline_4(**kwargs) : 
    return Pipeline(
        [
            node(
                func=predict,
                inputs=[
                        "Topic_distribution",
                        "Verbatim_soge_preprocessed",
                        "index_train_set",
                        "index_test_set",
                        "index_val_set"
                        ],
                outputs="verbatim_predicted_topics"
                    
                )
            ]
        )

def plot_word_evolution(**kwargs) : 
    return Pipeline(
        [
            node(
                func=plot_words,
                inputs=[
                        "params:query",
                        "Word_distribution",
                        "Verbatim_dictionary",
                        "mapper_date"
                        ],         
                 outputs=None
                )
            ]
        )
#def create_pipeline_1(**kwargs):
#
#		return Pipeline(
#			[
#				node(
#					func= get_data,
#					inputs=["BOW_train","BOW_test","BOW_test_h1","BOW_test_h2","BOW_val","timestamp_train",
#							"timestamp_test","timestamp_val","UN_dictionary"],
#					outputs=["train_rnn_inp","train_times","num_docs_train",
 #          "test_rnn_inp","test_times","num_docs_test",
  #         "valid_rnn_inp","valid_times","num_docs_valid",
 #          "test_1_rnn_inp","test_2_rnn_inp",
#           "vocab_size"],
#
#			
#					),
#				node(
#					func=get_model,
#					inputs=["params:num_topics", "params:num_times", "params:vocab_size", "params:t_hidden_size" , 
 #             "params:eta_hidden_size", "params:rho_size", "params:emb_size", "params:enc_drop", "params:eta_nlayers", 
  #            "params:train_embeddings", "params:theta_act", "params:delta", "embeddings"],
#					outputs=["DETM_model"]
#					
#				)
#			]
#		)'''
#		
