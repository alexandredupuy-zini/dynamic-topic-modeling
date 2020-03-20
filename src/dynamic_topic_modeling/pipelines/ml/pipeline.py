from kedro.pipeline import Pipeline, node 
from .train import get_model,train_model
from .eval import eval




def create_pipeline_1(**kwargs) : 

    return Pipeline(
        [
            node(
                func=get_model,
                inputs=["params:num_topics","train_num_times","train_vocab_size","params:t_hidden_size",
               "params:eta_hidden_size","params:rho_size","params:emb_size","params:enc_drop","params:eta_nlayers","params:eta_dropout",
               "params:theta_act","params:delta","params:GPU"],
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
                        "BOW_test_h1",'BOW_test_h2','timestamp_test',
                        "BOW_val","timestamp_val",
                        "params:log_interval", "params:batch_size","params:eval_batch_size","params:n_epochs","params:optimizer","params:learning_rate",
                        "params:wdecay","params:anneal_lr","params:nonmono","params:lr_factor", "params:clip_grad","params:seed"
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
                    )
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
