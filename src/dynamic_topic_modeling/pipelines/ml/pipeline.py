from kedro.pipeline import Pipeline, node 


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
