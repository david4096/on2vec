import optuna
import pandas as pd
import plotly
from evaluate import load_ontology, evaluate_embeddings
from on2vec.training import train_ontology_embeddings, train_text_augmented_ontology_embeddings
from on2vec.embedding import embed_ontology_with_model
from on2vec.evaluation import evaluate_embeddings as on2vec_evaluate_embeddings
def train_ontology_embeddings_wrapper(owl_file, 
        model_output, 
        model_type, 
        hidden_dim,
        out_dim,
        epochs,
        loss_fn_name,
        learning_rate,
        use_multi_relation,
        dropout,
        num_bases = None,
        include_text_features=False
        #text_model_type=None, 
        #text_model_name=None, 
        #fusion_method="concat"
        ):
    # Wrapper function to train ontology embeddings with fixed parameters
    model=None
    if include_text_features:    
    #    train_text_augmented_ontology_embeddings(owl_file, model_output,
    #                                       text_model_type='sentence_transformer',
    #                                       text_model_name='all-MiniLM-L6-v2',
    #                                       backbone_model='gcn', fusion_method='concat',
    #                                       hidden_dim=128, out_dim=64,
    #                                       epochs=100, loss_fn_name='triplet',
    #                                       learning_rate=0.01, dropout=0.0) """
        model =train_text_augmented_ontology_embeddings(
            owl_file=owl_file,
            model_output=model_output,
            #text_model_type=text_model_type,  # Default text model type, can be changed
            #text_model_name=text_model_name,  # Default text model name, can be changed
            backbone_model=model_type,  # Default backbone model, can be changed
            #fusion_method=fusion_method,  # Default fusion method, can be changed)
            hidden_dim=hidden_dim,    # Default hidden dimension, can be changed
            out_dim=out_dim,       # Default output dimension, can be changed
            epochs=epochs,
            loss_fn_name=loss_fn_name,  # Default loss function, can be changed
            learning_rate=learning_rate,     # Default learning rate, can be changed
            dropout=dropout)    
    else: 
        model=train_ontology_embeddings(
            owl_file=owl_file,
            model_output=model_output,
            model_type=model_type,  # Default model type, can be changed
            hidden_dim=hidden_dim,    # Default hidden dimension, can be changed
            out_dim=out_dim,       # Default output dimension, can be changed
            epochs=epochs,
            loss_fn_name=loss_fn_name,  # Default loss function, can be changed
            learning_rate=learning_rate,     # Default learning rate, can be changed
            use_multi_relation=use_multi_relation,
            dropout=dropout,
            num_bases=num_bases)
    return model



def define_objective(trial, owl_file, model_output, epochs, use_multi_relation=False, dropout=0.0, num_bases=None, parquet_file='embeddings.parquet', include_text_features=False, relationship=['rdfs:subClassOf'],evaluator='on2vec_eval'):
    # Hyperparameters to optimize
    # Load ontology
    ontology = load_ontology(owl_file)
    hidden_dim = trial.suggest_int("hidden_dim",4, 256)
    out_dim = trial.suggest_int("out_dim", 4, 256)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model_type = trial.suggest_categorical("model_type", ['gcn', 'gat', 'heterogeneous'])
    loss_fn_name = trial.suggest_categorical("loss_fn_name", ['cosine', 'cross_entropy'])
    #text_model_type = None
    #if include_text_features:
    #    text_model_type = trial.suggest_categorical("text_model_type", ['sentence_transformer', 'huggingface', 'openai', 'tfidf'])
    
    # Train embeddings

#   def train_ontology_embeddings(owl_file, model_output, model_type='gcn', hidden_dim=128, out_dim=64,
   #                         epochs=100, loss_fn_name='triplet', learning_rate=0.01, use_multi_relation=False,
    #                        dropout=0.0, num_bases=None):
  #  """
   # Complete training pipeline from OWL file to saved model.

#    Args:
 #       owl_file (str): Path to OWL ontology file
  #      model_output (str): Path to save trained model
   #     model_type (str): Type of GNN model ('gcn', 'gat', 'rgcn', 'weighted_gcn', 'heterogeneous')
    ##   out_dim (int): Output embedding dimension
      # loss_fn_name (str): Name of loss function
       # learning_rate (float): Learning rate
  #      use_multi_relation (bool): Use multi-relation graph building
  #      dropout (float): Dropout rate for multi-relation models
  #      num_bases (int, optional): Number of bases for RGCN decomposition
#"""
    model = train_ontology_embeddings_wrapper(
        owl_file=owl_file, 
        model_output=model_output, 
        model_type=model_type, 
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        epochs=epochs,
        loss_fn_name=loss_fn_name,
        learning_rate=learning_rate,
        use_multi_relation=use_multi_relation,
        dropout=dropout,
        num_bases=num_bases,
        include_text_features=include_text_features,
        #fusion_method=fusion_method
        )
    
    embed_ontology_with_model(owl_file=owl_file, 
                              model_path=model_output,
                            output_file=parquet_file
)
   
    roc_auc = 0.0
    #mean_rank = float('inf')
    if(evaluator=='on2vec_eval'):
        vals = on2vec_evaluate_embeddings(parquet_file,owl_file)
        roc_auc = vals["roc_auc"]
        #mean_rank = vals["mean_rank"]
    else:
         # Evaluate embeddings
        ontology = load_ontology(owl_file)
        embeddings_df = pd.read_parquet(parquet_file)
        metrics = evaluate_embeddings(ontology,embeddings_df, relationship=relationship)
        roc_auc = metrics["roc_auc"]
        #mean_rank = metrics["mean_rank"]
    # Evaluate model

    return roc_auc #, mean_rank
study = optuna.create_study(directions=["maximize"])
study.optimize(lambda trial: define_objective(trial, owl_file='EDAM.owl', model_output='model.pth', epochs=10, use_multi_relation=True, dropout=0.0, num_bases=5, parquet_file='embeddings.parquet',include_text_features=False, relationship= None, evaluator='on2vec_eval' 
                                              #,fusion_method='concat'
                                              ), n_trials=50)
paretoplot=optuna.visualization.plot_pareto_front(study, target_names=["roc_auc"])  
paretoplot.write_html("pareto_front.html")
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
# Print details of the trial with the highest accuracy
trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print("Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
#Plot of hyperparameter importance for roc_auc
hyperparameterimportanceroc=optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="roc_auc"
)
hyperparameterimportanceroc.write_html("hyperparameter_importance_roc.html")
#Plot of hyperparameter importance for mean_rank
#hyperparameterimportancemeanrank=optuna.visualization.plot_param_importances(
#    study, target=lambda t: t.values[0], target_name="mean_rank"
#)
