import optuna
import pandas as pd
import plotly
from evaluate import load_ontology, evaluate_embeddings
from on2vec.training import train_ontology_embeddings
from on2vec.embedding import embed_ontology_with_model

def define_objective(trial, owl_file, model_output, epochs = 100, loss_fn_name='triplet', use_multi_relation=False, dropout=0.0, num_bases=None, parquet_file='embeddings.parquet'):
    # Hyperparameters to optimize
    # Load ontology
    ontology = load_ontology(owl_file)
    hidden_dim = trial.suggest_int("hidden_dim",4, 256)
    out_dim = trial.suggest_int("out_dim", 4, 256)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model_type = trial.suggest_categorical("model_type", ['gcn', 'gat'])
    #loss_fun_nm = trial.suggest_categorical("loss_fun_nm", ['pairwise', 'nll', 'cross_entropy'])

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
    model = train_ontology_embeddings(
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
        num_bases=num_bases)
    
    embeddings = embed_ontology_with_model(owl_file=owl_file, 
                              model_path=model_output,
                            output_file=parquet_file
)
    # Evaluate embeddings
    ontology = load_ontology(owl_file)
    embeddings_df = pd.read_parquet(parquet_file)
    metrics = evaluate_embeddings(ontology,embeddings_df)
    roc_auc = metrics["roc_auc"]
    mean_rank = metrics["mean_rank"]
    # Evaluate model

    return roc_auc, mean_rank
study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(lambda trial: define_objective(trial, owl_file='test.owl', model_output='model.pth', epochs=10, loss_fn_name='triplet', use_multi_relation=True, dropout=0.2, num_bases=5, parquet_file='embeddings.parquet'), n_trials=50)
optuna.visualization.plot_pareto_front(study, target_names=["roc_auc", "mean_rank"])  
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print("Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="roc_auc"
)
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="mean_rank"
)
