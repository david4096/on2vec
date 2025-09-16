from owlready2 import get_ontology
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def load_ontology(ontology_path):
    """
    Load an ontology from the specified path.
    
    Parameters:
    ontology_path (str): Path to the ontology file.
    
    Returns:
    Ontology: Loaded ontology object.
    """
    return get_ontology(ontology_path).load()

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a model using various metrics.
    
    Parameters:
    y_true (list): True labels.
    y_pred (list): Predicted labels.
    
    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_pred)
    }


def main():
    # Example usage
    ontology_path = "ppi_yeast/test.owl"
    ontology = load_ontology(ontology_path)
    
    # Dummy data for evaluation
    y_true = [0, 1, 1, 1, 1]
    y_pred = [0.1, 0.9, 0.8, 0.1, 0.95]
    
    nodes = list(ontology.classes())
    nodes_dict = {node.name: idx for idx, node in enumerate(nodes)}
    print("Nodes Dictionary:", nodes_dict)
    print("Ontology Classes:", len(nodes))
    edge_types = list(ontology.object_properties())
    y_true = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    for node1 in nodes:
        for node2 in node1.interacts_with:
            y_true[nodes_dict[node1.name], nodes_dict[node2.name]] = 1
    # Load embeddings
    df = pd.read_parquet("ppi-yeast-embeddings.parquet")
    print(df.head())
    embeddings = {}
    y_pred = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for _, row in df.iterrows():
        embeddings[row['node_id']] = np.array(row['embedding'])
    for node1 in nodes:
        if node1.iri in embeddings:
            emb1 = embeddings[node1.iri]
            for node2 in node1.interacts_with:
                if node2.iri in embeddings:
                    emb2 = embeddings[node2.iri]
                    score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    y_pred[nodes_dict[node1.name], nodes_dict[node2.name]] = score
    metrics = evaluate_model(y_true, y_pred)
    print("Evaluation Metrics:", metrics)
    
if __name__ == "__main__":
    main()