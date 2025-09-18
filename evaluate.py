from owlready2 import get_ontology
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity

def load_ontology(ontology_path):
    """
    Load an ontology from the specified path.
    
    Parameters:
    ontology_path (str): Path to the ontology file.
    
    Returns:
    Ontology: Loaded ontology object.
    """
    return get_ontology(ontology_path).load()


def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc


def evaluate_predictions(pairs, y_true, y_pred):
    """
    Evaluate the performance of a model using various metrics.
    
    Parameters:
    y_true (list): True labels.
    y_pred (list): Predicted labels.
    
    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop1 = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    ranks = {}
    franks = {}
    n = len(pairs)
    
    for c, d in pairs:
        preds = y_pred[c]
        rank = rankdata(-preds, method='average')[d]
        mean_rank += rank
        if rank <= 1:
            top1 += 1
        if rank <= 10:
            top10 += 1
        if rank <= 100:
            top100 += 1
        if rank not in ranks:
            ranks[rank] = 0
        ranks[rank] += 1
        
        f_preds = y_pred[c].copy()
        f_preds[np.where(y_true[c] == 1)] = 0.0
        f_preds[d] = y_pred[c, d]
        frank = rankdata(-f_preds, method='average')[d]
        fmean_rank += frank
        if frank <= 1:
            ftop1 += 1
        if frank <= 10:
            ftop10 += 1
        if frank <= 100:
            ftop100 += 1
        if frank not in franks:
            franks[frank] = 0
        franks[frank] += 1
    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank /= n
    ftop1 /= n
    ftop10 /= n
    ftop100 /= n
    fmean_rank /= n

    rank_auc = compute_rank_roc(ranks, len(y_true))
    frank_auc = compute_rank_roc(franks, len(y_true))

    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "top1": top1,
        "top10": top10,
        "top100": top100,
        "mean_rank": mean_rank,
        "rank_auc": rank_auc,
        "ftop1": ftop1,
        "ftop10": ftop10,
        "ftop100": ftop100,
        "fmean_rank": fmean_rank,
        "frank_auc": frank_auc
    }
    
def evaluate_embeddings(ontology, embeddings_df, relationship =['interacts_with']):
    # Example usage
    embeddings_dict = {}
    for _, row in embeddings_df.iterrows():
        embeddings_dict[row['node_id']] = np.array(row['embedding'])
    nodes_set = set()
    nodes = []
    for cls in ontology.classes():
        for rel in relationship:
            if hasattr(cls, rel) and cls.iri in embeddings_dict:
                nodes_set.add(cls)
    nodes=list(nodes_set)
    print("Ontology Classes:", len(nodes))
    nodes_dict = {node.iri: idx for idx, node in enumerate(nodes)}
    embeds = np.zeros((len(nodes), len(embeddings_df['embedding'].iloc[0])), dtype=np.float32)
    for idx in range(len(nodes)):
        embeds[idx] = embeddings_dict[nodes[idx].iri]
    y_true = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    pairs = []
    for node1 in nodes:
        for rel in relationship:
            for node2 in getattr(node1, rel):
                if node1.iri != node2.iri:
                    y_true[nodes_dict[node1.iri], nodes_dict[node2.iri]] = 1
                    pairs.append((nodes_dict[node1.iri], nodes_dict[node2.iri]))
    print("Positive pairs:", len(pairs))
    # Load embeddings
    y_pred = cosine_similarity(embeds)
    for c in range(len(nodes)):
        y_pred[c, c] = 0.0  # No self-loops
        y_true[c, c] = 0.0  # No self-loops
    metrics = evaluate_predictions(pairs, y_true, y_pred)
    print("Evaluation Metrics:", metrics)
    return metrics
    
if __name__ == "__main__":
    ontology_path = "ppi_human/test.owl"
    embeddings_path = "ppi-human-embeddings-cross.parquet"
    ontology = load_ontology(ontology_path)
    embeddings_df = pd.read_parquet(embeddings_path)
    evaluate_embeddings(ontology, embeddings_df)