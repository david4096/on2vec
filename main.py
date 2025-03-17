import argparse
import logging
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from owlready2 import get_ontology
from torch.optim import Adam
import time
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the GNN model for node-level embeddings
class OntologyGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, model_type='gcn'):
        super(OntologyGNN, self).__init__()
        if model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, out_dim)
        else:
            raise ValueError("Unsupported model type. Use 'gcn' or 'gat'.")
            
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Return per-node embeddings

def build_graph_from_owl(owl_file):
    ontology = get_ontology(owl_file).load()
    classes = list(ontology.classes())
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    edges = []
    for cls in classes:
        for parent in cls.is_a:
            if hasattr(parent, "iri") and parent in class_to_index:
                edges.append((class_to_index[parent], class_to_index[cls]))
                edges.append((class_to_index[cls], class_to_index[parent]))
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = len(classes)
    x = torch.eye(num_nodes, dtype=torch.float)
    return x, edge_index, class_to_index

def cross_entropy_loss(embeddings, edge_index, num_neg_samples=1000):
    # TODO broken?
    logging.info("Calculating cross entropy loss...")

    num_nodes = embeddings.size(0)
    
    # Positive pairs (connected nodes)
    pos_src = edge_index[0]
    pos_dst = edge_index[1]
    
    # Generate negative pairs using random sampling
    neg_indices_src = []
    neg_indices_dst = []
    negative_samples_found = 0

    while negative_samples_found < num_neg_samples:
        candidate_src = torch.randint(0, num_nodes, (num_neg_samples,))
        candidate_dst = torch.randint(0, num_nodes, (num_neg_samples,))

        # Ensure negative samples are not edges
        mask = ~torch.isin(torch.stack([candidate_src, candidate_dst], dim=0).t(), edge_index.t()).any(dim=1)

        valid_src = candidate_src[mask]
        valid_dst = candidate_dst[mask]

        neg_indices_src.append(valid_src)
        neg_indices_dst.append(valid_dst)

        negative_samples_found += valid_src.size(0)

    neg_indices_src = torch.cat(neg_indices_src)[:num_neg_samples]
    neg_indices_dst = torch.cat(neg_indices_dst)[:num_neg_samples]

    # Prepare the labels: 1 for positive pairs, 0 for negative pairs
    labels = torch.cat([torch.ones(pos_src.size(0)), torch.zeros(neg_indices_src.size(0))])

    # Calculate scores for positive and negative pairs
    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)
    neg_scores = (embeddings[neg_indices_src] * embeddings[neg_indices_dst]).sum(dim=1)

    # Concatenate the positive and negative scores
    scores = torch.cat([pos_scores, neg_scores])

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    logging.info("Cross entropy loss calculated.")
    return loss

def contrastive_loss(embeddings, edge_index, num_neg_samples=1000):
    logging.info("Calculating contrastive loss...")

    # Positive pairs (connected nodes)
    pos_loss = torch.norm(embeddings[edge_index[0]] - embeddings[edge_index[1]], dim=1).pow(2).sum()

    # Efficient negative sampling
    num_nodes = embeddings.size(0)
    negative_samples_found = 0
    neg_indices_src = []
    neg_indices_dst = []

    while negative_samples_found < num_neg_samples:
        # Sample random pairs
        candidate_src = torch.randint(0, num_nodes, (num_neg_samples,))
        candidate_dst = torch.randint(0, num_nodes, (num_neg_samples,))

        # Check if the pairs are in edge_index (i.e., actual edges)
        mask = ~torch.isin(torch.stack([candidate_src, candidate_dst], dim=0).t(), edge_index.t()).any(dim=1)

        valid_src = candidate_src[mask]
        valid_dst = candidate_dst[mask]

        # Add valid pairs to the list
        neg_indices_src.append(valid_src)
        neg_indices_dst.append(valid_dst)

        negative_samples_found += valid_src.size(0)

    # Concatenate lists into tensors
    neg_indices_src = torch.cat(neg_indices_src)[:num_neg_samples]
    neg_indices_dst = torch.cat(neg_indices_dst)[:num_neg_samples]

    if len(neg_indices_src) == 0:
        logging.warning("No valid negative samples found after extended search.")
        return pos_loss  # Only positive loss is considered if no negative pairs are valid

    # Negative pairs (non-connected nodes)
    neg_loss = torch.relu(1 - torch.norm(embeddings[neg_indices_src] - embeddings[neg_indices_dst], dim=1).pow(2)).sum()

    logging.info("Contrastive loss calculated.")
    return pos_loss + neg_loss

def triplet_loss(embeddings, edge_index, margin=1.0):
    logging.info("Calculating triplet loss...")
    
    anchor = embeddings[edge_index[0]]
    positive = embeddings[edge_index[1]]
    
    # Generate random negative samples
    num_nodes = embeddings.size(0)
    neg_indices = torch.randint(0, num_nodes, (edge_index.size(1),))
    negative = embeddings[neg_indices]

    loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin)
    
    logging.info("Triplet loss calculated.")
    return loss

def cosine_embedding_loss(embeddings, edge_index):
    logging.info("Calculating cosine embedding loss...")
    
    positive = embeddings[edge_index[0]]
    negative = embeddings[edge_index[1]]

    # Cosine similarity for positive pairs (connected nodes)
    pos_loss = F.cosine_embedding_loss(positive, negative, torch.ones_like(edge_index[0], dtype=torch.float))

    # Negative pairs (random)
    num_nodes = embeddings.size(0)
    neg_indices = torch.randint(0, num_nodes, (edge_index.size(1),))
    neg_embeds = embeddings[neg_indices]
    
    # Cosine similarity for negative pairs (random, non-connected nodes)
    neg_loss = F.cosine_embedding_loss(positive, neg_embeds, -torch.ones_like(edge_index[0], dtype=torch.float))
    
    logging.info("Cosine embedding loss calculated.")
    return pos_loss + neg_loss

def train(model, x, edge_index, optimizer, loss_fn, epochs=100):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch}/{epochs} starting...")
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        out = model(x, edge_index)
        loss = loss_fn(out, edge_index)
        
        loss.backward()
        optimizer.step()
        
        logging.info(f"Epoch {epoch} complete, Loss: {loss.item()}, Time: {time.time() - epoch_start_time:.2f}s")
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Loss {loss.item()}")

    logging.info(f"Training complete. Total Time: {time.time() - start_time:.2f}s")

def save_embeddings_to_parquet(embeddings, node_ids, output_file):
    logging.info("Saving embeddings to Parquet file...")
    
    # Convert embeddings tensor to a Python list or torch array
    embeddings_list = embeddings.cpu().tolist()
    
    # Create a Polars DataFrame with the node IDs and embeddings
    data = {'node_id': node_ids, 'embedding': embeddings_list}
    df = pl.DataFrame(data)
    
    # Convert to Arrow Table and write to Parquet
    arrow_table = df.to_arrow()
    pq.write_table(arrow_table, output_file)
    
    logging.info(f"Embeddings saved to {output_file}.")

def extract_node_ids_from_ontology(class_to_index):
    # Extracts node identifiers (IRIs) from the class_to_index mapping
    node_ids = [cls.iri for cls in class_to_index.keys()]
    return node_ids

    
    return node_ids
def main():
    parser = argparse.ArgumentParser(description="Train a GNN to generate node embeddings from an OWL ontology.")
    parser.add_argument('owl_file', type=str, help="Path to the OWL ontology file")
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'], help="Type of GNN model to use (gcn or gat)")
    parser.add_argument('--hidden_dim', type=int, default=16, help="Dimension of the hidden layer")
    parser.add_argument('--out_dim', type=int, default=8, help="Dimension of the output node embeddings")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='embeddings.parquet', help='Output Parquet file for the embeddings')
    parser.add_argument('--loss_fn', type=str, default='triplet', choices=['contrastive', 'triplet', 'cosine', 'cross_entropy'], help="Loss function to use")
    args = parser.parse_args()

    logging.info(f"Loading OWL ontology from {args.owl_file}...")
    x, edge_index, class_to_index = build_graph_from_owl(args.owl_file)
    logging.info("Ontology loaded and graph built.")
    node_ids = extract_node_ids_from_ontology(class_to_index)

    input_dim = x.size(1)
    model = OntologyGNN(input_dim, args.hidden_dim, args.out_dim, model_type=args.model_type)
    optimizer = Adam(model.parameters(), lr=0.01)

    # Select the loss function
    if args.loss_fn == 'contrastive':
        loss_fn = contrastive_loss
    elif args.loss_fn == 'triplet':
        loss_fn = triplet_loss
    elif args.loss_fn == 'cosine':
        loss_fn = cosine_embedding_loss
    elif args.loss_fn == 'cross_entropy':
        loss_fn = cross_entropy_loss

    logging.info("Starting training...")
    train(model, x, edge_index, optimizer, loss_fn, epochs=args.epochs)
    # Get final embeddings for all nodes
    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index)

    # Save embeddings to Parquet
    save_embeddings_to_parquet(embeddings, node_ids, args.output)

if __name__ == "__main__":
    main()
