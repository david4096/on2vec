import argparse
import logging
from on2vec.training import train_ontology_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model on an OWL ontology.")
    parser.add_argument('owl_file', type=str, help="Path to the OWL ontology file")
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'], help="Type of GNN model to use (gcn or gat)")
    parser.add_argument('--hidden_dim', type=int, default=16, help="Dimension of the hidden layer")
    parser.add_argument('--out_dim', type=int, default=8, help="Dimension of the output node embeddings")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--model_output', type=str, default='model.pt', help='Output path for the trained model checkpoint')
    parser.add_argument('--loss_fn', type=str, default='triplet', choices=['contrastive', 'triplet', 'cosine', 'cross_entropy'], help="Loss function to use")
    args = parser.parse_args()

    # Train the model using the package function
    result = train_ontology_embeddings(
        owl_file=args.owl_file,
        model_output=args.model_output,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        epochs=args.epochs,
        loss_fn_name=args.loss_fn
    )

    logging.info(f"Training completed successfully!")
    logging.info(f"Model saved to: {result['model_path']}")
    logging.info(f"Processed {result['num_nodes']} classes with {result['num_edges']} edges")
    logging.info(f"Model config: {result['model_config']}")


if __name__ == "__main__":
    main()