import argparse
import logging
from on2vec.training import train_ontology_embeddings
from on2vec.embedding import embed_same_ontology

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description="Train a GNN and generate embeddings from an OWL ontology (integrated workflow).")
    parser.add_argument('owl_file', type=str, help="Path to the OWL ontology file")
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'],
                       help="Type of GNN model to use (gcn or gat)")
    parser.add_argument('--hidden_dim', type=int, default=16, help="Dimension of the hidden layer")
    parser.add_argument('--out_dim', type=int, default=8, help="Dimension of the output node embeddings")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='embeddings.parquet',
                       help='Output Parquet file for the embeddings')
    parser.add_argument('--model_output', type=str, default='model.pt',
                       help='Output path for the trained model checkpoint')
    parser.add_argument('--loss_fn', type=str, default='triplet',
                       choices=['contrastive', 'triplet', 'cosine', 'cross_entropy'],
                       help="Loss function to use")
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load existing model for embedding generation')
    args = parser.parse_args()

    if not args.skip_training:
        # Training phase
        logging.info("Starting training phase...")
        training_result = train_ontology_embeddings(
            owl_file=args.owl_file,
            model_output=args.model_output,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            epochs=args.epochs,
            loss_fn_name=args.loss_fn
        )
        logging.info(f"Training completed. Model saved to {training_result['model_path']}")

    # Embedding phase
    logging.info("Starting embedding generation phase...")
    embedding_result = embed_same_ontology(
        model_path=args.model_output,
        owl_file=args.owl_file,
        output_file=args.output
    )

    logging.info(f"Embeddings generated and saved to {args.output}")
    logging.info(f"Generated {len(embedding_result['node_ids'])} embeddings")



if __name__ == "__main__":
    main()