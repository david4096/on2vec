import argparse
import logging
from on2vec.training import train_ontology_embeddings, train_text_augmented_ontology_embeddings
from on2vec.embedding import embed_same_ontology

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description="Train a GNN and generate embeddings from an OWL ontology (integrated workflow).")
    parser.add_argument('owl_file', type=str, help="Path to the OWL ontology file")
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'rgcn', 'weighted_gcn', 'heterogeneous'],
                       help="Type of GNN model to use")
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
    parser.add_argument('--use_multi_relation', action='store_true',
                       help="Use multi-relation graph with all ObjectProperty relations")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (for multi-relation models)")
    parser.add_argument('--num_bases', type=int, help="Number of bases for RGCN decomposition (optional)")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for optimizer")

    # Text-augmented embedding arguments
    parser.add_argument('--use_text_features', action='store_true',
                       help="Enable text-augmented embeddings using semantic features from ontology")
    parser.add_argument('--text_model_type', type=str, default='sentence_transformer',
                       choices=['sentence_transformer', 'huggingface', 'openai', 'tfidf'],
                       help="Type of text embedding model to use")
    parser.add_argument('--text_model_name', type=str, default='all-MiniLM-L6-v2',
                       help="Name of the text model (e.g., 'all-MiniLM-L6-v2', 'bert-base-uncased')")
    parser.add_argument('--fusion_method', type=str, default='concat',
                       choices=['concat', 'add', 'weighted_sum', 'attention'],
                       help="Method to combine structural and text features")

    args = parser.parse_args()

    # Auto-enable multi-relation for models that require it
    if args.model_type in ['rgcn', 'weighted_gcn', 'heterogeneous']:
        args.use_multi_relation = True
        logging.info(f"Auto-enabling multi-relation graph for {args.model_type} model")

    if not args.skip_training:
        # Training phase
        if args.use_text_features:
            logging.info("Starting text-augmented training phase...")
            logging.info(f"Text model: {args.text_model_type} - {args.text_model_name}")
            logging.info(f"Fusion method: {args.fusion_method}")
            training_result = train_text_augmented_ontology_embeddings(
                owl_file=args.owl_file,
                model_output=args.model_output,
                text_model_type=args.text_model_type,
                text_model_name=args.text_model_name,
                backbone_model=args.model_type,
                fusion_method=args.fusion_method,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                epochs=args.epochs,
                loss_fn_name=args.loss_fn,
                learning_rate=args.learning_rate,
                dropout=args.dropout
            )
            logging.info(f"Text-augmented training completed. Model saved to {training_result['model_path']}")
            logging.info(f"Structural features: {training_result['structural_dim']}, Text features: {training_result['text_dim']}")
            logging.info(f"Text features extracted for {training_result['text_features_extracted']} classes")
        else:
            logging.info("Starting standard training phase...")
            training_result = train_ontology_embeddings(
                owl_file=args.owl_file,
                model_output=args.model_output,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                epochs=args.epochs,
                loss_fn_name=args.loss_fn,
                learning_rate=args.learning_rate,
                use_multi_relation=args.use_multi_relation,
                dropout=args.dropout,
                num_bases=args.num_bases
            )
            logging.info(f"Training completed. Model saved to {training_result['model_path']}")
            if training_result.get('num_relations', 0) > 0:
                logging.info(f"Multi-relation graph with {training_result['num_relations']} relation types")

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