import argparse
import logging
from on2vec.embedding import embed_ontology_with_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from OWL ontology using a trained model.")
    parser.add_argument('model_path', type=str, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument('owl_file', type=str, help="Path to the OWL ontology file to embed")
    parser.add_argument('--output', type=str, default='embeddings.parquet', help='Output Parquet file for the embeddings')
    args = parser.parse_args()

    # Generate embeddings using the package function
    result = embed_ontology_with_model(
        model_path=args.model_path,
        owl_file=args.owl_file,
        output_file=args.output
    )

    if result['embeddings'] is not None:
        logging.info(f"Successfully generated {len(result['node_ids'])} embeddings")
        logging.info(f"Alignment ratio: {result['alignment_info']['alignment_ratio']:.2%}")

    else:
        logging.error("Failed to generate embeddings!")
        if result['alignment_info']['aligned_classes'] == 0:
            logging.error("No classes could be aligned between training and target ontology.")


if __name__ == "__main__":
    main()