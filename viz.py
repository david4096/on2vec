import polars as pl
import umap
import matplotlib.pyplot as plt
import logging
import argparse
import logging
import pyarrow as pa
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_embeddings_from_parquet(parquet_file):
    logging.info(f"Loading embeddings from Parquet file: {parquet_file}...")

    # Read Parquet file into a Polars DataFrame
    df = pl.read_parquet(parquet_file)

    # Extract node IDs and embeddings
    node_ids = df['node_id'].to_list()

    # Convert embedding series (containing lists) into a 2D array
    # Each embedding is stored as a list, so we need to stack them into a 2D array
    embeddings = np.vstack(df['embedding'].to_list())

    logging.info("Embeddings and node IDs loaded successfully.")
    
    return node_ids, embeddings
    
    return node_ids, embeddings

def generate_umap(embeddings, n_neighbors=15, min_dist=0.1):
    logging.info(f"Starting UMAP with {n_neighbors} neighbors and min_dist={min_dist}...")

    # Initialize UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    
    # Fit UMAP to the embeddings
    embedding_2d = reducer.fit_transform(embeddings)
    
    logging.info("UMAP reduction complete.")
    
    return embedding_2d

import matplotlib.pyplot as plt

def generate_umap_plot(embedding_2d, node_ids, output_file, width=10, height=8):
    """
    Generates a 2D UMAP plot and saves it to a file.
    """
    plt.figure(figsize=(width, height))

    # Plot the 2D UMAP embedding
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', s=10, alpha=0.6)

    # Add labels for a few points (optional, to check if node IDs are aligned)
    for i, node_id in enumerate(node_ids):
        if i % 50 == 0:  # Label every 50th point for clarity
            plt.text(embedding_2d[i, 0], embedding_2d[i, 1], str(node_id), fontsize=8)

    plt.title("2D UMAP Projection of Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Save the plot to the output file
    plt.savefig(output_file)
    plt.close()

    logging.info(f"UMAP plot saved to {output_file}")



def main():
    parser = argparse.ArgumentParser(description='2D UMAP Visualizer for Node Embeddings')
    parser.add_argument('parquet_file', type=str, help='Path to the Parquet file with node embeddings')
    parser.add_argument('--neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='Minimum distance for UMAP')
    parser.add_argument('--width', type=int, default=15, help='Width of the output plot (in pixels)')
    parser.add_argument('--height', type=int, default=10, help='Height of the output plot (in pixels)')
    parser.add_argument('--output', type=str, default='umap_plot.png', help='Output file for the plot (PNG format)')
    args = parser.parse_args()

    # Load embeddings from Parquet file
    node_ids, embeddings = load_embeddings_from_parquet(args.parquet_file)

    # Generate 2D UMAP
    embedding_2d = generate_umap(embeddings, n_neighbors=args.neighbors, min_dist=args.min_dist)

    # Save the 2D plot to a file
    generate_umap_plot(embedding_2d, node_ids, args.output, args.width, args.height)

if __name__ == "__main__":
    main()
