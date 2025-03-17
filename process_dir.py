import os
import subprocess
import argparse

def process_owl_files(directory, model_type, hidden_dim, out_dim, epochs, loss_fn, neighbors, min_dist, width, height, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all OWL files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".owl"):
            owl_file_path = os.path.join(directory, file_name)
            
            # Define the output path for the embedding (Parquet file)
            parquet_output = os.path.join(output_dir, f"{file_name.replace('.owl', '')}_embeddings.parquet")
            
            print(f"Processing {owl_file_path}...")
            
            # Call main.py to generate embeddings for the current OWL file
            main_args = [
                "python", "main.py",
                "--model_type", model_type,
                "--hidden_dim", str(hidden_dim),
                "--out_dim", str(out_dim),
                "--epochs", str(epochs),
                "--output", parquet_output,
                "--loss_fn", loss_fn,
                owl_file_path
            ]

            try:
                subprocess.run(main_args, check=True)
                print(f"Generated embeddings for {owl_file_path} at {parquet_output}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {owl_file_path}: {e}")
                continue

            # Call viz.py to visualize the resulting Parquet file
            viz_args = [
                "python", "viz.py",
                "--neighbors", str(neighbors),
                "--min_dist", str(min_dist),
                "--width", str(width),
                "--height", str(height),
                "--output", os.path.join(output_dir, f"{file_name.replace('.owl', '')}_visualization.png"),
                parquet_output
            ]

            try:
                subprocess.run(viz_args, check=True)
                print(f"Visualization created for {parquet_output}")
            except subprocess.CalledProcessError as e:
                print(f"Error visualizing {parquet_output}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OWL files to generate embeddings and visualize them.")
    parser.add_argument("directory", help="Directory containing OWL files.")
    parser.add_argument("--model_type", default="gcn", choices=["gcn", "gat"], help="Type of model to use.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Hidden dimension size.")
    parser.add_argument("--out_dim", default=64, type=int, help="Output dimension size.")
    parser.add_argument("--epochs", default=256, type=int, help="Number of training epochs.")
    parser.add_argument("--loss_fn", default="cosine", choices=["contrastive", "triplet", "cosine", "cross_entropy"], help="Loss function to use.")
    parser.add_argument("--neighbors", default=10, type=int, help="Number of neighbors for UMAP visualization.")
    parser.add_argument("--min_dist", default=0.1, type=float, help="Minimum distance for UMAP visualization.")
    parser.add_argument("--width", default=16, type=int, help="Width of the visualization.")
    parser.add_argument("--height", default=12, type=int, help="Height of the visualization.")
    parser.add_argument("--output_dir", default="output", help="Directory to save the output Parquet and visualization files.")
    
    args = parser.parse_args()

    process_owl_files(
        args.directory,
        args.model_type,
        args.hidden_dim,
        args.out_dim,
        args.epochs,
        args.loss_fn,
        args.neighbors,
        args.min_dist,
        args.width,
        args.height,
        args.output_dir
    )
