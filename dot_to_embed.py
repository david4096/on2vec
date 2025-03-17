import argparse
import polars as pl
import umap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from owlready2 import get_ontology
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ontology(owl_file):
    logger.info(f"Loading OWL ontology from {owl_file}")
    onto = get_ontology(owl_file).load()
    return onto

def map_class_names(ontology):
    logger.info("Mapping full IRIs to class names")
    return {cls.name: cls.iri for cls in ontology.classes()}

def load_embeddings_parquet(parquet_file):
    logger.info(f"Loading GNN embeddings from {parquet_file}")
    df = pl.read_parquet(parquet_file)
    return df

def load_dot_layout_parquet(parquet_file):
    logger.info(f"Loading DOT layout from {parquet_file}")
    df = pl.read_parquet(parquet_file)
    return df

def normalize_data(data):
    # Normalize the data using MinMaxScaler to scale values between 0 and 1
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    logger.info("Normalized data to range [0, 1]")
    return normalized_data

def calculate_umap(embedding_df):
    embeddings = np.array(embedding_df["embedding"].to_list())
    
    logger.info("Calculating UMAP projection")
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    umap_embedding = umap_model.fit_transform(embeddings)
    return umap_embedding

def save_parquet(df, output_path):
    logger.info(f"Saving to Parquet file: {output_path}")
    df.write_parquet(output_path)

def create_animation(dot_layout, umap_embedding, output_path):
    logger.info("Creating animation")

    # Normalize the x-y data between 0 and 1 for colormap
    norm = plt.Normalize(dot_layout[:, 0].min(), dot_layout[:, 0].max())
    
    # Apply a colormap based on normalized x positions (can use y positions too)
    colors = plt.cm.viridis(norm(dot_layout[:, 0]))  # Colormap based on x values
    fig, ax = plt.subplots(figsize=(16, 10))
    scat = ax.scatter(dot_layout[:, 0], dot_layout[:, 1], s=3, c=colors)

    normed_umap = normalize_data(umap_embedding)

    total_frames = 100  # Number of frames for each direction
    pause_frames = 50    # Pause for 50 frames at each extreme (start and end)

    def update(frame):
        if frame < total_frames:  # Forward direction
            alpha = frame / total_frames
        elif frame < total_frames + pause_frames:  # Pause at end
            alpha = 1
        elif frame < 2 * total_frames + pause_frames:  # Reverse direction
            alpha = 1 - (frame - total_frames - pause_frames) / total_frames
        else:  # Pause at start
            alpha = 0

        interpolated = (1 - alpha) * dot_layout + alpha * normed_umap
        scat.set_offsets(interpolated)
        return scat,

    anim = FuncAnimation(fig, update, frames=2 * total_frames + 2 * pause_frames, interval=50, repeat=True)
    anim.save(output_path, writer='imagemagick')
    logger.info(f"Animation saved to {output_path}")

def map_iris_to_classnames(dot_layout, iri_to_classname):
    """
    Maps class names in the dot layout to full IRIs using plain Python
    and returns a list of full IRIs corresponding to the class names.
    """
    full_iri_list = []
    for class_name in dot_layout["node_id"]:
        full_iri = iri_to_classname.get(class_name, None)
        full_iri_list.append(full_iri)
    
    return full_iri_list

def main():
    parser = argparse.ArgumentParser(description="Transform from DOT layout to UMAP layout with animation")
    parser.add_argument('gnn_parquet', help="Parquet file with GNN embeddings")
    parser.add_argument('dot_layout_parquet', help="Parquet file with DOT layout coordinates")
    parser.add_argument('owl_file', help="OWL file for IRI mapping")
    parser.add_argument('output_parquet', help="Output Parquet file for UMAP embedding")
    parser.add_argument('output_animation', help="Output path for the animation file")
    args = parser.parse_args()

    # Load ontology to map IRI to class names
    ontology = load_ontology(args.owl_file)
    iri_to_classname = map_class_names(ontology)

    # Load GNN embeddings and DOT layout using Polars
    gnn_df = load_embeddings_parquet(args.gnn_parquet)
    dot_df = load_dot_layout_parquet(args.dot_layout_parquet)

    # Map class names to full IRIs using plain Python
    full_iri_list = map_iris_to_classnames(dot_df, iri_to_classname)
    
    # Now convert to a Polars DataFrame
    dot_df = dot_df.with_columns([
        pl.Series("full_iri", full_iri_list)
    ])

    # Normalize the DOT layout coordinates
    dot_layout_coords = dot_df.select(pl.col("x"), pl.col("y")).to_numpy()
    normalized_dot_layout_coords = normalize_data(dot_layout_coords)

    # Calculate UMAP embedding
    umap_embedding = calculate_umap(gnn_df)

    # Save UMAP embedding to Parquet
    umap_df = pl.DataFrame({
        "full_iri": gnn_df["node_id"],
        "umap_x": umap_embedding[:, 0],
        "umap_y": umap_embedding[:, 1]
    })
    save_parquet(umap_df, args.output_parquet)

    # Create animation between normalized DOT layout and UMAP embedding
    create_animation(normalized_dot_layout_coords, umap_embedding, args.output_animation)

if __name__ == "__main__":
    main()
