"""
Visualization utilities for ontology embeddings
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_embeddings_for_viz(parquet_file):
    """
    Load embeddings from parquet file for visualization.

    Args:
        parquet_file (str): Path to parquet file

    Returns:
        tuple: (embeddings_array, node_ids, metadata)
    """
    logger.info(f"Loading embeddings from {parquet_file}")

    # Load parquet file
    df = pl.read_parquet(parquet_file)

    # Extract metadata
    metadata = {}
    parquet_file_obj = pl.scan_parquet(parquet_file)
    try:
        metadata = parquet_file_obj.collect_schema().metadata or {}
    except:
        logger.warning("Could not load metadata from parquet file")

    # Get node IDs
    node_ids = df['node_id'].to_list()

    # Get embedding vectors - check for different storage formats
    if 'embedding' in df.columns:
        # Format: single 'embedding' column containing arrays/lists
        embeddings_series = df['embedding']
        # Convert to numpy array - each row should be a vector
        embeddings = np.array(embeddings_series.to_list())
    else:
        # Format: separate columns 'dim_0', 'dim_1', etc.
        embedding_cols = [col for col in df.columns if col.startswith('dim_')]
        embedding_cols.sort(key=lambda x: int(x.split('_')[1]))  # Sort by dimension number

        if not embedding_cols:
            raise ValueError("No embedding dimensions found in parquet file. Expected either 'embedding' column or columns named 'dim_0', 'dim_1', etc.")

        embeddings = df.select(embedding_cols).to_numpy()

    logger.info(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")

    return embeddings, node_ids, metadata


def plot_pca_2d(parquet_file, output_file=None, title=None, figsize=(12, 8),
                alpha=0.6, s=20, random_state=42):
    """
    Create a 2D PCA plot of embeddings.

    Args:
        parquet_file (str): Path to parquet file containing embeddings
        output_file (str, optional): Path to save the plot
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        alpha (float): Point transparency
        s (float): Point size
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (figure, axes, pca_embeddings)
    """
    # Load embeddings
    embeddings, node_ids, metadata = load_embeddings_for_viz(parquet_file)

    # Perform PCA
    logger.info("Computing PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    pca_embeddings = pca.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                        alpha=alpha, s=s, c='steelblue', edgecolors='none')

    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

    if title is None:
        # Extract filename for default title
        filename = Path(parquet_file).stem
        model_info = ""
        if 'model_config' in metadata:
            try:
                import json
                config = json.loads(metadata['model_config'])
                model_type = config.get('model_type', 'unknown').upper()
                model_info = f" ({model_type})"
            except:
                pass
        title = f'PCA Visualization: {filename}{model_info}'

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add variance explanation text
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    ax.text(0.02, 0.98, f'Total variance explained: {total_variance:.1%}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if output_file:
        logger.info(f"Saving plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    logger.info(f"PCA plot created with {len(pca_embeddings)} points")
    logger.info(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    return fig, ax, pca_embeddings


def plot_tsne_2d(parquet_file, output_file=None, title=None, figsize=(12, 8),
                 alpha=0.6, s=20, perplexity=30, random_state=42, max_iter=1000):
    """
    Create a 2D t-SNE plot of embeddings.

    Args:
        parquet_file (str): Path to parquet file containing embeddings
        output_file (str, optional): Path to save the plot
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        alpha (float): Point transparency
        s (float): Point size
        perplexity (int): t-SNE perplexity parameter
        random_state (int): Random state for reproducibility
        max_iter (int): Maximum iterations for t-SNE

    Returns:
        tuple: (figure, axes, tsne_embeddings)
    """
    # Load embeddings
    embeddings, node_ids, metadata = load_embeddings_for_viz(parquet_file)

    # Perform t-SNE
    logger.info(f"Computing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=max_iter, verbose=1)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                        alpha=alpha, s=s, c='darkorange', edgecolors='none')

    # Labels and title
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    if title is None:
        filename = Path(parquet_file).stem
        model_info = ""
        if 'model_config' in metadata:
            try:
                import json
                config = json.loads(metadata['model_config'])
                model_type = config.get('model_type', 'unknown').upper()
                model_info = f" ({model_type})"
            except:
                pass
        title = f't-SNE Visualization: {filename}{model_info}'

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add parameter info
    ax.text(0.02, 0.98, f'Perplexity: {perplexity}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if output_file:
        logger.info(f"Saving plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    logger.info(f"t-SNE plot created with {len(tsne_embeddings)} points")

    return fig, ax, tsne_embeddings


def plot_umap_2d(parquet_file, output_file=None, title=None, figsize=(12, 8),
                 alpha=0.6, s=20, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Create a 2D UMAP plot of embeddings.

    Args:
        parquet_file (str): Path to parquet file containing embeddings
        output_file (str, optional): Path to save the plot
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        alpha (float): Point transparency
        s (float): Point size
        n_neighbors (int): UMAP n_neighbors parameter
        min_dist (float): UMAP min_dist parameter
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (figure, axes, umap_embeddings)
    """
    # Load embeddings
    embeddings, node_ids, metadata = load_embeddings_for_viz(parquet_file)

    # Perform UMAP
    logger.info(f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=random_state)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1],
                        alpha=alpha, s=s, c='forestgreen', edgecolors='none')

    # Labels and title
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    if title is None:
        filename = Path(parquet_file).stem
        model_info = ""
        if 'model_config' in metadata:
            try:
                import json
                config = json.loads(metadata['model_config'])
                model_type = config.get('model_type', 'unknown').upper()
                model_info = f" ({model_type})"
            except:
                pass
        title = f'UMAP Visualization: {filename}{model_info}'

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add parameter info
    ax.text(0.02, 0.98, f'n_neighbors: {n_neighbors}, min_dist: {min_dist}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if output_file:
        logger.info(f"Saving plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    logger.info(f"UMAP plot created with {len(umap_embeddings)} points")

    return fig, ax, umap_embeddings


def plot_embedding_distribution(parquet_file, output_file=None, title=None, figsize=(15, 10)):
    """
    Create distribution plots for embedding dimensions.

    Args:
        parquet_file (str): Path to parquet file containing embeddings
        output_file (str, optional): Path to save the plot
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)

    Returns:
        tuple: (figure, axes)
    """
    # Load embeddings
    embeddings, node_ids, metadata = load_embeddings_for_viz(parquet_file)

    # Determine subplot layout
    n_dims = embeddings.shape[1]
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot histograms for each dimension
    for i in range(n_dims):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        ax.hist(embeddings[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_dims, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    if title is None:
        filename = Path(parquet_file).stem
        title = f'Embedding Distribution: {filename}'

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save if requested
    if output_file:
        logger.info(f"Saving distribution plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    logger.info(f"Distribution plot created for {n_dims} dimensions")

    return fig, axes


def create_visualization_comparison(parquet_file, output_file=None, figsize=(18, 6),
                                  random_state=42):
    """
    Create a comparison plot showing PCA, t-SNE, and UMAP side by side.

    Args:
        parquet_file (str): Path to parquet file containing embeddings
        output_file (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (figure, axes, results_dict)
    """
    # Load embeddings
    embeddings, node_ids, metadata = load_embeddings_for_viz(parquet_file)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    results = {}

    # PCA
    logger.info("Computing PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    pca_embeddings = pca.fit_transform(embeddings)

    axes[0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                   alpha=0.6, s=20, c='steelblue', edgecolors='none')
    axes[0].set_title('PCA')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].grid(True, alpha=0.3)

    results['pca'] = pca_embeddings

    # t-SNE
    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_state, verbose=0)
    tsne_embeddings = tsne.fit_transform(embeddings)

    axes[1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                   alpha=0.6, s=20, c='darkorange', edgecolors='none')
    axes[1].set_title('t-SNE')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)

    results['tsne'] = tsne_embeddings

    # UMAP
    logger.info("Computing UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    axes[2].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1],
                   alpha=0.6, s=20, c='forestgreen', edgecolors='none')
    axes[2].set_title('UMAP')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)

    results['umap'] = umap_embeddings

    # Overall title
    filename = Path(parquet_file).stem
    model_info = ""
    if 'model_config' in metadata:
        try:
            import json
            config = json.loads(metadata['model_config'])
            model_type = config.get('model_type', 'unknown').upper()
            model_info = f" ({model_type})"
        except:
            pass

    fig.suptitle(f'Dimensionality Reduction Comparison: {filename}{model_info}',
                fontsize=16)

    plt.tight_layout()

    # Save if requested
    if output_file:
        logger.info(f"Saving comparison plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    logger.info(f"Comparison plot created with {len(embeddings)} points")

    return fig, axes, results