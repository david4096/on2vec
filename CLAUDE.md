# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

on2vec is a toolkit for generating vector embeddings from OWL ontologies using Graph Neural Networks (GNNs). The project processes OWL files to create embeddings and various visualizations including UMAP projections and force-directed layouts.

## Dependencies and Environment

This is a Python project using UV for dependency management:
- Python >= 3.10 required
- Dependencies managed via `pyproject.toml` and `uv.lock`
- Key dependencies: PyTorch, torch-geometric, owlready2, UMAP, matplotlib, polars

To set up the environment:
```bash
uv sync
```

## Core Architecture

The codebase follows a pipeline architecture with distinct stages:

1. **OWL Processing** (`main.py`): Converts OWL ontologies to graphs and trains GNN models
2. **Embedding Generation**: Uses GCN or GAT models with various loss functions (triplet, contrastive, cosine, cross-entropy)
3. **Visualization Pipeline**: Multiple visualization approaches including UMAP and force-directed layouts
4. **Batch Processing**: Scripts for processing multiple files in directories

### Key Components

- `main.py`: Core GNN training and embedding generation
- `viz.py`: UMAP visualization of embeddings
- `dot_to_embed.py`: Animation between DOT layout and UMAP projections
- `force_layout.py`: Force-directed graph layout using Graphviz
- `process_dir.py`: Batch processing for multiple OWL files
- `force_directory.py`: Batch processing for force-directed layouts

## Common Commands

### Generate embeddings from single OWL file:
```bash
python main.py ONTOLOGY.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --loss_fn triplet --output embeddings.parquet
```

### Visualize embeddings:
```bash
python viz.py embeddings.parquet --neighbors 15 --min_dist 0.1 --output visualization.png
```

### Process entire directory of OWL files:
```bash
python process_dir.py owl_files/ --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --output_dir output/
```

### Generate force-directed layouts:
```bash
python force_layout.py ONTOLOGY.owl --output_image layout.png --output_parquet coordinates.parquet
```

### Create animation between layouts:
```bash
python dot_to_embed.py embeddings.parquet coordinates.parquet ONTOLOGY.owl output.parquet animation.gif
```

## File Structure Patterns

- OWL files typically stored in `owl_files/` directory
- Generated embeddings saved as Parquet files
- Visualizations output as PNG images
- Animated transitions saved as GIF files
- Batch outputs organized in `output/` directories

## Model Configuration

The GNN models support:
- **Architectures**: GCN (Graph Convolutional Networks), GAT (Graph Attention Networks)
- **Loss Functions**: triplet, contrastive, cosine, cross_entropy
- **Embedding Dimensions**: Configurable hidden_dim and output_dim
- **Training**: Configurable epochs with Adam optimizer

## Data Flow

1. OWL → Graph (nodes=classes, edges=subclass relationships)
2. Graph → GNN Training → Node Embeddings
3. Embeddings → UMAP/Visualization → 2D Projections
4. Multiple layouts can be interpolated for animations

## Output Formats

- Embeddings: Parquet files with node_id and embedding columns
- Visualizations: PNG images
- Coordinates: Parquet files with node_id, x, y columns
- Animations: GIF files showing layout transitions