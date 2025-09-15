# on2vec

A toolkit for generating vector embeddings from OWL ontologies using Graph Neural Networks (GNNs).

## Overview

on2vec converts OWL ontologies into graph representations and learns node embeddings using various GNN architectures (GCN, GAT) and loss functions. The toolkit provides visualization capabilities including UMAP projections, force-directed layouts, and animated transitions between different representations.

## Features

- **Multiple GNN Models**: Support for GCN and GAT architectures
- **Various Loss Functions**: Triplet, contrastive, cosine, and cross-entropy losses
- **Rich Visualizations**: UMAP projections, force-directed layouts, animated transitions
- **Batch Processing**: Process entire directories of OWL files
- **Flexible Output**: Parquet files for embeddings, PNG for visualizations, GIF for animations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd on2vec

# Install dependencies using UV
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Quick Start

### Generate embeddings from an OWL file:
```bash
python main.py EDAM.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --output embeddings.parquet
```

### Create UMAP visualization:
```bash
python viz.py embeddings.parquet --output visualization.png
```

### Process multiple OWL files:
```bash
python process_dir.py owl_files/ --output_dir output/
```

## Roadmap: Unified CLI Tool

To improve usability, the following tasks are planned to consolidate the separate scripts into a single, installable CLI tool:

### Phase 1: CLI Consolidation
- [ ] Create `on2vec/cli.py` with click-based command structure
- [ ] Combine useful tools into easy-to-use CLI

### Phase 2: Package Structure
- [ ] Refactor core functionality into `on2vec/` module structure:
  - [ ] `on2vec/models.py` - GNN model definitions
  - [ ] `on2vec/training.py` - Training loops and loss functions
  - [ ] `on2vec/visualization.py` - Plotting and animation functions
  - [ ] `on2vec/ontology.py` - OWL file processing utilities
  - [ ] `on2vec/io.py` - File I/O operations
- [ ] Update `pyproject.toml` with proper package structure and entry points
- [ ] Add comprehensive test suite

### Phase 3: Installation & Distribution
- [ ] Configure `pyproject.toml` for pip installation with entry point: `on2vec = on2vec.cli:main`
- [ ] Add development installation instructions: `pip install -e .`
- [ ] Create GitHub Actions for automated testing and releases
- [ ] Publish to PyPI for `pip install on2vec`
- [ ] Add shell completion support

### Phase 4: Enhanced Features
- [ ] Add configuration file support (YAML/TOML)
- [ ] Implement progress bars and better logging
- [ ] Add model checkpointing and resume functionality
- [ ] Support for different ontology formats beyond OWL
- [ ] Add embedding similarity search capabilities

## Current Usage

Until the CLI consolidation is complete, use the individual scripts as shown in the Quick Start section. See `CLAUDE.md` for detailed usage examples and architecture information.

## Requirements

- Python >= 3.10
- PyTorch
- torch-geometric
- owlready2
- UMAP
- matplotlib
- polars
- networkx

## License

[Add license information]