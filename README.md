# on2vec

A toolkit for generating vector embeddings from OWL ontologies using Graph Neural Networks (GNNs).

## Overview

on2vec converts OWL ontologies into graph representations and learns node embeddings using various GNN architectures (GCN, GAT) and loss functions. The toolkit uses a two-phase approach: first train a model, then generate embeddings for any ontology using the trained model.

## Features

- **Two-Phase Workflow**: Separate training and embedding steps for efficiency
- **Multiple GNN Models**: Support for GCN and GAT architectures
- **Various Loss Functions**: Triplet, contrastive, cosine, and cross-entropy losses
- **Model Checkpointing**: Save and reuse trained models across different ontologies
- **Rich Metadata**: Parquet files include comprehensive metadata about source ontologies
- **Package Structure**: Importable Python modules with CLI scripts

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

### Recommended Two-Phase Workflow

#### 1. Train a model (one-time setup):
```bash
python train.py EDAM.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --model_output edam_model.pt
```

#### 2. Generate embeddings using the trained model:
```bash
# Embed the same ontology used for training
python embed.py edam_model.pt EDAM.owl --output edam_embeddings.parquet

# Or embed a different ontology using the same trained model
python embed.py edam_model.pt other_ontology.owl --output other_embeddings.parquet
```

### Alternative: Integrated Workflow
```bash
# Train and embed in one step (backwards compatible)
python main.py EDAM.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --output embeddings.parquet

# Skip training and use existing model
python main.py EDAM.owl --skip_training --model_output edam_model.pt --output embeddings.parquet
```

### Using as Python Package
```python
from on2vec import train_model, generate_embeddings_from_model

# Train model
result = train_model(owl_file="EDAM.owl", model_output="model.pt")

# Generate embeddings
embeddings = generate_embeddings_from_model(
    model_path="model.pt",
    owl_file="EDAM.owl"
)
```

## Current Architecture

The toolkit is organized into a clean package structure:

### Core Modules (Completed ✅)
- **`on2vec/models.py`** - GNN model definitions (GCN, GAT)
- **`on2vec/training.py`** - Training loops, loss functions, model checkpointing
- **`on2vec/embedding.py`** - Embedding generation from trained models
- **`on2vec/ontology.py`** - OWL file processing utilities
- **`on2vec/io.py`** - File I/O operations with rich metadata
- **`on2vec/loss_functions.py`** - Various loss function implementations

### CLI Scripts (Completed ✅)
- **`train.py`** - Train GNN models on ontologies
- **`embed.py`** - Generate embeddings using trained models
- **`main.py`** - Integrated training + embedding workflow

### Package Features (Completed ✅)
- Importable Python modules for programmatic use
- Model checkpointing with complete metadata preservation
- Ontology alignment between training and target ontologies
- Rich Parquet metadata describing source ontologies
- Comprehensive logging and error handling

## Future Roadmap

### Phase 1: CLI Unification
- [ ] Create unified `on2vec` command with subcommands
- [ ] Add visualization commands (`on2vec viz`, `on2vec layout`)
- [ ] Implement configuration file support

### Phase 2: Distribution
- [ ] Add comprehensive test suite
- [ ] Configure for PyPI distribution
- [ ] Add GitHub Actions for CI/CD

### Phase 3: Enhanced Features
- [ ] Embedding similarity search
- [ ] Support for additional ontology formats
- [ ] Interactive visualization tools

## Key Benefits of the Current Architecture

- **Model Reuse**: Train once, embed multiple ontologies
- **Faster Iteration**: Skip training when experimenting with different ontologies
- **Model Persistence**: Save and share trained models with complete metadata
- **Memory Efficiency**: Load models only when needed for embedding
- **Ontology Alignment**: Automatically aligns classes between training and target ontologies
- **Rich Metadata**: Parquet files include source ontology information and model configuration
- **Dual Interface**: Use as CLI scripts or import as Python modules

## File Structure

```
on2vec/
├── on2vec/              # Package modules
│   ├── models.py        # GNN architectures
│   ├── training.py      # Training workflows
│   ├── embedding.py     # Embedding generation
│   ├── ontology.py      # OWL processing
│   ├── io.py           # File operations
│   └── loss_functions.py # Loss implementations
├── train.py            # CLI: Train models
├── embed.py            # CLI: Generate embeddings
├── main.py             # CLI: Integrated workflow
└── pyproject.toml      # Package configuration
```

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