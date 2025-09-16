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

### Phase 1: Enhanced Graph Modeling
- [ ] **Multi-Relation Graph Support**: Include object properties, data properties, and custom relations beyond subclass hierarchies
- [ ] **Semantic Text Integration**: Combine graph structure with text embeddings from class labels, definitions, and annotations
- [ ] **Configurable Text Models**: Support for different text embedding models (BERT, SentenceTransformers, OpenAI, etc.)
- [ ] **Rich Semantic Features**: Incorporate rdfs:comment, skos:definition, and other descriptive properties
- [ ] **Unified Embedding Space**: Merge structural graph embeddings with semantic text embeddings

### Phase 2: Evaluation & Benchmarking
- [ ] **Evaluation Framework**: Comprehensive metrics for embedding quality assessment
  - [ ] Intrinsic evaluation (clustering, visualization quality)
  - [ ] Extrinsic evaluation (downstream task performance)
  - [ ] Ontology-specific metrics (hierarchy preservation, semantic coherence)
- [ ] **Benchmark Datasets**: Curated evaluation sets for different ontology domains
- [ ] **Baseline Comparisons**: Compare against existing ontology embedding methods
- [ ] **Cross-Domain Evaluation**: Test generalization across different ontology types

### Phase 3: Example Use Cases & Applications
- [ ] **Semantic Search**: Find similar concepts across ontologies
- [ ] **Ontology Alignment**: Automated mapping between different ontologies
- [ ] **Knowledge Graph Completion**: Predict missing relations and concepts
- [ ] **Recommendation Systems**: Suggest relevant concepts based on embeddings
- [ ] **Clustering & Taxonomy**: Discover hidden concept groupings
- [ ] **Interactive Ontology Exploration**: Visual navigation through embedding space

### Phase 4: CLI Unification & Distribution
- [ ] Create unified `on2vec` command with subcommands
- [ ] Add visualization commands (`on2vec viz`, `on2vec layout`)
- [ ] Implement configuration file support
- [ ] Add comprehensive test suite
- [ ] Configure for PyPI distribution
- [ ] Add GitHub Actions for CI/CD

### Phase 5: Advanced Features
- [ ] **Multi-Modal Embeddings**: Combine structural, textual, and visual information
- [ ] **Dynamic Ontology Support**: Handle evolving ontologies over time
- [ ] **Federated Learning**: Train models across distributed ontology collections
- [ ] **Interactive Visualization Tools**: Web-based exploration interfaces
- [ ] **API Server**: REST/GraphQL API for embedding services

## Current Capabilities & Limitations

### ✅ What It Does Now
- **Structural Embeddings**: Learns from ontology class hierarchies (subclass relations)
- **Multiple GNN Architectures**: GCN and GAT models with various loss functions
- **Model Reuse**: Train once, embed multiple ontologies with automatic class alignment
- **Rich I/O**: Parquet output with comprehensive metadata about source ontologies
- **Dual Interface**: CLI scripts and importable Python modules

### 🔄 Current Limitations (Addressed in Roadmap)
- **Graph Relations**: Only uses subclass hierarchies, ignores other semantic relations
- **Text Information**: Doesn't incorporate class labels, descriptions, or annotations
- **Evaluation**: No built-in metrics for embedding quality assessment
- **Use Cases**: Limited examples of practical applications
- **Semantic Richness**: Focuses on structure, not semantic meaning of concepts

### 🎯 Vision: Comprehensive Ontology Embeddings
The roadmap addresses these limitations by evolving toward embeddings that capture both **structural relationships** and **semantic meaning**, enabling rich applications like cross-ontology search, automated alignment, and knowledge discovery.

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