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

# For notebook support, install optional dependencies
uv sync --extra notebook

# Activate the virtual environment
source .venv/bin/activate
```

### Jupyter Notebook Demo

For an interactive demonstration, check out the included notebook:

```bash
# Start Jupyter Lab
jupyter lab on2vec_demo.ipynb
```

The demo notebook walks through:
- Downloading a real ontology (Cardiovascular Disease Ontology)
- Training a GNN model
- Generating and analyzing embeddings
- Vector operations and format conversion

## Quick Start

### Recommended Two-Phase Workflow

#### 1. Train a model (one-time setup):
```bash
# Basic GCN model with subclass relations only
python train.py EDAM.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --model_output edam_model.pt

# Multi-relation RGCN model with all ObjectProperty relations (auto-enables multi-relation)
python train.py EDAM.owl --model_type rgcn --hidden_dim 128 --out_dim 64 --epochs 100 --model_output edam_rgcn_model.pt

# Heterogeneous model with relation-specific layers (auto-enables multi-relation)
python train.py EDAM.owl --model_type heterogeneous --hidden_dim 128 --out_dim 64 --epochs 100 --model_output edam_hetero_model.pt
```

#### 2. Generate embeddings using the trained model:
```bash
# Embed the same ontology used for training
python embed.py edam_model.pt EDAM.owl --output edam_embeddings.parquet

# Or embed a different ontology using the same trained model
python embed.py edam_model.pt other_ontology.owl --output other_embeddings.parquet

# Use multi-relation model for richer embeddings
python embed.py edam_rgcn_model.pt EDAM.owl --output edam_multi_rel_embeddings.parquet
```

### Alternative: Integrated Workflow
```bash
# Train and embed in one step (backwards compatible)
python main.py EDAM.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --output embeddings.parquet

# Skip training and use existing model
python main.py EDAM.owl --skip_training --model_output edam_model.pt --output embeddings.parquet

# NEW: Text-augmented embeddings with semantic features
python main.py EDAM.owl --use_text_features --text_model_type sentence_transformer --text_model_name all-MiniLM-L6-v2 --fusion_method concat --output text_embeddings.parquet

# Use different text models and fusion methods
python main.py EDAM.owl --use_text_features --text_model_type huggingface --text_model_name bert-base-uncased --fusion_method attention --output bert_embeddings.parquet
```

### Using as Python Package
```python
from on2vec import (
    train_ontology_embeddings,
    train_text_augmented_ontology_embeddings,
    embed_ontology_with_model,
    inspect_parquet_metadata,
    load_embeddings_as_dataframe,
    add_embedding_vectors
)

# Train standard model
result = train_ontology_embeddings(
    owl_file="EDAM.owl",
    model_output="model.pt",
    model_type="gcn",
    hidden_dim=64,
    out_dim=32
)

# NEW: Train text-augmented model with semantic features
text_result = train_text_augmented_ontology_embeddings(
    owl_file="EDAM.owl",
    model_output="text_model.pt",
    text_model_type="sentence_transformer",
    text_model_name="all-MiniLM-L6-v2",
    backbone_model="gcn",
    fusion_method="concat",
    hidden_dim=64,
    out_dim=32
)

# Generate embeddings
embeddings = embed_ontology_with_model(
    model_path="model.pt",
    owl_file="EDAM.owl",
    output_file="embeddings.parquet"
)

# Work with embedding files
inspect_parquet_metadata("embeddings.parquet")
df = load_embeddings_as_dataframe("embeddings.parquet")
result_vector = add_embedding_vectors("embeddings.parquet", "Class1", "embeddings.parquet", "Class2")
```

### Working with Embedding Files

The toolkit includes comprehensive utilities for working with the generated Parquet files:

#### Parquet Tools CLI
```bash
# Inspect metadata and basic info
python parquet_tools.py inspect embeddings.parquet

# List all concept IDs
python parquet_tools.py list embeddings.parquet --numbered

# Get specific embedding vector
python parquet_tools.py get embeddings.parquet "http://example.org/concept1" --format list

# Convert to CSV format
python parquet_tools.py convert embeddings.parquet --output embeddings.csv

# Vector arithmetic operations
python parquet_tools.py add embeddings.parquet "concept1" "concept2"
python parquet_tools.py subtract file1.parquet "concept1" file2.parquet "concept2"
```

#### Python API for Embedding Analysis
```python
from on2vec import (
    load_embeddings_as_dataframe,
    get_embedding_vector,
    convert_parquet_to_csv,
    add_embedding_vectors,
    subtract_embedding_vectors
)

# Load as DataFrame for analysis
df, metadata = load_embeddings_as_dataframe("embeddings.parquet", return_metadata=True)
print(f"Loaded {len(df)} embeddings with metadata: {list(metadata.keys())}")

# Get individual vectors
vector1 = get_embedding_vector("embeddings.parquet", "http://example.org/concept1")

# Perform vector operations
sum_vector = add_embedding_vectors("file1.parquet", "concept1", "file2.parquet", "concept2")
diff_vector = subtract_embedding_vectors("file1.parquet", "concept1", "file1.parquet", "concept2")

# Convert formats
csv_file = convert_parquet_to_csv("embeddings.parquet")
```

### Multi-Relation Graph Support

NEW! on2vec now supports capturing **all ObjectProperty relations** from ontologies, not just subclass hierarchies:

#### Build Multi-Relation Graphs
```python
from on2vec import build_multi_relation_graph_from_owl

# Build graph with all relation types
graph_data = build_multi_relation_graph_from_owl("ontology.owl")

print(f"Nodes: {graph_data['node_features'].shape[0]:,}")
print(f"Edges: {graph_data['edge_index'].shape[1]:,}")
print(f"Relation types: {len(graph_data['relation_names'])}")
print(f"Edge distribution: {graph_data['edge_type_counts']}")

# Build graph with only ObjectProperties (no subclass relations)
graph_data_obj_only = build_multi_relation_graph_from_owl(
    "ontology.owl",
    include_subclass=False
)
```

#### Train Models with Multi-Relation Graphs
```python
from on2vec import train_ontology_embeddings

# Train using multi-relation RGCN model
result = train_ontology_embeddings(
    owl_file="ontology.owl",
    model_output="multi_rel_model.pt",
    model_type="rgcn",  # Use RGCN for multi-relation support
    hidden_dim=64,
    out_dim=32,
    epochs=100,
    use_multi_relation=True  # Enable multi-relation graph building
)

# Train heterogeneous model
result_hetero = train_ontology_embeddings(
    owl_file="ontology.owl",
    model_output="hetero_model.pt",
    model_type="heterogeneous",  # Use heterogeneous model
    hidden_dim=64,
    out_dim=32,
    epochs=100,
    use_multi_relation=True
)
```

#### Advanced Multi-Relation Models
```python
from on2vec import MultiRelationOntologyGNN, HeterogeneousOntologyGNN

# RGCN model for handling multiple relation types
model = MultiRelationOntologyGNN(
    input_dim=graph_data['node_features'].shape[1],
    hidden_dim=64,
    out_dim=32,
    num_relations=len(graph_data['relation_names']),
    model_type='rgcn',
    dropout=0.2
)

# Alternative: Weighted GCN with learnable relation weights
weighted_model = MultiRelationOntologyGNN(
    input_dim=graph_data['node_features'].shape[1],
    hidden_dim=64,
    out_dim=32,
    num_relations=len(graph_data['relation_names']),
    model_type='weighted_gcn'
)

# Heterogeneous model with relation-specific layers and attention
hetero_model = HeterogeneousOntologyGNN(
    input_dim=graph_data['node_features'].shape[1],
    hidden_dim=64,
    out_dim=32,
    relation_types=graph_data['relation_names'],
    dropout=0.2
)

# Forward pass with edge types
embeddings = model(
    graph_data['node_features'],
    graph_data['edge_index'],
    graph_data['edge_types']
)
```

#### Multi-Relation Graph Analysis
```python
# Analyze relation type distribution
relation_stats = graph_data['edge_type_counts']
print(f"Total relation types found: {len(graph_data['relation_names'])}")
print(f"Relation types with edges: {len([r for r, c in relation_stats.items() if c > 0])}")

# Show top relation types by edge count
sorted_relations = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)
print("Top 5 relation types:")
for rel, count in sorted_relations[:5]:
    print(f"  {rel}: {count} edges")

# Compare with subclass-only graph
from on2vec import build_graph_from_owl
basic_x, basic_edge_index, basic_mapping = build_graph_from_owl("ontology.owl")

print(f"Basic graph edges: {basic_edge_index.shape[1]}")
print(f"Multi-relation edges: {graph_data['edge_index'].shape[1]}")
print(f"Improvement: {graph_data['edge_index'].shape[1] / basic_edge_index.shape[1]:.1f}x more edges")
```

#### Benefits of Multi-Relation Graphs
- **Richer Structure**: Captures domain-specific semantic relationships beyond hierarchies
- **Better Embeddings**: Relations like "causes", "part_of", "located_in" provide semantic context
- **Domain Knowledge**: Leverages complete ontology design with all ObjectProperty relations
- **Advanced Models**: Enables RGCN, heterogeneous, and attention-based architectures
- **Improved Coverage**: Typically provides 1.5-2x more edges than subclass-only graphs

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
- **`parquet_tools.py`** - Comprehensive utilities for working with embedding files

### Package Features (Completed ✅)
- Importable Python modules for programmatic use
- Model checkpointing with complete metadata preservation
- Ontology alignment between training and target ontologies
- Rich Parquet metadata describing source ontologies
- Comprehensive embedding file utilities (inspect, convert, vector operations)
- DataFrame integration for analysis workflows
- Comprehensive logging and error handling

## Future Roadmap

### Phase 1: Enhanced Graph Modeling
- [x] **Multi-Relation Graph Support**: Include object properties, data properties, and custom relations beyond subclass hierarchies ✅
  - [x] Extract all ObjectProperty relations from OWL ontologies
  - [x] RGCN and heterogeneous model architectures for multi-relation graphs
  - [x] Relation type analysis and edge distribution statistics
- [x] **Semantic Text Integration**: Combine graph structure with text embeddings from class labels, definitions, and annotations ✅
- [x] **Configurable Text Models**: Support for different text embedding models (BERT, SentenceTransformers, OpenAI, etc.) ✅
- [x] **Rich Semantic Features**: Incorporate rdfs:comment, skos:definition, and other descriptive properties ✅
- [x] **Unified Embedding Space**: Merge structural graph embeddings with semantic text embeddings ✅

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
- **Multi-Relation Graphs**: NEW! Capture all ObjectProperty relations, not just subclass
  - Extract and utilize all semantic relationships from OWL ontologies
  - Support for 79 different ObjectProperty types in complex ontologies like CVDO
  - Typically provides 1.5-2x more edges than hierarchy-only approaches
- **Text-Augmented Embeddings**: NEW! Combine structural and semantic text features ✅
  - **Rich Semantic Extraction**: Extract text from rdfs:comment, skos:definition, labels, and annotations
  - **Configurable Text Models**: Support for SentenceTransformers, HuggingFace, OpenAI, TF-IDF
  - **Flexible Fusion**: Multiple methods to combine structural + text features (concat, add, weighted, attention)
  - **CLI Integration**: Full command-line support with `--use_text_features` flag
- **Advanced GNN Architectures**: GCN, GAT, RGCN, and heterogeneous models
  - **RGCN**: Relational Graph Convolutional Networks for multi-relation support
  - **Heterogeneous**: Relation-specific layers with attention mechanisms
  - **Weighted GCN**: Learnable relation weights for different edge types
- **Model Reuse**: Train once, embed multiple ontologies with automatic class alignment
- **Rich I/O**: Parquet output with comprehensive metadata about source ontologies
  - **Parquet Tools CLI**: Inspect, convert, list, get, add, subtract operations
  - **DataFrame Integration**: Polars DataFrame loading with metadata preservation
  - **Vector Arithmetic**: Add and subtract embedding vectors for semantic operations
- **Dual Interface**: CLI scripts and importable Python modules
- **Jupyter Integration**: Interactive demonstration notebook with real ontology examples

### 🔄 Current Limitations (Addressed in Roadmap)
- **Evaluation**: No built-in metrics for embedding quality assessment
- **Use Cases**: Limited examples of practical applications

### 🎯 Vision: Comprehensive Ontology Embeddings
**ACHIEVED!** ✅ on2vec now captures both **structural relationships** and **semantic meaning** through text-augmented embeddings, enabling rich applications like cross-ontology search, automated alignment, and knowledge discovery.

## File Structure

```
on2vec/
├── on2vec/              # Package modules
│   ├── models.py        # GNN architectures
│   ├── training.py      # Training workflows
│   ├── embedding.py     # Embedding generation
│   ├── ontology.py      # OWL processing
│   ├── io.py           # File operations & parquet utilities
│   └── loss_functions.py # Loss implementations
├── train.py            # CLI: Train models
├── embed.py            # CLI: Generate embeddings
├── main.py             # CLI: Integrated workflow
├── parquet_tools.py    # CLI: Parquet file utilities
├── on2vec_demo.ipynb   # 📓 Interactive Jupyter demo
└── pyproject.toml      # Package configuration
```

## Requirements

### Core Dependencies
- Python >= 3.10
- PyTorch >= 2.6.0
- torch-geometric >= 2.6.1
- owlready2 >= 0.47
- polars >= 1.24.0
- pyarrow >= 19.0.1
- networkx >= 3.4.2
- matplotlib >= 3.10.1
- umap-learn >= 0.5.7

### Optional Dependencies
- **Notebook support:** `jupyter`, `jupyterlab`, `ipywidgets` (install with `--extra notebook`)
- **Development:** `pytest`, `black`, `isort`, `mypy` (install with `--extra dev`)

## Getting Started

1. **Try the interactive demo:** Open `on2vec_demo.ipynb` in Jupyter Lab for a complete walkthrough
2. **CLI workflow:** Use the two-phase train → embed approach for production workflows
3. **Python integration:** Import on2vec modules for custom analysis pipelines

## License

[Add license information]