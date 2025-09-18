# on2vec

A toolkit for generating vector embeddings from OWL ontologies using Graph Neural Networks (GNNs), with **HuggingFace Sentence Transformers integration** and **MTEB benchmarking**.

## 🚀 Quick Start

### Installation

```bash
pip install on2vec
```

Create production-ready Sentence Transformers models with ontology knowledge in **one command**:

```bash
# Complete end-to-end workflow
on2vec hf biomedical.owl my-biomedical-model
```

**Use like any sentence transformer:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./hf_models/my-biomedical-model')
embeddings = model.encode(['heart disease', 'cardiovascular problems'])
```

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📥 Installation](#-installation)
- [🤗 HuggingFace Integration](#-huggingface-integration)
- [🧪 MTEB Benchmarking](#-mteb-benchmarking)
- [💻 Core on2vec Usage](#-core-on2vec-usage)
- [🏗️ Architecture](#-architecture)
- [📚 Documentation](#-documentation)

## 📥 Installation

### From PyPI (Recommended)

```bash
# Basic installation
pip install on2vec

# With MTEB benchmarking support
pip install on2vec[benchmark]

# With all optional dependencies
pip install on2vec[all]
```

### From Source

```bash
git clone <repository-url>
cd on2vec
pip install -e .
```

### Dependencies
- Python >= 3.10
- PyTorch + torch-geometric
- owlready2, sentence-transformers
- polars, matplotlib, umap-learn

## 🤗 HuggingFace Integration

### One-Command Model Creation

```bash
# Create complete model with auto-generated documentation
on2vec hf ontology.owl model-name

# With custom settings
on2vec hf ontology.owl model-name \
  --base-model all-mpnet-base-v2 \
  --fusion gated \
  --epochs 200
```

### Step-by-Step Workflow

```bash
# 1. Train ontology embeddings
on2vec hf-train ontology.owl --output embeddings.parquet

# 2. Create HuggingFace model (auto-detects base model)
on2vec hf-create embeddings.parquet model-name

# 3. Test model functionality
on2vec hf-test ./hf_models/model-name

# 4. Inspect model details
on2vec inspect ./hf_models/model-name
```

### Batch Processing

```bash
# Process multiple ontologies
on2vec hf-batch owl_files/ ./output \
  --base-models all-MiniLM-L6-v2 all-mpnet-base-v2 \
  --fusion-methods concat gated \
  --max-workers 4
```

### Features

- ✅ **Auto-generated model cards** with comprehensive metadata
- ✅ **Smart base model detection** from embeddings
- ✅ **Upload instructions** and HuggingFace Hub preparation
- ✅ **Domain detection** and appropriate tagging
- ✅ **Multiple fusion methods**: concat, attention, gated, weighted_avg
- ✅ **Batch processing** for multiple ontologies

## 🧪 MTEB Benchmarking

Evaluate your models against the Massive Text Embedding Benchmark:

### Quick Benchmark

```bash
# Fast evaluation on subset of tasks
on2vec benchmark ./hf_models/my-model --quick

# Focus on specific task types
on2vec benchmark ./hf_models/my-model --task-types STS Classification

# Full MTEB benchmark
on2vec benchmark ./hf_models/my-model
```

### Compare Models

```bash
# Benchmark vanilla baseline
on2vec benchmark sentence-transformers/all-MiniLM-L6-v2 \
  --model-name vanilla-baseline --quick

# Compare ontology vs vanilla models
on2vec compare ./hf_models/my-model --detailed
```

### Features

- ✅ **Full MTEB integration** with 58+ evaluation tasks
- ✅ **Task filtering** by category (STS, Classification, Clustering, etc.)
- ✅ **Automated reporting** with JSON summaries and markdown reports
- ✅ **Resource management** with configurable batch sizes
- ✅ **Comparison tools** for baseline evaluation

## 💻 Core on2vec Usage

### Basic Training

```bash
# Train GCN model
on2vec train ontology.owl --output model.pt --model-type gcn --epochs 100

# Train with text features (for HuggingFace integration)
on2vec train ontology.owl --output embeddings.parquet --use-text-features

# Multi-relation models with all ObjectProperties
on2vec train ontology.owl --output model.pt --use-multi-relation --model-type rgcn
```

### Generate Embeddings

```bash
# Generate embeddings from trained model
on2vec embed model.pt ontology.owl --output embeddings.parquet
```

### Visualization

```bash
# Create UMAP visualization
on2vec visualize embeddings.parquet --output visualization.png
```

### Python API

```python
from sentence_transformers import SentenceTransformer
from on2vec import train_ontology_embeddings, embed_ontology_with_model

# Train model
result = train_ontology_embeddings(
    owl_file="ontology.owl",
    model_output="model.pt",
    model_type="gcn",
    hidden_dim=128,
    out_dim=64
)

# Generate embeddings
embeddings = embed_ontology_with_model(
    model_path="model.pt",
    owl_file="ontology.owl",
    output_file="embeddings.parquet"
)

# Use HuggingFace model
model = SentenceTransformer('./hf_models/my-model')
vectors = model.encode(['concept 1', 'concept 2'])
```

## 🏗️ Architecture

### Core Components

- **Graph Construction**: Converts OWL ontologies to graph representations
- **GNN Training**: Supports GCN, GAT, RGCN, and heterogeneous architectures
- **Text Integration**: Combines structural and semantic features using sentence transformers
- **Fusion Methods**: Multiple approaches to combine text + structural embeddings
- **HuggingFace Bridge**: Creates sentence-transformers compatible models

### Model Pipeline

```
OWL Ontology → Graph → GNN Training → Structural Embeddings
                                            ↓
Text Features → Sentence Transformer → Text Embeddings
                                            ↓
                              Fusion Layer → Final Model
                                            ↓
                              HuggingFace Model + Model Card
```

### Supported Architectures

- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **RGCN**: Relational GCN for multi-relation graphs
- **Heterogeneous**: Relation-specific layers with attention

## 📚 Documentation

- [📚 CLI Quick Reference](CLI_QUICK_REFERENCE.md) - All commands and examples
- [📖 HuggingFace Integration](docs/sentence_transformers_integration.md) - Complete workflow guide
- [🧪 MTEB Benchmarking](mteb_benchmarks/README.md) - Evaluation framework
- [🧬 Project Instructions](CLAUDE.md) - Development guidelines

## 🎯 Key Features

- **🤗 HuggingFace Ready**: One-command model creation with professional documentation
- **🧪 MTEB Integration**: Comprehensive benchmarking against standard tasks
- **📊 Rich Metadata**: Auto-generated model cards with complete technical details
- **🔧 Smart Automation**: Auto-detects base models, domains, and configurations
- **⚡ Batch Processing**: Handle multiple ontologies efficiently
- **🎨 Multiple Fusion Methods**: Flexible combination of text and structural features
- **📈 Comprehensive Evaluation**: Built-in comparison and testing tools

## 🚀 Example Workflow

```bash
# 1. Install on2vec
pip install on2vec[benchmark]

# 2. Create a model from biomedical ontology
on2vec hf EDAM.owl edam-biomedical

# 3. Quick benchmark evaluation
on2vec benchmark ./hf_models/edam-biomedical --quick

# 4. Compare with vanilla models
on2vec compare ./hf_models/edam-biomedical --detailed

# 5. Inspect model details
on2vec inspect ./hf_models/edam-biomedical

# 6. Upload to HuggingFace Hub (instructions auto-generated)
# See ./hf_models/edam-biomedical/UPLOAD_INSTRUCTIONS.md
```

The model is immediately usable as a drop-in replacement for any sentence-transformer, with the added benefit of ontological domain knowledge!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use on2vec in your research, please cite:

```bibtex
@software{on2vec2025,
  title={on2vec: Ontology Embeddings with Graph Neural Networks},
  author={David Steinberg},
  year={2025},
  url={https://github.com/david4096/on2vec}
}
```