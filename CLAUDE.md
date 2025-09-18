# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

on2vec is a toolkit for generating vector embeddings from OWL ontologies using Graph Neural Networks (GNNs), with comprehensive **HuggingFace Sentence Transformers integration** and **MTEB benchmarking capabilities**. The project creates production-ready models that combine ontological knowledge with semantic text understanding.

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

The codebase follows a modern pipeline architecture with distinct stages:

1. **OWL Processing** (`main.py`): Converts OWL ontologies to graphs and trains GNN models
2. **Text Integration**: Combines structural embeddings with semantic text features
3. **HuggingFace Model Creation** (`create_hf_model.py`): Creates sentence-transformers compatible models
4. **Model Documentation**: Auto-generates comprehensive model cards and upload instructions
5. **MTEB Benchmarking** (`mteb_benchmarks/`): Evaluation against standard benchmarks
6. **Visualization Pipeline**: UMAP projections and analysis tools

### Key Components

#### HuggingFace Integration (Primary Interface)
- `create_hf_model.py`: Main CLI for end-to-end workflows
- `batch_hf_models.py`: Batch processing for multiple ontologies
- `on2vec/sentence_transformer_hub.py`: Core HuggingFace model creation
- `on2vec/model_card_generator.py`: Comprehensive documentation generation
- `on2vec/metadata_utils.py`: Smart metadata extraction and auto-detection

#### MTEB Benchmarking
- `mteb_benchmarks/benchmark_runner.py`: Full MTEB evaluation framework
- `mteb_benchmarks/README.md`: Benchmarking documentation
- `test_edam_model.py`: Domain-specific model comparison

#### Core on2vec
- `main.py`: Core GNN training and embedding generation
- `viz.py`: UMAP visualization of embeddings
- `process_dir.py`: Batch processing for multiple OWL files
- `on2vec/` package: Modular components for programmatic use

## Common Commands

### HuggingFace Model Creation (Primary Workflow)

#### One-command model creation:
```bash
python create_hf_model.py e2e ONTOLOGY.owl model-name
```

#### Step-by-step workflow:
```bash
# 1. Train ontology with text features
python create_hf_model.py train ONTOLOGY.owl --output embeddings.parquet

# 2. Create HuggingFace model (auto-detects base model)
python create_hf_model.py create embeddings.parquet model-name

# 3. Test model
python create_hf_model.py test ./hf_models/model-name

# 4. Get upload instructions
python create_hf_model.py upload-info ./hf_models/model-name
```

#### Batch processing:
```bash
python batch_hf_models.py process owl_files/ ./output --max-workers 4
```

### MTEB Benchmarking

#### Quick benchmark:
```bash
python mteb_benchmarks/benchmark_runner.py ./hf_models/model-name --quick
```

#### Full MTEB evaluation:
```bash
python mteb_benchmarks/benchmark_runner.py ./hf_models/model-name
```

### Core on2vec Commands

#### Generate embeddings from single OWL file:
```bash
python main.py ONTOLOGY.owl --model_type gcn --hidden_dim 128 --out_dim 64 --epochs 100 --output embeddings.parquet
```

#### Visualize embeddings:
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