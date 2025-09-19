# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-18

### Added
- Initial release of on2vec toolkit
- Unified CLI with `on2vec` command and subcommands
- Core GNN training workflows (`train`, `embed`)
- HuggingFace Sentence Transformers integration (`hf`, `hf-train`, `hf-create`, `hf-test`)
- MTEB benchmarking support (`benchmark`, `compare`)
- Multiple GNN architectures: GCN, GAT, RGCN, Heterogeneous
- Text-augmented training with sentence transformers
- Multiple fusion methods for combining structural and text embeddings
- Auto-generated model cards with comprehensive metadata
- Batch processing capabilities for multiple ontologies
- Comprehensive test suite with 30+ tests
- Rich CLI with help, error handling, and progress indicators

### Features
- **Graph Neural Networks**: Support for GCN, GAT, RGCN, and heterogeneous models
- **Text Integration**: Combines structural graph embeddings with text features
- **HuggingFace Compatibility**: Creates drop-in replacement sentence transformer models
- **MTEB Benchmarking**: Full integration with Massive Text Embedding Benchmark
- **Smart Automation**: Auto-detects base models, domains, and configurations
- **Professional Output**: Auto-generated model cards, upload instructions, and documentation
- **Batch Processing**: Efficient processing of multiple ontologies
- **Multiple Loss Functions**: Triplet, contrastive, cosine, cross-entropy
- **Visualization**: UMAP-based embedding visualization
- **Cross-Format Support**: Parquet, CSV conversion utilities

### Architecture
- Modular Python package structure
- CLI-first design with unified entry point
- Comprehensive error handling and logging
- Test-driven development with pytest
- Clean separation between core functionality and CLI

### Dependencies
- Python >= 3.10
- PyTorch + torch-geometric for GNN training
- owlready2 for OWL ontology processing
- sentence-transformers for text features
- polars for efficient data handling
- MTEB for benchmarking (optional)
- HuggingFace Hub for model sharing

### Documentation
- Complete README with quickstart guide
- CLI reference documentation
- HuggingFace integration guide
- Comprehensive docstrings
- Usage examples and tutorials