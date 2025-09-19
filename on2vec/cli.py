#!/usr/bin/env python3
"""
Unified CLI for on2vec - Ontology Embeddings with Graph Neural Networks

This provides a single entry point for all on2vec functionality:
- Core embedding workflows (train, embed, visualize)
- HuggingFace model creation and management
- MTEB benchmarking and evaluation
- Batch processing and utilities
"""

import sys
import argparse
import time
from typing import List, Optional
from pathlib import Path

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CLI utilities
from .cli_utils import (
    validate_file_exists,
    validate_directory_exists,
    validate_model_compatibility,
    display_configuration,
    print_success_summary,
    handle_cli_error,
    ProgressTracker,
    with_progress_tracking,
    print_command_examples,
    Emoji
)
from .config import load_config_file, find_default_config, merge_config_with_args, save_sample_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog='on2vec',
        description='Generate vector embeddings from OWL ontologies using Graph Neural Networks with HuggingFace integration',
        epilog='''For more help on a specific command, use: on2vec <command> --help

🚀 Quick start:
  on2vec hf biomedical.owl my-model     # Create HuggingFace model
  on2vec train onto.owl -o model.pt     # Train GNN model
  on2vec benchmark ./hf_models/model    # Run MTEB benchmarks

💡 Command shortcuts: t=train, e=embed, v=visualize, b=benchmark, i=inspect''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Get version info with dependencies
    try:
        from . import __version__
        import torch
        import sentence_transformers
        version_info = f'on2vec {__version__} (PyTorch {torch.__version__}, sentence-transformers {sentence_transformers.__version__})'
    except ImportError:
        try:
            from . import __version__
            version_info = f'on2vec {__version__}'
        except ImportError:
            version_info = 'on2vec 0.1.1'

    parser.add_argument('--version', action='version', version=version_info)
    parser.add_argument('--config', help='Configuration file path (YAML or JSON)')
    parser.add_argument('--save-config', help='Save sample configuration file and exit', metavar='PATH')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available on2vec commands',
        dest='command',
        help='on2vec command to run'
    )

    # Core embedding commands
    setup_train_parser(subparsers)
    setup_embed_parser(subparsers)
    setup_visualize_parser(subparsers)

    # HuggingFace integration commands
    setup_hf_parser(subparsers)
    setup_hf_train_parser(subparsers)
    setup_hf_create_parser(subparsers)
    setup_hf_test_parser(subparsers)
    setup_hf_batch_parser(subparsers)

    # Evaluation and benchmarking
    setup_evaluate_parser(subparsers)
    setup_evaluate_batch_parser(subparsers)
    setup_benchmark_parser(subparsers)
    setup_compare_parser(subparsers)

    # Utilities
    setup_inspect_parser(subparsers)
    setup_convert_parser(subparsers)

    return parser


def setup_train_parser(subparsers):
    """Set up the train command parser."""
    train_parser = subparsers.add_parser(
        'train',
        help='Train GNN models on OWL ontologies',
        description='Train Graph Neural Network models on OWL ontology structures',
        epilog='''examples:
  on2vec train ontology.owl --output model.pt
  on2vec train bio.owl -o bio_model.pt --model-type gat --epochs 200
  on2vec train onto.owl -o text_model.pt --use-text-features --text-model all-mpnet-base-v2''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    train_parser.add_argument('ontology', help='Path to OWL ontology file')
    train_parser.add_argument('--output', '-o', required=True, help='Output model file path')
    train_parser.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                              default='gcn', help='GNN model architecture (default: %(default)s)')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimensions (default: %(default)s)')
    train_parser.add_argument('--out-dim', type=int, default=64, help='Output embedding dimensions (default: %(default)s)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: %(default)s)')
    train_parser.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                              default='triplet', help='Loss function (default: %(default)s)')
    train_parser.add_argument('--use-multi-relation', action='store_true',
                              help='Include all ObjectProperty relations')
    train_parser.add_argument('--use-text-features', action='store_true',
                              help='Include text features from ontology')
    train_parser.add_argument('--text-model', default='all-MiniLM-L6-v2',
                              help='Text model for semantic features (default: %(default)s)')
    train_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                              help='Device for training (default: %(default)s - auto-detects best available)')
    train_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    train_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')


def setup_embed_parser(subparsers):
    """Set up the embed command parser."""
    embed_parser = subparsers.add_parser(
        'embed',
        help='Generate embeddings using trained models',
        description='Generate embeddings for ontology concepts using pre-trained GNN models',
        epilog='''examples:
  on2vec embed model.pt ontology.owl --output embeddings.parquet
  on2vec embed trained_model.pt different_onto.owl -o cross_embeddings.parquet''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    embed_parser.add_argument('model', help='Path to trained model file')
    embed_parser.add_argument('ontology', help='Path to OWL ontology file')
    embed_parser.add_argument('--output', '-o', required=True, help='Output embeddings file (.parquet)')
    embed_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                              help='Device for embedding generation (default: %(default)s - auto-detects best available)')
    embed_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    embed_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')


def setup_visualize_parser(subparsers):
    """Set up the visualize command parser."""
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Create visualizations of embeddings',
        description='Generate UMAP visualizations and other plots from embedding files',
        epilog='''examples:
  on2vec visualize embeddings.parquet
  on2vec visualize embeddings.parquet --output viz.png --neighbors 20 --min-dist 0.05''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    viz_parser.add_argument('embeddings', help='Path to embeddings file (.parquet)')
    viz_parser.add_argument('--output', '-o', help='Output visualization file (.png)')
    viz_parser.add_argument('--neighbors', type=int, default=15, help='UMAP n_neighbors parameter (default: %(default)s)')
    viz_parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter (default: %(default)s)')


def setup_hf_parser(subparsers):
    """Set up the HuggingFace end-to-end workflow parser."""
    hf_parser = subparsers.add_parser(
        'hf',
        help='Create HuggingFace sentence-transformers models (end-to-end)',
        description='Complete workflow: train ontology → create HuggingFace model → test → prepare for upload',
        epilog='''examples:
  on2vec hf biomedical.owl my-bio-model
  on2vec hf ontology.owl advanced-model --base-model all-mpnet-base-v2 --fusion attention --epochs 200
  on2vec hf onto.owl public-model --upload --hub-name username/my-ontology-model
  on2vec hf complex.owl model --author "Your Name" --description "Custom ontology embeddings"''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    hf_parser.add_argument('ontology', help='Path to OWL ontology file')
    hf_parser.add_argument('model_name', help='Name for the HuggingFace model')
    hf_parser.add_argument('--output-dir', default='./hf_models', help='Output directory for models (default: %(default)s)')
    hf_parser.add_argument('--base-model', help='Base sentence transformer model')
    hf_parser.add_argument('--fusion', choices=['concat', 'attention', 'gated', 'weighted_avg'],
                           default='concat', help='Fusion method for combining embeddings (default: %(default)s)')
    hf_parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    hf_parser.add_argument('--skip-testing', action='store_true', help='Skip testing step')

    # Training configuration
    training_group = hf_parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', type=int, default=100, help='Training epochs (default: %(default)s)')
    training_group.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                               default='gcn', help='GNN model architecture (default: %(default)s)')
    training_group.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimensions (default: %(default)s)')
    training_group.add_argument('--out-dim', type=int, default=64, help='Output embedding dimensions (default: %(default)s)')
    training_group.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                               default='triplet', help='Loss function (default: %(default)s)')
    training_group.add_argument('--use-multi-relation', action='store_true',
                               help='Include all ObjectProperty relations')
    training_group.add_argument('--text-model', help='Text model for semantic features (overrides base-model for training)')

    # Model details configuration
    details_group = hf_parser.add_argument_group('Model Details')
    details_group.add_argument('--author', help='Model author name')
    details_group.add_argument('--author-email', help='Model author email')
    details_group.add_argument('--description', help='Custom model description')
    details_group.add_argument('--domain', help='Ontology domain (auto-detected if not specified)')
    details_group.add_argument('--license', default='apache-2.0', help='Model license (default: %(default)s)')
    details_group.add_argument('--tags', nargs='+', help='Additional custom tags')

    # HuggingFace upload options
    upload_group = hf_parser.add_argument_group('HuggingFace Upload')
    upload_group.add_argument('--upload', action='store_true', help='Automatically upload to HuggingFace Hub')
    upload_group.add_argument('--hub-name', help='HuggingFace Hub model name (e.g., username/model-name)')
    upload_group.add_argument('--private', action='store_true', help='Make the uploaded model private')
    upload_group.add_argument('--commit-message', help='Commit message for upload')

    # Global options
    hf_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                           help='Device for training and processing (default: %(default)s - auto-detects best available)')
    hf_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    hf_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')


def setup_hf_train_parser(subparsers):
    """Set up the HuggingFace train command parser."""
    hf_train_parser = subparsers.add_parser(
        'hf-train',
        help='Train ontology embeddings for HuggingFace integration',
        description='Train ontology models with text features for HuggingFace model creation'
    )

    hf_train_parser.add_argument('ontology', help='Path to OWL ontology file')
    hf_train_parser.add_argument('--output', '-o', required=True, help='Output embeddings file (.parquet)')
    hf_train_parser.add_argument('--text-model', default='all-MiniLM-L6-v2', help='Base text model (default: %(default)s)')
    hf_train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: %(default)s)')
    hf_train_parser.add_argument('--model-type', choices=['gcn', 'gat'], default='gcn', help='GNN architecture (default: %(default)s)')
    hf_train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimensions (default: %(default)s)')
    hf_train_parser.add_argument('--out-dim', type=int, default=64, help='Output dimensions (default: %(default)s)')
    hf_train_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                                 help='Device for training (default: %(default)s - auto-detects best available)')


def setup_hf_create_parser(subparsers):
    """Set up the HuggingFace create command parser."""
    hf_create_parser = subparsers.add_parser(
        'hf-create',
        help='Create HuggingFace model from embeddings',
        description='Create sentence-transformers compatible models from ontology embeddings'
    )

    hf_create_parser.add_argument('embeddings', help='Path to embeddings file (.parquet)')
    hf_create_parser.add_argument('model_name', help='Name for the HuggingFace model')
    hf_create_parser.add_argument('--output-dir', default='./hf_models', help='Output directory (default: %(default)s)')
    hf_create_parser.add_argument('--base-model', help='Base sentence transformer (auto-detected if not specified)')
    hf_create_parser.add_argument('--fusion', choices=['concat', 'attention', 'gated', 'weighted_avg'],
                                  default='concat', help='Fusion method (default: %(default)s)')
    hf_create_parser.add_argument('--ontology', help='Original ontology file (for model card generation)')

    # Model details configuration
    hf_create_parser.add_argument('--author', help='Model author name')
    hf_create_parser.add_argument('--author-email', help='Model author email')
    hf_create_parser.add_argument('--description', help='Custom model description')
    hf_create_parser.add_argument('--domain', help='Ontology domain (auto-detected if not specified)')
    hf_create_parser.add_argument('--license', default='apache-2.0', help='Model license')
    hf_create_parser.add_argument('--tags', nargs='+', help='Additional custom tags')

    # HuggingFace upload options
    hf_create_parser.add_argument('--upload', action='store_true', help='Automatically upload to HuggingFace Hub')
    hf_create_parser.add_argument('--hub-name', help='HuggingFace Hub model name (e.g., username/model-name)')
    hf_create_parser.add_argument('--private', action='store_true', help='Make the uploaded model private')
    hf_create_parser.add_argument('--commit-message', help='Commit message for upload')


def setup_hf_test_parser(subparsers):
    """Set up the HuggingFace test command parser."""
    hf_test_parser = subparsers.add_parser(
        'hf-test',
        help='Test HuggingFace models',
        description='Test sentence-transformers models with sample queries'
    )

    hf_test_parser.add_argument('model_path', help='Path to HuggingFace model directory')
    hf_test_parser.add_argument('--queries', nargs='+', help='Custom test queries')


def setup_hf_batch_parser(subparsers):
    """Set up the HuggingFace batch processing parser."""
    hf_batch_parser = subparsers.add_parser(
        'hf-batch',
        help='Batch process multiple ontologies for HuggingFace models',
        description='Process multiple OWL files to create HuggingFace models in batch',
        epilog='''examples:
  on2vec hf-batch owl_files/ ./output
  on2vec hf-batch owl_files/ ./output --max-workers 4 --base-models all-MiniLM-L6-v2 all-mpnet-base-v2
  on2vec hf-batch owl_files/ ./output --fusion-methods concat attention --epochs 100 200''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    hf_batch_parser.add_argument('input_dir', help='Directory containing OWL files')
    hf_batch_parser.add_argument('output_dir', help='Output directory for results')
    hf_batch_parser.add_argument('--base-models', nargs='+', default=['all-MiniLM-L6-v2'],
                                 help='Base models to test (default: %(default)s)')
    hf_batch_parser.add_argument('--fusion-methods', nargs='+', default=['concat'],
                                 help='Fusion methods to test (default: %(default)s)')
    hf_batch_parser.add_argument('--max-workers', type=int, default=2, help='Parallel workers (default: %(default)s)')

    # Processing options
    hf_batch_parser.add_argument('--force-retrain', action='store_true',
                                 help='Force retraining even if embeddings exist')
    hf_batch_parser.add_argument('--owl-pattern', default='*.owl',
                                 help='Pattern for finding OWL files (default: %(default)s)')
    hf_batch_parser.add_argument('--limit', type=int,
                                 help='Limit number of OWL files to process')

    # Training configuration
    training_group = hf_batch_parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', nargs='+', type=int, default=[100],
                               help='Training epochs to test (can specify multiple)')
    training_group.add_argument('--model-types', nargs='+', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                               default=['gcn'], help='GNN model architectures to test (can specify multiple)')
    training_group.add_argument('--hidden-dims', nargs='+', type=int, default=[128],
                               help='Hidden layer dimensions to test (can specify multiple)')
    training_group.add_argument('--out-dims', nargs='+', type=int, default=[64],
                               help='Output embedding dimensions to test (can specify multiple)')
    training_group.add_argument('--loss-fns', nargs='+', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                               default=['triplet'], help='Loss functions to test (can specify multiple)')
    training_group.add_argument('--use-multi-relation', action='store_true',
                               help='Include all ObjectProperty relations')
    training_group.add_argument('--text-model', help='Text model for semantic features (overrides base-model for training)')

    # Keep backward compatibility with singular versions
    training_group.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                               help='Single GNN model architecture (deprecated, use --model-types)')
    training_group.add_argument('--hidden-dim', type=int,
                               help='Single hidden layer dimension (deprecated, use --hidden-dims)')
    training_group.add_argument('--out-dim', type=int,
                               help='Single output embedding dimension (deprecated, use --out-dims)')
    training_group.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                               help='Single loss function (deprecated, use --loss-fns)')

    # Model details configuration
    details_group = hf_batch_parser.add_argument_group('Model Details')
    details_group.add_argument('--author', help='Model author name')
    details_group.add_argument('--author-email', help='Model author email')
    details_group.add_argument('--description', help='Custom model description template (use {ontology_name} placeholder)')
    details_group.add_argument('--domain', help='Ontology domain (auto-detected if not specified)')
    details_group.add_argument('--license', default='apache-2.0', help='Model license (default: %(default)s)')
    details_group.add_argument('--tags', nargs='+', help='Additional custom tags')

    # HuggingFace upload options
    upload_group = hf_batch_parser.add_argument_group('HuggingFace Upload')
    upload_group.add_argument('--upload', action='store_true', help='Automatically upload to HuggingFace Hub')
    upload_group.add_argument('--hub-name-template', help='HuggingFace Hub model name template (e.g., username/{ontology_name}-{config_id})')
    upload_group.add_argument('--private', action='store_true', help='Make the uploaded models private')
    upload_group.add_argument('--commit-message', help='Commit message template for uploads')

    # Collection creation options
    collection_group = hf_batch_parser.add_argument_group('Collection Creation')
    collection_group.add_argument('--create-collection', help='Create model collection with this name')
    collection_group.add_argument('--collection-criteria', choices=['best_test', 'fastest', 'smallest'],
                                  default='best_test', help='Selection criteria for collection')


def setup_evaluate_parser(subparsers):
    """Set up the evaluate command parser."""
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Comprehensively evaluate ontology embeddings',
        description='Perform intrinsic and extrinsic evaluation of ontology embeddings',
        epilog='''examples:
  on2vec evaluate embeddings.parquet
  on2vec evaluate embeddings.parquet --ontology ontology.owl --output-dir results
  on2vec evaluate embeddings.parquet --intrinsic --clustering-methods kmeans dbscan''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    evaluate_parser.add_argument('embeddings', help='Path to embeddings parquet file')
    evaluate_parser.add_argument('--ontology', help='Path to original OWL ontology file (enables structural evaluation)')
    evaluate_parser.add_argument('--output-dir', help='Directory to save evaluation results and visualizations')
    evaluate_parser.add_argument('--no-plots', action='store_true', help='Skip generating visualization plots')

    # Evaluation subset selection
    eval_group = evaluate_parser.add_argument_group('Evaluation Selection')
    eval_group.add_argument('--intrinsic', action='store_true', help='Run only intrinsic evaluation')
    eval_group.add_argument('--extrinsic', action='store_true', help='Run only extrinsic evaluation')
    eval_group.add_argument('--ontology-specific', action='store_true', help='Run only ontology-specific evaluation')
    eval_group.add_argument('--skip-clustering', action='store_true', help='Skip clustering analysis')
    eval_group.add_argument('--skip-link-prediction', action='store_true', help='Skip link prediction evaluation')
    eval_group.add_argument('--skip-hierarchy', action='store_true', help='Skip hierarchy preservation evaluation')

    # Clustering configuration
    cluster_group = evaluate_parser.add_argument_group('Clustering Configuration')
    cluster_group.add_argument('--clustering-methods', nargs='+',
                              choices=['kmeans', 'dbscan', 'hierarchical'],
                              default=['kmeans', 'dbscan', 'hierarchical'],
                              help='Clustering methods to evaluate')
    cluster_group.add_argument('--n-clusters', nargs='+', type=int,
                              default=[5, 10, 15, 20],
                              help='Number of clusters to test for k-means and hierarchical')

    evaluate_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')


def setup_evaluate_batch_parser(subparsers):
    """Set up the evaluate-batch command parser."""
    batch_parser = subparsers.add_parser(
        'evaluate-batch',
        help='Evaluate multiple embedding files in batch',
        description='Run comprehensive evaluation on multiple embedding files and create comparison reports'
    )

    batch_parser.add_argument('embeddings', nargs='*', help='Paths to embedding parquet files')
    batch_parser.add_argument('--embeddings-list', help='File containing list of embedding file paths (one per line)')
    batch_parser.add_argument('--ontology', nargs='*', help='Paths to corresponding OWL ontology files (optional)')
    batch_parser.add_argument('--ontology-list', help='File containing list of ontology file paths (one per line)')
    batch_parser.add_argument('--output-dir', default='evaluation_batch_results',
                             help='Directory to save batch evaluation results (default: %(default)s)')

    # Evaluation subset selection (same as single evaluation)
    eval_group = batch_parser.add_argument_group('Evaluation Selection')
    eval_group.add_argument('--intrinsic', action='store_true', help='Run only intrinsic evaluation')
    eval_group.add_argument('--extrinsic', action='store_true', help='Run only extrinsic evaluation')
    eval_group.add_argument('--ontology-specific', action='store_true', help='Run only ontology-specific evaluation')
    eval_group.add_argument('--skip-clustering', action='store_true', help='Skip clustering analysis')
    eval_group.add_argument('--skip-link-prediction', action='store_true', help='Skip link prediction evaluation')
    eval_group.add_argument('--skip-hierarchy', action='store_true', help='Skip hierarchy preservation evaluation')

    batch_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')


def setup_benchmark_parser(subparsers):
    """Set up the MTEB benchmark command parser."""
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run MTEB benchmarks on models',
        description='Evaluate sentence-transformers models against MTEB benchmark tasks',
        epilog='''examples:
  on2vec benchmark ./hf_models/my-model
  on2vec benchmark my-model --quick --output-dir ./results
  on2vec benchmark model-path --task-types STS Classification --batch-size 16''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    benchmark_parser.add_argument('model_path', help='Path to model or model name')
    benchmark_parser.add_argument('--output-dir', default='./mteb_results', help='Results output directory (default: %(default)s)')
    benchmark_parser.add_argument('--model-name', help='Model name for results')
    benchmark_parser.add_argument('--tasks', nargs='+', help='Specific tasks to run')
    benchmark_parser.add_argument('--task-types', nargs='+', choices=[
        'Classification', 'Clustering', 'PairClassification', 'Reranking',
        'Retrieval', 'STS', 'Summarization'
    ], help='Task types to run')
    benchmark_parser.add_argument('--quick', action='store_true', help='Run quick subset of tasks')
    benchmark_parser.add_argument('--batch-size', type=int, default=32, help='Evaluation batch size (default: %(default)s)')


def setup_compare_parser(subparsers):
    """Set up the model comparison parser."""
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare ontology-augmented vs vanilla models',
        description='Compare performance of ontology models against vanilla sentence transformers'
    )

    compare_parser.add_argument('model_path', help='Path to ontology-augmented model')
    compare_parser.add_argument('--vanilla-model', default='all-MiniLM-L6-v2', help='Vanilla model for comparison (default: %(default)s)')
    compare_parser.add_argument('--domain-terms', nargs='+', help='Domain-specific terms for testing')
    compare_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')


def setup_inspect_parser(subparsers):
    """Set up the inspect command parser."""
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect embedding files and models',
        description='Display metadata and statistics for embedding files and models'
    )

    inspect_parser.add_argument('file', help='Path to embeddings file (.parquet) or model directory')
    inspect_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def setup_convert_parser(subparsers):
    """Set up the convert command parser."""
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert between embedding file formats',
        description='Convert embedding files between different formats (parquet, csv, etc.)'
    )

    convert_parser.add_argument('input_file', help='Input file path')
    convert_parser.add_argument('output_file', help='Output file path')
    convert_parser.add_argument('--format', choices=['csv', 'parquet'], help='Output format (auto-detected if not specified)')


def run_train_command(args):
    """Execute the train command."""
    from .workflows import train_model_only

    # Validate inputs
    if not validate_file_exists(args.ontology, "OWL ontology file"):
        return 1

    if not validate_directory_exists(Path(args.output).parent, create_if_missing=True):
        return 1

    # Display configuration
    config = {
        'ontology_file': args.ontology,
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'out_dim': args.out_dim,
        'epochs': args.epochs,
        'loss_function': args.loss_fn,
        'use_multi_relation': args.use_multi_relation,
        'use_text_features': args.use_text_features,
        'text_model': args.text_model if args.use_text_features else 'N/A'
    }
    display_configuration(config, "Training Configuration")

    try:
        start_time = time.time()

        def train_operation():
            return train_model_only(
                owl_file=args.ontology,
                model_output=args.output,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                epochs=args.epochs,
                loss_fn=args.loss_fn,
                use_multi_relation=args.use_multi_relation,
                use_text_features=args.use_text_features,
                text_model_name=args.text_model or 'all-MiniLM-L6-v2',
                device=args.device
            )

        result = with_progress_tracking("Model training", train_operation)

        # Add timing info
        result['elapsed_time'] = time.time() - start_time

        # Add file size if model exists
        if Path(args.output).exists():
            result['file_size'] = Path(args.output).stat().st_size

        print_success_summary("Model Training", {
            'model_path': result['model_path'],
            'elapsed_time': result['elapsed_time'],
            'file_size': result.get('file_size')
        })
        return 0
    except Exception as e:
        return handle_cli_error(e, "Model training", verbose=getattr(args, 'verbose', False))


def run_embed_command(args):
    """Execute the embed command."""
    from .workflows import embed_with_trained_model

    # Validate inputs
    if not validate_file_exists(args.model, "model file"):
        return 1

    if not validate_file_exists(args.ontology, "OWL ontology file"):
        return 1

    if not validate_directory_exists(Path(args.output).parent, create_if_missing=True):
        return 1

    # Display configuration
    config = {
        'model_file': args.model,
        'ontology_file': args.ontology,
        'output_file': args.output
    }
    display_configuration(config, "Embedding Generation Configuration")

    try:
        start_time = time.time()

        def embed_operation():
            return embed_with_trained_model(
                model_path=args.model,
                owl_file=args.ontology,
                output_file=args.output,
                device=args.device
            )

        result = with_progress_tracking("Embedding generation", embed_operation)

        # Add timing and file size info
        result['elapsed_time'] = time.time() - start_time
        if Path(args.output).exists():
            result['file_size'] = Path(args.output).stat().st_size

        print_success_summary("Embedding Generation", {
            'output_file': result['output_file'],
            'num_embeddings': result['num_embeddings'],
            'elapsed_time': result['elapsed_time'],
            'file_size': result.get('file_size')
        })
        return 0
    except Exception as e:
        return handle_cli_error(e, "Embedding generation", verbose=getattr(args, 'verbose', False))


def run_visualize_command(args):
    """Execute the visualize command."""
    import subprocess
    import os

    # Use the comprehensive visualization script
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
    viz_script = os.path.join(script_dir, 'visualize_embeddings.py')

    if not os.path.exists(viz_script):
        print(f"❌ Visualization script not found: {viz_script}")
        return 1

    # Build command arguments
    cmd = ['python', viz_script, args.embeddings]

    if hasattr(args, 'output') and args.output:
        cmd.extend(['--output-dir', args.output])
    if hasattr(args, 'neighbors') and args.neighbors:
        cmd.extend(['--umap-neighbors', str(args.neighbors)])
    if hasattr(args, 'min_dist') and args.min_dist:
        cmd.extend(['--umap-min-dist', str(args.min_dist)])

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization failed: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running visualization: {e}")
        return 1


def run_hf_command(args):
    """Execute the HuggingFace end-to-end command."""
    from .huggingface_workflows import end_to_end_workflow

    # Validate inputs
    if not validate_file_exists(args.ontology, "OWL ontology file"):
        return 1

    if not validate_directory_exists(args.output_dir, create_if_missing=True):
        return 1

    # Build training configuration
    training_config = {
        'epochs': args.epochs,
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'out_dim': args.out_dim,
        'loss_fn': args.loss_fn,
        'use_multi_relation': args.use_multi_relation,
        'device': args.device,
    }

    # Use text-model if specified, otherwise use base-model
    text_model = args.text_model or args.base_model or "all-MiniLM-L6-v2"
    training_config['text_model'] = text_model

    # Build model details
    model_details = {}
    if args.author:
        model_details['author'] = args.author
    if args.author_email:
        model_details['author_email'] = args.author_email
    if args.description:
        model_details['description'] = args.description
    if args.domain:
        model_details['domain'] = args.domain
    if args.license:
        model_details['license'] = args.license
    if args.tags:
        model_details['tags'] = args.tags

    # Build upload options
    upload_options = {}
    if args.upload:
        upload_options['upload'] = True
        upload_options['hub_name'] = args.hub_name or f"your-username/{args.model_name}"
        upload_options['private'] = args.private
        upload_options['commit_message'] = args.commit_message

    # Display comprehensive configuration
    display_config = {
        **training_config,
        'ontology_file': args.ontology,
        'model_name': args.model_name,
        'output_directory': args.output_dir,
        'base_model': args.base_model or "all-MiniLM-L6-v2",
        'fusion_method': args.fusion,
        'skip_training': args.skip_training,
        'skip_testing': args.skip_testing,
        'upload_to_hub': args.upload
    }
    display_configuration(display_config, "HuggingFace Workflow Configuration")

    try:
        start_time = time.time()

        def hf_workflow():
            return end_to_end_workflow(
                owl_file=args.ontology,
                model_name=args.model_name,
                output_dir=args.output_dir,
                base_model=args.base_model or "all-MiniLM-L6-v2",
                fusion_method=args.fusion,
                skip_training=args.skip_training,
                skip_testing=args.skip_testing,
                training_config=training_config,
                model_details=model_details,
                upload_options=upload_options
            )

        success = with_progress_tracking("HuggingFace model creation", hf_workflow)

        if success:
            elapsed_time = time.time() - start_time
            model_path = Path(args.output_dir) / args.model_name
            file_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) if model_path.exists() else None

            print_success_summary("HuggingFace Model Creation", {
                'model_path': str(model_path),
                'elapsed_time': elapsed_time,
                'file_size': file_size
            })

        return 0 if success else 1
    except Exception as e:
        return handle_cli_error(e, "HuggingFace workflow", verbose=getattr(args, 'verbose', False))


def run_hf_train_command(args):
    """Execute the HuggingFace train command."""
    from .huggingface_workflows import train_ontology_with_text

    try:
        success = train_ontology_with_text(
            owl_file=args.ontology,
            output_file=args.output,
            text_model=args.text_model or "all-MiniLM-L6-v2",
            epochs=args.epochs,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            device=args.device
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ HuggingFace training failed: {e}")
        return 1


def run_hf_create_command(args):
    """Execute the HuggingFace create command."""
    from .huggingface_workflows import create_hf_model

    # Build model details
    model_details = {}
    if args.author:
        model_details['author'] = args.author
    if args.author_email:
        model_details['author_email'] = args.author_email
    if args.description:
        model_details['description'] = args.description
    if args.domain:
        model_details['domain'] = args.domain
    if args.license:
        model_details['license'] = args.license
    if args.tags:
        model_details['tags'] = args.tags

    # Build upload options
    upload_options = {}
    if args.upload:
        upload_options['upload'] = True
        upload_options['hub_name'] = args.hub_name or f"your-username/{args.model_name}"
        upload_options['private'] = args.private
        upload_options['commit_message'] = args.commit_message

    try:
        model_path = create_hf_model(
            embeddings_file=args.embeddings,
            model_name=args.model_name,
            output_dir=args.output_dir,
            base_model=args.base_model,
            fusion_method=args.fusion,
            ontology_file=args.ontology,
            model_details=model_details,
            upload_options=upload_options
        )
        print(f"✅ HuggingFace model created: {model_path}")
        return 0
    except Exception as e:
        print(f"❌ HuggingFace model creation failed: {e}")
        return 1


def run_hf_test_command(args):
    """Execute the HuggingFace test command."""
    from .huggingface_workflows import validate_hf_model

    try:
        success = validate_hf_model(args.model_path, args.queries)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ HuggingFace model testing failed: {e}")
        return 1


def run_hf_batch_command(args):
    """Execute the HuggingFace batch command."""
    from pathlib import Path
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Build training configuration with backward compatibility
    # Handle backward compatibility - convert singular to plural
    model_types = args.model_types
    if args.model_type and args.model_type not in model_types:
        model_types = [args.model_type]

    hidden_dims = args.hidden_dims
    if args.hidden_dim and args.hidden_dim not in hidden_dims:
        hidden_dims = [args.hidden_dim]

    out_dims = args.out_dims
    if args.out_dim and args.out_dim not in out_dims:
        out_dims = [args.out_dim]

    loss_fns = args.loss_fns
    if args.loss_fn and args.loss_fn not in loss_fns:
        loss_fns = [args.loss_fn]

    training_config = {
        'model_types': model_types,
        'hidden_dims': hidden_dims,
        'out_dims': out_dims,
        'loss_fns': loss_fns,
        'use_multi_relation': args.use_multi_relation,
    }

    # Use text-model if specified, otherwise use base-models
    if args.text_model:
        training_config['text_model'] = args.text_model

    # Build model details
    model_details = {}
    if args.author:
        model_details['author'] = args.author
    if args.author_email:
        model_details['author_email'] = args.author_email
    if args.description:
        model_details['description'] = args.description
    if args.domain:
        model_details['domain'] = args.domain
    if args.license:
        model_details['license'] = args.license
    if args.tags:
        model_details['tags'] = args.tags

    # Build upload options
    upload_options = {}
    if args.upload:
        upload_options['upload'] = True
        upload_options['hub_name_template'] = args.hub_name_template or "your-username/{ontology_name}-{config_id}"
        upload_options['private'] = args.private
        upload_options['commit_message'] = args.commit_message
    elif args.hub_name_template:
        # Even if not uploading, preserve template for upload instructions
        upload_options['hub_name_template'] = args.hub_name_template

    # Build collection options
    collection_options = {}
    if args.create_collection:
        collection_options['name'] = args.create_collection
        collection_options['criteria'] = args.collection_criteria

    try:
        # Log what advanced features are enabled
        if upload_options and upload_options.get('upload'):
            logger.info("Upload to HuggingFace Hub enabled")
        if model_details:
            logger.info("Custom model metadata configured")

        # Use existing batch processing
        from .batch_hf_models import batch_process_ontologies
        results = batch_process_ontologies(
            owl_directory=args.input_dir,
            output_directory=args.output_dir,
            base_models=args.base_models,
            fusion_methods=args.fusion_methods,
            epochs_list=args.epochs,
            max_workers=args.max_workers,
            force_retrain=args.force_retrain,
            owl_pattern=args.owl_pattern,
            limit=args.limit,
            training_config=training_config,  # ADD THIS!
            model_details=model_details if model_details else None,
            upload_options=upload_options if upload_options else None
        )

        # Print summary
        from .batch_hf_models import print_summary_report
        print_summary_report(results)

        # Create collection if requested
        if args.create_collection:
            from .batch_hf_models import create_model_collection
            collection_path = create_model_collection(
                summary=results,
                collection_name=args.create_collection,
                output_dir=args.output_dir,
                selection_criteria=args.collection_criteria
            )
            print(f"✅ Collection created: {collection_path}")

        return 0 if results.get('success_rate', 0) > 0 else 1

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


def run_benchmark_command(args):
    """Execute the benchmark command."""
    import subprocess

    cmd = ['python', 'mteb_benchmarks/benchmark_runner.py', args.model_path]

    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    if args.model_name:
        cmd.extend(['--model-name', args.model_name])
    if args.tasks:
        cmd.extend(['--tasks'] + args.tasks)
    if args.task_types:
        cmd.extend(['--task-types'] + args.task_types)
    if args.quick:
        cmd.append('--quick')
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])

    return subprocess.run(cmd).returncode


def run_evaluate_command(args):
    """Execute the evaluate command."""
    from .evaluation import EmbeddingEvaluator
    import logging
    from pathlib import Path

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating embeddings: {args.embeddings}")

    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(args.output_dir) / "evaluation_report.json"
    else:
        report_path = None

    try:
        # Create evaluator
        evaluator = EmbeddingEvaluator(args.embeddings, args.ontology)

        # Determine which evaluations to run
        run_intrinsic = not args.extrinsic and not args.ontology_specific
        run_extrinsic = not args.intrinsic and not args.ontology_specific
        run_ontology_specific = not args.intrinsic and not args.extrinsic

        # If specific flags are set, override defaults
        if args.intrinsic:
            run_intrinsic = True
        if args.extrinsic:
            run_extrinsic = True
        if args.ontology_specific:
            run_ontology_specific = True

        results = {'metadata': {
            'embeddings_file': args.embeddings,
            'ontology_file': args.ontology,
            'embedding_shape': evaluator.embeddings.shape,
            'embedding_metadata': evaluator.metadata
        }}

        # Run evaluations based on selection
        if run_intrinsic:
            logger.info("Running intrinsic evaluation...")
            clustering_methods = args.clustering_methods if not args.skip_clustering else []
            intrinsic_results = evaluator.evaluate_intrinsic(
                clustering_methods=clustering_methods,
                n_clusters_range=args.n_clusters
            )
            results['intrinsic_evaluation'] = intrinsic_results

        if run_extrinsic:
            logger.info("Running extrinsic evaluation...")
            extrinsic_results = evaluator.evaluate_extrinsic(
                link_prediction=not args.skip_link_prediction
            )
            results['extrinsic_evaluation'] = extrinsic_results

        if run_ontology_specific:
            logger.info("Running ontology-specific evaluation...")
            ont_specific_results = evaluator.evaluate_ontology_specific()
            results['ontology_specific_evaluation'] = ont_specific_results

        # Save report if path provided
        if report_path:
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            results_serializable = convert_numpy(results)

            with open(report_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)

            logger.info(f"Evaluation report saved to {report_path}")

        # Create visualizations if output directory specified
        if args.output_dir and not args.no_plots:
            logger.info("Creating visualizations...")
            viz_paths = evaluator.visualize_evaluation_results(results, args.output_dir)
            logger.info(f"Visualizations saved to: {args.output_dir}")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Embedding info
        shape = results['metadata']['embedding_shape']
        print(f"Embeddings shape: {shape[0]} nodes × {shape[1]} dimensions")

        # Intrinsic metrics
        if 'intrinsic_evaluation' in results:
            intrinsic = results['intrinsic_evaluation']

            if 'distribution' in intrinsic:
                dist = intrinsic['distribution']
                print(f"Mean embedding norm: {dist['norms']['mean_norm']:.3f}")
                print(f"Mean cosine similarity: {dist['similarities']['mean_similarity']:.3f}")

            if 'dimensionality' in intrinsic:
                dim = intrinsic['dimensionality']
                eff_dims = dim['effective_dimensions']
                print(f"Effective dimensions (95% variance): {eff_dims['95_percent_variance']}")

            if 'clustering' in intrinsic and isinstance(intrinsic['clustering'], dict):
                clustering = intrinsic['clustering']
                if 'kmeans' in clustering:
                    kmeans = clustering['kmeans']
                    best_silhouette = max([
                        result['silhouette_score']
                        for result in kmeans.values()
                        if isinstance(result, dict) and 'silhouette_score' in result
                    ], default=0.0)
                    print(f"Best K-means silhouette score: {best_silhouette:.3f}")

        # Extrinsic metrics
        if 'extrinsic_evaluation' in results:
            extrinsic = results['extrinsic_evaluation']

            if 'link_prediction' in extrinsic:
                link_pred = extrinsic['link_prediction']
                if 'classifiers' in link_pred:
                    classifiers = link_pred['classifiers']
                    if 'logistic_regression' in classifiers:
                        lr_results = classifiers['logistic_regression']
                        if 'roc_auc' in lr_results:
                            print(f"Link prediction ROC-AUC: {lr_results['roc_auc']:.3f}")

            if 'hierarchy' in extrinsic:
                hierarchy = extrinsic['hierarchy']
                if 'similarity_difference' in hierarchy:
                    print(f"Hierarchy preservation (similarity diff): {hierarchy['similarity_difference']:.3f}")

        # Ontology-specific metrics
        if 'ontology_specific_evaluation' in results:
            ont_specific = results['ontology_specific_evaluation']

            if 'structural_consistency' in ont_specific:
                structural = ont_specific['structural_consistency']
                if 'centrality_correlations' in structural:
                    centrality = structural['centrality_correlations']
                    if 'degree' in centrality and isinstance(centrality['degree'], dict) and 'pearson_correlation' in centrality['degree']:
                        degree_corr = centrality['degree']['pearson_correlation']
                        print(f"Degree centrality correlation: {degree_corr:.3f}")

        print("="*60)

        if report_path:
            print(f"\nDetailed report saved to: {report_path}")

        logger.info("Evaluation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_evaluate_batch_command(args):
    """Execute the evaluate-batch command."""
    from .evaluation import create_evaluation_benchmark
    import logging
    from pathlib import Path

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.info("Running batch evaluation")

    # Parse embedding files
    embeddings_files = []
    if args.embeddings_list:
        with open(args.embeddings_list, 'r') as f:
            embeddings_files = [line.strip() for line in f if line.strip()]
    else:
        embeddings_files = args.embeddings or []

    # Parse ontology files
    ontology_files = None
    if args.ontology_list:
        with open(args.ontology_list, 'r') as f:
            ontology_files = [line.strip() for line in f if line.strip()]
    elif args.ontology:
        ontology_files = args.ontology

    if not embeddings_files:
        logger.error("No embedding files provided")
        return 1

    try:
        # Run benchmark
        results = create_evaluation_benchmark(
            embeddings_files,
            ontology_files,
            args.output_dir
        )

        # Print summary
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)

        successful_evaluations = [k for k, v in results.items() if 'error' not in v]
        failed_evaluations = [k for k, v in results.items() if 'error' in v]

        print(f"Successfully evaluated: {len(successful_evaluations)} files")
        print(f"Failed evaluations: {len(failed_evaluations)} files")

        if failed_evaluations:
            print("\nFailed files:")
            for failed_file in failed_evaluations:
                error = results[failed_file]['error']
                print(f"  - {failed_file}: {error}")

        print(f"\nBatch results saved to: {args.output_dir}")
        print("="*60)

        logger.info("Batch evaluation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_compare_command(args):
    """Execute the compare command."""
    from .model_comparison import main_compare

    # Use the generic model comparison module
    try:
        success = main_compare(
            ontology_model_path=args.model_path,
            vanilla_model_path=args.vanilla_model,
            domain_terms=args.domain_terms,
            detailed=args.detailed
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return 1


def run_inspect_command(args):
    """Execute the inspect command."""
    from on2vec.io import inspect_parquet_metadata
    from pathlib import Path

    file_path = Path(args.file)

    if file_path.suffix == '.parquet':
        # Inspect embeddings file
        inspect_parquet_metadata(str(file_path))
    elif file_path.is_dir() and (file_path / 'config.json').exists():
        # Inspect HuggingFace model
        print(f"🤗 HuggingFace Model: {file_path}")

        # Read model metadata
        import json
        config_path = file_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"📐 Model type: {config.get('architectures', ['Unknown'])[0]}")

        metadata_path = file_path / 'on2vec_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"🧬 Source: {metadata.get('ontology_source', 'Unknown')}")
                print(f"🤖 Base model: {metadata.get('base_model', 'Unknown')}")
                print(f"🔗 Fusion: {metadata.get('fusion_method', 'Unknown')}")
                print(f"📊 Concepts: {metadata.get('ontology_concepts', 'Unknown')}")
    else:
        print(f"❌ Unknown file type: {file_path}")
        return 1

    return 0


def run_convert_command(args):
    """Execute the convert command."""
    from on2vec.io import convert_parquet_to_csv
    from pathlib import Path

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if input_path.suffix == '.parquet' and output_path.suffix == '.csv':
        csv_file = convert_parquet_to_csv(str(input_path), str(output_path))
        print(f"✅ Converted to: {csv_file}")
        return 0
    else:
        print(f"❌ Unsupported conversion: {input_path.suffix} → {output_path.suffix}")
        return 1


def main(args: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = create_parser()

    if args is None:
        args = sys.argv[1:]

    # Handle special arguments before parsing
    if '--save-config' in args:
        # Parse just to get the save-config argument
        temp_parser = argparse.ArgumentParser()
        temp_parser.add_argument('--save-config', metavar='PATH')
        temp_args, _ = temp_parser.parse_known_args(args)
        save_sample_config(temp_args.save_config)
        return 0

    # If no command provided, show help
    if not args or args[0] in ['-h', '--help']:
        parser.print_help()
        print("\n💡 Quick shortcuts: t=train, e=embed, v=viz, eval=evaluate, b=bench, i=inspect")
        print("📝 Configuration: --config file.yml or --save-config file.yml")
        return 0

    # Handle alias expansion before parsing
    if len(args) > 0:
        alias_map = {
            't': 'train',
            'e': 'embed',
            'viz': 'visualize',
            'v': 'visualize',
            'eval': 'evaluate',
            'bench': 'benchmark',
            'b': 'benchmark',
            'i': 'inspect'
        }
        if args[0] in alias_map:
            args[0] = alias_map[args[0]]

    parsed_args = parser.parse_args(args)

    # Load configuration if specified or find default
    config = {}
    config_path = getattr(parsed_args, 'config', None)
    if not config_path:
        config_path = find_default_config()

    if config_path:
        try:
            config = load_config_file(config_path)
            print(f"📝 Using configuration from: {config_path}")
        except Exception as e:
            print(f"⚠️ Configuration file error: {e}")
            return 1

    # Merge config with command line arguments
    if hasattr(parsed_args, 'command') and parsed_args.command:
        parsed_args = merge_config_with_args(config, parsed_args, parsed_args.command)

    # If no subcommand was selected, show help
    if not hasattr(parsed_args, 'command') or parsed_args.command is None:
        parser.print_help()
        return 0

    # Route to appropriate command handler with aliases
    command_map = {
        # Full command names
        'train': run_train_command,
        'embed': run_embed_command,
        'visualize': run_visualize_command,
        'hf': run_hf_command,
        'hf-train': run_hf_train_command,
        'hf-create': run_hf_create_command,
        'hf-test': run_hf_test_command,
        'hf-batch': run_hf_batch_command,
        'evaluate': run_evaluate_command,
        'evaluate-batch': run_evaluate_batch_command,
        'benchmark': run_benchmark_command,
        'compare': run_compare_command,
        'inspect': run_inspect_command,
        'convert': run_convert_command,
        # Aliases
        't': run_train_command,
        'e': run_embed_command,
        'viz': run_visualize_command,
        'v': run_visualize_command,
        'eval': run_evaluate_command,
        'bench': run_benchmark_command,
        'b': run_benchmark_command,
        'i': run_inspect_command,
    }

    if parsed_args.command in command_map:
        try:
            return command_map[parsed_args.command](parsed_args)
        except Exception as e:
            print(f"❌ Error running {parsed_args.command}: {e}")
            return 1
    else:
        print(f"❌ Unknown command: {parsed_args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())