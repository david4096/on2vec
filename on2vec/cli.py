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
from typing import List, Optional
from pathlib import Path

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog='on2vec',
        description='Generate vector embeddings from OWL ontologies using Graph Neural Networks',
        epilog='For more help on a specific command, use: on2vec <command> --help'
    )

    parser.add_argument('--version', action='version', version='on2vec 0.1.0')

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
        description='Train Graph Neural Network models on OWL ontology structures'
    )

    train_parser.add_argument('ontology', help='Path to OWL ontology file')
    train_parser.add_argument('--output', '-o', required=True, help='Output model file path')
    train_parser.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                              default='gcn', help='GNN model architecture')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimensions')
    train_parser.add_argument('--out-dim', type=int, default=64, help='Output embedding dimensions')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                              default='triplet', help='Loss function')
    train_parser.add_argument('--use-multi-relation', action='store_true',
                              help='Include all ObjectProperty relations')
    train_parser.add_argument('--use-text-features', action='store_true',
                              help='Include text features from ontology')
    train_parser.add_argument('--text-model', default='all-MiniLM-L6-v2',
                              help='Text model for semantic features')


def setup_embed_parser(subparsers):
    """Set up the embed command parser."""
    embed_parser = subparsers.add_parser(
        'embed',
        help='Generate embeddings using trained models',
        description='Generate embeddings for ontology concepts using pre-trained GNN models'
    )

    embed_parser.add_argument('model', help='Path to trained model file')
    embed_parser.add_argument('ontology', help='Path to OWL ontology file')
    embed_parser.add_argument('--output', '-o', required=True, help='Output embeddings file (.parquet)')


def setup_visualize_parser(subparsers):
    """Set up the visualize command parser."""
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Create visualizations of embeddings',
        description='Generate UMAP visualizations and other plots from embedding files'
    )

    viz_parser.add_argument('embeddings', help='Path to embeddings file (.parquet)')
    viz_parser.add_argument('--output', '-o', help='Output visualization file (.png)')
    viz_parser.add_argument('--neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    viz_parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')


def setup_hf_parser(subparsers):
    """Set up the HuggingFace end-to-end workflow parser."""
    hf_parser = subparsers.add_parser(
        'hf',
        help='Create HuggingFace sentence-transformers models (end-to-end)',
        description='Complete workflow: train ontology → create HuggingFace model → test → prepare for upload'
    )

    hf_parser.add_argument('ontology', help='Path to OWL ontology file')
    hf_parser.add_argument('model_name', help='Name for the HuggingFace model')
    hf_parser.add_argument('--output-dir', default='./hf_models', help='Output directory for models')
    hf_parser.add_argument('--base-model', help='Base sentence transformer model')
    hf_parser.add_argument('--fusion', choices=['concat', 'attention', 'gated', 'weighted_avg'],
                           default='concat', help='Fusion method for combining embeddings')
    hf_parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    hf_parser.add_argument('--skip-testing', action='store_true', help='Skip testing step')

    # Training configuration
    training_group = hf_parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', type=int, default=100, help='Training epochs')
    training_group.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn', 'heterogeneous'],
                               default='gcn', help='GNN model architecture')
    training_group.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimensions')
    training_group.add_argument('--out-dim', type=int, default=64, help='Output embedding dimensions')
    training_group.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                               default='triplet', help='Loss function')
    training_group.add_argument('--use-multi-relation', action='store_true',
                               help='Include all ObjectProperty relations')
    training_group.add_argument('--text-model', help='Text model for semantic features (overrides base-model for training)')

    # Model details configuration
    details_group = hf_parser.add_argument_group('Model Details')
    details_group.add_argument('--author', help='Model author name')
    details_group.add_argument('--author-email', help='Model author email')
    details_group.add_argument('--description', help='Custom model description')
    details_group.add_argument('--domain', help='Ontology domain (auto-detected if not specified)')
    details_group.add_argument('--license', default='apache-2.0', help='Model license')
    details_group.add_argument('--tags', nargs='+', help='Additional custom tags')

    # HuggingFace upload options
    upload_group = hf_parser.add_argument_group('HuggingFace Upload')
    upload_group.add_argument('--upload', action='store_true', help='Automatically upload to HuggingFace Hub')
    upload_group.add_argument('--hub-name', help='HuggingFace Hub model name (e.g., username/model-name)')
    upload_group.add_argument('--private', action='store_true', help='Make the uploaded model private')
    upload_group.add_argument('--commit-message', help='Commit message for upload')


def setup_hf_train_parser(subparsers):
    """Set up the HuggingFace train command parser."""
    hf_train_parser = subparsers.add_parser(
        'hf-train',
        help='Train ontology embeddings for HuggingFace integration',
        description='Train ontology models with text features for HuggingFace model creation'
    )

    hf_train_parser.add_argument('ontology', help='Path to OWL ontology file')
    hf_train_parser.add_argument('--output', '-o', required=True, help='Output embeddings file (.parquet)')
    hf_train_parser.add_argument('--text-model', default='all-MiniLM-L6-v2', help='Base text model')
    hf_train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    hf_train_parser.add_argument('--model-type', choices=['gcn', 'gat'], default='gcn', help='GNN architecture')
    hf_train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimensions')
    hf_train_parser.add_argument('--out-dim', type=int, default=64, help='Output dimensions')


def setup_hf_create_parser(subparsers):
    """Set up the HuggingFace create command parser."""
    hf_create_parser = subparsers.add_parser(
        'hf-create',
        help='Create HuggingFace model from embeddings',
        description='Create sentence-transformers compatible models from ontology embeddings'
    )

    hf_create_parser.add_argument('embeddings', help='Path to embeddings file (.parquet)')
    hf_create_parser.add_argument('model_name', help='Name for the HuggingFace model')
    hf_create_parser.add_argument('--output-dir', default='./hf_models', help='Output directory')
    hf_create_parser.add_argument('--base-model', help='Base sentence transformer (auto-detected if not specified)')
    hf_create_parser.add_argument('--fusion', choices=['concat', 'attention', 'gated', 'weighted_avg'],
                                  default='concat', help='Fusion method')
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
        description='Process multiple OWL files to create HuggingFace models in batch'
    )

    hf_batch_parser.add_argument('input_dir', help='Directory containing OWL files')
    hf_batch_parser.add_argument('output_dir', help='Output directory for results')
    hf_batch_parser.add_argument('--base-models', nargs='+', default=['all-MiniLM-L6-v2'],
                                 help='Base models to test')
    hf_batch_parser.add_argument('--fusion-methods', nargs='+', default=['concat'],
                                 help='Fusion methods to test')
    hf_batch_parser.add_argument('--max-workers', type=int, default=2, help='Parallel workers')


def setup_benchmark_parser(subparsers):
    """Set up the MTEB benchmark command parser."""
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run MTEB benchmarks on models',
        description='Evaluate sentence-transformers models against MTEB benchmark tasks'
    )

    benchmark_parser.add_argument('model_path', help='Path to model or model name')
    benchmark_parser.add_argument('--output-dir', default='./mteb_results', help='Results output directory')
    benchmark_parser.add_argument('--model-name', help='Model name for results')
    benchmark_parser.add_argument('--tasks', nargs='+', help='Specific tasks to run')
    benchmark_parser.add_argument('--task-types', nargs='+', choices=[
        'Classification', 'Clustering', 'PairClassification', 'Reranking',
        'Retrieval', 'STS', 'Summarization'
    ], help='Task types to run')
    benchmark_parser.add_argument('--quick', action='store_true', help='Run quick subset of tasks')
    benchmark_parser.add_argument('--batch-size', type=int, default=32, help='Evaluation batch size')


def setup_compare_parser(subparsers):
    """Set up the model comparison parser."""
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare ontology-augmented vs vanilla models',
        description='Compare performance of ontology models against vanilla sentence transformers'
    )

    compare_parser.add_argument('model_path', help='Path to ontology-augmented model')
    compare_parser.add_argument('--vanilla-model', default='all-MiniLM-L6-v2', help='Vanilla model for comparison')
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

    try:
        result = train_model_only(
            owl_file=args.ontology,
            model_output=args.output,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            epochs=args.epochs,
            loss_fn=args.loss_fn,
            use_multi_relation=args.use_multi_relation,
            use_text_features=args.use_text_features,
            text_model_name=args.text_model or 'all-MiniLM-L6-v2'
        )
        print(f"✅ Training completed: {result['model_path']}")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


def run_embed_command(args):
    """Execute the embed command."""
    from .workflows import embed_with_trained_model

    try:
        result = embed_with_trained_model(
            model_path=args.model,
            owl_file=args.ontology,
            output_file=args.output
        )
        print(f"✅ Embeddings generated: {result['output_file']} ({result['num_embeddings']:,} embeddings)")
        return 0
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return 1


def run_visualize_command(args):
    """Execute the visualize command."""
    # For now, still use the script since it's not moved to module yet
    from viz import main as viz_main

    # Convert args to format expected by viz.py
    sys.argv = ['viz.py', args.embeddings]
    if args.output:
        sys.argv.extend(['--output', args.output])
    if args.neighbors:
        sys.argv.extend(['--neighbors', str(args.neighbors)])
    if args.min_dist:
        sys.argv.extend(['--min_dist', str(args.min_dist)])

    return viz_main()


def run_hf_command(args):
    """Execute the HuggingFace end-to-end command."""
    from .huggingface_workflows import end_to_end_workflow

    # Build training configuration
    training_config = {
        'epochs': args.epochs,
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'out_dim': args.out_dim,
        'loss_fn': args.loss_fn,
        'use_multi_relation': args.use_multi_relation,
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

    try:
        success = end_to_end_workflow(
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
        return 0 if success else 1
    except Exception as e:
        print(f"❌ HuggingFace workflow failed: {e}")
        return 1


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
            out_dim=args.out_dim
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
    from batch_hf_models import batch_process_ontologies

    return batch_process_ontologies(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        base_models=args.base_models,
        fusion_methods=args.fusion_methods,
        max_workers=args.max_workers
    )


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


def run_compare_command(args):
    """Execute the compare command."""
    from test_edam_model import main as compare_main

    # For now, use the existing comparison script
    # Could be enhanced to be more generic
    sys.argv = ['test_edam_model.py']
    if args.detailed:
        sys.argv.append('--detailed')

    return compare_main()


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

    # If no command provided, show help
    if not args or args[0] in ['-h', '--help']:
        parser.print_help()
        return 0

    parsed_args = parser.parse_args(args)

    # If no subcommand was selected, show help
    if not hasattr(parsed_args, 'command') or parsed_args.command is None:
        parser.print_help()
        return 0

    # Route to appropriate command handler
    command_map = {
        'train': run_train_command,
        'embed': run_embed_command,
        'visualize': run_visualize_command,
        'hf': run_hf_command,
        'hf-train': run_hf_train_command,
        'hf-create': run_hf_create_command,
        'hf-test': run_hf_test_command,
        'hf-batch': run_hf_batch_command,
        'benchmark': run_benchmark_command,
        'compare': run_compare_command,
        'inspect': run_inspect_command,
        'convert': run_convert_command,
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