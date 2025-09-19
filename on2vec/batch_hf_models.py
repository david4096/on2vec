#!/usr/bin/env python3
"""
Batch processing CLI for creating multiple HuggingFace models from ontology directories.

This tool allows you to:
1. Process entire directories of OWL files
2. Create multiple HuggingFace models with different configurations
3. Run comparative evaluations across models
4. Generate model collections for specific domains
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import from on2vec module
from .huggingface_workflows import train_ontology_with_text, create_hf_model, validate_hf_model as test_model
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_owl_files(directory: str, pattern: str = "*.owl") -> List[Path]:
    """Find all OWL files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    owl_files = list(directory.glob(pattern))
    if not owl_files:
        # Try recursive search
        owl_files = list(directory.rglob(pattern))

    logger.info(f"Found {len(owl_files)} OWL files in {directory}")
    return sorted(owl_files)


def generate_model_configs(
    base_models: List[str],
    fusion_methods: List[str],
    epochs_list: List[int],
    training_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generate all combinations of model configurations."""
    configs = []

    # Extract training parameters with defaults
    model_types = training_config.get('model_types', ['gcn']) if training_config else ['gcn']
    hidden_dims = training_config.get('hidden_dims', [128]) if training_config else [128]
    out_dims = training_config.get('out_dims', [64]) if training_config else [64]
    loss_fns = training_config.get('loss_fns', ['triplet']) if training_config else ['triplet']
    use_multi_relation = training_config.get('use_multi_relation', False) if training_config else False
    text_model = training_config.get('text_model') if training_config else None

    for base_model in base_models:
        for fusion in fusion_methods:
            for epochs in epochs_list:
                for model_type in model_types:
                    for hidden_dim in hidden_dims:
                        for out_dim in out_dims:
                            for loss_fn in loss_fns:
                                # Use text_model if specified, otherwise use base_model
                                actual_text_model = text_model or base_model

                                config = {
                                    'base_model': base_model,
                                    'fusion_method': fusion,
                                    'epochs': epochs,
                                    'model_type': model_type,
                                    'hidden_dim': hidden_dim,
                                    'out_dim': out_dim,
                                    'loss_fn': loss_fn,
                                    'use_multi_relation': use_multi_relation,
                                    'text_model': actual_text_model,
                                    'config_id': f"{base_model.split('/')[-1]}_{fusion}_{model_type}_h{hidden_dim}_o{out_dim}_{loss_fn}_e{epochs}"
                                }
                                configs.append(config)
    return configs


def process_single_ontology(
    owl_file: Path,
    config: Dict[str, Any],
    output_base_dir: Path,
    force_retrain: bool = False,
    model_details: Optional[Dict[str, Any]] = None,
    upload_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a single ontology with given configuration."""
    ontology_name = owl_file.stem
    config_id = config['config_id']

    logger.info(f"Processing {ontology_name} with config {config_id}")

    # Create directories
    ontology_dir = output_base_dir / ontology_name
    embeddings_file = ontology_dir / f"{ontology_name}_{config_id}_embeddings.parquet"
    model_dir = ontology_dir / f"{ontology_name}_{config_id}_model"

    ontology_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'ontology': str(owl_file),
        'ontology_name': ontology_name,
        'config': config,
        'embeddings_file': str(embeddings_file),
        'model_dir': str(model_dir),
        'success': False,
        'error': None,
        'metrics': {}
    }

    try:
        start_time = time.time()

        # Step 1: Train embeddings (if needed)
        if force_retrain or not embeddings_file.exists():
            logger.info(f"Training embeddings for {ontology_name}")
            if not train_ontology_with_text(
                owl_file=str(owl_file),
                output_file=str(embeddings_file),
                text_model=config['text_model'],
                epochs=config['epochs'],
                model_type=config['model_type'],
                hidden_dim=config['hidden_dim'],
                out_dim=config['out_dim'],
                loss_fn=config['loss_fn'],
                use_multi_relation=config['use_multi_relation']
            ):
                raise Exception("Training failed")
        else:
            logger.info(f"Using existing embeddings: {embeddings_file}")

        # Step 2: Create HF model
        logger.info(f"Creating HF model for {ontology_name}")

        # Process upload options template if present
        processed_upload_options = None
        if upload_options:
            processed_upload_options = upload_options.copy()
            if 'hub_name_template' in upload_options:
                # Replace template placeholders with actual values
                hub_name = upload_options['hub_name_template'].format(
                    ontology_name=ontology_name,
                    config_id=config_id
                )
                processed_upload_options['hub_name'] = hub_name
                # Remove the template key since we now have the actual hub_name
                processed_upload_options.pop('hub_name_template', None)

        actual_model_dir = create_hf_model(
            embeddings_file=str(embeddings_file),
            model_name=f"{ontology_name}_{config_id}",
            output_dir=str(ontology_dir),
            base_model=config['base_model'],
            fusion_method=config['fusion_method'],
            ontology_file=str(owl_file),
            model_details=model_details,
            upload_options=processed_upload_options
        )

        # Step 3: Test model
        logger.info(f"Testing model for {ontology_name}")
        test_success = test_model(actual_model_dir)

        # Collect metrics
        result['metrics'] = {
            'processing_time': time.time() - start_time,
            'embeddings_size': embeddings_file.stat().st_size if embeddings_file.exists() else 0,
            'model_size': sum(f.stat().st_size for f in Path(actual_model_dir).rglob('*') if f.is_file()),
            'test_passed': test_success
        }

        result['model_dir'] = actual_model_dir
        result['success'] = True
        logger.info(f"✅ Completed {ontology_name} with config {config_id}")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Failed {ontology_name} with config {config_id}: {e}")

    return result


def batch_process_ontologies(
    owl_directory: str,
    output_directory: str,
    base_models: List[str],
    fusion_methods: List[str],
    epochs_list: List[int],
    max_workers: int = 2,
    force_retrain: bool = False,
    owl_pattern: str = "*.owl",
    limit: Optional[int] = None,
    training_config: Optional[Dict[str, Any]] = None,
    model_details: Optional[Dict[str, Any]] = None,
    upload_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process multiple ontologies with multiple configurations."""
    logger.info(f"Starting batch processing")
    logger.info(f"Input directory: {owl_directory}")
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Base models: {base_models}")
    logger.info(f"Fusion methods: {fusion_methods}")
    logger.info(f"Epochs: {epochs_list}")
    logger.info(f"Max workers: {max_workers}")

    # Find OWL files
    owl_files = find_owl_files(owl_directory, owl_pattern)
    if limit:
        owl_files = owl_files[:limit]
        logger.info(f"Limited to first {limit} files")

    # Generate configurations
    configs = generate_model_configs(base_models, fusion_methods, epochs_list, training_config)
    logger.info(f"Generated {len(configs)} configurations")

    # Create output directory
    output_base_dir = Path(output_directory)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Generate all tasks
    tasks = []
    for owl_file in owl_files:
        for config in configs:
            tasks.append((owl_file, config))

    total_tasks = len(tasks)
    logger.info(f"Total tasks: {total_tasks}")

    # Process tasks
    results = []
    completed = 0

    if max_workers == 1:
        # Sequential processing
        for owl_file, config in tasks:
            result = process_single_ontology(owl_file, config, output_base_dir, force_retrain, model_details, upload_options)
            results.append(result)
            completed += 1
            logger.info(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_ontology, owl_file, config, output_base_dir, force_retrain, model_details, upload_options): (owl_file, config)
                for owl_file, config in tasks
            }

            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                logger.info(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")

    # Compile summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    summary = {
        'total_tasks': total_tasks,
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / total_tasks if total_tasks > 0 else 0,
        'configs_tested': len(configs),
        'ontologies_processed': len(owl_files),
        'results': results,
        'summary_stats': compile_summary_stats(successful)
    }

    # Save results
    results_file = output_base_dir / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch processing completed!")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Results saved to: {results_file}")

    return summary


def compile_summary_stats(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile summary statistics from successful results."""
    if not successful_results:
        return {}

    processing_times = [r['metrics']['processing_time'] for r in successful_results]
    model_sizes = [r['metrics']['model_size'] for r in successful_results]
    test_passes = [r['metrics']['test_passed'] for r in successful_results]

    return {
        'avg_processing_time': sum(processing_times) / len(processing_times),
        'total_processing_time': sum(processing_times),
        'avg_model_size_mb': sum(model_sizes) / len(model_sizes) / (1024*1024),
        'total_model_size_gb': sum(model_sizes) / (1024*1024*1024),
        'test_pass_rate': sum(test_passes) / len(test_passes) if test_passes else 0,
        'config_performance': compile_config_performance(successful_results)
    }


def compile_config_performance(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compile performance stats by configuration."""
    config_stats = {}

    for result in results:
        config_id = result['config']['config_id']
        if config_id not in config_stats:
            config_stats[config_id] = {
                'count': 0,
                'avg_time': 0,
                'avg_size': 0,
                'test_pass_rate': 0
            }

        stats = config_stats[config_id]
        stats['count'] += 1
        stats['avg_time'] += result['metrics']['processing_time']
        stats['avg_size'] += result['metrics']['model_size']
        stats['test_pass_rate'] += int(result['metrics']['test_passed'])

    # Compute averages
    for config_id, stats in config_stats.items():
        count = stats['count']
        stats['avg_time'] /= count
        stats['avg_size'] /= count * (1024*1024)  # Convert to MB
        stats['test_pass_rate'] /= count

    return config_stats


def print_summary_report(summary: Dict[str, Any]):
    """Print a formatted summary report."""
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)

    print(f"📋 Total Tasks: {summary['total_tasks']}")
    print(f"✅ Successful: {summary['successful']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"🎯 Success Rate: {summary['success_rate']:.1%}")

    if 'summary_stats' in summary and summary['summary_stats']:
        stats = summary['summary_stats']
        print(f"\n⏱️ Processing Time:")
        print(f"   Average: {stats['avg_processing_time']:.1f}s")
        print(f"   Total: {stats['total_processing_time']:.1f}s")

        print(f"\n💾 Model Sizes:")
        print(f"   Average: {stats['avg_model_size_mb']:.1f} MB")
        print(f"   Total: {stats['total_model_size_gb']:.2f} GB")

        print(f"\n🧪 Test Pass Rate: {stats['test_pass_rate']:.1%}")

        if 'config_performance' in stats:
            print(f"\n📈 Configuration Performance:")
            for config_id, perf in stats['config_performance'].items():
                print(f"   {config_id}:")
                print(f"     Count: {perf['count']}, Time: {perf['avg_time']:.1f}s, Size: {perf['avg_size']:.1f}MB, Pass: {perf['test_pass_rate']:.1%}")

    # Show failed tasks
    failed_results = [r for r in summary['results'] if not r['success']]
    if failed_results:
        print(f"\n❌ Failed Tasks:")
        for result in failed_results[:5]:  # Show first 5
            print(f"   {result['ontology_name']} ({result['config']['config_id']}): {result['error']}")
        if len(failed_results) > 5:
            print(f"   ... and {len(failed_results) - 5} more")

    print("\n" + "="*60)


def create_model_collection(
    summary: Dict[str, Any],
    collection_name: str,
    output_dir: str,
    selection_criteria: str = "best_test"
) -> str:
    """Create a curated collection of the best models."""
    logger.info(f"Creating model collection: {collection_name}")

    successful_results = [r for r in summary['results'] if r['success']]

    if selection_criteria == "best_test":
        # Select models that passed tests and have good performance
        selected = [r for r in successful_results if r['metrics']['test_passed']]
    elif selection_criteria == "fastest":
        # Select fastest processing models
        selected = sorted(successful_results, key=lambda x: x['metrics']['processing_time'])[:10]
    elif selection_criteria == "smallest":
        # Select smallest models
        selected = sorted(successful_results, key=lambda x: x['metrics']['model_size'])[:10]
    else:
        selected = successful_results

    collection_dir = Path(output_dir) / f"{collection_name}_collection"
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Create collection metadata
    collection_info = {
        'name': collection_name,
        'description': f"Curated collection of ontology-augmented models (selection: {selection_criteria})",
        'models': [],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'selection_criteria': selection_criteria,
        'total_models': len(selected)
    }

    # Copy or link selected models
    for result in selected:
        model_name = f"{result['ontology_name']}_{result['config']['config_id']}"

        model_info = {
            'name': model_name,
            'ontology': result['ontology_name'],
            'config': result['config'],
            'metrics': result['metrics'],
            'path': result['model_dir']
        }
        collection_info['models'].append(model_info)

    # Save collection metadata
    collection_file = collection_dir / "collection.json"
    with open(collection_file, 'w') as f:
        json.dump(collection_info, f, indent=2)

    # Create README
    readme_content = f"""# {collection_name} Model Collection

This collection contains {len(selected)} ontology-augmented Sentence Transformer models.

## Selection Criteria
{selection_criteria}

## Models Included

"""
    for model in collection_info['models']:
        readme_content += f"### {model['name']}\n"
        readme_content += f"- **Ontology**: {model['ontology']}\n"
        readme_content += f"- **Base Model**: {model['config']['base_model']}\n"
        readme_content += f"- **Fusion**: {model['config']['fusion_method']}\n"
        readme_content += f"- **Size**: {model['metrics']['model_size'] / (1024*1024):.1f} MB\n"
        readme_content += f"- **Test Passed**: {'✅' if model['metrics']['test_passed'] else '❌'}\n\n"

    readme_content += f"""
## Usage

```python
from sentence_transformers import SentenceTransformer

# Load any model from the collection
model = SentenceTransformer('path/to/model')
embeddings = model.encode(['your', 'sentences'])
```

Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_file = collection_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    logger.info(f"Collection created: {collection_dir}")
    return str(collection_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process ontologies to create multiple HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all OWL files with default settings
  python batch_hf_models.py process owl_files/ ./batch_output

  # Custom configuration with multiple base models
  python batch_hf_models.py process owl_files/ ./output \\
    --base-models all-MiniLM-L6-v2 all-mpnet-base-v2 \\
    --fusion-methods concat gated \\
    --epochs 50 100

  # Limited parallel processing
  python batch_hf_models.py process owl_files/ ./output \\
    --max-workers 4 \\
    --limit 10

  # Create model collection from results
  python batch_hf_models.py collection ./output/batch_results.json \\
    --name "biomedical-models" \\
    --criteria best_test
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Batch processing command
    process_parser = subparsers.add_parser('process', help='Process directory of OWL files')
    process_parser.add_argument('owl_directory', help='Directory containing OWL files')
    process_parser.add_argument('output_directory', help='Output directory for models')
    process_parser.add_argument('--base-models', nargs='+',
                               default=['all-MiniLM-L6-v2'],
                               help='Base Sentence Transformer models')
    process_parser.add_argument('--fusion-methods', nargs='+',
                               choices=['concat', 'weighted_avg', 'attention', 'gated'],
                               default=['concat'],
                               help='Fusion methods to test')
    process_parser.add_argument('--epochs', nargs='+', type=int,
                               default=[100],
                               help='Training epochs to test')
    process_parser.add_argument('--max-workers', type=int, default=2,
                               help='Maximum parallel workers')
    process_parser.add_argument('--force-retrain', action='store_true',
                               help='Force retraining even if embeddings exist')
    process_parser.add_argument('--owl-pattern', default='*.owl',
                               help='Pattern for finding OWL files')
    process_parser.add_argument('--limit', type=int,
                               help='Limit number of OWL files to process')

    # Collection creation command
    collection_parser = subparsers.add_parser('collection', help='Create model collection')
    collection_parser.add_argument('results_file', help='Path to batch_results.json file')
    collection_parser.add_argument('--name', required=True, help='Collection name')
    collection_parser.add_argument('--criteria', choices=['best_test', 'fastest', 'smallest'],
                                  default='best_test', help='Selection criteria')
    collection_parser.add_argument('--output-dir', help='Output directory (defaults to results directory)')

    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show summary from results file')
    summary_parser.add_argument('results_file', help='Path to batch_results.json file')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == 'process':
            summary = batch_process_ontologies(
                owl_directory=args.owl_directory,
                output_directory=args.output_directory,
                base_models=args.base_models,
                fusion_methods=args.fusion_methods,
                epochs_list=args.epochs,
                max_workers=args.max_workers,
                force_retrain=args.force_retrain,
                owl_pattern=args.owl_pattern,
                limit=args.limit
            )
            print_summary_report(summary)

        elif args.command == 'collection':
            with open(args.results_file, 'r') as f:
                summary = json.load(f)

            output_dir = args.output_dir or Path(args.results_file).parent
            collection_path = create_model_collection(
                summary=summary,
                collection_name=args.name,
                output_dir=output_dir,
                selection_criteria=args.criteria
            )
            print(f"✅ Collection created: {collection_path}")

        elif args.command == 'summary':
            with open(args.results_file, 'r') as f:
                summary = json.load(f)
            print_summary_report(summary)

        return 0

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
