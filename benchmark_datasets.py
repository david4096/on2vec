#!/usr/bin/env python3
"""
Command-line interface for managing benchmark datasets and baseline comparisons.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from on2vec.benchmarks import (
    OntologyBenchmarkDatasets,
    BaselineComparison,
    compare_with_baselines
)

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def list_datasets(args):
    """List available benchmark datasets."""
    setup_logging(args.verbose)

    datasets = OntologyBenchmarkDatasets()
    available = datasets.list_available_datasets(domain=args.domain, size=args.size)

    print("\n" + "="*60)
    print("AVAILABLE BENCHMARK DATASETS")
    print("="*60)

    if not available:
        print("No datasets match the specified criteria.")
        return 0

    for key, info in available.items():
        print(f"\n{key.upper()}: {info['name']}")
        print(f"  Domain: {info['domain']}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
        print(f"  Relations: {', '.join(info['relations'])}")
        print(f"  URL: {info['url']}")

    print("="*60)
    return 0

def download_datasets(args):
    """Download benchmark datasets."""
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    datasets = OntologyBenchmarkDatasets(cache_dir=args.cache_dir)

    if args.all:
        # Download all datasets
        logger.info("Downloading all available datasets...")
        downloaded = datasets.download_benchmark_suite(
            domains=args.domains,
            sizes=args.sizes,
            force_redownload=args.force
        )
    elif args.datasets:
        # Download specific datasets
        downloaded = {}
        for dataset_key in args.datasets:
            try:
                path = datasets.download_dataset(dataset_key, force_redownload=args.force)
                downloaded[dataset_key] = path
            except Exception as e:
                logger.error(f"Failed to download {dataset_key}: {e}")
    else:
        logger.error("No datasets specified. Use --all or provide dataset names.")
        return 1

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    if downloaded:
        print(f"Successfully downloaded {len(downloaded)} datasets:")
        for key, path in downloaded.items():
            print(f"  {key}: {path}")

        print(f"\nCache directory: {datasets.cache_dir}")
    else:
        print("No datasets were downloaded.")

    print("="*60)
    return 0

def create_baselines(args):
    """Create baseline embeddings."""
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    baseline_comp = BaselineComparison()

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    baseline_files = {}

    for method in args.methods:
        logger.info(f"Creating {method} baseline for {args.ontology}")

        try:
            baseline_file = baseline_comp.create_baseline_embeddings(
                method,
                args.ontology,
                args.embedding_dim,
                **args.method_kwargs if hasattr(args, 'method_kwargs') else {}
            )

            # Move to output directory if specified
            if args.output_dir:
                output_path = Path(args.output_dir) / f"{method}_baseline_embeddings.parquet"
                Path(baseline_file).rename(output_path)
                baseline_files[method] = str(output_path)
            else:
                baseline_files[method] = baseline_file

        except Exception as e:
            logger.error(f"Failed to create {method} baseline: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("BASELINE CREATION SUMMARY")
    print("="*60)

    if baseline_files:
        print(f"Successfully created {len(baseline_files)} baseline embeddings:")
        for method, path in baseline_files.items():
            print(f"  {method}: {path}")
    else:
        print("No baseline embeddings were created.")

    print("="*60)
    return 0

def compare_baselines(args):
    """Compare embeddings against baselines."""
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info(f"Comparing {args.embeddings} against baselines")

    try:
        results = compare_with_baselines(
            args.embeddings,
            args.ontology,
            args.methods,
            args.embedding_dim,
            args.output_dir
        )

        # Print summary
        print("\n" + "="*60)
        print("BASELINE COMPARISON SUMMARY")
        print("="*60)

        print(f"Original embeddings: {args.embeddings}")
        print(f"Ontology: {args.ontology}")
        print(f"Compared against: {', '.join(args.methods)}")

        # Show key metrics if available
        original_eval = results['original']['evaluation']
        if 'intrinsic_evaluation' in original_eval and 'clustering' in original_eval['intrinsic_evaluation']:
            clustering = original_eval['intrinsic_evaluation']['clustering']
            if 'kmeans' in clustering:
                kmeans = clustering['kmeans']
                best_silhouette = max([
                    result['silhouette_score']
                    for result in kmeans.values()
                    if 'silhouette_score' in result
                ])
                print(f"Original best silhouette score: {best_silhouette:.3f}")

        for method, baseline_data in results['baselines'].items():
            if 'error' not in baseline_data:
                eval_results = baseline_data['evaluation']
                if 'intrinsic_evaluation' in eval_results and 'clustering' in eval_results['intrinsic_evaluation']:
                    clustering = eval_results['intrinsic_evaluation']['clustering']
                    if 'kmeans' in clustering:
                        kmeans = clustering['kmeans']
                        best_silhouette = max([
                            result['silhouette_score']
                            for result in kmeans.values()
                            if 'silhouette_score' in result
                        ])
                        print(f"{method} best silhouette score: {best_silhouette:.3f}")

        print(f"\nDetailed results saved to: {args.output_dir}")
        print("="*60)

        return 0

    except Exception as e:
        logger.error(f"Baseline comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def setup_cross_domain_evaluation(args):
    """Set up cross-domain evaluation datasets."""
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    datasets = OntologyBenchmarkDatasets(cache_dir=args.cache_dir)

    try:
        cross_domain_sets = datasets.create_cross_domain_evaluation_set(
            args.train_domains,
            args.test_domains
        )

        # Print summary
        print("\n" + "="*60)
        print("CROSS-DOMAIN EVALUATION SETUP")
        print("="*60)

        print(f"Training domains: {', '.join(args.train_domains)}")
        print(f"Test domains: {', '.join(args.test_domains)}")

        print(f"\nTraining datasets ({len(cross_domain_sets['train'])}):")
        for key, path in cross_domain_sets['train'].items():
            print(f"  {key}: {path}")

        print(f"\nTest datasets ({len(cross_domain_sets['test'])}):")
        for key, path in cross_domain_sets['test'].items():
            print(f"  {key}: {path}")

        print("="*60)

        return 0

    except Exception as e:
        logger.error(f"Failed to setup cross-domain evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage benchmark datasets and baseline comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python benchmark_datasets.py list

  # Download specific datasets
  python benchmark_datasets.py download --datasets go hp edam

  # Download all small datasets
  python benchmark_datasets.py download --all --sizes small

  # Create baseline embeddings
  python benchmark_datasets.py create-baselines ontology.owl --methods random structural --embedding-dim 64

  # Compare against baselines
  python benchmark_datasets.py compare embeddings.parquet ontology.owl --methods random structural text_only

  # Setup cross-domain evaluation
  python benchmark_datasets.py cross-domain --train-domains biology chemistry --test-domains medicine
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List datasets command
    list_parser = subparsers.add_parser('list', help='List available benchmark datasets')
    list_parser.add_argument('--domain', help='Filter by domain')
    list_parser.add_argument('--size', help='Filter by size (small, medium, large, very_large)')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Download datasets command
    download_parser = subparsers.add_parser('download', help='Download benchmark datasets')
    download_parser.add_argument('--datasets', nargs='+', help='Specific dataset names to download')
    download_parser.add_argument('--all', action='store_true', help='Download all available datasets')
    download_parser.add_argument('--domains', nargs='+', help='Filter by domains')
    download_parser.add_argument('--sizes', nargs='+', help='Filter by sizes')
    download_parser.add_argument('--cache-dir', help='Cache directory for downloaded datasets')
    download_parser.add_argument('--force', action='store_true', help='Force redownload existing files')
    download_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Create baselines command
    baseline_parser = subparsers.add_parser('create-baselines', help='Create baseline embeddings')
    baseline_parser.add_argument('ontology', help='Path to ontology file')
    baseline_parser.add_argument('--methods', nargs='+',
                                choices=['random', 'structural', 'text_only', 'node2vec', 'deepwalk'],
                                default=['random', 'structural'],
                                help='Baseline methods to create')
    baseline_parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    baseline_parser.add_argument('--output-dir', help='Directory to save baseline embeddings')
    baseline_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Compare baselines command
    compare_parser = subparsers.add_parser('compare', help='Compare embeddings against baselines')
    compare_parser.add_argument('embeddings', help='Path to embeddings file')
    compare_parser.add_argument('ontology', help='Path to ontology file')
    compare_parser.add_argument('--methods', nargs='+',
                               choices=['random', 'structural', 'text_only', 'node2vec', 'deepwalk'],
                               default=['random', 'structural'],
                               help='Baseline methods to compare against')
    compare_parser.add_argument('--embedding-dim', type=int, help='Embedding dimension (inferred if not provided)')
    compare_parser.add_argument('--output-dir', default='baseline_comparison', help='Output directory')
    compare_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Cross-domain evaluation command
    cross_parser = subparsers.add_parser('cross-domain', help='Setup cross-domain evaluation')
    cross_parser.add_argument('--train-domains', nargs='+', required=True,
                             help='Domains for training (biology, chemistry, medicine, etc.)')
    cross_parser.add_argument('--test-domains', nargs='+', required=True,
                             help='Domains for testing')
    cross_parser.add_argument('--cache-dir', help='Cache directory for datasets')
    cross_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run appropriate command
    if args.command == 'list':
        return list_datasets(args)
    elif args.command == 'download':
        return download_datasets(args)
    elif args.command == 'create-baselines':
        return create_baselines(args)
    elif args.command == 'compare':
        return compare_baselines(args)
    elif args.command == 'cross-domain':
        return setup_cross_domain_evaluation(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())