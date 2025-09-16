#!/usr/bin/env python3
"""
Command-line interface for evaluating ontology embeddings.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from on2vec.evaluation import (
    EmbeddingEvaluator,
    evaluate_embeddings,
    create_evaluation_benchmark
)

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def evaluate_single_embedding(args):
    """Evaluate a single embedding file."""
    setup_logging(args.verbose)

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

        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        results = evaluator.create_evaluation_report(str(report_path) if report_path else None)

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

            if 'clustering' in intrinsic:
                clustering = intrinsic['clustering']
                if 'kmeans' in clustering:
                    kmeans = clustering['kmeans']
                    best_silhouette = max([
                        result['silhouette_score']
                        for result in kmeans.values()
                        if 'silhouette_score' in result
                    ])
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
                    if 'degree' in centrality and 'pearson_correlation' in centrality['degree']:
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

def benchmark_embeddings(args):
    """Benchmark multiple embedding files."""
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Running benchmark evaluation")

    # Parse embedding files
    embeddings_files = []
    if args.embeddings_list:
        with open(args.embeddings_list, 'r') as f:
            embeddings_files = [line.strip() for line in f if line.strip()]
    else:
        embeddings_files = args.embeddings

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
        print("BENCHMARK SUMMARY")
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

        print(f"\nBenchmark results saved to: {args.output_dir}")
        print("="*60)

        logger.info("Benchmark completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ontology embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single embedding file
  python evaluate_embeddings.py single embeddings.parquet --ontology ontology.owl

  # Evaluate with output directory for visualizations
  python evaluate_embeddings.py single embeddings.parquet --output-dir evaluation_results/

  # Benchmark multiple embedding files
  python evaluate_embeddings.py benchmark emb1.parquet emb2.parquet emb3.parquet --output-dir benchmark_results/

  # Benchmark from file list
  python evaluate_embeddings.py benchmark --embeddings-list embedding_files.txt --output-dir benchmark_results/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single evaluation command
    single_parser = subparsers.add_parser(
        'single',
        help='Evaluate a single embedding file'
    )
    single_parser.add_argument(
        'embeddings',
        help='Path to embeddings parquet file'
    )
    single_parser.add_argument(
        '--ontology',
        help='Path to original OWL ontology file (optional)'
    )
    single_parser.add_argument(
        '--output-dir',
        help='Directory to save evaluation results and visualizations'
    )
    single_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    single_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Benchmark multiple embedding files'
    )
    benchmark_parser.add_argument(
        'embeddings',
        nargs='*',
        help='Paths to embedding parquet files'
    )
    benchmark_parser.add_argument(
        '--embeddings-list',
        help='File containing list of embedding file paths (one per line)'
    )
    benchmark_parser.add_argument(
        '--ontology',
        nargs='*',
        help='Paths to corresponding OWL ontology files (optional)'
    )
    benchmark_parser.add_argument(
        '--ontology-list',
        help='File containing list of ontology file paths (one per line)'
    )
    benchmark_parser.add_argument(
        '--output-dir',
        default='benchmark_results',
        help='Directory to save benchmark results (default: benchmark_results)'
    )
    benchmark_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run appropriate command
    if args.command == 'single':
        return evaluate_single_embedding(args)
    elif args.command == 'benchmark':
        return benchmark_embeddings(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())