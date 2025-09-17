#!/usr/bin/env python3
"""
Cross-domain evaluation framework for testing ontology embedding generalization.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from on2vec.benchmarks import OntologyBenchmarkDatasets, BaselineComparison
from on2vec.evaluation import EmbeddingEvaluator
from on2vec import train_ontology_embeddings, embed_ontology_with_model

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class CrossDomainEvaluator:
    """
    Framework for evaluating embedding generalization across domains.
    """

    def __init__(self, output_dir: str = "cross_domain_evaluation"):
        """
        Initialize cross-domain evaluator.

        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Setup benchmark datasets
        self.datasets = OntologyBenchmarkDatasets(
            cache_dir=str(self.output_dir / 'benchmark_cache')
        )

        # Setup baseline comparison
        self.baseline_comp = BaselineComparison()

    def run_cross_domain_experiment(self,
                                  train_domains: List[str],
                                  test_domains: List[str],
                                  model_configs: List[Dict[str, Any]],
                                  baseline_methods: List[str] = ['random', 'structural'],
                                  embedding_dim: int = 64) -> Dict[str, Any]:
        """
        Run comprehensive cross-domain evaluation experiment.

        Args:
            train_domains (list): Domains for training
            test_domains (list): Domains for testing
            model_configs (list): List of model configurations to test
            baseline_methods (list): Baseline methods to compare against
            embedding_dim (int): Embedding dimension

        Returns:
            dict: Complete experiment results
        """
        self.logger.info("Starting cross-domain evaluation experiment")

        # Download datasets
        cross_domain_sets = self.datasets.create_cross_domain_evaluation_set(
            train_domains, test_domains
        )

        train_datasets = cross_domain_sets['train']
        test_datasets = cross_domain_sets['test']

        if not train_datasets:
            raise ValueError(f"No training datasets found for domains: {train_domains}")
        if not test_datasets:
            raise ValueError(f"No test datasets found for domains: {test_domains}")

        experiment_results = {
            'experiment_config': {
                'train_domains': train_domains,
                'test_domains': test_domains,
                'model_configs': model_configs,
                'baseline_methods': baseline_methods,
                'embedding_dim': embedding_dim
            },
            'datasets': {
                'train': train_datasets,
                'test': test_datasets
            },
            'results': {}
        }

        # For each model configuration
        for config_idx, model_config in enumerate(model_configs):
            config_name = f"config_{config_idx:02d}_{model_config.get('model_type', 'unknown')}"
            self.logger.info(f"Evaluating configuration: {config_name}")

            config_results = {
                'config': model_config,
                'train_results': {},
                'cross_domain_results': {},
                'baseline_comparisons': {}
            }

            # Train models on training datasets
            trained_models = {}
            for train_dataset_key, train_dataset_path in train_datasets.items():
                self.logger.info(f"Training on {train_dataset_key}")

                try:
                    # Train model
                    model_output = self.output_dir / f"models/{config_name}_{train_dataset_key}_model.pt"
                    model_output.parent.mkdir(parents=True, exist_ok=True)

                    train_result = train_ontology_embeddings(
                        owl_file=train_dataset_path,
                        model_output=str(model_output),
                        out_dim=embedding_dim,
                        **model_config
                    )

                    # Generate embeddings for training dataset
                    train_embeddings_path = self.output_dir / f"embeddings/{config_name}_{train_dataset_key}_train.parquet"
                    train_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

                    embed_result = embed_ontology_with_model(
                        model_path=str(model_output),
                        owl_file=train_dataset_path,
                        output_file=str(train_embeddings_path)
                    )

                    # Evaluate training performance
                    train_evaluator = EmbeddingEvaluator(
                        str(train_embeddings_path), train_dataset_path
                    )
                    train_evaluation = train_evaluator.create_evaluation_report()

                    config_results['train_results'][train_dataset_key] = {
                        'model_path': str(model_output),
                        'embeddings_path': str(train_embeddings_path),
                        'evaluation': train_evaluation
                    }

                    trained_models[train_dataset_key] = str(model_output)

                except Exception as e:
                    self.logger.error(f"Failed to train on {train_dataset_key}: {e}")
                    config_results['train_results'][train_dataset_key] = {
                        'error': str(e)
                    }

            # Test cross-domain generalization
            for test_dataset_key, test_dataset_path in test_datasets.items():
                self.logger.info(f"Testing cross-domain generalization on {test_dataset_key}")

                test_results = {}

                # Test each trained model on this test dataset
                for train_dataset_key, model_path in trained_models.items():
                    cross_domain_key = f"{train_dataset_key}_to_{test_dataset_key}"

                    try:
                        # Generate embeddings using trained model
                        test_embeddings_path = self.output_dir / f"embeddings/{config_name}_{cross_domain_key}.parquet"
                        test_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

                        embed_result = embed_ontology_with_model(
                            model_path=model_path,
                            owl_file=test_dataset_path,
                            output_file=str(test_embeddings_path)
                        )

                        # Evaluate cross-domain performance
                        test_evaluator = EmbeddingEvaluator(
                            str(test_embeddings_path), test_dataset_path
                        )
                        test_evaluation = test_evaluator.create_evaluation_report()

                        test_results[cross_domain_key] = {
                            'trained_on': train_dataset_key,
                            'tested_on': test_dataset_key,
                            'embeddings_path': str(test_embeddings_path),
                            'evaluation': test_evaluation
                        }

                    except Exception as e:
                        self.logger.error(f"Failed cross-domain test {cross_domain_key}: {e}")
                        test_results[cross_domain_key] = {
                            'error': str(e)
                        }

                config_results['cross_domain_results'][test_dataset_key] = test_results

            # Create baseline comparisons for test datasets
            for test_dataset_key, test_dataset_path in test_datasets.items():
                self.logger.info(f"Creating baseline comparisons for {test_dataset_key}")

                baseline_results = {}

                for baseline_method in baseline_methods:
                    try:
                        baseline_embeddings = self.baseline_comp.create_baseline_embeddings(
                            baseline_method, test_dataset_path, embedding_dim
                        )

                        # Evaluate baseline
                        baseline_evaluator = EmbeddingEvaluator(
                            baseline_embeddings, test_dataset_path
                        )
                        baseline_evaluation = baseline_evaluator.create_evaluation_report()

                        baseline_results[baseline_method] = {
                            'embeddings_path': baseline_embeddings,
                            'evaluation': baseline_evaluation
                        }

                    except Exception as e:
                        self.logger.error(f"Failed to create {baseline_method} baseline for {test_dataset_key}: {e}")
                        baseline_results[baseline_method] = {
                            'error': str(e)
                        }

                config_results['baseline_comparisons'][test_dataset_key] = baseline_results

            experiment_results['results'][config_name] = config_results

        # Save experiment results
        results_file = self.output_dir / 'cross_domain_experiment_results.json'

        def convert_numpy(obj):
            """Convert numpy arrays to lists for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        experiment_serializable = convert_numpy(experiment_results)

        with open(results_file, 'w') as f:
            json.dump(experiment_serializable, f, indent=2)

        self.logger.info(f"Experiment results saved to {results_file}")
        return experiment_results

    def analyze_cross_domain_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cross-domain evaluation results and extract key insights.

        Args:
            results (dict): Experiment results from run_cross_domain_experiment

        Returns:
            dict: Analysis summary
        """
        analysis = {
            'domain_transfer_analysis': {},
            'model_comparison': {},
            'baseline_comparison': {},
            'generalization_scores': {}
        }

        # Extract key metrics from results
        for config_name, config_results in results['results'].items():
            model_analysis = {
                'within_domain_performance': {},
                'cross_domain_performance': {},
                'transfer_effectiveness': {}
            }

            # Analyze within-domain performance
            for train_dataset, train_result in config_results['train_results'].items():
                if 'evaluation' in train_result:
                    eval_data = train_result['evaluation']
                    model_analysis['within_domain_performance'][train_dataset] = self._extract_key_metrics(eval_data)

            # Analyze cross-domain performance
            for test_dataset, test_results in config_results['cross_domain_results'].items():
                test_analysis = {}

                for transfer_key, transfer_result in test_results.items():
                    if 'evaluation' in transfer_result:
                        eval_data = transfer_result['evaluation']
                        test_analysis[transfer_key] = self._extract_key_metrics(eval_data)

                model_analysis['cross_domain_performance'][test_dataset] = test_analysis

            # Calculate transfer effectiveness
            for test_dataset, test_results in config_results['cross_domain_results'].items():
                transfer_scores = []

                for transfer_key, transfer_result in test_results.items():
                    if 'evaluation' in transfer_result:
                        # Compare against baseline performance
                        baseline_comparison = config_results['baseline_comparisons'].get(test_dataset, {})
                        if 'random' in baseline_comparison and 'evaluation' in baseline_comparison['random']:
                            transfer_score = self._calculate_transfer_score(
                                transfer_result['evaluation'],
                                baseline_comparison['random']['evaluation']
                            )
                            transfer_scores.append(transfer_score)

                if transfer_scores:
                    model_analysis['transfer_effectiveness'][test_dataset] = {
                        'mean_transfer_score': np.mean(transfer_scores),
                        'std_transfer_score': np.std(transfer_scores),
                        'individual_scores': transfer_scores
                    }

            analysis['model_comparison'][config_name] = model_analysis

        return analysis

    def _extract_key_metrics(self, evaluation: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from evaluation results."""
        metrics = {}

        try:
            # Intrinsic metrics
            if 'intrinsic_evaluation' in evaluation:
                intrinsic = evaluation['intrinsic_evaluation']

                # Clustering metrics
                if 'clustering' in intrinsic and 'kmeans' in intrinsic['clustering']:
                    kmeans_results = intrinsic['clustering']['kmeans']
                    best_silhouette = max([
                        result['silhouette_score']
                        for result in kmeans_results.values()
                        if 'silhouette_score' in result
                    ])
                    metrics['best_silhouette_score'] = best_silhouette

                # Distribution metrics
                if 'distribution' in intrinsic:
                    dist = intrinsic['distribution']
                    metrics['mean_embedding_norm'] = dist['norms']['mean_norm']
                    metrics['mean_cosine_similarity'] = dist['similarities']['mean_similarity']

                # Dimensionality metrics
                if 'dimensionality' in intrinsic:
                    dim = intrinsic['dimensionality']
                    metrics['effective_dimensions_95'] = dim['effective_dimensions']['95_percent_variance']

            # Extrinsic metrics
            if 'extrinsic_evaluation' in evaluation:
                extrinsic = evaluation['extrinsic_evaluation']

                # Link prediction
                if 'link_prediction' in extrinsic and 'classifiers' in extrinsic['link_prediction']:
                    classifiers = extrinsic['link_prediction']['classifiers']
                    if 'logistic_regression' in classifiers and 'roc_auc' in classifiers['logistic_regression']:
                        metrics['link_prediction_roc_auc'] = classifiers['logistic_regression']['roc_auc']

                # Hierarchy preservation
                if 'hierarchy' in extrinsic and 'similarity_difference' in extrinsic['hierarchy']:
                    metrics['hierarchy_preservation'] = extrinsic['hierarchy']['similarity_difference']

            # Ontology-specific metrics
            if 'ontology_specific_evaluation' in evaluation:
                ont_specific = evaluation['ontology_specific_evaluation']

                # Structural consistency
                if 'structural_consistency' in ont_specific and 'centrality_correlations' in ont_specific['structural_consistency']:
                    centrality_corrs = ont_specific['structural_consistency']['centrality_correlations']
                    if 'degree' in centrality_corrs and 'pearson_correlation' in centrality_corrs['degree']:
                        metrics['degree_centrality_correlation'] = centrality_corrs['degree']['pearson_correlation']

        except Exception as e:
            logging.getLogger(__name__).warning(f"Error extracting metrics: {e}")

        return metrics

    def _calculate_transfer_score(self, transfer_eval: Dict[str, Any], baseline_eval: Dict[str, Any]) -> float:
        """Calculate transfer effectiveness score compared to baseline."""
        transfer_metrics = self._extract_key_metrics(transfer_eval)
        baseline_metrics = self._extract_key_metrics(baseline_eval)

        # Calculate improvement over baseline
        improvements = []

        for metric_name in ['best_silhouette_score', 'link_prediction_roc_auc', 'hierarchy_preservation']:
            if metric_name in transfer_metrics and metric_name in baseline_metrics:
                transfer_value = transfer_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]

                if baseline_value != 0:
                    improvement = (transfer_value - baseline_value) / abs(baseline_value)
                    improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def create_cross_domain_report(self, results: Dict[str, Any]) -> str:
        """
        Create a comprehensive cross-domain evaluation report.

        Args:
            results (dict): Experiment results

        Returns:
            str: Path to generated report
        """
        analysis = self.analyze_cross_domain_results(results)

        report_path = self.output_dir / 'cross_domain_evaluation_report.md'

        with open(report_path, 'w') as f:
            f.write("# Cross-Domain Ontology Embedding Evaluation Report\n\n")

            # Experiment overview
            config = results['experiment_config']
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Training Domains**: {', '.join(config['train_domains'])}\n")
            f.write(f"- **Test Domains**: {', '.join(config['test_domains'])}\n")
            f.write(f"- **Model Configurations**: {len(config['model_configs'])}\n")
            f.write(f"- **Baseline Methods**: {', '.join(config['baseline_methods'])}\n")
            f.write(f"- **Embedding Dimension**: {config['embedding_dim']}\n\n")

            # Dataset information
            f.write("## Datasets\n\n")
            f.write("### Training Datasets\n")
            for key, path in results['datasets']['train'].items():
                f.write(f"- **{key}**: {path}\n")

            f.write("\n### Test Datasets\n")
            for key, path in results['datasets']['test'].items():
                f.write(f"- **{key}**: {path}\n")

            # Results summary
            f.write("\n## Results Summary\n\n")

            for config_name, model_analysis in analysis['model_comparison'].items():
                f.write(f"### {config_name}\n\n")

                # Within-domain performance
                if model_analysis['within_domain_performance']:
                    f.write("#### Within-Domain Performance\n")
                    for dataset, metrics in model_analysis['within_domain_performance'].items():
                        f.write(f"**{dataset}**:\n")
                        for metric_name, metric_value in metrics.items():
                            f.write(f"- {metric_name}: {metric_value:.3f}\n")
                        f.write("\n")

                # Transfer effectiveness
                if model_analysis['transfer_effectiveness']:
                    f.write("#### Transfer Effectiveness\n")
                    for test_dataset, transfer_data in model_analysis['transfer_effectiveness'].items():
                        f.write(f"**{test_dataset}**:\n")
                        f.write(f"- Mean Transfer Score: {transfer_data['mean_transfer_score']:.3f}\n")
                        f.write(f"- Std Transfer Score: {transfer_data['std_transfer_score']:.3f}\n")
                        f.write("\n")

        self.logger.info(f"Cross-domain evaluation report saved to {report_path}")
        return str(report_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-domain ontology embedding evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--train-domains', nargs='+', required=True,
        help='Training domains (e.g., biology chemistry)'
    )
    parser.add_argument(
        '--test-domains', nargs='+', required=True,
        help='Test domains (e.g., medicine food_science)'
    )
    parser.add_argument(
        '--model-configs', default='configs/cross_domain_models.json',
        help='Path to JSON file with model configurations'
    )
    parser.add_argument(
        '--baseline-methods', nargs='+',
        default=['random', 'structural'],
        help='Baseline methods to compare against'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=64,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--output-dir', default='cross_domain_evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load model configurations
        if Path(args.model_configs).exists():
            with open(args.model_configs) as f:
                model_configs = json.load(f)
        else:
            # Default configurations
            model_configs = [
                {
                    'model_type': 'gcn',
                    'hidden_dim': 128,
                    'epochs': 100,
                    'loss_fn': 'triplet'
                },
                {
                    'model_type': 'gat',
                    'hidden_dim': 128,
                    'epochs': 100,
                    'loss_fn': 'triplet'
                },
                {
                    'model_type': 'rgcn',
                    'hidden_dim': 128,
                    'epochs': 100,
                    'loss_fn': 'triplet',
                    'use_multi_relation': True
                }
            ]

        # Run cross-domain evaluation
        evaluator = CrossDomainEvaluator(args.output_dir)

        results = evaluator.run_cross_domain_experiment(
            train_domains=args.train_domains,
            test_domains=args.test_domains,
            model_configs=model_configs,
            baseline_methods=args.baseline_methods,
            embedding_dim=args.embedding_dim
        )

        # Generate report
        report_path = evaluator.create_cross_domain_report(results)

        print(f"\nCross-domain evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report available at: {report_path}")

        return 0

    except Exception as e:
        logger.error(f"Cross-domain evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())