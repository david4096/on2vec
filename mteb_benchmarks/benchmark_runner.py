#!/usr/bin/env python3
"""
MTEB Benchmark Runner for on2vec Sentence Transformer Models

This module provides functionality to evaluate sentence transformer models
created with on2vec ontology augmentation against the MTEB (Massive Text
Embedding Benchmark) suite.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime

try:
    import mteb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install mteb sentence-transformers")
    sys.exit(1)


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('mteb_runner')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_model(model_path: str) -> SentenceTransformer:
    """Load a sentence transformer model."""
    try:
        model = SentenceTransformer(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Dimensions: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        raise


def get_available_tasks(task_types: Optional[List[str]] = None) -> List[str]:
    """Get list of available MTEB tasks, optionally filtered by type."""
    all_tasks = mteb.get_tasks()

    if task_types:
        filtered_tasks = []
        for task in all_tasks:
            task_obj = mteb.get_task(task)
            if hasattr(task_obj, 'metadata') and task_obj.metadata.type in task_types:
                filtered_tasks.append(task)
        return filtered_tasks

    return all_tasks


def run_benchmark(
    model: SentenceTransformer,
    tasks: List[str],
    output_dir: str,
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Run MTEB benchmark on specified tasks."""

    logger = logging.getLogger('mteb_runner')
    logger.info(f"Starting benchmark for {model_name} on {len(tasks)} tasks")

    # Create evaluation object
    evaluation = mteb.MTEB(tasks=tasks)

    # Run evaluation
    results = evaluation.run(
        model,
        output_folder=output_dir,
        batch_size=batch_size,
        device=device,
        overwrite_results=True
    )

    logger.info(f"Benchmark completed. Results saved to {output_dir}")
    return results


def summarize_results(results_dir: str, model_name: str) -> Dict[str, Any]:
    """Summarize benchmark results from output directory."""
    results_path = Path(results_dir)
    summary = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'task_results': {},
        'averages': {}
    }

    # Find all result JSON files
    for json_file in results_path.glob("**/*.json"):
        try:
            with open(json_file, 'r') as f:
                task_data = json.load(f)

            task_name = json_file.stem
            summary['task_results'][task_name] = task_data

        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")

    # Calculate category averages
    task_categories = {}
    for task_name, task_data in summary['task_results'].items():
        if 'mteb_dataset_name' in task_data:
            # Try to get task metadata for categorization
            try:
                task_obj = mteb.get_task(task_data['mteb_dataset_name'])
                category = task_obj.metadata.type if hasattr(task_obj, 'metadata') else 'unknown'

                if category not in task_categories:
                    task_categories[category] = []

                # Extract main score
                if 'test' in task_data and len(task_data['test']) > 0:
                    main_score = task_data['test'][0].get('main_score', 0)
                    task_categories[category].append(main_score)

            except Exception:
                continue

    # Calculate averages per category
    for category, scores in task_categories.items():
        if scores:
            summary['averages'][category] = {
                'mean': sum(scores) / len(scores),
                'count': len(scores),
                'scores': scores
            }

    return summary


def generate_benchmark_report(summary: Dict[str, Any], output_file: str):
    """Generate a markdown report from benchmark summary."""
    report = f"""# MTEB Benchmark Report

**Model:** {summary['model_name']}
**Date:** {summary['timestamp']}
**Total Tasks:** {len(summary['task_results'])}

## Category Averages

"""

    if summary['averages']:
        report += "| Category | Average Score | Task Count |\n"
        report += "|----------|---------------|------------|\n"

        for category, data in summary['averages'].items():
            report += f"| {category} | {data['mean']:.3f} | {data['count']} |\n"

    report += "\n## Individual Task Results\n\n"

    for task_name, task_data in summary['task_results'].items():
        report += f"### {task_name}\n\n"

        if 'test' in task_data and len(task_data['test']) > 0:
            test_result = task_data['test'][0]
            main_score = test_result.get('main_score', 'N/A')
            report += f"**Main Score:** {main_score}\n\n"

            # Add other metrics
            for key, value in test_result.items():
                if key != 'main_score' and isinstance(value, (int, float)):
                    report += f"- {key}: {value:.3f}\n"

        report += "\n"

    # Save report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"📊 Benchmark report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run MTEB benchmarks on sentence transformer models")
    parser.add_argument('model_path', help='Path to sentence transformer model')
    parser.add_argument('--output-dir', default='./mteb_results', help='Output directory for results')
    parser.add_argument('--model-name', help='Model name for results (default: inferred from path)')
    parser.add_argument('--tasks', nargs='+', help='Specific tasks to run (default: all)')
    parser.add_argument('--task-types', nargs='+', choices=[
        'Classification', 'Clustering', 'PairClassification', 'Reranking',
        'Retrieval', 'STS', 'Summarization'
    ], help='Task types to run')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tasks')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    # Infer model name if not provided
    model_name = args.model_name or Path(args.model_path).name

    print(f"🧪 MTEB Benchmark Runner")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)

    # Load model
    model = load_model(args.model_path)

    # Get tasks to run
    if args.tasks:
        tasks = args.tasks
    elif args.quick:
        # Quick subset for testing
        tasks = [
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',  # Semantic similarity
            'Banking77Classification',  # Classification
            'SprintDuplicateQuestions',  # Pair classification
        ]
    else:
        tasks = get_available_tasks(args.task_types)

    print(f"📋 Running {len(tasks)} tasks:")
    for i, task in enumerate(tasks[:5]):  # Show first 5
        print(f"  {i+1}. {task}")
    if len(tasks) > 5:
        print(f"  ... and {len(tasks) - 5} more")

    # Create output directory
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    try:
        results = run_benchmark(
            model=model,
            tasks=tasks,
            output_dir=str(output_dir),
            model_name=model_name,
            batch_size=args.batch_size,
            device=args.device
        )

        # Generate summary and report
        summary = summarize_results(str(output_dir), model_name)

        # Save summary JSON
        summary_file = output_dir / 'benchmark_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate markdown report
        report_file = output_dir / 'benchmark_report.md'
        generate_benchmark_report(summary, str(report_file))

        print(f"\n✅ Benchmark completed successfully!")
        print(f"📁 Results: {output_dir}")
        print(f"📊 Summary: {summary_file}")
        print(f"📝 Report: {report_file}")

        # Print quick summary
        if summary['averages']:
            print(f"\n🏆 Quick Summary:")
            for category, data in summary['averages'].items():
                print(f"  {category}: {data['mean']:.3f} (avg of {data['count']} tasks)")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())