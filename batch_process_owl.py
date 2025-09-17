#!/usr/bin/env python3
"""
Batch process OWL files with training and visualization pipeline.
Processes all OWL files in owl_files/ directory with:
- 8-dimension output embeddings
- 64 hidden dimensions
- Cosine loss function
- All visualization types (PCA, t-SNE, UMAP, distribution, comparison)
- 4 parallel workers
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_file_size_mb(file_path):
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0


def run_command(cmd, timeout=600):
    """Run a command with timeout and return (success, output, error)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def process_single_owl_file(owl_file_path):
    """
    Process a single OWL file: train model, generate embeddings, create visualizations.

    Args:
        owl_file_path (str): Path to the OWL file

    Returns:
        dict: Processing results and statistics
    """
    start_time = time.time()
    owl_file = Path(owl_file_path)
    base_name = owl_file.stem

    # Create output directory for this ontology
    output_dir = Path("output") / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Processing {base_name} (Size: {get_file_size_mb(owl_file_path):.1f} MB)")

    results = {
        'file': base_name,
        'file_size_mb': get_file_size_mb(owl_file_path),
        'start_time': datetime.now().isoformat(),
        'success': False,
        'steps_completed': [],
        'errors': [],
        'timings': {},
        'model_stats': {},
        'visualization_stats': {}
    }

    try:
        # Step 1: Train model
        step_start = time.time()
        model_path = output_dir / f"{base_name}_model.pt"
        embeddings_path = output_dir / f"{base_name}_embeddings.parquet"

        train_cmd = (
            f"uv run main.py {owl_file_path} "
            f"--model_type heterogeneous "
            f"--hidden_dim 64 "
            f"--out_dim 8 "
            f"--loss_fn cosine "
            f"--epochs 100 "
            f"--model_output {model_path} "
            f"--output {embeddings_path}"
        )

        logger.info(f"🏋️  Training model for {base_name}...")
        success, stdout, stderr = run_command(train_cmd, timeout=1200)  # 20 min timeout

        if not success:
            results['errors'].append(f"Training failed: {stderr}")
            logger.error(f"❌ Training failed for {base_name}: {stderr}")
            return results

        results['steps_completed'].append('training')
        results['timings']['training'] = time.time() - step_start

        # Extract model stats from stdout
        lines = stdout.split('\n')
        for line in lines:
            if 'Processed' in line and 'classes with' in line and 'edges' in line:
                # Parse "Processed X classes with Y edges"
                parts = line.split()
                try:
                    nodes_idx = parts.index('Processed') + 1
                    edges_idx = parts.index('edges') - 1
                    results['model_stats']['nodes'] = int(parts[nodes_idx])
                    results['model_stats']['edges'] = int(parts[edges_idx])
                except:
                    pass
            if 'embeddings' in line.lower() and 'Generated' in line:
                # Parse embedding count
                parts = line.split()
                try:
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0:
                            results['model_stats']['embeddings'] = int(part)
                            break
                except:
                    pass

        logger.info(f"✅ Training completed for {base_name} ({results['timings']['training']:.1f}s)")

        # Check if embeddings file was created
        if not embeddings_path.exists():
            results['errors'].append("Embeddings file not created")
            logger.error(f"❌ Embeddings file not found for {base_name}")
            return results

        # Step 2: Create all visualizations
        visualization_commands = [
            ("pca", f"uv run parquet_tools.py plot-pca {embeddings_path} --output {output_dir}/{base_name}_pca.png"),
            ("tsne", f"uv run parquet_tools.py plot-tsne {embeddings_path} --output {output_dir}/{base_name}_tsne.png"),
            ("umap", f"uv run parquet_tools.py plot-umap {embeddings_path} --output {output_dir}/{base_name}_umap.png"),
            ("distribution", f"uv run parquet_tools.py plot-dist {embeddings_path} --output {output_dir}/{base_name}_distribution.png"),
            ("comparison", f"uv run parquet_tools.py plot-compare {embeddings_path} --output {output_dir}/{base_name}_comparison.png")
        ]

        for viz_name, viz_cmd in visualization_commands:
            step_start = time.time()
            logger.info(f"📊 Creating {viz_name} plot for {base_name}...")

            success, stdout, stderr = run_command(viz_cmd, timeout=600)  # 10 min timeout

            if success:
                results['steps_completed'].append(f'viz_{viz_name}')
                results['timings'][f'viz_{viz_name}'] = time.time() - step_start
                results['visualization_stats'][viz_name] = 'success'
                logger.info(f"✅ {viz_name} plot completed for {base_name}")
            else:
                results['errors'].append(f"{viz_name} visualization failed: {stderr}")
                results['visualization_stats'][viz_name] = 'failed'
                logger.warning(f"⚠️  {viz_name} plot failed for {base_name}: {stderr}")

        # Step 3: Create summary report
        summary_path = output_dir / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"On2Vec Processing Summary: {base_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Source file: {owl_file_path}\n")
            f.write(f"File size: {results['file_size_mb']:.1f} MB\n")
            f.write(f"Processing date: {results['start_time']}\n\n")

            f.write("Model Configuration:\n")
            f.write("- Model type: GCN\n")
            f.write("- Hidden dimensions: 64\n")
            f.write("- Output dimensions: 8\n")
            f.write("- Loss function: Cosine\n")
            f.write("- Epochs: 100\n\n")

            if results['model_stats']:
                f.write("Model Statistics:\n")
                for key, value in results['model_stats'].items():
                    f.write(f"- {key.title()}: {value:,}\n")
                f.write("\n")

            f.write("Processing Times:\n")
            for step, duration in results['timings'].items():
                f.write(f"- {step.replace('_', ' ').title()}: {duration:.1f}s\n")
            f.write(f"- Total: {time.time() - start_time:.1f}s\n\n")

            f.write("Generated Files:\n")
            f.write(f"- Model: {model_path.name}\n")
            f.write(f"- Embeddings: {embeddings_path.name}\n")
            for viz_name, status in results['visualization_stats'].items():
                status_icon = "✅" if status == 'success' else "❌"
                f.write(f"- {viz_name.title()} plot: {base_name}_{viz_name}.png {status_icon}\n")

        results['success'] = True
        results['total_time'] = time.time() - start_time
        results['end_time'] = datetime.now().isoformat()

        logger.info(f"🎉 Successfully processed {base_name} in {results['total_time']:.1f}s")

    except Exception as e:
        results['errors'].append(f"Unexpected error: {str(e)}")
        results['traceback'] = traceback.format_exc()
        logger.error(f"💥 Unexpected error processing {base_name}: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")

    return results


def main():
    """Main batch processing function."""
    start_time = time.time()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Find all OWL files
    owl_files_dir = Path("owl_files")
    if not owl_files_dir.exists():
        logger.error("❌ owl_files directory not found!")
        return 1

    owl_files = list(owl_files_dir.glob("*.owl"))
    if not owl_files:
        logger.error("❌ No OWL files found in owl_files directory!")
        return 1

    # Sort by file size (smallest first for faster initial results)
    owl_files.sort(key=lambda f: get_file_size_mb(f))

    logger.info(f"🔍 Found {len(owl_files)} OWL files to process")
    logger.info(f"📊 Size range: {get_file_size_mb(owl_files[0]):.1f} MB to {get_file_size_mb(owl_files[-1]):.1f} MB")
    logger.info(f"🔧 Configuration: GCN, 64 hidden dims, 8 output dims, cosine loss, 100 epochs")
    logger.info(f"🚀 Starting processing with 4 parallel workers...")

    # Process files in parallel
    completed_results = []
    failed_results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_owl_file, str(owl_file)): owl_file
            for owl_file in owl_files
        }

        # Process completed jobs
        for future in as_completed(future_to_file):
            owl_file = future_to_file[future]
            try:
                result = future.result()
                if result['success']:
                    completed_results.append(result)
                    logger.info(f"✅ [{len(completed_results)}/{len(owl_files)}] Completed {result['file']}")
                else:
                    failed_results.append(result)
                    logger.error(f"❌ [{len(completed_results) + len(failed_results)}/{len(owl_files)}] Failed {result['file']}")

            except Exception as e:
                failed_results.append({
                    'file': owl_file.stem,
                    'success': False,
                    'errors': [f"Future exception: {str(e)}"]
                })
                logger.error(f"💥 Exception processing {owl_file.stem}: {e}")

    # Generate final report
    total_time = time.time() - start_time

    report_path = output_dir / "batch_processing_report.txt"
    with open(report_path, 'w') as f:
        f.write("On2Vec Batch Processing Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing date: {datetime.now().isoformat()}\n")
        f.write(f"Total files: {len(owl_files)}\n")
        f.write(f"Successfully processed: {len(completed_results)}\n")
        f.write(f"Failed: {len(failed_results)}\n")
        f.write(f"Success rate: {len(completed_results)/len(owl_files)*100:.1f}%\n")
        f.write(f"Total processing time: {total_time:.1f}s ({total_time/3600:.1f}h)\n\n")

        if completed_results:
            f.write("Successfully Processed Files:\n")
            f.write("-" * 30 + "\n")
            completed_results.sort(key=lambda x: x['total_time'])
            for result in completed_results:
                f.write(f"{result['file']:<20} {result['total_time']:>6.1f}s")
                if 'model_stats' in result and 'embeddings' in result['model_stats']:
                    f.write(f" ({result['model_stats']['embeddings']:,} embeddings)")
                f.write("\n")

        if failed_results:
            f.write(f"\nFailed Files ({len(failed_results)}):\n")
            f.write("-" * 20 + "\n")
            for result in failed_results:
                f.write(f"{result['file']}: {', '.join(result['errors'])}\n")

        # Statistics
        if completed_results:
            times = [r['total_time'] for r in completed_results]
            f.write(f"\nProcessing Time Statistics:\n")
            f.write(f"- Fastest: {min(times):.1f}s\n")
            f.write(f"- Slowest: {max(times):.1f}s\n")
            f.write(f"- Average: {sum(times)/len(times):.1f}s\n")
            f.write(f"- Median: {sorted(times)[len(times)//2]:.1f}s\n")

    # Print final summary
    logger.info("🎉 Batch processing completed!")
    logger.info(f"📊 Results: {len(completed_results)}/{len(owl_files)} successful ({len(completed_results)/len(owl_files)*100:.1f}%)")
    logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.info(f"📄 Full report saved to: {report_path}")

    return 0 if len(failed_results) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())