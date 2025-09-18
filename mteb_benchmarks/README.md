# MTEB Benchmarking for on2vec Models

This directory contains tools for evaluating sentence transformer models created with on2vec ontology augmentation against the MTEB (Massive Text Embedding Benchmark) suite.

## Setup

Install MTEB dependencies:

```bash
pip install mteb
# or with uv
uv add mteb
```

## Quick Start

### Run Full Benchmark

```bash
# Run all MTEB tasks on your model
python benchmark_runner.py ./hf_models/my-model --model-name my-ontology-model

# Run specific task types
python benchmark_runner.py ./hf_models/my-model --task-types STS Classification

# Quick test run (subset of tasks)
python benchmark_runner.py ./hf_models/my-model --quick
```

### Results Structure

```
mteb_results/
├── my-model/
│   ├── benchmark_summary.json      # Complete results data
│   ├── benchmark_report.md         # Human-readable report
│   └── task_results/              # Individual task JSON files
│       ├── STS12.json
│       ├── Banking77Classification.json
│       └── ...
```

## Command Reference

### Basic Usage

```bash
python benchmark_runner.py MODEL_PATH [options]
```

### Options

- `--output-dir DIR`: Output directory (default: `./mteb_results`)
- `--model-name NAME`: Model name for results organization
- `--tasks TASK1 TASK2`: Run specific tasks only
- `--task-types TYPE1 TYPE2`: Run specific task categories
- `--batch-size N`: Batch size for evaluation (default: 32)
- `--device DEVICE`: Device to use (cuda/cpu)
- `--quick`: Run quick subset for testing
- `--log-file FILE`: Save logs to file

### Task Types Available

- `Classification`: Text classification tasks
- `Clustering`: Text clustering tasks
- `PairClassification`: Sentence pair classification
- `Reranking`: Document reranking tasks
- `Retrieval`: Information retrieval tasks
- `STS`: Semantic textual similarity
- `Summarization`: Text summarization evaluation

## Examples

### Compare Multiple Models

```bash
# Benchmark vanilla model
python benchmark_runner.py sentence-transformers/all-MiniLM-L6-v2 \
  --model-name vanilla-miniLM --quick

# Benchmark your ontology-augmented model
python benchmark_runner.py ./hf_models/edam-text-model \
  --model-name edam-augmented --quick

# Compare results in mteb_results/ directory
```

### Domain-Specific Evaluation

```bash
# Focus on semantic similarity tasks (good for ontology models)
python benchmark_runner.py ./hf_models/bio-model \
  --task-types STS \
  --model-name biomedical-ontology

# Classification-heavy evaluation
python benchmark_runner.py ./hf_models/legal-model \
  --task-types Classification PairClassification \
  --model-name legal-ontology
```

### Production Benchmarking

```bash
# Full benchmark with logging
python benchmark_runner.py ./hf_models/production-model \
  --model-name production-ontology-v1 \
  --log-file benchmark.log \
  --batch-size 64 \
  --device cuda
```

## Understanding Results

### Benchmark Report

The generated `benchmark_report.md` includes:

- **Category Averages**: Mean scores across task types
- **Individual Results**: Detailed metrics per task
- **Task Counts**: Number of tasks evaluated per category

### Key Metrics

- **STS Tasks**: Measure semantic similarity understanding
- **Classification**: Domain knowledge application
- **Retrieval**: Information finding capabilities
- **Clustering**: Concept grouping abilities

### Interpreting Ontology Model Performance

Ontology-augmented models typically show:

- **Higher STS scores**: Better semantic understanding
- **Improved classification**: Domain-specific knowledge helps categorization
- **Better clustering**: Ontological relationships improve concept grouping
- **Mixed retrieval**: May depend on domain alignment

## Integration with on2vec Workflow

### After Creating a Model

```bash
# 1. Create ontology-augmented model
python create_hf_model.py e2e biomedical.owl bio-model

# 2. Quick benchmark test
python mteb_benchmarks/benchmark_runner.py ./hf_models/bio-model --quick

# 3. Full benchmark if results look promising
python mteb_benchmarks/benchmark_runner.py ./hf_models/bio-model
```

### Batch Model Comparison

```bash
# Create multiple fusion variants
python batch_hf_models.py process owl_files/ ./output

# Benchmark all variants
for model in ./output/models/*/; do
    python mteb_benchmarks/benchmark_runner.py "$model" \
        --model-name "$(basename "$model")" --quick
done
```

## Tips for Better Results

1. **Use domain-relevant tasks**: Focus on task types that align with your ontology domain
2. **Compare against base model**: Always benchmark the base text model for comparison
3. **Start with --quick**: Test subset first before running full benchmark
4. **Monitor resource usage**: MTEB can be memory and compute intensive
5. **Save logs**: Use `--log-file` for debugging and progress tracking

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python benchmark_runner.py model --batch-size 8

# Use CPU if GPU memory limited
python benchmark_runner.py model --device cpu
```

### Task Failures
```bash
# Run specific working tasks only
python benchmark_runner.py model --tasks STS12 STS13 Banking77Classification
```

### Performance Issues
```bash
# Quick subset for testing
python benchmark_runner.py model --quick

# Focus on one task type
python benchmark_runner.py model --task-types STS
```