# Batch Processing OWL Files with On2Vec

This guide explains how to use the batch processing scripts to train models and generate visualizations for multiple OWL files in parallel.

## Overview

The batch processing system will:
1. **Train GCN models** with your specified parameters (8-dim output, 64 hidden dims, cosine loss)
2. **Generate embeddings** for each ontology
3. **Create all visualizations**: PCA, t-SNE, UMAP, distribution plots, and comparison plots
4. **Run 4 files in parallel** for optimal performance
5. **Generate comprehensive reports** and summaries

## Files

- `batch_process_owl.py` - **Main script**: Processes all 103 OWL files
- `demo_batch_process.py` - **Demo script**: Processes just the duo.owl file for testing
- `test_batch_process.py` - **Test script**: Processes the 3 smallest files for testing

## Quick Start

### 1. Demo (recommended first step)
```bash
# Process just duo.owl to see the complete pipeline
uv run python demo_batch_process.py
```

**Expected output**: Completes in ~40s, creates `output/duo/` with 8 files:
- `duo_model.pt` - Trained model
- `duo_embeddings.parquet` - Generated embeddings
- `duo_pca.png` - PCA visualization
- `duo_tsne.png` - t-SNE visualization
- `duo_umap.png` - UMAP visualization
- `duo_distribution.png` - Distribution plots
- `duo_comparison.png` - Side-by-side comparison
- `duo_summary.txt` - Processing summary

### 2. Test Run (3 smallest files)
```bash
# Process 3 smallest valid OWL files
uv run python test_batch_process.py
```

### 3. Full Batch Processing (all 103 files)
```bash
# Process all OWL files with 4 parallel workers
uv run python batch_process_owl.py
```

⚠️ **Warning**: This will process 103 files and may take several hours depending on file sizes.

## Configuration

All scripts use these fixed parameters:
- **Model type**: GCN (Graph Convolutional Network)
- **Hidden dimensions**: 64
- **Output dimensions**: 8
- **Loss function**: Cosine
- **Epochs**: 100
- **Parallel workers**: 4

## Output Structure

```
output/
├── batch_processing_report.txt          # Overall summary
├── <ontology_name>/                      # One directory per OWL file
│   ├── <name>_model.pt                  # Trained model
│   ├── <name>_embeddings.parquet        # Generated embeddings
│   ├── <name>_pca.png                   # PCA plot
│   ├── <name>_tsne.png                  # t-SNE plot
│   ├── <name>_umap.png                  # UMAP plot
│   ├── <name>_distribution.png          # Distribution plots
│   ├── <name>_comparison.png            # Comparison plot (all methods)
│   └── <name>_summary.txt               # Processing summary
└── batch_process.log                    # Detailed processing log
```

## Sample Processing Times

Based on the demo with duo.owl (45 nodes, 76 edges):
- **Training**: ~6s
- **PCA visualization**: ~5s
- **t-SNE visualization**: ~5s
- **UMAP visualization**: ~9s
- **Distribution plots**: ~6s
- **Comparison plot**: ~10s
- **Total**: ~40s

Larger ontologies will take proportionally longer.

## Monitoring Progress

The batch processing provides detailed logging:
```bash
# Watch the log file in real time
tail -f batch_process.log

# Check current status
grep "✅\|❌" batch_process.log | tail -10
```

## Error Handling

The scripts handle various error conditions:
- **Invalid OWL files**: Skipped with warning
- **Training failures**: Logged and continued with next file
- **Visualization failures**: Individual plots may fail but processing continues
- **Timeouts**: 20min for training, 10min per visualization

## Performance Tips

1. **Start with demo**: Always run the demo first to verify everything works
2. **Monitor disk space**: Each ontology generates ~1-2MB of output files
3. **Check memory usage**: Large ontologies may require significant RAM
4. **Parallel processing**: The script automatically uses 4 workers for optimal performance

## File Size Expectations

The 103 OWL files range from very small (HTML redirects) to very large:
- **Small files** (duo.owl): ~100KB, 45 nodes → ~40s processing
- **Medium files**: ~1MB, ~1000 nodes → ~2-5 minutes
- **Large files**: ~10MB+, ~10,000+ nodes → ~10-30 minutes

## Customization

To modify parameters, edit the relevant script:

```python
# In batch_process_owl.py, modify the train_cmd:
train_cmd = (
    f"uv run main.py {owl_file_path} "
    f"--model_type gcn "           # Change model type
    f"--hidden_dim 64 "            # Change hidden dimensions
    f"--out_dim 8 "                # Change output dimensions
    f"--loss_fn cosine "           # Change loss function
    f"--epochs 100 "               # Change epochs
    f"--model_output {model_path} "
    f"--output {embeddings_path}"
)
```

## Results Analysis

After completion, check:
1. **`output/batch_processing_report.txt`** - Overall success rate and statistics
2. **Individual summaries** - Processing details for each ontology
3. **Visualization files** - Compare embeddings across different ontologies
4. **`batch_process.log`** - Detailed logs for debugging any issues

## Example Usage Workflow

```bash
# 1. Start with demo to verify setup
uv run python demo_batch_process.py

# 2. Check demo results
ls output/duo/
cat output/duo/duo_summary.txt

# 3. Run test with 3 files
uv run python test_batch_process.py

# 4. If satisfied, run full batch
nohup uv run python batch_process_owl.py > batch_output.log 2>&1 &

# 5. Monitor progress
tail -f batch_process.log
```

This will process all your OWL files with the exact specifications: 8-dim output, 64 hidden dimensions, cosine loss, and generate all visualization types with 4 parallel workers!