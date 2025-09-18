# on2vec CLI Quick Reference

## 🚀 HuggingFace Sentence Transformers Integration

### One-Command Model Creation

```bash
# Create complete HuggingFace model from OWL file (with auto model card generation)
python create_hf_model.py e2e biomedical.owl my-biomedical-model

# With custom settings (auto-generates comprehensive model card)
python create_hf_model.py e2e ontology.owl my-model \
  --base-model all-mpnet-base-v2 \
  --fusion gated \
  --epochs 200
```

### Step-by-Step Workflow

```bash
# 1. Train ontology with text features
python create_hf_model.py train ontology.owl --output embeddings.parquet

# 2. Create HuggingFace model (auto-detects base model from embeddings)
python create_hf_model.py create embeddings.parquet my-model --fusion concat

# 3. Test model
python create_hf_model.py test ./hf_models/my-model

# 4. Show upload instructions
python create_hf_model.py upload-info ./hf_models/my-model my-model
```

### 🧠 Smart Auto-Detection

The CLI automatically detects the base model used to create embeddings:

```bash
# ✅ Auto-detects all-MiniLM-L6-v2 from embeddings metadata
python create_hf_model.py create embeddings.parquet my-model

# ⚠️  Warns about mismatches and uses the correct model
python create_hf_model.py create embeddings.parquet my-model --base-model all-mpnet-base-v2
# WARNING: Base model mismatch! Using detected model: all-MiniLM-L6-v2
```

### Batch Processing

```bash
# Process directory of OWL files
python batch_hf_models.py process owl_files/ ./output \
  --base-models all-MiniLM-L6-v2 all-mpnet-base-v2 \
  --fusion-methods concat gated \
  --max-workers 4

# Create model collection
python batch_hf_models.py collection ./output/batch_results.json \
  --name "biomedical-models" --criteria best_test
```

### Utilities

```bash
# Validate embeddings
python create_hf_model.py validate embeddings.parquet

# Test existing model
python create_hf_model.py test ./models/my-model

# Show help
python create_hf_model.py --help
python batch_hf_models.py --help
```

## 📊 Original on2vec Commands

### Basic Training

```bash
# Train GCN model
python main.py ontology.owl --model_type gcn --epochs 100

# Train with text features (for HF integration)
python main.py ontology.owl --use_text_features --output embeddings.parquet

# Custom configuration
python main.py ontology.owl \
  --model_type gat \
  --hidden_dim 256 \
  --out_dim 128 \
  --epochs 200 \
  --loss_fn contrastive
```

### Visualization

```bash
# Create UMAP visualization
python viz.py embeddings.parquet --output visualization.png

# Batch visualization
python process_dir.py owl_files/ --model_type gcn --epochs 100
```

### Multi-relation Models

```bash
# Train multi-relation model
python main.py ontology.owl --use_multi_relation --model_type rgcn

# Heterogeneous model
python main.py ontology.owl --use_multi_relation --model_type heterogeneous
```

## 🔧 Advanced Workflows

### Domain-Specific Models

```bash
# Biomedical domain
python create_hf_model.py e2e biomedical_ontology.owl bio-embedder \
  --base-model dmis-lab/biobert-v1.1 \
  --fusion gated \
  --epochs 150

# Legal domain
python create_hf_model.py e2e legal_ontology.owl legal-embedder \
  --base-model nlpaueb/legal-bert-base-uncased \
  --fusion attention
```

### Comparative Analysis

```bash
# Create multiple fusion variants
for method in concat weighted_avg attention gated; do
  python create_hf_model.py create embeddings.parquet "model-$method" \
    --fusion $method --output-dir ./comparison
done

# Test all variants
for model in ./comparison/model-*; do
  python create_hf_model.py test "$model"
done
```

### Production Pipeline

```bash
# 1. Comprehensive training
python create_hf_model.py train ontology.owl \
  --output production_embeddings.parquet \
  --text-model all-mpnet-base-v2 \
  --epochs 300 \
  --model-type gat \
  --hidden-dim 512

# 2. Create production model
python create_hf_model.py create production_embeddings.parquet production-model \
  --fusion gated \
  --base-model all-mpnet-base-v2 \
  --output-dir ./production

# 3. Validate thoroughly
python create_hf_model.py test ./production/production-model \
  --queries "domain term 1" "domain term 2" "domain term 3"
```

## 📤 HuggingFace Hub Upload

```bash
# Get upload instructions (auto-generated during model creation)
python create_hf_model.py upload-info ./hf_models/my-model my-model

# Manual upload process (instructions in model's UPLOAD_INSTRUCTIONS.md):
# 1. pip install huggingface_hub
# 2. huggingface-cli login
# 3. Upload model:
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./hf_models/my-model')
model.push_to_hub('your-username/my-model')
"

# Model automatically includes comprehensive README.md with:
# - YAML frontmatter with proper tags and metadata
# - Detailed model description and training process
# - Usage examples and code snippets
# - Domain-specific information and limitations
# - Proper citations and licensing
```

## 🧪 MTEB Benchmarking

```bash
# Quick benchmark test
python mteb_benchmarks/benchmark_runner.py ./hf_models/my-model --quick

# Full MTEB benchmark
python mteb_benchmarks/benchmark_runner.py ./hf_models/my-model

# Focus on specific task types
python mteb_benchmarks/benchmark_runner.py ./hf_models/my-model \
  --task-types STS Classification

# Compare with vanilla model
python mteb_benchmarks/benchmark_runner.py sentence-transformers/all-MiniLM-L6-v2 \
  --model-name vanilla-baseline --quick
```

## ⚡ Quick Tips

- **Start simple**: Use `e2e` command for first attempts
- **Test fusion methods**: `concat` (fast), `gated` (smart), `attention` (sophisticated)
- **Monitor resources**: Use `--max-workers 1` for limited memory
- **Validate first**: Use `validate` command before creating models
- **Batch process**: Use `batch_hf_models.py` for multiple ontologies
- **Check compatibility**: Models work with standard `sentence-transformers` API
- **Model cards**: Automatically generated with comprehensive documentation
- **Benchmark early**: Use `--quick` for fast evaluation during development

For detailed documentation: `docs/sentence_transformers_integration.md`