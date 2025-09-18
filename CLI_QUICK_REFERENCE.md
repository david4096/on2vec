# on2vec CLI Quick Reference

## 📥 Installation

```bash
# Basic installation
pip install on2vec

# With benchmarking support
pip install on2vec[benchmark]

# All features
pip install on2vec[all]
```

## 🚀 HuggingFace Sentence Transformers Integration

### One-Command Model Creation

```bash
# Create complete HuggingFace model from OWL file (with auto model card generation)
on2vec hf biomedical.owl my-biomedical-model

# With custom settings (auto-generates comprehensive model card)
on2vec hf ontology.owl my-model \
  --base-model all-mpnet-base-v2 \
  --fusion gated \
  --epochs 200
```

### Step-by-Step Workflow

```bash
# 1. Train ontology with text features
on2vec hf-train ontology.owl --output embeddings.parquet

# 2. Create HuggingFace model (auto-detects base model from embeddings)
on2vec hf-create embeddings.parquet my-model --fusion concat

# 3. Test model
on2vec hf-test ./hf_models/my-model

# 4. Inspect model details
on2vec inspect ./hf_models/my-model
```

### 🧠 Smart Auto-Detection

The CLI automatically detects the base model used to create embeddings:

```bash
# ✅ Auto-detects all-MiniLM-L6-v2 from embeddings metadata
on2vec hf-create embeddings.parquet my-model

# ⚠️  Warns about mismatches and uses the correct model
on2vec hf-create embeddings.parquet my-model --base-model all-mpnet-base-v2
# WARNING: Base model mismatch! Using detected model: all-MiniLM-L6-v2
```

### Batch Processing

```bash
# Process directory of OWL files
on2vec hf-batch owl_files/ ./output \
  --base-models all-MiniLM-L6-v2 all-mpnet-base-v2 \
  --fusion-methods concat gated \
  --max-workers 4
```

### Utilities

```bash
# Inspect embeddings or models
on2vec inspect embeddings.parquet
on2vec inspect ./hf_models/my-model

# Convert formats
on2vec convert embeddings.parquet embeddings.csv

# Show help
on2vec --help
on2vec hf --help
```

## 📊 Core on2vec Commands

### Basic Training

```bash
# Train GCN model
on2vec train ontology.owl --output model.pt --model-type gcn --epochs 100

# Train with text features (for HF integration)
on2vec train ontology.owl --output embeddings.parquet --use-text-features

# Custom configuration
on2vec train ontology.owl --output model.pt \
  --model-type gat \
  --hidden-dim 256 \
  --out-dim 128 \
  --epochs 200 \
  --loss-fn contrastive
```

### Generate Embeddings

```bash
# Generate embeddings from trained model
on2vec embed model.pt ontology.owl --output embeddings.parquet
```

### Visualization

```bash
# Create UMAP visualization
on2vec visualize embeddings.parquet --output visualization.png
```

### Multi-relation Models

```bash
# Train multi-relation model
on2vec train ontology.owl --output model.pt --use-multi-relation --model-type rgcn

# Heterogeneous model
on2vec train ontology.owl --output model.pt --use-multi-relation --model-type heterogeneous
```

## 🔧 Advanced Workflows

### Domain-Specific Models

```bash
# Biomedical domain
on2vec hf biomedical_ontology.owl bio-embedder \
  --base-model dmis-lab/biobert-v1.1 \
  --fusion gated \
  --epochs 150

# Legal domain
on2vec hf legal_ontology.owl legal-embedder \
  --base-model nlpaueb/legal-bert-base-uncased \
  --fusion attention
```

### Comparative Analysis

```bash
# Create multiple fusion variants
for method in concat weighted_avg attention gated; do
  on2vec hf-create embeddings.parquet "model-$method" \
    --fusion $method --output-dir ./comparison
done

# Test all variants
for model in ./comparison/model-*; do
  on2vec hf-test "$model"
done
```

### Production Pipeline

```bash
# 1. Comprehensive training
on2vec hf-train ontology.owl \
  --output production_embeddings.parquet \
  --text-model all-mpnet-base-v2 \
  --epochs 300 \
  --model-type gat \
  --hidden-dim 512

# 2. Create production model
on2vec hf-create production_embeddings.parquet production-model \
  --fusion gated \
  --base-model all-mpnet-base-v2 \
  --output-dir ./production

# 3. Validate thoroughly
on2vec hf-test ./production/production-model \
  --queries "domain term 1" "domain term 2" "domain term 3"
```

## 📤 HuggingFace Hub Upload

```bash
# Upload instructions auto-generated in each model directory:
# See ./hf_models/my-model/UPLOAD_INSTRUCTIONS.md

# Manual upload process:
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
on2vec benchmark ./hf_models/my-model --quick

# Full MTEB benchmark
on2vec benchmark ./hf_models/my-model

# Focus on specific task types
on2vec benchmark ./hf_models/my-model --task-types STS Classification

# Compare with vanilla model
on2vec benchmark sentence-transformers/all-MiniLM-L6-v2 \
  --model-name vanilla-baseline --quick

# Compare ontology vs vanilla models
on2vec compare ./hf_models/my-model --detailed
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