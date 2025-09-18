# Sentence Transformers Integration with on2vec

This guide shows how to create production-ready Sentence Transformers models that incorporate ontology knowledge from on2vec embeddings.

## Installation

```bash
# Install on2vec
pip install on2vec

# With benchmarking support
pip install on2vec[benchmark]
```

## Overview

The integration allows you to:
1. **Train ontology embeddings** using on2vec with text features
2. **Create custom Sentence Transformers models** that combine semantic text similarity with ontology structural knowledge
3. **Upload and share models** on Hugging Face Hub for community use
4. **Use models seamlessly** with the standard `sentence-transformers` library

## Quick Start

### Option 1: One-Command Workflow (Recommended)

Create a complete HuggingFace model in one command:

```bash
# Complete end-to-end workflow
on2vec hf biomedical.owl my-biomedical-model
```

This single command:
1. ✅ Trains ontology with text features
2. ✅ Generates multi-embedding files
3. ✅ Creates HuggingFace compatible model
4. ✅ Auto-generates comprehensive model card with metadata
5. ✅ Creates upload instructions
6. ✅ Tests the model
7. ✅ Ready for HuggingFace Hub upload

### Option 2: Step-by-Step Workflow

#### Step 1: Train Ontology Model

```bash
# Train with custom configuration
on2vec hf-train biomedical.owl \
    --output embeddings.parquet \
    --text-model all-MiniLM-L6-v2 \
    --epochs 100 \
    --model-type gcn \
    --hidden-dim 128 \
    --out-dim 64
```

#### Step 2: Create HuggingFace Model

```bash
# Create model from embeddings (auto-detects base model)
on2vec hf-create embeddings.parquet my-model \
    --fusion concat \
    --output-dir ./hf_models
```

**🧠 Smart Auto-Detection**: The CLI automatically detects the base model used to create the embeddings from the parquet metadata, so you don't need to specify `--base-model` unless you want to override it.

#### Step 3: Test Model

```bash
# Test the created model
uv run python create_hf_model.py test ./hf_models/my-model
```

#### Step 4: Get Upload Instructions

```bash
# Show how to upload to HuggingFace Hub
uv run python create_hf_model.py upload-info ./hf_models/my-model my-model
```

### Option 3: Programmatic API

```python
from on2vec.sentence_transformer_hub import create_and_save_hf_model

# Create model programmatically
model_path = create_and_save_hf_model(
    ontology_embeddings_file="embeddings.parquet",
    model_name="my-ontology-model",
    output_dir="./models",
    fusion_method="concat"
)
```

### Using Your Model

```python
from sentence_transformers import SentenceTransformer

# Load your custom model
model = SentenceTransformer("./hf_models/my-model")

# Use like any sentence transformer
sentences = ["heart disease", "cardiovascular problems", "protein folding"]
embeddings = model.encode(sentences)

# Compute similarities
from sentence_transformers.util import cos_sim
similarities = cos_sim(embeddings, embeddings)
```

## Model Architecture Types

### Basic Ontology-Augmented Model

**Best for**: General semantic similarity with ontology knowledge

```python
from on2vec.sentence_transformer_integration import create_ontology_augmented_model

model = create_ontology_augmented_model(
    base_model='all-MiniLM-L6-v2',
    ontology_embeddings_file='embeddings.parquet',
    fusion_method='concat',  # 'concat', 'weighted_avg', 'attention'
    top_k_matches=3,
    structural_weight=0.3
)

# Usage
result = model(["protein folding disorders"])
embeddings = result['sentence_embedding']  # Shape: [1, 392]
```

**Dimensions**: Text (384) + Structural (8) = 392 output dimensions

### Query/Document Retrieval Model

**Best for**: Asymmetric search where queries are fast and documents are rich

```python
from on2vec.query_document_ontology_model import create_retrieval_model_with_ontology

model = create_retrieval_model_with_ontology(
    ontology_embeddings_file='embeddings.parquet',
    fusion_method='gated',  # Learns optimal text/structure weighting
    projection_dim=256      # Common embedding space
)

# Encode queries (fast, text-only)
query_embeds = model.encode_queries(["heart disease"])

# Encode documents (rich, with ontology)
doc_embeds = model.encode_documents([
    "Cardiovascular disease affects cardiac function...",
    "Protein misfolding causes neurodegeneration..."
])

# Compute retrieval scores
import torch
scores = torch.mm(query_embeds, doc_embeds.t())
```

## Fusion Methods

### 1. Concatenation (`concat`)
- **Simple**: Combines text and structural embeddings by concatenation
- **Output**: `text_dim + structural_dim` (e.g., 384 + 8 = 392)
- **Best for**: When you want to preserve all information

### 2. Weighted Average (`weighted_avg`)
- **Balanced**: Learns optimal weighting between text and structure
- **Output**: `min(text_dim, structural_dim)` (projected to common space)
- **Best for**: When embeddings have similar importance

### 3. Attention (`attention`)
- **Sophisticated**: Multi-head attention to focus on relevant aspects
- **Output**: Learned hidden dimension
- **Best for**: Complex domain-specific applications

### 4. Gated Fusion (`gated`)
- **Adaptive**: Neural gate learns when to use text vs structural info
- **Output**: `min(text_dim, structural_dim)`
- **Best for**: When text and structure have different relevance per query

## Creating HuggingFace Hub Models

### Step 1: Create Hub-Compatible Architecture

```python
# on2vec/sentence_transformer_hub.py
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
import torch.nn as nn

class OntologyAugmentedSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path, ontology_embeddings_file, **kwargs):
        # Initialize base transformer
        transformer = Transformer(model_name_or_path)
        pooling = Pooling(transformer.get_word_embedding_dimension())

        # Add ontology fusion module
        ontology_module = OntologyFusionModule(ontology_embeddings_file)

        super().__init__(modules=[transformer, pooling, ontology_module], **kwargs)
```

### Step 2: Package for Upload

```python
# Create model
model = create_hf_model("embeddings.parquet", "biomedical-ontology-embedder")

# Save with proper structure
model.save("./biomedical-ontology-embedder")

# Upload to Hub (requires huggingface_hub login)
model.push_to_hub("your-username/biomedical-ontology-embedder")
```

### Step 3: Usage After Upload

```python
from sentence_transformers import SentenceTransformer

# Anyone can now use your model
model = SentenceTransformer("your-username/biomedical-ontology-embedder")

# Works with all sentence-transformers features
embeddings = model.encode(["heart disease", "protein folding"])
```

## Model Cards and Documentation

Every model created with on2vec automatically includes comprehensive documentation:

### Auto-Generated Model Card

```bash
# Model card automatically created during model generation
ls ./hf_models/my-model/README.md
```

The model card includes:
- ✅ **Complete technical specifications** extracted from training metadata
- ✅ **HuggingFace YAML frontmatter** with proper tags for discoverability
- ✅ **Architecture details** including GNN type, dimensions, fusion method
- ✅ **Domain information** auto-detected from ontology filename
- ✅ **Training statistics** including concept count, alignment ratios
- ✅ **Usage examples** and code snippets
- ✅ **Performance characteristics** including model and ontology sizes

### Upload Instructions

```bash
# Upload instructions automatically generated
ls ./hf_models/my-model/UPLOAD_INSTRUCTIONS.md
```

Contains step-by-step instructions for:
- Installing dependencies
- HuggingFace Hub authentication
- Python upload script
- Manual upload alternatives

## MTEB Benchmarking

Evaluate your ontology-augmented models against standard benchmarks:

### Quick Evaluation

```bash
# Fast benchmark on subset of tasks
on2vec benchmark ./hf_models/my-model --quick

# Focus on semantic similarity tasks (ideal for ontology models)
on2vec benchmark ./hf_models/my-model --task-types STS

# Full MTEB benchmark (58+ tasks)
on2vec benchmark ./hf_models/my-model
```

### Comparative Analysis

```bash
# Benchmark vanilla baseline
on2vec benchmark sentence-transformers/all-MiniLM-L6-v2 \
  --model-name vanilla-baseline --quick

# Compare with your ontology model
on2vec benchmark ./hf_models/my-model \
  --model-name ontology-augmented --quick

# Compare ontology vs vanilla models
on2vec compare ./hf_models/my-model --detailed

# Results saved in mteb_results/ with detailed reports
```

### Benchmark Results

Each benchmark generates:
- **JSON summary** with detailed metrics per task
- **Markdown report** with category averages and interpretations
- **Task-specific results** for granular analysis

Example results structure:
```
mteb_results/
├── my-model/
│   ├── benchmark_summary.json      # Complete results
│   ├── benchmark_report.md         # Human-readable report
│   └── STS12.json                 # Individual task results
└── vanilla-baseline/
    ├── benchmark_summary.json
    └── benchmark_report.md
```

## Advanced Usage Examples

### Biomedical Search Engine

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# Load biomedical ontology model
model = SentenceTransformer("./biomedical-ontology-model")

# Encode a corpus of biomedical documents
documents = [
    "Cardiovascular disease results from atherosclerosis...",
    "Protein misfolding leads to neurodegeneration...",
    "Oncogenic mutations cause uncontrolled cell growth...",
]
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Search with ontology-aware embeddings
queries = ["heart problems", "alzheimer disease", "cancer mutations"]
query_embeddings = model.encode(queries, convert_to_tensor=True)

# Find most relevant documents
for query, query_embed in zip(queries, query_embeddings):
    results = semantic_search(query_embed, doc_embeddings, top_k=1)
    print(f"Query: {query}")
    print(f"Best match: {documents[results[0][0]['corpus_id']]}")
    print(f"Score: {results[0][0]['score']:.3f}\n")
```

### Concept Clustering with Ontology

```python
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./ontology-model")

# Biological concepts
concepts = [
    "cardiovascular disease", "heart failure", "myocardial infarction",
    "protein folding", "alzheimer disease", "neurodegeneration",
    "gene mutation", "cancer", "tumor suppressor"
]

# Get ontology-aware embeddings
embeddings = model.encode(concepts)

# Cluster with ontology knowledge
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Display clusters
for i, concept in enumerate(concepts):
    print(f"Cluster {clusters[i]}: {concept}")
```

### Model Evaluation and Comparison

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import Dataset

# Load models for comparison
standard_model = SentenceTransformer("all-MiniLM-L6-v2")
ontology_model = SentenceTransformer("./my-ontology-model")

# Create evaluation dataset
eval_data = Dataset.from_dict({
    "sentence1": ["heart disease", "protein folding"],
    "sentence2": ["cardiovascular problems", "protein misfolding"],
    "score": [0.9, 0.85]  # Human-annotated similarity
})

# Evaluate both models
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_data["sentence1"],
    sentences2=eval_data["sentence2"],
    scores=eval_data["score"],
    name="ontology-eval"
)

standard_score = evaluator(standard_model)
ontology_score = evaluator(ontology_model)

print(f"Standard model score: {standard_score}")
print(f"Ontology model score: {ontology_score}")
```

## Model Configuration Options

### Text Model Selection

```python
# Different base text models
base_models = [
    "all-MiniLM-L6-v2",      # Fast, 384 dims
    "all-mpnet-base-v2",     # Best quality, 768 dims
    "distilbert-base-nli-mean-tokens",  # 768 dims
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
]
```

### Ontology-Specific Tuning

```python
# Fine-tune for specific ontology domains
model = create_ontology_augmented_model(
    base_model='all-MiniLM-L6-v2',
    ontology_embeddings_file='go_embeddings.parquet',  # Gene Ontology
    fusion_method='attention',
    top_k_matches=5,        # More concept matches for GO
    structural_weight=0.4   # Higher weight for structured knowledge
)
```

### Performance Optimization

```python
# For production deployment
model = create_retrieval_model_with_ontology(
    ontology_embeddings_file='embeddings.parquet',
    fusion_method='concat',     # Fastest fusion
    projection_dim=128,         # Smaller common space
    query_model='distilbert-base-nli-mean-tokens',  # Faster queries
    document_model='all-MiniLM-L6-v2'               # Balanced docs
)
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch Errors**
   ```python
   # Ensure compatible fusion settings
   if fusion_method == 'gated':
       # Use projection_dim to align dimensions
       projection_dim = min(text_dim, structural_dim)
   ```

2. **Memory Issues with Large Ontologies**
   ```python
   # Reduce concept matching for large ontologies
   top_k_matches = 3  # Instead of 10
   ```

3. **Slow Inference**
   ```python
   # Use query/document architecture for retrieval
   # Use concat fusion for speed
   # Consider smaller base models
   ```

### Performance Tips

- **Development**: Use `fusion_method='concat'` for fastest prototyping
- **Production**: Use `fusion_method='gated'` for best quality
- **Large Scale**: Consider Query/Document architecture
- **Memory**: Set `top_k_matches=3` for large ontologies

## CLI Reference

### create_hf_model.py - Main CLI Tool

Complete command-line interface for creating HuggingFace models:

#### Available Commands

```bash
# See all commands
python create_hf_model.py --help

# Command-specific help
python create_hf_model.py e2e --help
python create_hf_model.py train --help
python create_hf_model.py create --help
```

#### Command Reference

**End-to-End Workflow**
```bash
python create_hf_model.py e2e OWL_FILE MODEL_NAME [options]

# Options:
  --output-dir DIR          # Output directory (default: ./hf_models)
  --base-model MODEL        # Base transformer (default: all-MiniLM-L6-v2)
  --fusion METHOD           # Fusion method (concat/weighted_avg/attention/gated)
  --epochs N                # Training epochs (default: 100)
  --skip-training           # Use existing embeddings
  --skip-testing            # Skip model validation
```

**Training Only**
```bash
python create_hf_model.py train OWL_FILE --output PARQUET_FILE [options]

# Options:
  --text-model MODEL        # Text model (default: all-MiniLM-L6-v2)
  --epochs N                # Training epochs (default: 100)
  --model-type TYPE         # GNN type (gcn/gat/rgcn)
  --hidden-dim N            # Hidden dimensions (default: 128)
  --out-dim N               # Output dimensions (default: 64)
  --loss-fn LOSS            # Loss function (triplet/contrastive/cosine)
```

**Model Creation Only**
```bash
python create_hf_model.py create EMBEDDINGS_FILE MODEL_NAME [options]

# Options:
  --output-dir DIR          # Output directory
  --base-model MODEL        # Base transformer model (auto-detected if not specified)
  --fusion METHOD           # Fusion method
  --no-validate             # Skip embeddings validation
```

**Model Testing**
```bash
python create_hf_model.py test MODEL_PATH [options]

# Options:
  --queries "query1" "query2"  # Custom test queries
```

**Validation & Upload Info**
```bash
# Validate embeddings file
python create_hf_model.py validate EMBEDDINGS_FILE

# Show upload instructions
python create_hf_model.py upload-info MODEL_PATH MODEL_NAME
```

### batch_hf_models.py - Batch Processing

Process multiple ontologies or configurations:

#### Batch Processing
```bash
python batch_hf_models.py process OWL_DIR OUTPUT_DIR [options]

# Options:
  --base-models MODEL1 MODEL2     # Multiple base models
  --fusion-methods METHOD1 METHOD2 # Multiple fusion methods
  --epochs N1 N2                  # Multiple epoch counts
  --max-workers N                 # Parallel processing
  --limit N                       # Limit number of files
  --force-retrain                 # Force retraining
```

#### Model Collections
```bash
# Create curated model collection
python batch_hf_models.py collection RESULTS_FILE --name COLLECTION_NAME

# Options:
  --criteria best_test/fastest/smallest  # Selection criteria
  --output-dir DIR                       # Output directory
```

#### Results Analysis
```bash
# Show batch processing summary
python batch_hf_models.py summary RESULTS_FILE
```

### 🧠 Smart Auto-Detection

The CLI automatically infers the base model from embeddings metadata, eliminating the need to remember which text model was used:

```bash
# ✅ Automatically detects all-MiniLM-L6-v2 from embeddings
python create_hf_model.py create embeddings.parquet my-model --fusion concat

# ⚠️ Warns about mismatches and uses the correct model
python create_hf_model.py create embeddings.parquet my-model \
  --base-model all-mpnet-base-v2
# Output: WARNING: Base model mismatch! Using detected model: all-MiniLM-L6-v2

# 🔍 View embeddings metadata
python create_hf_model.py validate embeddings.parquet
# Shows: Text model: all-MiniLM-L6-v2 (384 dims)
```

### Example Workflows

**Single Model Creation**
```bash
# Quick biomedical model (auto-detects everything)
python create_hf_model.py e2e biomedical.owl biomedical-embedder

# Advanced configuration
python create_hf_model.py e2e ontology.owl custom-model \
  --base-model all-mpnet-base-v2 \
  --fusion gated \
  --epochs 200 \
  --output-dir ./production_models
```

**Batch Processing**
```bash
# Process directory with multiple configurations
python batch_hf_models.py process owl_files/ ./batch_output \
  --base-models all-MiniLM-L6-v2 all-mpnet-base-v2 \
  --fusion-methods concat gated attention \
  --epochs 50 100 \
  --max-workers 4

# Create collection from results
python batch_hf_models.py collection ./batch_output/batch_results.json \
  --name "biomedical-collection" \
  --criteria best_test
```

**Custom Training Pipeline**
```bash
# Step 1: Custom training
python create_hf_model.py train ontology.owl \
  --output custom_embeddings.parquet \
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --epochs 150 \
  --model-type gat \
  --hidden-dim 256

# Step 2: Create multiple fusion variants
python create_hf_model.py create custom_embeddings.parquet model-concat --fusion concat
python create_hf_model.py create custom_embeddings.parquet model-gated --fusion gated
python create_hf_model.py create custom_embeddings.parquet model-attention --fusion attention

# Step 3: Test all variants
python create_hf_model.py test ./hf_models/model-concat
python create_hf_model.py test ./hf_models/model-gated
python create_hf_model.py test ./hf_models/model-attention
```

## Next Steps

1. **Start with CLI**: Use `create_hf_model.py e2e` for your first model
2. **Experiment with fusion**: Try different fusion methods for your domain
3. **Batch process**: Use `batch_hf_models.py` for multiple ontologies
4. **Create collections**: Curate your best models for sharing
5. **Upload to Hub**: Share successful models with the community
6. **Integrate in apps**: Use with existing sentence-transformers workflows

For more examples and advanced usage, see the `examples/` directory and the comprehensive test suite.