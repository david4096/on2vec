# Sentence Transformers Integration with on2vec

This guide shows how to create production-ready Sentence Transformers models that incorporate ontology knowledge from on2vec embeddings.

## Overview

The integration allows you to:
1. **Train ontology embeddings** using on2vec with text features
2. **Create custom Sentence Transformers models** that combine semantic text similarity with ontology structural knowledge
3. **Upload and share models** on Hugging Face Hub for community use
4. **Use models seamlessly** with the standard `sentence-transformers` library

## Quick Start

### 1. Generate Ontology Embeddings

First, train an on2vec model with text features enabled:

```bash
# Train text-augmented model on your ontology
uv run python main.py your_ontology.owl \
    --use_text_features \
    --text_model_type sentence_transformer \
    --text_model_name all-MiniLM-L6-v2 \
    --output embeddings.parquet \
    --epochs 100
```

This creates a parquet file containing:
- **Fused embeddings**: Combined text + structural knowledge
- **Text embeddings**: Pure semantic text features (384 dims)
- **Structural embeddings**: Pure ontology graph features (varies)
- **Metadata**: Model info, ontology details, text model used

### 2. Create Sentence Transformers Model

Use the CLI tool to create and test your model:

```bash
# Create and test both model types
uv run python create_ontology_sentence_transformer.py embeddings.parquet \
    --demo both \
    --save-model saved_models
```

### 3. Build HuggingFace Compatible Model

```python
from on2vec.sentence_transformer_hub import create_hf_model

# Create a proper HF sentence-transformers model
model = create_hf_model(
    ontology_embeddings_file="embeddings.parquet",
    model_name="my-ontology-model",
    fusion_method="concat"
)

# Save for HuggingFace Hub
model.save("./my-ontology-model")
```

### 4. Use with Standard sentence-transformers

```python
from sentence_transformers import SentenceTransformer

# Load your custom model
model = SentenceTransformer("./my-ontology-model")

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

## Next Steps

1. **Train your model**: Follow the quick start with your ontology
2. **Experiment with fusion methods**: Try different approaches for your domain
3. **Evaluate performance**: Compare against baseline models
4. **Share on Hub**: Upload successful models for community use
5. **Integrate in applications**: Use with existing sentence-transformers workflows

For more examples and advanced usage, see the `examples/` directory and the comprehensive test suite.