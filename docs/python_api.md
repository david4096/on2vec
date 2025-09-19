# on2vec Python API Guide

This guide covers using on2vec programmatically through its Python API instead of the CLI. All CLI functionality is accessible through the Python API with full feature parity.

## Installation

```bash
# Basic installation
pip install on2vec

# With benchmarking support
pip install on2vec[benchmark]

# All features including development dependencies
pip install on2vec[all]
```

## Core Imports

```python
import on2vec
from on2vec import (
    # Core models
    OntologyGNN, MultiRelationOntologyGNN, HeterogeneousOntologyGNN,

    # Training and embedding functions
    train_ontology_embeddings, embed_ontology_with_model,

    # Graph building
    build_graph_from_owl, build_multi_relation_graph_from_owl,

    # I/O operations
    save_embeddings_to_parquet, load_embeddings_from_parquet,

    # Visualization
    plot_umap_2d, create_visualization_comparison,

    # Evaluation
    EmbeddingEvaluator, evaluate_embeddings
)
```

## 1. Basic Training and Embedding Workflow

### Simple GNN Training

```python
from on2vec.workflows import train_and_embed_workflow

# Complete workflow: train + embed
result = train_and_embed_workflow(
    owl_file="ontology.owl",
    model_type="gcn",  # or "gat", "rgcn", "heterogeneous"
    hidden_dim=128,
    out_dim=64,
    epochs=100,
    output="embeddings.parquet",
    model_output="model.pt",
    loss_fn="triplet"  # or "contrastive", "cosine", "cross_entropy"
)

print(f"Model saved: {result['model_path']}")
print(f"Embeddings saved: {result['embeddings_path']}")
print(f"Generated {result['num_embeddings']} embeddings")
```

### Training with Text Features

```python
# Train with text features for richer embeddings
result = train_and_embed_workflow(
    owl_file="ontology.owl",
    model_type="gcn",
    hidden_dim=128,
    out_dim=64,
    epochs=100,
    output="text_embeddings.parquet",
    model_output="text_model.pt",
    use_text_features=True,
    text_model_name="all-MiniLM-L6-v2",  # or other sentence transformer
    fusion_method="concat"  # or "weighted_avg", "attention"
)
```

### Two-Phase Training (Train Once, Embed Multiple)

```python
from on2vec.workflows import train_model_only, embed_with_trained_model

# Phase 1: Train model only
training_result = train_model_only(
    owl_file="ontology1.owl",
    model_output="shared_model.pt",
    model_type="gcn",
    hidden_dim=128,
    out_dim=64,
    epochs=100
)

# Phase 2: Generate embeddings for different ontologies
embedding_result1 = embed_with_trained_model(
    model_path="shared_model.pt",
    owl_file="ontology1.owl",
    output_file="embeddings1.parquet"
)

embedding_result2 = embed_with_trained_model(
    model_path="shared_model.pt",
    owl_file="ontology2.owl",
    output_file="embeddings2.parquet"
)
```

## 2. HuggingFace Integration

### End-to-End HuggingFace Model Creation

```python
from on2vec.huggingface_workflows import end_to_end_workflow

# Complete workflow: train → create HF model → test
success = end_to_end_workflow(
    owl_file="ontology.owl",
    model_name="my-ontology-model",
    output_dir="./hf_models",
    base_model="all-MiniLM-L6-v2",
    fusion_method="concat",
    training_config={
        'epochs': 100,
        'model_type': 'gcn',
        'hidden_dim': 128,
        'out_dim': 64,
        'loss_fn': 'triplet'
    },
    model_details={
        'author': 'Your Name',
        'description': 'Custom ontology model',
        'domain': 'biomedicine'
    }
)
```

### Step-by-Step HuggingFace Workflow

```python
from on2vec.huggingface_workflows import (
    train_ontology_with_text, create_hf_model, validate_hf_model
)

# Step 1: Train with text features
success = train_ontology_with_text(
    owl_file="ontology.owl",
    output_file="embeddings.parquet",
    text_model="all-MiniLM-L6-v2",
    epochs=100
)

# Step 2: Create HuggingFace model
model_path = create_hf_model(
    embeddings_file="embeddings.parquet",
    model_name="my-model",
    output_dir="./hf_models",
    fusion_method="concat"
)

# Step 3: Test the model
success = validate_hf_model(model_path)
```

### Upload to HuggingFace Hub

```python
from on2vec.huggingface_workflows import upload_to_hf_hub

# Upload model to HuggingFace Hub
success = upload_to_hf_hub(
    model_path="./hf_models/my-model",
    hub_name="your-username/my-ontology-model",
    private=False,
    commit_message="Initial release of ontology model"
)
```

## 3. Advanced Training Options

### Multi-Relation Graphs

```python
# Use all object properties as relations (for RGCN models)
result = train_and_embed_workflow(
    owl_file="ontology.owl",
    model_type="rgcn",  # Requires multi-relation
    use_multi_relation=True,
    num_bases=4,  # For RGCN regularization
    hidden_dim=128,
    out_dim=64
)
```

### Custom Loss Functions and Hyperparameters

```python
result = train_and_embed_workflow(
    owl_file="ontology.owl",
    model_type="gat",  # Graph Attention Network
    hidden_dim=256,
    out_dim=128,
    epochs=200,
    loss_fn="cross_entropy",
    learning_rate=0.001,
    dropout=0.1
)
```

## 4. Working with Embeddings

### Loading and Inspecting Embeddings

```python
from on2vec.io import (
    load_embeddings_from_parquet,
    inspect_parquet_metadata,
    load_embeddings_as_dataframe
)

# Load embeddings
embeddings_dict = load_embeddings_from_parquet("embeddings.parquet")
node_ids = embeddings_dict['node_ids']
vectors = embeddings_dict['embeddings']

# Inspect metadata
inspect_parquet_metadata("embeddings.parquet")

# Load as DataFrame for analysis
df = load_embeddings_as_dataframe("embeddings.parquet")
print(df.head())
```

### Embedding Operations

```python
from on2vec.io import (
    get_embedding_vector,
    add_embedding_vectors,
    subtract_embedding_vectors
)

# Get specific embedding
concept_embedding = get_embedding_vector(
    "embeddings.parquet",
    "http://example.org/ontology#Disease"
)

# Embedding arithmetic
result = add_embedding_vectors(
    "embeddings.parquet",
    ["concept1", "concept2"],
    weights=[0.7, 0.3]
)

difference = subtract_embedding_vectors(
    "embeddings.parquet",
    "concept1",
    "concept2"
)
```

## 5. Visualization

### UMAP Visualization

```python
from on2vec.visualization import plot_umap_2d, create_visualization_comparison

# Create UMAP plot
plot_path = plot_umap_2d(
    embeddings_file="embeddings.parquet",
    output_file="visualization.png",
    n_neighbors=15,
    min_dist=0.1,
    title="Ontology Concepts"
)

# Compare multiple embeddings
comparison_path = create_visualization_comparison(
    embeddings_files=["embeddings1.parquet", "embeddings2.parquet"],
    labels=["Model 1", "Model 2"],
    output_file="comparison.png"
)
```

### Other Visualization Methods

```python
from on2vec.visualization import plot_pca_2d, plot_tsne_2d

# PCA visualization
plot_pca_2d("embeddings.parquet", "pca_plot.png")

# t-SNE visualization
plot_tsne_2d("embeddings.parquet", "tsne_plot.png", perplexity=30)
```

## 6. Evaluation and Benchmarking

### Comprehensive Evaluation

```python
from on2vec.evaluation import EmbeddingEvaluator

# Create evaluator
evaluator = EmbeddingEvaluator(
    embeddings_file="embeddings.parquet",
    ontology_file="ontology.owl"  # Optional, enables structural evaluation
)

# Run comprehensive evaluation
results = evaluator.evaluate_all()

# Individual evaluation components
intrinsic_results = evaluator.evaluate_intrinsic()
extrinsic_results = evaluator.evaluate_extrinsic()
ontology_results = evaluator.evaluate_ontology_specific()
```

### Batch Evaluation

```python
from on2vec.evaluation import create_evaluation_benchmark

# Evaluate multiple embedding files
results = create_evaluation_benchmark(
    embeddings_files=[
        "model1_embeddings.parquet",
        "model2_embeddings.parquet",
        "model3_embeddings.parquet"
    ],
    ontology_files=[
        "ontology.owl",
        "ontology.owl",
        "ontology.owl"
    ],
    output_dir="evaluation_results"
)
```

### MTEB Benchmarking

```python
from mteb_benchmarks.benchmark_runner import run_mteb_evaluation

# Run MTEB benchmarks on HuggingFace model
results = run_mteb_evaluation(
    model_path="./hf_models/my-model",
    output_dir="./mteb_results",
    tasks=["STS", "Classification"],  # Specific task types
    batch_size=32
)
```

## 7. Low-Level API Usage

### Manual Graph Building

```python
from on2vec.ontology import build_graph_from_owl, build_multi_relation_graph_from_owl
import torch_geometric

# Build simple graph (subclass relations only)
data = build_graph_from_owl("ontology.owl")
print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")

# Build multi-relation graph
multi_data = build_multi_relation_graph_from_owl("ontology.owl")
print(f"Relations: {len(multi_data.edge_types)}")
```

### Manual Model Training

```python
from on2vec.models import OntologyGNN
from on2vec.training import train_model
from on2vec.loss_functions import TripletLoss
import torch

# Create model
model = OntologyGNN(
    input_dim=64,  # Node feature dimensions
    hidden_dim=128,
    output_dim=64,
    model_type="gcn"
)

# Create loss function
loss_fn = TripletLoss(margin=1.0)

# Load graph data
data = build_graph_from_owl("ontology.owl")

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    embeddings = model(data.x, data.edge_index)

    # Compute loss (simplified)
    loss = loss_fn(embeddings, data.edge_index)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Custom Text Integration

```python
from on2vec.sentence_transformer_integration import create_ontology_augmented_model

# Create custom ontology-augmented model
model = create_ontology_augmented_model(
    base_model='all-MiniLM-L6-v2',
    ontology_embeddings_file='text_embeddings.parquet',
    fusion_method='attention',
    top_k_matches=5,
    structural_weight=0.4
)

# Use for similarity tasks
queries = ["heart disease", "cardiovascular problems", "cardiac arrest"]
embeddings = model(queries)['sentence_embedding']
```

## 8. Batch Processing

### Batch Training Multiple Ontologies

```python
from batch_hf_models import batch_process_ontologies

# Process multiple OWL files
results = batch_process_ontologies(
    owl_directory="owl_files/",
    output_directory="batch_output/",
    base_models=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    fusion_methods=["concat", "attention"],
    epochs_list=[100, 200],
    max_workers=4,
    training_config={
        'model_types': ['gcn', 'gat'],
        'hidden_dims': [128],
        'out_dims': [64],
        'loss_fns': ['triplet']
    }
)
```

## 9. Model Management and Conversion

### Format Conversion

```python
from on2vec.io import convert_parquet_to_csv

# Convert embeddings to CSV
csv_file = convert_parquet_to_csv(
    "embeddings.parquet",
    "embeddings.csv"
)
```

### Model Inspection

```python
from on2vec.io import inspect_parquet_metadata
import json

# Inspect embedding file metadata
inspect_parquet_metadata("embeddings.parquet")

# For HuggingFace models, check config
with open("./hf_models/my-model/config.json", 'r') as f:
    config = json.load(f)
    print(f"Model architecture: {config['architectures']}")

# Check on2vec-specific metadata
with open("./hf_models/my-model/on2vec_metadata.json", 'r') as f:
    metadata = json.load(f)
    print(f"Base model: {metadata['base_model']}")
    print(f"Fusion method: {metadata['fusion_method']}")
```

## 10. Advanced Configuration

### Custom Training Configuration

```python
training_config = {
    'epochs': 150,
    'model_type': 'gat',  # Graph Attention Networks
    'hidden_dim': 256,
    'out_dim': 128,
    'loss_fn': 'cross_entropy',
    'use_multi_relation': True,
    'learning_rate': 0.001,
    'dropout': 0.1,
    'text_model': 'sentence-transformers/all-mpnet-base-v2'
}

# Use in workflows
result = train_and_embed_workflow(
    owl_file="complex_ontology.owl",
    **training_config
)
```

### Model Details for HuggingFace

```python
model_details = {
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'description': 'Biomedical ontology embeddings with graph attention',
    'domain': 'biomedicine',
    'license': 'apache-2.0',
    'tags': ['ontology', 'biomedicine', 'graph-neural-networks']
}

# Create model with metadata
model_path = create_hf_model(
    embeddings_file="embeddings.parquet",
    model_name="biomedical-ontology-gat",
    model_details=model_details
)
```

## Error Handling and Best Practices

### Robust Training with Error Handling

```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Validate inputs
    owl_path = Path("ontology.owl")
    if not owl_path.exists():
        raise FileNotFoundError(f"OWL file not found: {owl_path}")

    # Train with error handling
    result = train_and_embed_workflow(
        owl_file=str(owl_path),
        model_type="gcn",
        hidden_dim=128,
        out_dim=64,
        epochs=100,
        output="embeddings.parquet"
    )

    logger.info(f"Training successful: {result['num_embeddings']} embeddings generated")

except Exception as e:
    logger.error(f"Training failed: {e}")
    raise
```

### Memory Management for Large Ontologies

```python
import torch
import gc

# For large ontologies, consider these settings
torch.cuda.empty_cache()  # If using GPU
gc.collect()  # Clean up Python objects

# Use smaller batch sizes or dimensions for large graphs
result = train_and_embed_workflow(
    owl_file="large_ontology.owl",
    hidden_dim=64,  # Smaller dimensions
    out_dim=32,
    epochs=50,  # Fewer epochs
    learning_rate=0.001  # Smaller learning rate
)
```

This comprehensive guide covers all major functionality available through the on2vec Python API. All CLI commands have corresponding Python functions, allowing for flexible programmatic usage and integration into larger pipelines.