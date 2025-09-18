# Installation and CLI Test Guide

## 🧪 Testing the Refactored on2vec Package

### 1. Install from Local Development

```bash
# Install in development mode
pip install -e .

# Or install with all features
pip install -e .[all]
```

### 2. Test Basic Commands

```bash
# Show help
on2vec --help

# Show version
on2vec --version

# Help for specific commands
on2vec hf --help
on2vec benchmark --help
```

### 3. Test End-to-End Workflow

```bash
# Create a complete HuggingFace model (requires OWL file)
on2vec hf your-ontology.owl my-model

# Test the created model
on2vec hf-test ./hf_models/my-model

# Inspect the model
on2vec inspect ./hf_models/my-model

# Compare with vanilla models
on2vec compare ./hf_models/my-model --detailed
```

### 4. Test Core Functionality

```bash
# Train a model
on2vec train ontology.owl --output model.pt --model-type gcn

# Generate embeddings
on2vec embed model.pt ontology.owl --output embeddings.parquet

# Visualize embeddings
on2vec visualize embeddings.parquet --output viz.png

# Convert formats
on2vec convert embeddings.parquet embeddings.csv
```

### 5. Test Benchmarking (requires MTEB)

```bash
# Install benchmarking dependencies
pip install on2vec[benchmark]

# Quick benchmark
on2vec benchmark ./hf_models/my-model --quick

# Full MTEB benchmark
on2vec benchmark ./hf_models/my-model
```

### 6. Python API Test

```python
# Test that Python API still works
from sentence_transformers import SentenceTransformer
from on2vec import train_ontology_embeddings

# Use on2vec Python functions
result = train_ontology_embeddings(
    owl_file="ontology.owl",
    model_output="model.pt",
    model_type="gcn"
)

# Use created HuggingFace models
model = SentenceTransformer('./hf_models/my-model')
embeddings = model.encode(['test sentence 1', 'test sentence 2'])
print(f"Generated embeddings: {embeddings.shape}")
```

## ✅ Expected Results

- All CLI commands should work without errors
- Models should be created with comprehensive documentation
- Python API should remain fully functional
- Installation should be clean with proper dependency resolution

## 🐛 Troubleshooting

### Command Not Found
If `on2vec` command is not found after installation:
```bash
# Check if installed
pip list | grep on2vec

# Reinstall
pip install -e . --force-reinstall
```

### Missing Dependencies
For MTEB benchmarking:
```bash
pip install on2vec[benchmark]
```

For Jupyter notebooks:
```bash
pip install on2vec[notebook]
```

For everything:
```bash
pip install on2vec[all]
```

## 📦 Distribution Test

To test as if installing from PyPI:

```bash
# Build the package
python -m build

# Install from built wheel
pip install dist/on2vec-0.1.0-py3-none-any.whl

# Test CLI
on2vec --help
```