#!/usr/bin/env python3
"""
Model card generation for on2vec Sentence Transformer models.

This module creates comprehensive model cards describing how ontology-augmented
models were created, their base components, and usage instructions.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import platform

from .metadata_utils import get_base_model_from_embeddings, get_embedding_info


def get_system_info() -> Dict[str, str]:
    """Get system information for reproducibility."""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'created_at': datetime.now().isoformat()
    }


def detect_ontology_domain(ontology_file: str) -> str:
    """Attempt to detect domain from ontology filename or path."""
    filename = Path(ontology_file).name.lower()

    domain_keywords = {
        'biomedical': ['bio', 'medical', 'edam', 'gene', 'protein', 'disease', 'anatomy'],
        'legal': ['legal', 'law', 'court', 'statute', 'regulation'],
        'financial': ['financial', 'finance', 'bank', 'economic', 'accounting'],
        'scientific': ['scientific', 'research', 'academic', 'scholar'],
        'technical': ['technical', 'engineering', 'computer', 'software', 'tech'],
        'environmental': ['environmental', 'climate', 'ecology', 'green', 'sustainability'],
        'educational': ['education', 'school', 'university', 'learning', 'curriculum']
    }

    for domain, keywords in domain_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return domain

    return 'general'


def extract_complete_metadata(
    model_path: str,
    embeddings_file: Optional[str] = None,
    fusion_method: str = "concat"
) -> Dict[str, Any]:
    """Extract complete metadata from all available sources."""

    complete_metadata = {}

    # 1. Extract from model's on2vec_metadata.json if it exists
    model_metadata_path = Path(model_path) / "on2vec_metadata.json"
    if model_metadata_path.exists():
        try:
            with open(model_metadata_path, 'r') as f:
                model_metadata = json.load(f)
                complete_metadata['hf_model_metadata'] = model_metadata
        except Exception:
            pass

    # 2. Extract from embeddings file if available
    if embeddings_file and Path(embeddings_file).exists():
        try:
            from .metadata_utils import get_embedding_info
            embed_info = get_embedding_info(embeddings_file)
            complete_metadata['embedding_info'] = embed_info
        except Exception:
            pass

    # 3. Extract fusion method from model metadata or use provided
    if 'hf_model_metadata' in complete_metadata:
        detected_fusion = complete_metadata['hf_model_metadata'].get('fusion_method')
        if detected_fusion:
            fusion_method = detected_fusion

    complete_metadata['detected_fusion_method'] = fusion_method

    return complete_metadata


def generate_model_metadata(
    model_path: str,
    ontology_file: Optional[str] = None,
    embeddings_file: Optional[str] = None,
    fusion_method: str = "concat",
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate metadata for the model card using all available information."""

    # Extract complete metadata from all sources
    complete_meta = extract_complete_metadata(model_path, embeddings_file, fusion_method)

    metadata = {
        'library_name': 'sentence-transformers',
        'tags': [
            'sentence-transformers',
            'sentence-similarity',
            'feature-extraction',
            'ontology',
            'on2vec',
            'graph-neural-networks'
        ],
        'pipeline_tag': 'sentence-similarity',
        'license': 'apache-2.0'
    }

    # Base model detection from embeddings metadata
    base_model = None
    embedding_info = complete_meta.get('embedding_info', {})
    if embedding_info.get('base_model'):
        base_model = embedding_info['base_model']
        metadata['base_model'] = base_model
        # Add base model to tags for discoverability
        base_model_short = base_model.split('/')[-1] if '/' in base_model else base_model
        metadata['tags'].append(f'base-{base_model_short}')

    # Ontology domain detection - use actual ontology file from metadata if available
    actual_ontology_file = None
    if embedding_info.get('metadata', {}).get('source_ontology_file'):
        actual_ontology_file = embedding_info['metadata']['source_ontology_file']
    elif ontology_file:
        actual_ontology_file = ontology_file

    if actual_ontology_file:
        domain = detect_ontology_domain(actual_ontology_file)
        metadata['tags'].extend([domain, f'{domain}-ontology'])

    # Fusion method from detected metadata
    actual_fusion = complete_meta.get('detected_fusion_method', fusion_method)
    metadata['tags'].append(f'fusion-{actual_fusion}')

    # GNN model type from embeddings metadata
    if embedding_info.get('metadata', {}).get('model_config', {}).get('backbone_model'):
        gnn_model = embedding_info['metadata']['model_config']['backbone_model']
        metadata['tags'].append(f"gnn-{gnn_model}")
    elif training_config and training_config.get('model_type'):
        metadata['tags'].append(f"gnn-{training_config['model_type']}")

    # Add concept count for discoverability
    if embedding_info.get('num_embeddings'):
        concept_count = embedding_info['num_embeddings']
        if concept_count < 1000:
            metadata['tags'].append('small-ontology')
        elif concept_count < 10000:
            metadata['tags'].append('medium-ontology')
        else:
            metadata['tags'].append('large-ontology')

    return metadata


def generate_model_card_content(
    model_name: str,
    base_model: Optional[str] = None,
    ontology_file: Optional[str] = None,
    embeddings_file: Optional[str] = None,
    fusion_method: str = "concat",
    training_config: Optional[Dict[str, Any]] = None,
    benchmark_results: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None
) -> str:
    """Generate the main content of the model card using all available metadata."""

    # Extract complete metadata
    complete_meta = {}
    if model_path:
        complete_meta = extract_complete_metadata(model_path, embeddings_file, fusion_method)

    embedding_info = complete_meta.get('embedding_info', {})
    model_config = embedding_info.get('metadata', {}).get('model_config', {})
    alignment_info = embedding_info.get('metadata', {}).get('alignment_info', {})

    # Header
    content = f"""# {model_name}

This is a sentence-transformers model created with [on2vec](https://github.com/david4096/on2vec), which augments text embeddings with ontological knowledge using Graph Neural Networks.

## Model Details

"""

    # Base model info - prefer from metadata
    actual_base_model = embedding_info.get('base_model') or base_model
    if actual_base_model:
        content += f"- **Base Text Model**: {actual_base_model}\n"
        text_dim = embedding_info.get('text_dim')
        if text_dim:
            content += f"  - Text Embedding Dimension: {text_dim}\n"

    # Ontology info - prefer from metadata
    actual_ontology_file = None
    if embedding_info.get('metadata', {}).get('source_ontology_file'):
        actual_ontology_file = embedding_info['metadata']['source_ontology_file']
        # Also include the full path if available
        ontology_path = embedding_info.get('metadata', {}).get('source_ontology_path')
    elif ontology_file:
        actual_ontology_file = ontology_file

    if actual_ontology_file:
        ontology_name = Path(actual_ontology_file).name
        domain = detect_ontology_domain(actual_ontology_file)
        content += f"- **Ontology**: {ontology_name}\n"
        content += f"- **Domain**: {domain}\n"

        # Ontology stats from metadata
        total_concepts = embedding_info.get('num_embeddings')
        if total_concepts:
            content += f"- **Ontology Concepts**: {total_concepts:,}\n"

        if alignment_info:
            aligned_classes = alignment_info.get('aligned_classes')
            total_classes = alignment_info.get('total_classes')
            alignment_ratio = alignment_info.get('alignment_ratio', 0) * 100
            if aligned_classes and total_classes:
                content += f"- **Concept Alignment**: {aligned_classes:,}/{total_classes:,} ({alignment_ratio:.1f}%)\n"

    # Fusion method - prefer from metadata
    actual_fusion = complete_meta.get('detected_fusion_method', fusion_method)
    content += f"- **Fusion Method**: {actual_fusion}\n"

    # GNN Architecture from metadata
    if model_config:
        gnn_model = model_config.get('backbone_model')
        if gnn_model:
            content += f"- **GNN Architecture**: {gnn_model.upper()}\n"

        # Structural embeddings info
        structural_dim = model_config.get('structural_dim')
        if structural_dim:
            content += f"- **Structural Embedding Dimension**: {structural_dim}\n"

        # Final output dimensions
        out_dim = model_config.get('out_dim')
        if out_dim:
            content += f"- **Output Embedding Dimension**: {out_dim}\n"

        hidden_dim = model_config.get('hidden_dim')
        if hidden_dim:
            content += f"- **Hidden Dimensions**: {hidden_dim}\n"

        dropout = model_config.get('dropout')
        if dropout is not None:
            content += f"- **Dropout**: {dropout}\n"

    # Training details from provided config (fallback if not in metadata)
    if training_config and not model_config:
        content += f"- **Training Epochs**: {training_config.get('epochs', 'N/A')}\n"
        content += f"- **Loss Function**: {training_config.get('loss_fn', 'N/A')}\n"

    # Generation timestamp from metadata
    generation_timestamp = embedding_info.get('metadata', {}).get('generation_timestamp')
    if generation_timestamp:
        # Parse ISO timestamp and format nicely
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(generation_timestamp.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%Y-%m-%d')
            content += f"- **Training Date**: {formatted_date}\n"
        except:
            content += f"- **Training Date**: {generation_timestamp[:10]}\n"
    else:
        system_info = get_system_info()
        content += f"- **Created**: {system_info['created_at'][:10]}\n"

    # on2vec version
    on2vec_version = embedding_info.get('metadata', {}).get('on2vec_version', '0.1.0')
    content += f"- **on2vec Version**: {on2vec_version}\n"

    # Source ontology file size
    source_file_size = embedding_info.get('metadata', {}).get('source_file_size')
    if source_file_size:
        size_mb = source_file_size / (1024 * 1024)
        content += f"- **Source Ontology Size**: {size_mb:.1f} MB\n"

    # Model file size (if model_path is available)
    if model_path:
        try:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                total_size = sum(f.stat().st_size for f in model_path_obj.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                content += f"- **Model Size**: {size_mb:.1f} MB\n"
        except:
            pass

    content += f"- **Library**: on2vec + sentence-transformers\n\n"

    # Technical Architecture Details
    content += """## Technical Architecture

This model uses a multi-stage architecture:

1. **Text Encoding**: Input text is encoded using the base sentence-transformer model
2. **Ontological Embedding**: Pre-trained GNN embeddings capture structural relationships
3. **Fusion Layer**: """

    if actual_fusion == 'attention':
        content += "Attention mechanism learns to weight text vs ontological information\n"
    elif actual_fusion == 'gated':
        content += "Gated fusion learns when to rely on ontological vs textual knowledge\n"
    elif actual_fusion == 'weighted_avg':
        content += "Weighted average with learnable combination weights\n"
    else:
        content += "Simple concatenation of text and ontological embeddings\n"

    if model_config:
        content += f"""
**Embedding Flow:**
- Text: {embedding_info.get('text_dim', 'N/A')} dimensions → {model_config.get('hidden_dim', 'N/A')} hidden → {model_config.get('out_dim', 'N/A')} output
- Structure: {model_config.get('structural_dim', 'N/A')} concepts → GNN → {model_config.get('out_dim', 'N/A')} output
- Fusion: {actual_fusion} → Final embedding

"""

    # Description section
    content += """## How It Works

This model combines:
1. **Text Embeddings**: Generated using the base sentence-transformer model
2. **Ontological Embeddings**: Created by training Graph Neural Networks on OWL ontology structure
3. **Fusion Layer**: Combines both embedding types using the specified fusion method

The ontological knowledge helps the model better understand domain-specific relationships and concepts.

## Usage

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('""" + model_name + """')

# Generate embeddings
sentences = ['Example sentence 1', 'Example sentence 2']
embeddings = model.encode(sentences)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
```

"""

    # Fusion method explanation
    fusion_descriptions = {
        'concat': 'Simple concatenation of text and ontology embeddings',
        'weighted_avg': 'Weighted average of text and ontology embeddings with learnable weights',
        'attention': 'Attention-based fusion that learns to focus on relevant embedding components',
        'gated': 'Gated fusion mechanism that learns when to use ontological vs textual information'
    }

    if fusion_method in fusion_descriptions:
        content += f"""## Fusion Method: {fusion_method}

{fusion_descriptions[fusion_method]}

"""

    # Benchmark results
    if benchmark_results:
        content += """## Performance

"""
        if 'averages' in benchmark_results:
            content += "### MTEB Benchmark Results\n\n"
            content += "| Task Type | Score | Count |\n"
            content += "|-----------|-------|-------|\n"

            for task_type, data in benchmark_results['averages'].items():
                content += f"| {task_type} | {data['mean']:.3f} | {data['count']} |\n"
            content += "\n"

    # Training details
    content += """## Training Process

This model was created using the on2vec pipeline:

1. **Ontology Processing**: The OWL ontology was converted to a graph structure
2. **GNN Training**: Graph Neural Networks were trained to learn ontological relationships
3. **Text Integration**: Base model text embeddings were combined with ontological embeddings
4. **Fusion Training**: The fusion layer was trained to optimally combine both embedding types

"""

    # Usage recommendations
    domain = detect_ontology_domain(ontology_file) if ontology_file else 'general'

    content += f"""## Intended Use

This model is particularly effective for:
- {domain.title()} domain text processing
- Tasks requiring understanding of domain-specific relationships
- Semantic similarity in specialized domains
- Classification tasks with domain knowledge requirements

"""

    # Limitations
    content += """## Limitations

- Performance may vary on domains different from the training ontology
- Ontological knowledge is limited to concepts present in the source OWL file
- May have higher computational requirements than vanilla text models

## Citation

If you use this model, please cite the on2vec framework:

```bibtex
@software{on2vec,
  title={on2vec: Ontology Embeddings with Graph Neural Networks},
  author={David Steinberg},
  url={https://github.com/david4096/on2vec},
  year={2024}
}
```

---

Created with [on2vec](https://github.com/david4096/on2vec) 🧬→🤖
"""

    return content


def create_model_card(
    model_path: str,
    model_name: Optional[str] = None,
    ontology_file: Optional[str] = None,
    embeddings_file: Optional[str] = None,
    fusion_method: str = "concat",
    training_config: Optional[Dict[str, Any]] = None,
    benchmark_results: Optional[Dict[str, Any]] = None
) -> str:
    """Create a complete model card for the model."""

    model_path = Path(model_path)
    if not model_name:
        model_name = model_path.name

    # Generate metadata using all available information
    metadata = generate_model_metadata(
        str(model_path),
        ontology_file,
        embeddings_file,
        fusion_method,
        training_config
    )

    # Generate content using all available information
    content = generate_model_card_content(
        model_name,
        metadata.get('base_model'),
        ontology_file,
        embeddings_file,
        fusion_method,
        training_config,
        benchmark_results,
        model_path=str(model_path)  # Pass model path for metadata extraction
    )

    # Combine YAML frontmatter and content
    yaml_front = "---\n" + yaml.dump(metadata, default_flow_style=False) + "---\n\n"
    full_card = yaml_front + content

    # Save to model directory
    card_path = model_path / "README.md"
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(full_card)

    print(f"📄 Model card created: {card_path}")
    return str(card_path)


def update_model_card_with_benchmarks(
    model_path: str,
    benchmark_results: Dict[str, Any]
) -> str:
    """Update an existing model card with benchmark results."""

    model_path = Path(model_path)
    card_path = model_path / "README.md"

    if not card_path.exists():
        print(f"⚠️  No existing model card found at {card_path}")
        return ""

    # Read existing card
    with open(card_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract YAML frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1]
            markdown_content = parts[2]

            # Parse existing metadata
            try:
                metadata = yaml.safe_load(yaml_content)
            except:
                metadata = {}

            # Add benchmark info to metadata if not present
            if 'datasets' not in metadata:
                metadata['datasets'] = ['mteb']

            # Regenerate content with benchmarks
            new_markdown = generate_model_card_content(
                model_path.name,
                metadata.get('base_model'),
                None,  # ontology_file not available
                None,  # embeddings_file not available
                'concat',  # default fusion method
                None,  # training_config not available
                benchmark_results
            )

            # Combine updated parts
            new_content = "---\n" + yaml.dump(metadata, default_flow_style=False) + "---\n\n" + new_markdown

            # Save updated card
            with open(card_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"📊 Model card updated with benchmark results: {card_path}")
            return str(card_path)

    print(f"⚠️  Could not parse existing model card format")
    return ""


def create_upload_instructions(
    model_path: str,
    model_name: str,
    hub_name: Optional[str] = None
) -> str:
    """Create instructions for uploading to HuggingFace Hub."""

    if not hub_name:
        hub_name = f"your-username/{model_name}"

    instructions = f"""# HuggingFace Hub Upload Instructions

## Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login
```

## Upload Model

```python
from sentence_transformers import SentenceTransformer

# Load your model
model = SentenceTransformer('{model_path}')

# Push to Hub
model.push_to_hub('{hub_name}')
```

## Alternative: Manual Upload

```bash
# Clone/create repository
git clone https://huggingface.co/{hub_name}
cd {model_name}

# Copy model files
cp -r {model_path}/* ./

# Commit and push
git add .
git commit -m "Add {model_name} model"
git push
```

## After Upload

Your model will be available at:
- **Model Page**: https://huggingface.co/{hub_name}
- **Usage**: `SentenceTransformer('{hub_name}')`

## Update Model Card

Edit the README.md on the Hub to:
- Add model description
- Include benchmark results
- Provide usage examples
- Add citations

The model card has been pre-generated with comprehensive information about the on2vec training process.
"""

    # Save instructions
    model_path = Path(model_path)
    instructions_path = model_path / "UPLOAD_INSTRUCTIONS.md"

    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)

    print(f"📤 Upload instructions created: {instructions_path}")
    return str(instructions_path)