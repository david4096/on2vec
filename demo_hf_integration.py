#!/usr/bin/env python3
"""
Demo script showing how to create and use HuggingFace compatible
ontology-augmented Sentence Transformer models.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from on2vec.sentence_transformer_hub import (
    create_hf_sentence_transformer,
    create_and_save_hf_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_model_creation(embeddings_file: str):
    """Demonstrate creating a HuggingFace compatible model."""
    print("\n" + "="*60)
    print("🚀 Creating HuggingFace Compatible Model")
    print("="*60)

    # Create the model
    model = create_hf_sentence_transformer(
        ontology_embeddings_file=embeddings_file,
        base_model='all-MiniLM-L6-v2',
        fusion_method='concat',
        top_k_matches=3,
        model_name="biomedical-ontology-embedder"
    )

    print(f"✅ Model created with {len(model._modules)} modules:")
    for i, module in enumerate(model._modules):
        print(f"  {i+1}. {module.__class__.__name__}")

    print(f"📐 Output dimensions: {model.get_sentence_embedding_dimension()}")

    return model


def demo_standard_usage(model):
    """Demonstrate usage with standard sentence-transformers API."""
    print("\n" + "="*60)
    print("🧪 Standard Sentence-Transformers Usage")
    print("="*60)

    # Test sentences
    sentences = [
        "heart disease and cardiovascular problems",
        "genetic mutations causing cancer",
        "protein folding and misfolding disorders",
        "neurodegenerative brain diseases",
        "inflammatory immune responses",
        "weather patterns and climate change"  # Non-biomedical for comparison
    ]

    print("Encoding sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True)

    print(f"✅ Generated embeddings: {embeddings.shape}")

    # Compute similarities
    from sentence_transformers.util import cos_sim
    similarities = cos_sim(embeddings, embeddings)

    print("\n📊 Similarity Matrix (top triangle):")
    print("-" * 60)
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = similarities[i][j].item()
            s1 = sentences[i][:25] + "..." if len(sentences[i]) > 25 else sentences[i]
            s2 = sentences[j][:25] + "..." if len(sentences[j]) > 25 else sentences[j]
            print(f"{s1:28} <-> {s2:28}: {sim:.3f}")


def demo_semantic_search(model):
    """Demonstrate semantic search capabilities."""
    print("\n" + "="*60)
    print("🔍 Semantic Search Demo")
    print("="*60)

    # Document corpus
    documents = [
        "Cardiovascular disease results from atherosclerosis, hypertension, and cardiac arrhythmias affecting heart function.",
        "Oncogenic mutations in tumor suppressor genes like p53 and BRCA1 lead to uncontrolled cell proliferation.",
        "Protein misfolding causes aggregation and cellular toxicity in neurodegenerative diseases like Alzheimer's.",
        "Neurodegeneration involves progressive loss of neurons and cognitive decline in aging populations.",
        "Autoimmune disorders result from dysregulated immune responses targeting healthy tissue.",
        "Machine learning algorithms optimize objective functions using gradient-based optimization techniques.",
        "Economic markets fluctuate based on supply, demand, and investor sentiment in financial systems.",
        "Climate change affects global temperature patterns through greenhouse gas emissions."
    ]

    # Queries
    queries = [
        "heart problems",
        "cancer genetics",
        "alzheimer disease",
        "artificial intelligence"
    ]

    print("🗂️  Encoding document corpus...")
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    print("❓ Processing queries...")
    query_embeddings = model.encode(queries, convert_to_tensor=True)

    # Perform semantic search
    from sentence_transformers.util import semantic_search

    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 50)

        query_embed = model.encode([query], convert_to_tensor=True)
        results = semantic_search(query_embed, doc_embeddings, top_k=3)

        for rank, result in enumerate(results[0], 1):
            doc_idx = result['corpus_id']
            score = result['score']
            doc = documents[doc_idx]
            doc_preview = doc[:70] + "..." if len(doc) > 70 else doc
            print(f"  {rank}. Score: {score:.3f}")
            print(f"     {doc_preview}")


def demo_model_comparison(embeddings_file: str):
    """Compare ontology-augmented vs standard models."""
    print("\n" + "="*60)
    print("⚖️  Model Comparison")
    print("="*60)

    from sentence_transformers import SentenceTransformer

    # Load standard model
    print("Loading standard model...")
    standard_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create ontology-augmented model
    print("Creating ontology-augmented model...")
    ontology_model = create_hf_sentence_transformer(
        ontology_embeddings_file=embeddings_file,
        fusion_method='concat'
    )

    # Test sentences (biomedical domain)
    test_sentences = [
        "myocardial infarction",
        "heart attack",
        "cardiac arrest",
        "protein aggregation",
        "amyloid plaques"
    ]

    print("\n📏 Comparing embeddings...")
    standard_embeds = standard_model.encode(test_sentences)
    ontology_embeds = ontology_model.encode(test_sentences)

    print(f"Standard model dims: {standard_embeds.shape[1]}")
    print(f"Ontology model dims: {ontology_embeds.shape[1]}")

    # Compare similarities for biomedical terms
    from sentence_transformers.util import cos_sim

    std_sim = cos_sim(standard_embeds[0:1], standard_embeds[1:2]).item()  # myocardial infarction vs heart attack
    ont_sim = cos_sim(ontology_embeds[0:1], ontology_embeds[1:2]).item()

    print(f"\n🔬 Similarity: 'myocardial infarction' <-> 'heart attack'")
    print(f"Standard model:  {std_sim:.3f}")
    print(f"Ontology model:  {ont_sim:.3f}")
    print(f"Improvement:     {ont_sim - std_sim:+.3f}")


def demo_save_and_load(embeddings_file: str):
    """Demonstrate saving and loading models."""
    print("\n" + "="*60)
    print("💾 Save and Load Demo")
    print("="*60)

    # Create and save model
    model_name = "demo-biomedical-embedder"
    output_dir = "./hf_models"

    print(f"Creating and saving model: {model_name}")
    model_path = create_and_save_hf_model(
        ontology_embeddings_file=embeddings_file,
        model_name=model_name,
        output_dir=output_dir,
        fusion_method='concat'
    )

    print(f"✅ Model saved to: {model_path}")

    # Load with standard sentence-transformers
    print("\n📥 Loading with SentenceTransformer...")
    from sentence_transformers import SentenceTransformer

    loaded_model = SentenceTransformer(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"📐 Dimensions: {loaded_model.get_sentence_embedding_dimension()}")

    # Test consistency
    test_sentences = ["heart disease", "protein folding"]

    # Create fresh model for comparison
    fresh_model = create_hf_sentence_transformer(
        ontology_embeddings_file=embeddings_file,
        fusion_method='concat'
    )

    fresh_embeds = fresh_model.encode(test_sentences)
    loaded_embeds = loaded_model.encode(test_sentences)

    # Check consistency
    if np.allclose(fresh_embeds, loaded_embeds, atol=1e-5):
        print("✅ Save/load consistency verified!")
    else:
        print("❌ Warning: Save/load inconsistency detected")

    return model_path


def demo_hub_preparation(model_path: str):
    """Show how to prepare model for HuggingFace Hub upload."""
    print("\n" + "="*60)
    print("🌐 HuggingFace Hub Preparation")
    print("="*60)

    model_path = Path(model_path)

    # List all files that would be uploaded
    print("📁 Files ready for Hub upload:")
    for file in model_path.rglob("*"):
        if file.is_file():
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"  {file.name:<25} {size:>6.1f} MB")

    # Show upload command
    print(f"\n🚀 To upload to HuggingFace Hub:")
    print(f"   pip install huggingface_hub")
    print(f"   huggingface-cli login")
    print(f"   python -c \"")
    print(f"   from sentence_transformers import SentenceTransformer")
    print(f"   model = SentenceTransformer('{model_path}')")
    print(f"   model.push_to_hub('your-username/{model_path.name}')")
    print(f"   \"")

    print(f"\n📖 After upload, users can access with:")
    print(f"   from sentence_transformers import SentenceTransformer")
    print(f"   model = SentenceTransformer('your-username/{model_path.name}')")


def main():
    parser = argparse.ArgumentParser(
        description="Demo HuggingFace integration for ontology-augmented Sentence Transformers"
    )
    parser.add_argument(
        'embeddings_file',
        help='Path to on2vec generated parquet file (with text + structural embeddings)'
    )
    parser.add_argument(
        '--demo',
        choices=['create', 'usage', 'search', 'compare', 'save', 'hub', 'all'],
        default='all',
        help='Which demo to run'
    )

    args = parser.parse_args()

    # Validate embeddings file
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        print("\n🔧 To create embeddings file:")
        print("python main.py ontology.owl --use_text_features --output embeddings.parquet")
        return 1

    print("🧬 HuggingFace Sentence Transformers Integration Demo")
    print("=" * 60)

    try:
        model = None
        model_path = None

        if args.demo in ['create', 'all']:
            model = demo_model_creation(args.embeddings_file)

        if args.demo in ['usage', 'all']:
            if model is None:
                model = create_hf_sentence_transformer(args.embeddings_file)
            demo_standard_usage(model)

        if args.demo in ['search', 'all']:
            if model is None:
                model = create_hf_sentence_transformer(args.embeddings_file)
            demo_semantic_search(model)

        if args.demo in ['compare', 'all']:
            demo_model_comparison(args.embeddings_file)

        if args.demo in ['save', 'all']:
            model_path = demo_save_and_load(args.embeddings_file)

        if args.demo in ['hub', 'all']:
            if model_path is None:
                model_path = create_and_save_hf_model(
                    args.embeddings_file,
                    "demo-model",
                    "./hf_models"
                )
            demo_hub_preparation(model_path)

        print("\n" + "="*60)
        print("✨ Demo completed successfully!")
        print("📚 See docs/sentence_transformers_integration.md for full documentation")
        print("="*60)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())