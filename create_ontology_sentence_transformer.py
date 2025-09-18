#!/usr/bin/env python3
"""
CLI tool to create and use ontology-augmented Sentence Transformer models.

This script demonstrates how to:
1. Load on2vec generated embeddings (with text + structural components)
2. Create custom Sentence Transformer models that incorporate ontology knowledge
3. Use the models for enhanced semantic similarity and retrieval
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from on2vec.sentence_transformer_integration import create_ontology_augmented_model
from on2vec.query_document_ontology_model import create_retrieval_model_with_ontology
from on2vec.io import inspect_parquet_metadata
import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_embeddings_file(embeddings_file: str) -> bool:
    """Validate that the embeddings file has the required format."""
    try:
        df = pl.read_parquet(embeddings_file)
        required_cols = ['node_id', 'text_embedding', 'structural_embedding']

        if not all(col in df.columns for col in required_cols):
            logger.error(f"Embeddings file missing required columns: {required_cols}")
            logger.error(f"Found columns: {df.columns}")
            return False

        logger.info(f"✅ Valid embeddings file with {len(df)} ontology concepts")
        return True

    except Exception as e:
        logger.error(f"❌ Error reading embeddings file: {e}")
        return False


def demo_basic_augmented_similarity(embeddings_file: str, queries: list):
    """Demonstrate basic ontology-augmented similarity."""
    print("\\n" + "="*60)
    print("🧬 Basic Ontology-Augmented Similarity")
    print("="*60)

    # Create ontology-augmented model
    model = create_ontology_augmented_model(
        base_model='all-MiniLM-L6-v2',
        ontology_embeddings_file=embeddings_file,
        fusion_method='concat',  # Simple concatenation
        top_k_matches=3,
        structural_weight=0.3
    )

    print(f"Model text dimension: {model.text_dim}")
    print(f"Model structural dimension: {model.structural_dim}")
    print(f"Model output dimension: {model.output_dim}")

    # Get embeddings
    embeddings = model(queries)['sentence_embedding']

    print(f"\\nGenerated embeddings shape: {embeddings.shape}")

    # Compute similarities
    similarities = cosine_similarity(embeddings)

    print("\\nSimilarity Matrix:")
    print("-" * 40)
    for i, query1 in enumerate(queries):
        for j, query2 in enumerate(queries):
            if i < j:  # Only show upper triangle
                sim = similarities[i, j]
                print(f"{query1[:25]:25} <-> {query2[:25]:25}: {sim:.3f}")

    # Show which ontology concepts were matched
    print("\\nOntology Concept Matching:")
    print("-" * 40)
    for query in queries[:3]:  # Limit to first 3 for brevity
        matches = model.find_similar_concepts(query)
        print(f"\\nQuery: '{query}'")
        for idx, score in matches:
            concept_iri = model.node_ids[idx]
            concept_desc = model.concept_descriptions[idx]
            print(f"  {score:.3f}: {concept_desc} ({concept_iri.split('/')[-1]})")


def demo_query_document_retrieval(embeddings_file: str, queries: list, documents: list):
    """Demonstrate query/document retrieval with ontology."""
    print("\\n" + "="*60)
    print("🔍 Query/Document Retrieval with Ontology")
    print("="*60)

    # Create retrieval model
    model = create_retrieval_model_with_ontology(
        ontology_embeddings_file=embeddings_file,
        fusion_method='concat',  # Simple concatenation
        projection_dim=None      # No projection for now
    )

    print(f"Query encoder dim: {model.query_encoder.output_dim}")
    print(f"Document encoder dim: {model.doc_encoder.output_dim}")

    # Encode queries and documents separately
    query_embeds = model.encode_queries(queries)
    doc_embeds = model.encode_documents(documents)

    print(f"Query embeddings: {query_embeds.shape}")
    print(f"Document embeddings: {doc_embeds.shape}")

    # Compute retrieval scores
    scores = np.dot(query_embeds.detach().numpy(), doc_embeds.detach().numpy().T)

    print("\\nRetrieval Results:")
    print("=" * 50)

    for i, query in enumerate(queries):
        query_scores = scores[i]
        top_indices = np.argsort(query_scores)[::-1][:3]  # Top 3

        print(f"\\n🔎 Query: '{query}'")
        print("-" * 50)

        for rank, doc_idx in enumerate(top_indices):
            score = query_scores[doc_idx]
            doc_preview = documents[doc_idx][:80] + "..." if len(documents[doc_idx]) > 80 else documents[doc_idx]
            print(f"  {rank+1}. Score: {score:.3f}")
            print(f"     {doc_preview}")


def create_and_save_model(embeddings_file: str, output_dir: str, model_type: str = 'basic'):
    """Create and save a custom ontology model for later use."""
    print(f"\\n🏗️  Creating and saving {model_type} ontology model...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_type == 'basic':
        model = create_ontology_augmented_model(
            base_model='all-MiniLM-L6-v2',
            ontology_embeddings_file=embeddings_file,
            fusion_method='concat'
        )
        save_path = output_path / "ontology_augmented_model"

    elif model_type == 'retrieval':
        model = create_retrieval_model_with_ontology(
            ontology_embeddings_file=embeddings_file,
            fusion_method='gated',
            projection_dim=256
        )
        save_path = output_path / "query_document_ontology_model"

    # Save model state
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'embeddings_file': embeddings_file,
    }, str(save_path) + ".pt")

    print(f"✅ Model saved to: {save_path}.pt")
    print(f"📋 Model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ontology-augmented Sentence Transformer models"
    )
    parser.add_argument(
        'embeddings_file',
        help='Path to on2vec generated parquet file (with text + structural embeddings)'
    )
    parser.add_argument(
        '--demo',
        choices=['basic', 'retrieval', 'both'],
        default='both',
        help='Which demo to run'
    )
    parser.add_argument(
        '--save-model',
        help='Directory to save the created models'
    )
    parser.add_argument(
        '--queries',
        nargs='+',
        default=[
            "heart disease and cardiovascular problems",
            "genetic mutations causing cancer",
            "protein folding and misfolding disorders",
            "neurodegenerative brain diseases",
            "inflammatory immune responses"
        ],
        help='Query texts for demonstration'
    )
    parser.add_argument(
        '--documents',
        nargs='+',
        default=[
            "Cardiovascular disease results from atherosclerosis, hypertension, and cardiac arrhythmias affecting heart function.",
            "Oncogenic mutations in tumor suppressor genes like p53 and BRCA1 lead to uncontrolled cell proliferation.",
            "Protein misfolding causes aggregation and cellular toxicity in neurodegenerative diseases like Alzheimer's.",
            "Neurodegeneration involves progressive loss of neurons and cognitive decline in aging populations.",
            "Autoimmune disorders result from dysregulated immune responses targeting healthy tissue.",
            "Weather patterns significantly impact agricultural yields through temperature and precipitation changes.",
            "Machine learning algorithms optimize objective functions using gradient-based optimization techniques.",
            "Economic markets fluctuate based on supply, demand, and investor sentiment in financial systems."
        ],
        help='Document texts for retrieval demonstration'
    )

    args = parser.parse_args()

    # Validate embeddings file
    if not Path(args.embeddings_file).exists():
        logger.error(f"Embeddings file not found: {args.embeddings_file}")
        print("\\n🔧 To create embeddings file:")
        print("1. Use on2vec to train a text-augmented model:")
        print("   python main.py ontology.owl --use_text_features --output embeddings.parquet")
        print("2. Ensure the output has text_embedding and structural_embedding columns")
        return 1

    if not validate_embeddings_file(args.embeddings_file):
        return 1

    # Show embeddings file info
    print("\\n📊 Embeddings File Info:")
    inspect_parquet_metadata(args.embeddings_file)

    # Run demos
    if args.demo in ['basic', 'both']:
        try:
            demo_basic_augmented_similarity(args.embeddings_file, args.queries)
        except Exception as e:
            logger.error(f"Basic demo failed: {e}")

    if args.demo in ['retrieval', 'both']:
        try:
            demo_query_document_retrieval(args.embeddings_file, args.queries, args.documents)
        except Exception as e:
            logger.error(f"Retrieval demo failed: {e}")

    # Save models if requested
    if args.save_model:
        try:
            if args.demo in ['basic', 'both']:
                create_and_save_model(args.embeddings_file, args.save_model, 'basic')
            if args.demo in ['retrieval', 'both']:
                create_and_save_model(args.embeddings_file, args.save_model, 'retrieval')
        except Exception as e:
            logger.error(f"Model saving failed: {e}")

    print("\\n✨ Demo completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())