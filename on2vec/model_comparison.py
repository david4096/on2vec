"""
Generic model comparison utilities for on2vec.

This module provides utilities to compare ontology-augmented models against
vanilla sentence transformers on various tasks.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def load_model_safe(model_path: str):
    """
    Safely load a model with error handling.

    Args:
        model_path: Path to model (HuggingFace directory or sentence transformer name)

    Returns:
        SentenceTransformer model or None if failed
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Check if it's a local path
        if Path(model_path).exists():
            logger.info(f"Loading local model from: {model_path}")
            return SentenceTransformer(model_path)
        else:
            logger.info(f"Loading model from HuggingFace Hub: {model_path}")
            return SentenceTransformer(model_path)

    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for embeddings.

    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]

    Returns:
        Similarity matrix [n_samples, n_samples]
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)

    # Compute cosine similarity
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix


def analyze_semantic_clusters(embeddings: np.ndarray, terms: List[str]) -> Dict[str, Any]:
    """
    Analyze semantic clustering in embeddings.

    Args:
        embeddings: Embeddings array
        terms: List of terms corresponding to embeddings

    Returns:
        Dictionary with clustering analysis results
    """
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Remove diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, -1)

    # Find most similar terms for each term
    top_similarities = []
    for i, term in enumerate(terms):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1][:3]  # Top 3 most similar
        top_terms = [(terms[j], similarities[j]) for j in top_indices if similarities[j] > -1]
        top_similarities.append((term, top_terms))

    # Compute average inter-term similarity
    upper_triangle = np.triu(similarity_matrix, k=1)
    non_zero_similarities = upper_triangle[upper_triangle > -1]
    avg_similarity = np.mean(non_zero_similarities) if len(non_zero_similarities) > 0 else 0.0

    # Compute similarity distribution statistics
    similarity_stats = {
        'mean': float(avg_similarity),
        'std': float(np.std(non_zero_similarities)) if len(non_zero_similarities) > 0 else 0.0,
        'min': float(np.min(non_zero_similarities)) if len(non_zero_similarities) > 0 else 0.0,
        'max': float(np.max(non_zero_similarities)) if len(non_zero_similarities) > 0 else 0.0,
        'median': float(np.median(non_zero_similarities)) if len(non_zero_similarities) > 0 else 0.0
    }

    return {
        'similarity_matrix': similarity_matrix,
        'top_similarities': top_similarities,
        'similarity_stats': similarity_stats,
        'num_terms': len(terms)
    }


def compare_models_on_terms(
    ontology_model_path: str,
    vanilla_model_path: str,
    test_terms: List[str],
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Compare two models on a set of test terms.

    Args:
        ontology_model_path: Path to ontology-augmented model
        vanilla_model_path: Path to vanilla model
        test_terms: List of terms to test on
        detailed: Whether to include detailed analysis

    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing models:")
    logger.info(f"  Ontology model: {ontology_model_path}")
    logger.info(f"  Vanilla model: {vanilla_model_path}")
    logger.info(f"  Test terms: {len(test_terms)}")

    # Load models
    ontology_model = load_model_safe(ontology_model_path)
    vanilla_model = load_model_safe(vanilla_model_path)

    if ontology_model is None:
        raise ValueError(f"Failed to load ontology model: {ontology_model_path}")
    if vanilla_model is None:
        raise ValueError(f"Failed to load vanilla model: {vanilla_model_path}")

    results = {
        'ontology_model_path': ontology_model_path,
        'vanilla_model_path': vanilla_model_path,
        'test_terms': test_terms,
        'num_terms': len(test_terms)
    }

    try:
        # Generate embeddings for both models
        logger.info("Generating embeddings with ontology model...")
        start_time = time.time()
        ontology_embeddings = ontology_model.encode(test_terms, convert_to_numpy=True)
        ontology_time = time.time() - start_time

        logger.info("Generating embeddings with vanilla model...")
        start_time = time.time()
        vanilla_embeddings = vanilla_model.encode(test_terms, convert_to_numpy=True)
        vanilla_time = time.time() - start_time

        results['embedding_times'] = {
            'ontology': ontology_time,
            'vanilla': vanilla_time
        }

        results['embedding_dimensions'] = {
            'ontology': ontology_embeddings.shape[1],
            'vanilla': vanilla_embeddings.shape[1]
        }

        # Analyze semantic clustering for both models
        logger.info("Analyzing semantic clustering...")
        ontology_analysis = analyze_semantic_clusters(ontology_embeddings, test_terms)
        vanilla_analysis = analyze_semantic_clusters(vanilla_embeddings, test_terms)

        results['ontology_analysis'] = ontology_analysis
        results['vanilla_analysis'] = vanilla_analysis

        # Compare similarity statistics
        ont_stats = ontology_analysis['similarity_stats']
        van_stats = vanilla_analysis['similarity_stats']

        results['comparison'] = {
            'mean_similarity_diff': ont_stats['mean'] - van_stats['mean'],
            'std_similarity_diff': ont_stats['std'] - van_stats['std'],
            'ontology_more_similar': ont_stats['mean'] > van_stats['mean'],
            'ontology_more_consistent': ont_stats['std'] < van_stats['std']
        }

        # If detailed analysis requested, include top similarities
        if detailed:
            results['detailed_similarities'] = {
                'ontology_top': ontology_analysis['top_similarities'],
                'vanilla_top': vanilla_analysis['top_similarities']
            }

        logger.info("Model comparison completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        raise


def generate_default_test_terms(domain: str = "biomedical") -> List[str]:
    """
    Generate default test terms for comparison.

    Args:
        domain: Domain for test terms ('biomedical', 'general', 'computer_science')

    Returns:
        List of test terms
    """
    if domain == "biomedical":
        return [
            "heart disease", "cardiovascular disorder", "cardiac dysfunction",
            "protein folding", "enzyme kinetics", "metabolic pathway",
            "gene expression", "transcription factor", "DNA methylation",
            "cell division", "mitosis", "apoptosis",
            "immune response", "antibody", "T cell activation",
            "cancer", "tumor", "oncogene",
            "diabetes", "insulin resistance", "glucose metabolism",
            "neuron", "synapse", "neurotransmitter"
        ]
    elif domain == "computer_science":
        return [
            "machine learning", "artificial intelligence", "neural network",
            "algorithm", "data structure", "computational complexity",
            "database", "relational model", "query optimization",
            "software engineering", "object oriented programming", "design pattern",
            "computer graphics", "rendering", "3D visualization",
            "cybersecurity", "encryption", "authentication",
            "distributed systems", "cloud computing", "microservices",
            "natural language processing", "text mining", "information retrieval"
        ]
    else:  # general
        return [
            "happiness", "joy", "contentment",
            "sadness", "grief", "melancholy",
            "anger", "rage", "frustration",
            "fear", "anxiety", "worry",
            "love", "affection", "care",
            "hope", "optimism", "confidence",
            "knowledge", "wisdom", "understanding",
            "beauty", "elegance", "grace"
        ]


def print_comparison_summary(results: Dict[str, Any], detailed: bool = False):
    """
    Print a formatted summary of model comparison results.

    Args:
        results: Results from compare_models_on_terms
        detailed: Whether to show detailed results
    """
    print("\n" + "="*60)
    print("🔬 MODEL COMPARISON RESULTS")
    print("="*60)

    print(f"📊 Test Configuration:")
    print(f"   Ontology Model: {Path(results['ontology_model_path']).name}")
    print(f"   Vanilla Model:  {results['vanilla_model_path']}")
    print(f"   Test Terms:     {results['num_terms']}")

    print(f"\n🎯 Embedding Properties:")
    ont_dim = results['embedding_dimensions']['ontology']
    van_dim = results['embedding_dimensions']['vanilla']
    print(f"   Ontology Dim:   {ont_dim}")
    print(f"   Vanilla Dim:    {van_dim}")

    ont_time = results['embedding_times']['ontology']
    van_time = results['embedding_times']['vanilla']
    print(f"   Ontology Time:  {ont_time:.3f}s")
    print(f"   Vanilla Time:   {van_time:.3f}s")

    print(f"\n📈 Similarity Analysis:")
    ont_stats = results['ontology_analysis']['similarity_stats']
    van_stats = results['vanilla_analysis']['similarity_stats']
    comp = results['comparison']

    print(f"   Average Inter-term Similarity:")
    print(f"     Ontology: {ont_stats['mean']:.4f} (±{ont_stats['std']:.4f})")
    print(f"     Vanilla:  {van_stats['mean']:.4f} (±{van_stats['std']:.4f})")
    print(f"     Difference: {comp['mean_similarity_diff']:+.4f}")

    print(f"\n🏆 Comparison Summary:")
    if comp['ontology_more_similar']:
        print("   ✅ Ontology model produces higher inter-term similarity")
    else:
        print("   ❌ Vanilla model produces higher inter-term similarity")

    if comp['ontology_more_consistent']:
        print("   ✅ Ontology model has more consistent similarities")
    else:
        print("   ❌ Vanilla model has more consistent similarities")

    if detailed and 'detailed_similarities' in results:
        print(f"\n🔍 Top Similar Term Pairs:")

        print(f"\n   Ontology Model:")
        for term, similarities in results['detailed_similarities']['ontology_top'][:5]:
            print(f"     '{term}' → ", end="")
            for similar_term, score in similarities[:2]:
                print(f"'{similar_term}' ({score:.3f}) ", end="")
            print()

        print(f"\n   Vanilla Model:")
        for term, similarities in results['detailed_similarities']['vanilla_top'][:5]:
            print(f"     '{term}' → ", end="")
            for similar_term, score in similarities[:2]:
                print(f"'{similar_term}' ({score:.3f}) ", end="")
            print()

    print(f"\n💡 Interpretation:")
    if comp['ontology_more_similar'] and comp['ontology_more_consistent']:
        print("   🎉 The ontology model shows both higher and more consistent")
        print("      semantic similarity, suggesting better domain knowledge.")
    elif comp['ontology_more_similar']:
        print("   👍 The ontology model captures stronger semantic relationships,")
        print("      though with more variability.")
    elif comp['ontology_more_consistent']:
        print("   👍 The ontology model provides more consistent similarities,")
        print("      though with lower average similarity.")
    else:
        print("   🤔 The vanilla model performed better on both metrics.")
        print("      Consider checking if the ontology model was properly trained.")

    print("="*60 + "\n")


def main_compare(
    ontology_model_path: str,
    vanilla_model_path: str = "all-MiniLM-L6-v2",
    domain_terms: Optional[List[str]] = None,
    detailed: bool = False
) -> bool:
    """
    Main comparison function that can be called from CLI.

    Args:
        ontology_model_path: Path to ontology-augmented model
        vanilla_model_path: Path to vanilla model for comparison
        domain_terms: Custom domain terms, if None uses biomedical defaults
        detailed: Whether to show detailed analysis

    Returns:
        True if comparison completed successfully, False otherwise
    """
    try:
        # Use provided terms or generate defaults
        if domain_terms is None:
            domain_terms = generate_default_test_terms("biomedical")
            logger.info(f"Using default biomedical test terms ({len(domain_terms)} terms)")
        else:
            logger.info(f"Using provided test terms ({len(domain_terms)} terms)")

        # Perform comparison
        results = compare_models_on_terms(
            ontology_model_path=ontology_model_path,
            vanilla_model_path=vanilla_model_path,
            test_terms=domain_terms,
            detailed=detailed
        )

        # Print results
        print_comparison_summary(results, detailed=detailed)

        return True

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        print(f"\n❌ Model comparison failed: {e}")
        return False