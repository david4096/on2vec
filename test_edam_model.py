#!/usr/bin/env python3
"""
Test script comparing EDAM ontology-augmented Sentence Transformer vs vanilla models.

This script evaluates how well ontology knowledge improves semantic similarity
for bioinformatics terms compared to vanilla text-only models.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_models():
    """Load both vanilla and EDAM ontology-augmented models."""
    from sentence_transformers import SentenceTransformer

    print("🤖 Loading models...")

    # Load vanilla model
    try:
        vanilla_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ Vanilla model loaded: all-MiniLM-L6-v2 ({vanilla_model.get_sentence_embedding_dimension()} dims)")
    except Exception as e:
        print(f"❌ Failed to load vanilla model: {e}")
        return None, None

    # Try to find EDAM model
    edam_model = None
    possible_paths = [
        './hf_models/edam-model',
        './hf_models/edam-text-model',
        './hf_models/EDAM-model',
        './edam_model'
    ]

    for path in possible_paths:
        if Path(path).exists():
            try:
                edam_model = SentenceTransformer(path)
                print(f"✅ EDAM model loaded: {path} ({edam_model.get_sentence_embedding_dimension()} dims)")
                break
            except Exception as e:
                print(f"⚠️  Failed to load EDAM model from {path}: {e}")
                continue

    if edam_model is None:
        print("❌ No EDAM model found. Create one first:")
        print("   python create_hf_model.py e2e EDAM.owl edam-model")
        return vanilla_model, None

    return vanilla_model, edam_model


def get_test_terms() -> Dict[str, List[str]]:
    """Get bioinformatics test terms organized by category."""
    return {
        "alignment": [
            "sequence alignment",
            "multiple sequence alignment",
            "pairwise alignment",
            "global alignment",
            "local alignment",
            "BLAST alignment",
            "structural alignment"
        ],
        "variant": [
            "genetic variant",
            "single nucleotide polymorphism",
            "SNP",
            "mutation",
            "insertion deletion",
            "copy number variation",
            "structural variant"
        ],
        "gene_expression": [
            "gene expression",
            "RNA expression",
            "transcription",
            "mRNA levels",
            "expression profiling",
            "differential expression",
            "gene regulation"
        ],
        "nucleic_acid": [
            "DNA",
            "RNA",
            "nucleotide",
            "oligonucleotide",
            "double helix",
            "base pair",
            "nucleic acid sequence"
        ],
        "protein": [
            "protein structure",
            "amino acid",
            "protein folding",
            "protein domain",
            "enzyme",
            "peptide",
            "protein sequence"
        ],
        "genomics": [
            "genome",
            "chromosome",
            "genomic DNA",
            "gene",
            "locus",
            "allele",
            "genotype"
        ]
    }


def compute_category_similarities(model, terms: Dict[str, List[str]], model_name: str) -> Dict[str, Dict[str, float]]:
    """Compute intra-category similarities for each term category."""
    print(f"\n📊 Computing similarities for {model_name}...")

    results = {}

    for category, term_list in terms.items():
        print(f"  Processing {category}...")

        # Encode all terms in this category
        embeddings = model.encode(term_list)

        # Compute pairwise similarities
        from sentence_transformers.util import cos_sim
        similarities = cos_sim(embeddings, embeddings)

        # Extract upper triangle (excluding diagonal)
        sim_values = []
        for i in range(len(term_list)):
            for j in range(i+1, len(term_list)):
                sim_values.append(similarities[i][j].item())

        results[category] = {
            'mean_similarity': np.mean(sim_values),
            'std_similarity': np.std(sim_values),
            'max_similarity': np.max(sim_values),
            'min_similarity': np.min(sim_values),
            'similarities': sim_values,
            'terms': term_list
        }

    return results


def compute_cross_category_similarities(model, terms: Dict[str, List[str]], model_name: str) -> Dict[str, float]:
    """Compute similarities between different categories."""
    print(f"\n🔀 Computing cross-category similarities for {model_name}...")

    categories = list(terms.keys())
    cross_sims = {}

    from sentence_transformers.util import cos_sim

    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:  # Only compute upper triangle
                # Take first term from each category as representative
                term1 = terms[cat1][0]
                term2 = terms[cat2][0]

                emb1 = model.encode([term1])
                emb2 = model.encode([term2])

                similarity = cos_sim(emb1, emb2)[0][0].item()
                cross_sims[f"{cat1}_vs_{cat2}"] = similarity

    return cross_sims


def find_most_similar_terms(model, query_terms: List[str], candidate_terms: List[str], model_name: str) -> List[Tuple[str, str, float]]:
    """Find most similar terms for given queries."""
    print(f"\n🎯 Finding most similar terms for {model_name}...")

    query_embeds = model.encode(query_terms)
    candidate_embeds = model.encode(candidate_terms)

    from sentence_transformers.util import cos_sim
    similarities = cos_sim(query_embeds, candidate_embeds)

    results = []
    for i, query in enumerate(query_terms):
        # Find most similar candidate (excluding exact matches)
        best_idx = -1
        best_sim = -1

        for j, candidate in enumerate(candidate_terms):
            sim = similarities[i][j].item()
            if candidate.lower() != query.lower() and sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_idx >= 0:
            results.append((query, candidate_terms[best_idx], best_sim))

    return results


def print_comparison_results(vanilla_results: Dict, edam_results: Dict):
    """Print detailed comparison between vanilla and EDAM models."""
    print("\n" + "="*80)
    print("📈 DETAILED MODEL COMPARISON")
    print("="*80)

    print(f"\n{'Category':<15} {'Vanilla Mean':<12} {'EDAM Mean':<12} {'Improvement':<12} {'Status'}")
    print("-" * 65)

    improvements = []

    for category in vanilla_results:
        if category in edam_results:
            vanilla_mean = vanilla_results[category]['mean_similarity']
            edam_mean = edam_results[category]['mean_similarity']
            improvement = edam_mean - vanilla_mean
            improvements.append(improvement)

            status = "🟢 Better" if improvement > 0.01 else "🟡 Similar" if improvement > -0.01 else "🔴 Worse"

            print(f"{category:<15} {vanilla_mean:<12.3f} {edam_mean:<12.3f} {improvement:<+12.3f} {status}")

    print("-" * 65)
    avg_improvement = np.mean(improvements) if improvements else 0
    print(f"{'AVERAGE':<15} {'':<12} {'':<12} {avg_improvement:<+12.3f}")

    # Statistical significance
    if avg_improvement > 0.02:
        print("\n🎉 EDAM model shows significant improvement over vanilla!")
    elif avg_improvement > 0.005:
        print("\n✅ EDAM model shows modest improvement over vanilla")
    else:
        print("\n📊 Results are mixed - EDAM model shows limited improvement")


def print_detailed_category_analysis(vanilla_results: Dict, edam_results: Dict, category: str):
    """Print detailed analysis for a specific category."""
    if category not in vanilla_results or category not in edam_results:
        return

    print(f"\n📋 Detailed Analysis: {category.upper()}")
    print("-" * 50)

    vanilla_data = vanilla_results[category]
    edam_data = edam_results[category]

    terms = vanilla_data['terms']
    vanilla_sims = vanilla_data['similarities']
    edam_sims = edam_data['similarities']

    print(f"Terms: {', '.join(terms[:3])}{'...' if len(terms) > 3 else ''}")
    print(f"Vanilla: μ={vanilla_data['mean_similarity']:.3f}, σ={vanilla_data['std_similarity']:.3f}")
    print(f"EDAM:    μ={edam_data['mean_similarity']:.3f}, σ={edam_data['std_similarity']:.3f}")

    # Show top improved pairs
    improvements = [(edam_sims[i] - vanilla_sims[i], i) for i in range(len(vanilla_sims))]
    improvements.sort(reverse=True)

    print(f"\nTop improved term pairs:")
    term_pairs = []
    idx = 0
    for i, term1 in enumerate(terms):
        for j, term2 in enumerate(terms):
            if i < j:
                term_pairs.append((term1, term2))
                idx += 1

    for improvement, idx in improvements[:3]:
        if idx < len(term_pairs):
            term1, term2 = term_pairs[idx]
            vanilla_sim = vanilla_sims[idx]
            edam_sim = edam_sims[idx]
            print(f"  {term1[:20]:20} <-> {term2[:20]:20}: {vanilla_sim:.3f} → {edam_sim:.3f} ({improvement:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="Compare EDAM ontology model vs vanilla model")
    parser.add_argument('--edam-model', help='Path to EDAM model directory')
    parser.add_argument('--vanilla-model', default='all-MiniLM-L6-v2', help='Vanilla model name')
    parser.add_argument('--category', help='Focus on specific category', choices=[
        'alignment', 'variant', 'gene_expression', 'nucleic_acid', 'protein', 'genomics'
    ])
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')

    args = parser.parse_args()

    print("🧬 EDAM Ontology Model vs Vanilla Model Comparison")
    print("=" * 60)

    # Load models
    vanilla_model, edam_model = load_models()

    if vanilla_model is None:
        print("❌ Failed to load vanilla model")
        return 1

    if edam_model is None:
        print("❌ Failed to load EDAM model - comparison not possible")
        return 1

    # Get test terms
    all_terms = get_test_terms()

    if args.category:
        # Focus on single category
        test_terms = {args.category: all_terms[args.category]}
    else:
        # Test all categories
        test_terms = all_terms

    # Run comparisons
    vanilla_results = compute_category_similarities(vanilla_model, test_terms, "Vanilla")
    edam_results = compute_category_similarities(edam_model, test_terms, "EDAM")

    # Print results
    print_comparison_results(vanilla_results, edam_results)

    # Detailed analysis if requested
    if args.detailed:
        for category in test_terms:
            print_detailed_category_analysis(vanilla_results, edam_results, category)

    # Test specific query similarity
    print(f"\n🔍 Query Similarity Test")
    print("-" * 40)

    queries = ["alignment", "variant", "gene expression", "nucleic acid"]
    all_candidates = [term for terms_list in all_terms.values() for term in terms_list]

    vanilla_matches = find_most_similar_terms(vanilla_model, queries, all_candidates, "Vanilla")
    edam_matches = find_most_similar_terms(edam_model, queries, all_candidates, "EDAM")

    print(f"\n{'Query':<15} {'Vanilla Best Match':<25} {'Score':<6} {'EDAM Best Match':<25} {'Score':<6}")
    print("-" * 85)

    for i, query in enumerate(queries):
        if i < len(vanilla_matches) and i < len(edam_matches):
            _, v_match, v_score = vanilla_matches[i]
            _, e_match, e_score = edam_matches[i]

            print(f"{query:<15} {v_match:<25} {v_score:<6.3f} {e_match:<25} {e_score:<6.3f}")

    print("\n" + "="*60)
    print("💡 Interpretation:")
    print("• Higher intra-category similarities = better domain understanding")
    print("• Different best matches = ontology provides domain-specific knowledge")
    print("• EDAM model should show higher similarities for related bioinfo terms")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())