#!/usr/bin/env python3
"""
Complete usage example for ontology-augmented Sentence Transformers.

This script demonstrates the complete workflow from training to deployment
of custom Sentence Transformers models with on2vec ontology integration.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from on2vec.sentence_transformer_hub import create_and_save_hf_model


def create_biomedical_model():
    """Create a biomedical ontology-augmented model."""
    print("🧬 Creating Biomedical Ontology Model")
    print("=" * 50)

    # Step 1: Create the model from on2vec embeddings
    model_path = create_and_save_hf_model(
        ontology_embeddings_file="test_embeddings.parquet",
        model_name="biomedical-ontology-embedder",
        output_dir="./models",
        fusion_method="concat"
    )

    print(f"✅ Model created and saved to: {model_path}")
    return model_path


def demo_semantic_similarity(model_path: str):
    """Demonstrate semantic similarity with ontology knowledge."""
    print("\n🔬 Semantic Similarity Demo")
    print("=" * 50)

    # Load the model with standard sentence-transformers
    model = SentenceTransformer(model_path)
    print(f"Model dimensions: {model.get_sentence_embedding_dimension()}")

    # Biomedical terms
    biomedical_terms = [
        "myocardial infarction",
        "heart attack",
        "cardiac arrest",
        "stroke",
        "cerebrovascular accident",
        "protein misfolding",
        "amyloid plaques",
        "alzheimer disease",
        "neurodegeneration",
        "diabetes mellitus",
        "hyperglycemia"
    ]

    # Encode all terms
    embeddings = model.encode(biomedical_terms)
    similarities = cos_sim(embeddings, embeddings)

    # Show most similar pairs
    print("\n🎯 Top Similar Term Pairs:")
    print("-" * 40)

    similar_pairs = []
    for i in range(len(biomedical_terms)):
        for j in range(i+1, len(biomedical_terms)):
            sim = similarities[i][j].item()
            similar_pairs.append((sim, biomedical_terms[i], biomedical_terms[j]))

    # Sort by similarity and show top 5
    similar_pairs.sort(reverse=True)
    for sim, term1, term2 in similar_pairs[:5]:
        print(f"{sim:.3f}: {term1} <-> {term2}")


def demo_biomedical_search(model_path: str):
    """Demonstrate biomedical document search."""
    print("\n🔍 Biomedical Document Search")
    print("=" * 50)

    model = SentenceTransformer(model_path)

    # Medical documents
    documents = [
        "Acute myocardial infarction results from coronary artery occlusion leading to cardiac tissue necrosis.",
        "Alzheimer's disease is characterized by progressive neurodegeneration with amyloid plaque accumulation.",
        "Type 2 diabetes mellitus involves insulin resistance and impaired glucose metabolism.",
        "Stroke occurs when blood flow to brain tissue is interrupted, causing neurological deficits.",
        "Cancer develops through oncogenic mutations that lead to uncontrolled cellular proliferation.",
        "Hypertension is a chronic condition involving elevated arterial blood pressure readings.",
        "Protein misfolding diseases include conditions where proteins adopt incorrect conformations.",
        "Autoimmune disorders result from immune system attacks on healthy tissue components.",
    ]

    # Medical queries
    queries = [
        "heart attack symptoms",
        "alzheimer memory loss",
        "diabetes blood sugar",
        "brain stroke recovery",
        "cancer treatment options"
    ]

    # Encode documents
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    print("Search Results:")
    print("-" * 30)

    for query in queries:
        print(f"\n📋 Query: '{query}'")
        query_embed = model.encode([query], convert_to_tensor=True)

        # Find most relevant documents
        results = semantic_search(query_embed, doc_embeddings, top_k=2)

        for rank, result in enumerate(results[0], 1):
            doc_idx = result['corpus_id']
            score = result['score']
            doc = documents[doc_idx]
            doc_preview = doc[:60] + "..." if len(doc) > 60 else doc
            print(f"  {rank}. ({score:.3f}) {doc_preview}")


def demo_comparison_with_standard():
    """Compare with standard sentence-transformers model."""
    print("\n⚖️ Comparison with Standard Model")
    print("=" * 50)

    # Load both models
    standard_model = SentenceTransformer('all-MiniLM-L6-v2')
    ontology_model = SentenceTransformer('./models/biomedical-ontology-embedder')

    # Test medical terminology
    medical_pairs = [
        ("myocardial infarction", "heart attack"),
        ("cerebrovascular accident", "stroke"),
        ("diabetes mellitus", "hyperglycemia"),
        ("alzheimer disease", "neurodegeneration"),
    ]

    print("Medical Term Similarities:")
    print("-" * 40)
    print(f"{'Term Pair':<35} {'Standard':<10} {'Ontology':<10} {'Δ':<8}")
    print("-" * 65)

    for term1, term2 in medical_pairs:
        # Standard model
        std_embeds = standard_model.encode([term1, term2])
        std_sim = cos_sim(std_embeds[0:1], std_embeds[1:2]).item()

        # Ontology model
        ont_embeds = ontology_model.encode([term1, term2])
        ont_sim = cos_sim(ont_embeds[0:1], ont_embeds[1:2]).item()

        improvement = ont_sim - std_sim
        pair_name = f"{term1} <-> {term2}"[:34]

        print(f"{pair_name:<35} {std_sim:<10.3f} {ont_sim:<10.3f} {improvement:<+8.3f}")


def demo_clustering_with_ontology():
    """Demonstrate concept clustering with ontology knowledge."""
    print("\n🎯 Ontology-Aware Clustering")
    print("=" * 50)

    model = SentenceTransformer('./models/biomedical-ontology-embedder')

    # Diverse biomedical concepts
    concepts = [
        # Cardiovascular
        "heart disease", "myocardial infarction", "hypertension", "cardiac arrest",
        # Neurological
        "alzheimer disease", "stroke", "neurodegeneration", "brain injury",
        # Metabolic
        "diabetes", "obesity", "metabolic syndrome", "insulin resistance",
        # Cancer
        "lung cancer", "breast cancer", "oncology", "tumor growth",
        # Infectious
        "pneumonia", "sepsis", "bacterial infection", "viral disease"
    ]

    # Generate embeddings
    embeddings = model.encode(concepts)

    # Simple clustering by similarity
    from sklearn.cluster import KMeans
    import numpy as np

    # Cluster into 5 groups
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    print("Ontology-Based Clusters:")
    print("-" * 30)

    # Group by cluster
    cluster_groups = {}
    for i, concept in enumerate(concepts):
        cluster_id = clusters[i]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(concept)

    for cluster_id, group in cluster_groups.items():
        print(f"\n🏷️ Cluster {cluster_id}:")
        for concept in group:
            print(f"   • {concept}")


def main():
    """Run complete demonstration."""
    print("🚀 Ontology-Augmented Sentence Transformers Demo")
    print("=" * 60)

    # Check if we have test embeddings
    if not Path("test_embeddings.parquet").exists():
        print("❌ Missing test_embeddings.parquet")
        print("Run: python main.py ado.owl --use_text_features --output test_embeddings.parquet")
        return 1

    try:
        # Create the model
        model_path = create_biomedical_model()

        # Run demonstrations
        demo_semantic_similarity(model_path)
        demo_biomedical_search(model_path)
        demo_comparison_with_standard()
        demo_clustering_with_ontology()

        print("\n" + "=" * 60)
        print("✨ All demos completed successfully!")
        print("\n📚 Next Steps:")
        print("• Upload to HuggingFace Hub: model.push_to_hub('your-username/model-name')")
        print("• Integrate into applications: SentenceTransformer('your-username/model-name')")
        print("• Fine-tune for specific domains with additional training data")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())