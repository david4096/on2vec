#!/usr/bin/env python3
"""
Demo script for text-augmented ontology embeddings.
This demonstrates the unified embedding space combining structural and semantic features.
"""

import sys
import os
import logging

# Add on2vec to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from on2vec.training import train_text_augmented_ontology_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """
    Demonstrate text-augmented ontology embedding training.
    """
    # Demo configuration
    owl_file = "owl_files/duo.owl"  # Small ontology for demo
    model_output = "output/demo_text_augmented_model.pt"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    print("🔬 Starting Text-Augmented Ontology Embedding Demo")
    print("=" * 60)
    print(f"📂 Input ontology: {owl_file}")
    print(f"💾 Output model: {model_output}")
    print()

    # Training configuration
    config = {
        'text_model_type': 'sentence_transformer',
        'text_model_name': 'all-MiniLM-L6-v2',  # Lightweight but effective
        'backbone_model': 'gcn',
        'fusion_method': 'concat',  # Simple concatenation of features
        'hidden_dim': 64,
        'out_dim': 8,
        'epochs': 50,  # Reduced for demo
        'loss_fn_name': 'cosine',
        'learning_rate': 0.01,
        'dropout': 0.1
    }

    print("⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    try:
        # Train text-augmented model
        print("🚀 Starting text-augmented training...")
        results = train_text_augmented_ontology_embeddings(
            owl_file=owl_file,
            model_output=model_output,
            **config
        )

        print("✅ Training completed successfully!")
        print()
        print("📊 Results:")
        print(f"   📦 Model saved to: {results['model_path']}")
        print(f"   🔢 Number of nodes: {results['num_nodes']}")
        print(f"   🔗 Number of edges: {results['num_edges']}")
        print(f"   📏 Structural features: {results['structural_dim']}")
        print(f"   📝 Text features: {results['text_dim']}")
        print(f"   📰 Text features extracted: {results['text_features_extracted']}")
        print()

        print("🎯 Model Configuration:")
        model_config = results['model_config']
        for key, value in model_config.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("🎉 Text-augmented ontology embedding demo completed!")
    print("   This model now combines structural graph information")
    print("   with rich semantic text features from ontology annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())