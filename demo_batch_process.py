#!/usr/bin/env python3
"""
Demo batch processing - processes just the duo.owl file to show the complete pipeline.
"""

import sys
from pathlib import Path

# Import the main processing function
sys.path.append(str(Path(__file__).parent))
from batch_process_owl import process_single_owl_file

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Demo the batch processing pipeline with one file."""

    # Process duo.owl as a demo
    owl_file = "owl_files/duo.owl"

    if not Path(owl_file).exists():
        print(f"❌ Demo file {owl_file} not found!")
        return 1

    print("🚀 Starting demo batch processing with duo.owl...")
    print("📊 Configuration: GCN, 64 hidden dims, 8 output dims, cosine loss")
    print("🎨 Will generate: PCA, t-SNE, UMAP, distribution, and comparison plots")
    print()

    result = process_single_owl_file(owl_file)

    if result['success']:
        print("✅ Demo completed successfully!")
        print(f"⏱️  Processing time: {result['total_time']:.1f}s")
        print(f"📊 Generated embeddings: {result['model_stats'].get('embeddings', 'N/A')}")
        print(f"🎨 Visualization results:")
        for viz_type, status in result['visualization_stats'].items():
            status_icon = "✅" if status == 'success' else "❌"
            print(f"   {status_icon} {viz_type.title()}")

        print(f"\n📁 Output files in: output/duo/")
        output_dir = Path("output/duo")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            for f in sorted(files):
                print(f"   - {f.name}")

    else:
        print("❌ Demo failed!")
        for error in result['errors']:
            print(f"   Error: {error}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())