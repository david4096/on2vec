#!/usr/bin/env python3
"""
CLI tool for creating HuggingFace compatible Sentence Transformer models from on2vec embeddings.

This provides a streamlined command-line interface for the complete workflow:
1. Train ontology model with text features
2. Generate multi-embedding files
3. Create HuggingFace compatible models
4. Test and validate models
5. Prepare for Hub upload
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from on2vec.sentence_transformer_hub import create_and_save_hf_model
from on2vec.io import inspect_parquet_metadata
from on2vec.metadata_utils import get_base_model_from_embeddings, get_embedding_info, validate_text_embeddings_compatibility
from on2vec.model_card_generator import create_model_card, create_upload_instructions
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_ontology_with_text(
    owl_file: str,
    output_file: str,
    text_model: str = "all-MiniLM-L6-v2",
    epochs: int = 100,
    model_type: str = "gcn",
    hidden_dim: int = 128,
    out_dim: int = 64,
    loss_fn: str = "triplet"
) -> bool:
    """Train an ontology model with text features enabled."""
    print(f"🧬 Training ontology model: {owl_file}")
    print(f"📊 Output: {output_file}")
    print(f"🤖 Text model: {text_model}")
    print(f"⚙️ Config: {model_type}, hidden={hidden_dim}, out={out_dim}, loss={loss_fn}, epochs={epochs}")

    cmd = [
        "uv", "run", "python", "main.py", owl_file,
        "--use_text_features",
        "--text_model_name", text_model,
        "--output", output_file,
        "--model_type", model_type,
        "--hidden_dim", str(hidden_dim),
        "--out_dim", str(out_dim),
        "--epochs", str(epochs),
        "--loss_fn", loss_fn
    ]

    print(f"🔧 Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False


def validate_embeddings(embeddings_file: str) -> Dict[str, Any]:
    """Validate and inspect embeddings file."""
    print(f"🔍 Validating embeddings: {embeddings_file}")

    if not Path(embeddings_file).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    # Inspect metadata
    try:
        inspect_parquet_metadata(embeddings_file)

        # Additional validation
        import polars as pl
        df = pl.read_parquet(embeddings_file)

        required_cols = ['node_id', 'text_embedding', 'structural_embedding']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        metadata = {
            'num_concepts': len(df),
            'has_text_embeddings': 'text_embedding' in df.columns,
            'has_structural_embeddings': 'structural_embedding' in df.columns,
            'text_dim': len(df['text_embedding'][0]) if 'text_embedding' in df.columns else None,
            'structural_dim': len(df['structural_embedding'][0]) if 'structural_embedding' in df.columns else None
        }

        print("✅ Embeddings validation passed!")
        return metadata

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


def create_hf_model(
    embeddings_file: str,
    model_name: str,
    output_dir: str = "./hf_models",
    base_model: Optional[str] = None,
    fusion_method: str = "concat",
    validate_first: bool = True,
    ontology_file: Optional[str] = None,  # For model card generation
    training_config: Optional[Dict[str, Any]] = None  # For model card generation
) -> str:
    """Create HuggingFace compatible model."""
    print(f"🏗️ Creating HuggingFace model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 Fusion method: {fusion_method}")

    if validate_first:
        validate_embeddings(embeddings_file)

    # Auto-infer base model from embeddings if not provided
    if base_model is None:
        print("🔍 Auto-detecting base model from embeddings metadata...")
        inferred_model = get_base_model_from_embeddings(embeddings_file)
        if inferred_model:
            base_model = inferred_model
            print(f"✅ Detected base model: {base_model}")
        else:
            base_model = "all-MiniLM-L6-v2"  # Default fallback
            print(f"⚠️  Could not detect base model, using default: {base_model}")
    else:
        # Validate compatibility if user specified a base model
        if not validate_text_embeddings_compatibility(embeddings_file, base_model):
            inferred_model = get_base_model_from_embeddings(embeddings_file)
            if inferred_model:
                print(f"⚠️  WARNING: Base model mismatch!")
                print(f"    Embeddings were created with: {inferred_model}")
                print(f"    You specified: {base_model}")
                print(f"    Using detected model: {inferred_model}")
                base_model = inferred_model
            else:
                print(f"⚠️  Could not verify base model compatibility, proceeding with: {base_model}")

    print(f"🤖 Using base model: {base_model}")

    try:
        model_path = create_and_save_hf_model(
            ontology_embeddings_file=embeddings_file,
            model_name=model_name,
            output_dir=output_dir,
            base_model=base_model,
            fusion_method=fusion_method
        )

        print(f"✅ Model created successfully: {model_path}")

        # Generate model card
        print("📄 Generating model card...")
        create_model_card(
            model_path=model_path,
            model_name=model_name,
            ontology_file=ontology_file,
            embeddings_file=embeddings_file,
            fusion_method=fusion_method,
            training_config=training_config
        )

        # Generate upload instructions
        print("📤 Generating upload instructions...")
        create_upload_instructions(
            model_path=model_path,
            model_name=model_name
        )

        return model_path

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise


def test_model(model_path: str, test_queries: Optional[List[str]] = None) -> bool:
    """Test the created model with sample queries."""
    print(f"🧪 Testing model: {model_path}")

    if test_queries is None:
        test_queries = [
            "heart disease",
            "cardiovascular problems",
            "protein folding",
            "neurodegenerative disorders",
            "genetic mutations"
        ]

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim

        # Load model
        model = SentenceTransformer(model_path)
        print(f"📐 Model dimensions: {model.get_sentence_embedding_dimension()}")

        # Test encoding
        embeddings = model.encode(test_queries)
        print(f"✅ Encoded {len(test_queries)} queries: {embeddings.shape}")

        # Test similarity
        similarities = cos_sim(embeddings, embeddings)

        print("\n📊 Sample similarities:")
        for i in range(min(3, len(test_queries))):
            for j in range(i+1, min(3, len(test_queries))):
                sim = similarities[i][j].item()
                print(f"  {test_queries[i][:20]:20} <-> {test_queries[j][:20]:20}: {sim:.3f}")

        print("✅ Model testing completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False


def show_upload_instructions(model_path: str, model_name: str):
    """Show instructions for uploading to HuggingFace Hub."""
    print("\n🌐 HuggingFace Hub Upload Instructions")
    print("=" * 50)

    # Show model size
    model_path_obj = Path(model_path)
    total_size = sum(f.stat().st_size for f in model_path_obj.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"📦 Model size: {size_mb:.1f} MB")

    # Show files
    print("\n📁 Model files:")
    for file in sorted(model_path_obj.rglob("*")):
        if file.is_file():
            file_size = file.stat().st_size / (1024 * 1024)
            rel_path = file.relative_to(model_path_obj)
            print(f"  {str(rel_path):30} {file_size:>6.1f} MB")

    print(f"\n🚀 Upload commands:")
    print("# Install huggingface_hub if needed")
    print("pip install huggingface_hub")
    print()
    print("# Login to HuggingFace (one time setup)")
    print("huggingface-cli login")
    print()
    print("# Upload via Python")
    print("python -c \"")
    print("from sentence_transformers import SentenceTransformer")
    print(f"model = SentenceTransformer('{model_path}')")
    print(f"model.push_to_hub('your-username/{model_name}')")
    print("\"")
    print()
    print("# Or upload via CLI")
    print(f"huggingface-cli upload your-username/{model_name} {model_path}")
    print()
    print("📖 After upload, users can access with:")
    print("from sentence_transformers import SentenceTransformer")
    print(f"model = SentenceTransformer('your-username/{model_name}')")


def end_to_end_workflow(
    owl_file: str,
    model_name: str,
    output_dir: str = "./hf_models",
    embeddings_file: Optional[str] = None,
    base_model: str = "all-MiniLM-L6-v2",
    fusion_method: str = "concat",
    epochs: int = 100,
    skip_training: bool = False,
    skip_testing: bool = False
) -> bool:
    """Run the complete end-to-end workflow."""
    print("🚀 Starting End-to-End Workflow")
    print("=" * 50)
    print(f"🧬 OWL file: {owl_file}")
    print(f"🏷️ Model name: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Base model: {base_model}")
    print(f"🔗 Fusion method: {fusion_method}")
    print()

    try:
        # Step 1: Train ontology model (if not skipping)
        if embeddings_file is None:
            embeddings_file = f"{Path(owl_file).stem}_embeddings.parquet"

        if not skip_training:
            print("📚 Step 1: Training ontology model with text features")
            if not train_ontology_with_text(owl_file, embeddings_file, base_model, epochs):
                return False
        else:
            print("📚 Step 1: Skipping training (using existing embeddings)")

        # Step 2: Validate embeddings
        print("\n🔍 Step 2: Validating embeddings")
        metadata = validate_embeddings(embeddings_file)
        print(f"   Concepts: {metadata['num_concepts']:,}")
        print(f"   Text dim: {metadata['text_dim']}")
        print(f"   Structural dim: {metadata['structural_dim']}")

        # Step 3: Create HuggingFace model
        print(f"\n🏗️ Step 3: Creating HuggingFace model")

        # For end-to-end workflow, auto-detect unless explicitly specified
        # If user didn't specify base_model in e2e, let create_hf_model auto-detect
        create_base_model = base_model if base_model != 'all-MiniLM-L6-v2' else None

        # Prepare training config for model card
        training_config = {
            'model_type': 'gcn',  # Default - could be enhanced to detect from embeddings
            'epochs': epochs,
            'hidden_dim': 128,  # Default values - could be enhanced
            'out_dim': 64,
            'loss_fn': 'triplet'
        }

        model_path = create_hf_model(
            embeddings_file=embeddings_file,
            model_name=model_name,
            output_dir=output_dir,
            base_model=create_base_model,
            fusion_method=fusion_method,
            validate_first=False,  # Already validated
            ontology_file=owl_file,
            training_config=training_config
        )

        # Step 4: Test model (if not skipping)
        if not skip_testing:
            print(f"\n🧪 Step 4: Testing model")
            if not test_model(model_path):
                return False
        else:
            print(f"\n🧪 Step 4: Skipping model testing")

        # Step 5: Show upload instructions
        print(f"\n📤 Step 5: Upload preparation")
        show_upload_instructions(model_path, model_name)

        print("\n" + "=" * 50)
        print("✅ End-to-End Workflow Completed Successfully!")
        print(f"📦 Model ready at: {model_path}")
        print("🌐 Ready for HuggingFace Hub upload!")

        return True

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create HuggingFace Sentence Transformer models from on2vec ontology embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete end-to-end workflow
  python create_hf_model.py e2e biomedical.owl my-biomedical-model

  # Train ontology with text features
  python create_hf_model.py train biomedical.owl --output embeddings.parquet

  # Create HF model from existing embeddings
  python create_hf_model.py create embeddings.parquet my-model --fusion concat

  # Test existing model
  python create_hf_model.py test ./hf_models/my-model

Advanced examples:
  # Custom training configuration
  python create_hf_model.py train ontology.owl \\
    --text-model all-mpnet-base-v2 \\
    --epochs 200 \\
    --model-type gat \\
    --hidden-dim 256 \\
    --out-dim 128

  # Different fusion methods
  python create_hf_model.py create embeddings.parquet my-model \\
    --fusion gated \\
    --base-model distilbert-base-nli-mean-tokens
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # End-to-end workflow
    e2e_parser = subparsers.add_parser('e2e', help='Run complete end-to-end workflow')
    e2e_parser.add_argument('owl_file', help='Path to OWL ontology file')
    e2e_parser.add_argument('model_name', help='Name for the HuggingFace model')
    e2e_parser.add_argument('--output-dir', default='./hf_models', help='Output directory for models')
    e2e_parser.add_argument('--embeddings-file', help='Path to embeddings file (will be generated if not provided)')
    e2e_parser.add_argument('--base-model', default='all-MiniLM-L6-v2', help='Base Sentence Transformer model')
    e2e_parser.add_argument('--fusion', choices=['concat', 'weighted_avg', 'attention', 'gated'],
                            default='concat', help='Fusion method')
    e2e_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    e2e_parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    e2e_parser.add_argument('--skip-testing', action='store_true', help='Skip model testing')

    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train ontology model with text features')
    train_parser.add_argument('owl_file', help='Path to OWL ontology file')
    train_parser.add_argument('--output', required=True, help='Output parquet file path')
    train_parser.add_argument('--text-model', default='all-MiniLM-L6-v2', help='Text model name')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--model-type', choices=['gcn', 'gat', 'rgcn'], default='gcn', help='GNN model type')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    train_parser.add_argument('--out-dim', type=int, default=64, help='Output dimension')
    train_parser.add_argument('--loss-fn', choices=['triplet', 'contrastive', 'cosine', 'cross_entropy'],
                              default='triplet', help='Loss function')

    # Model creation subcommand
    create_parser = subparsers.add_parser('create', help='Create HuggingFace model from embeddings')
    create_parser.add_argument('embeddings_file', help='Path to embeddings parquet file')
    create_parser.add_argument('model_name', help='Name for the model')
    create_parser.add_argument('--output-dir', default='./hf_models', help='Output directory')
    create_parser.add_argument('--base-model', help='Base Sentence Transformer model (auto-detected if not specified)')
    create_parser.add_argument('--fusion', choices=['concat', 'weighted_avg', 'attention', 'gated'],
                               default='concat', help='Fusion method')
    create_parser.add_argument('--no-validate', action='store_true', help='Skip embeddings validation')

    # Model testing subcommand
    test_parser = subparsers.add_parser('test', help='Test HuggingFace model')
    test_parser.add_argument('model_path', help='Path to model directory')
    test_parser.add_argument('--queries', nargs='+', help='Custom test queries')

    # Validation subcommand
    validate_parser = subparsers.add_parser('validate', help='Validate embeddings file')
    validate_parser.add_argument('embeddings_file', help='Path to embeddings parquet file')

    # Upload instructions subcommand
    upload_parser = subparsers.add_parser('upload-info', help='Show upload instructions for model')
    upload_parser.add_argument('model_path', help='Path to model directory')
    upload_parser.add_argument('model_name', help='Model name for Hub')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == 'e2e':
            success = end_to_end_workflow(
                owl_file=args.owl_file,
                model_name=args.model_name,
                output_dir=args.output_dir,
                embeddings_file=args.embeddings_file,
                base_model=args.base_model,
                fusion_method=args.fusion,
                epochs=args.epochs,
                skip_training=args.skip_training,
                skip_testing=args.skip_testing
            )
            return 0 if success else 1

        elif args.command == 'train':
            success = train_ontology_with_text(
                owl_file=args.owl_file,
                output_file=args.output,
                text_model=args.text_model,
                epochs=args.epochs,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                loss_fn=args.loss_fn
            )
            return 0 if success else 1

        elif args.command == 'create':
            model_path = create_hf_model(
                embeddings_file=args.embeddings_file,
                model_name=args.model_name,
                output_dir=args.output_dir,
                base_model=args.base_model,
                fusion_method=args.fusion,
                validate_first=not args.no_validate
            )
            print(f"\n🎉 Success! Model created at: {model_path}")
            return 0

        elif args.command == 'test':
            success = test_model(args.model_path, args.queries)
            return 0 if success else 1

        elif args.command == 'validate':
            validate_embeddings(args.embeddings_file)
            return 0

        elif args.command == 'upload-info':
            show_upload_instructions(args.model_path, args.model_name)
            return 0

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())