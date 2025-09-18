#!/usr/bin/env python3
"""
HuggingFace integration workflows for on2vec.

This module contains all the HuggingFace Sentence Transformers integration
functionality that was previously in create_hf_model.py and batch_hf_models.py.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .workflows import train_and_embed_workflow
from .sentence_transformer_hub import create_and_save_hf_model
from .io import inspect_parquet_metadata
from .metadata_utils import get_base_model_from_embeddings, get_embedding_info, validate_text_embeddings_compatibility
from .model_card_generator import create_model_card, create_upload_instructions

logger = logging.getLogger(__name__)


def train_ontology_with_text(
    owl_file: str,
    output_file: str,
    text_model: str = "all-MiniLM-L6-v2",
    epochs: int = 100,
    model_type: str = "gcn",
    hidden_dim: int = 128,
    out_dim: int = 64,
    loss_fn: str = "triplet",
    use_multi_relation: bool = False
) -> bool:
    """Train an ontology model with text features enabled."""
    print(f"🧬 Training ontology model: {owl_file}")
    print(f"📊 Output: {output_file}")
    print(f"🤖 Text model: {text_model}")
    print(f"⚙️ Config: {model_type}, hidden={hidden_dim}, out={out_dim}, loss={loss_fn}, epochs={epochs}")

    try:
        # Use the workflow function instead of subprocess
        result = train_and_embed_workflow(
            owl_file=owl_file,
            model_type=model_type,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
            output=output_file,
            model_output=f"{Path(output_file).stem}_model.pt",
            loss_fn=loss_fn,
            use_text_features=True,
            text_model_name=text_model,
            fusion_method="concat",
            use_multi_relation=use_multi_relation
        )

        print("✅ Training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        return False


def validate_embeddings(embeddings_file: str) -> Dict[str, Any]:
    """Validate embeddings file and return metadata."""
    print(f"🔍 Validating embeddings: {embeddings_file}")

    try:
        # Use the existing inspection function
        inspect_parquet_metadata(embeddings_file)

        # Get detailed metadata
        embedding_info = get_embedding_info(embeddings_file)

        print("✅ Embeddings validation passed!")
        print(f"   Concepts: {embedding_info.get('num_embeddings', 'Unknown'):,}")
        print(f"   Text dim: {embedding_info.get('text_dim', 'Unknown')}")
        print(f"   Structural dim: {embedding_info.get('structural_dim', 'Unknown')}")

        return embedding_info

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logger.error(f"Validation failed: {e}")
        raise


def create_hf_model(
    embeddings_file: str,
    model_name: str,
    output_dir: str = "./hf_models",
    base_model: Optional[str] = None,
    fusion_method: str = "concat",
    validate_first: bool = True,
    ontology_file: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None,
    model_details: Optional[Dict[str, Any]] = None,
    upload_options: Optional[Dict[str, Any]] = None
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

        # Generate model card with enhanced details
        print("📄 Generating model card...")
        create_model_card(
            model_path=model_path,
            model_name=model_name,
            ontology_file=ontology_file,
            embeddings_file=embeddings_file,
            fusion_method=fusion_method,
            training_config=training_config
        )

        # Handle upload if requested
        if upload_options and upload_options.get('upload'):
            print("🚀 Uploading to HuggingFace Hub...")
            success = upload_to_hf_hub(
                model_path=model_path,
                hub_name=upload_options.get('hub_name', f'your-username/{model_name}'),
                private=upload_options.get('private', False),
                commit_message=upload_options.get('commit_message')
            )
            if success:
                print(f"✅ Model uploaded successfully to {upload_options.get('hub_name')}")
            else:
                print("⚠️  Upload failed, but model was created locally")
        else:
            # Generate upload instructions
            print("📤 Generating upload instructions...")
            create_upload_instructions(
                model_path=model_path,
                model_name=model_name,
                hub_name=upload_options.get('hub_name') if upload_options else None
            )

        return model_path

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        logger.error(f"Model creation failed: {e}")
        raise


def validate_hf_model(model_path: str, test_queries: Optional[List[str]] = None) -> bool:
    """Test the created model with sample queries."""
    print(f"🧪 Testing model: {model_path}")

    if test_queries is None:
        test_queries = [
            "heart disease",
            "cardiovascular problems",
            "protein folding",
            "gene expression",
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
        logger.error(f"Model testing failed: {e}")
        return False


def upload_to_hf_hub(
    model_path: str,
    hub_name: str,
    private: bool = False,
    commit_message: Optional[str] = None
) -> bool:
    """Upload model to HuggingFace Hub."""
    try:
        from sentence_transformers import SentenceTransformer

        # Load the model
        model = SentenceTransformer(model_path)

        # Set commit message
        if not commit_message:
            commit_message = f"Upload {Path(model_path).name} model created with on2vec"

        # Push to hub
        model.push_to_hub(
            repo_id=hub_name,
            private=private,
            commit_message=commit_message
        )

        print(f"✅ Model successfully uploaded to: https://huggingface.co/{hub_name}")
        return True

    except ImportError:
        print("❌ Upload failed: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        logger.error(f"Upload failed: {e}")
        return False


def show_upload_instructions(model_path: str, model_name: str, hub_name: Optional[str] = None):
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
    skip_training: bool = False,
    skip_testing: bool = False,
    training_config: Optional[Dict[str, Any]] = None,
    model_details: Optional[Dict[str, Any]] = None,
    upload_options: Optional[Dict[str, Any]] = None
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

            # Extract training parameters
            train_epochs = training_config.get('epochs', 100) if training_config else 100
            train_model_type = training_config.get('model_type', 'gcn') if training_config else 'gcn'
            train_hidden_dim = training_config.get('hidden_dim', 128) if training_config else 128
            train_out_dim = training_config.get('out_dim', 64) if training_config else 64
            train_loss_fn = training_config.get('loss_fn', 'triplet') if training_config else 'triplet'
            train_use_multi_relation = training_config.get('use_multi_relation', False) if training_config else False
            train_text_model = training_config.get('text_model', base_model) if training_config else base_model

            if not train_ontology_with_text(
                owl_file=owl_file,
                output_file=embeddings_file,
                text_model=train_text_model,
                epochs=train_epochs,
                model_type=train_model_type,
                hidden_dim=train_hidden_dim,
                out_dim=train_out_dim,
                loss_fn=train_loss_fn,
                use_multi_relation=train_use_multi_relation
            ):
                return False
        else:
            print("📚 Step 1: Skipping training (using existing embeddings)")

            # Still create training_config for model card if not provided
            if not training_config:
                training_config = {
                    'model_type': 'gcn',
                    'epochs': 100,
                    'hidden_dim': 128,
                    'out_dim': 64,
                    'loss_fn': 'triplet'
                }

        # Step 2: Validate embeddings
        print("\n🔍 Step 2: Validating embeddings")
        metadata = validate_embeddings(embeddings_file)
        print(f"   Concepts: {metadata['num_embeddings']:,}")
        print(f"   Text dim: {metadata['text_dim']}")
        print(f"   Structural dim: {metadata['structural_dim']}")

        # Step 3: Create HuggingFace model
        print(f"\n🏗️ Step 3: Creating HuggingFace model")

        # For end-to-end workflow, auto-detect unless explicitly specified
        # If user didn't specify base_model in e2e, let create_hf_model auto-detect
        create_base_model = base_model if base_model != 'all-MiniLM-L6-v2' else None

        model_path = create_hf_model(
            embeddings_file=embeddings_file,
            model_name=model_name,
            output_dir=output_dir,
            base_model=create_base_model,
            fusion_method=fusion_method,
            validate_first=False,  # Already validated
            ontology_file=owl_file,
            training_config=training_config,
            model_details=model_details,
            upload_options=upload_options
        )

        # Step 4: Test model (if not skipping)
        if not skip_testing:
            print(f"\n🧪 Step 4: Testing model")
            if not validate_hf_model(model_path):
                return False
        else:
            print(f"\n🧪 Step 4: Skipping model testing")

        # Step 5: Show upload instructions (if not uploaded automatically)
        if not (upload_options and upload_options.get('upload')):
            print(f"\n📤 Step 5: Upload preparation")
            show_upload_instructions(model_path, model_name)

        print("\n" + "=" * 50)
        print("✅ End-to-End Workflow Completed Successfully!")
        print(f"📦 Model ready at: {model_path}")
        print("🌐 Ready for HuggingFace Hub upload!")

        return True

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        logger.error(f"Workflow failed: {e}")
        return False