"""
Embedding generation utilities
"""

import torch
import logging
from .training import load_model_checkpoint
from .ontology import build_graph_from_owl, build_multi_relation_graph_from_owl, align_ontology_with_training
from .device_utils import get_device, move_to_device

logger = logging.getLogger(__name__)


def generate_embeddings_from_model(model, x, edge_index, new_to_training_idx=None, node_ids=None, edge_type=None, relation_to_index=None, text_x=None, device=None):
    """
    Generate embeddings using a trained model.

    Args:
        model (torch.nn.Module): Trained model
        x (torch.Tensor): Node features (structural)
        edge_index (torch.Tensor): Graph edge indices
        new_to_training_idx (dict, optional): Mapping from new indices to training indices
        node_ids (list, optional): List of node IDs to use
        edge_type (torch.Tensor, optional): Edge types for multi-relation models
        relation_to_index (dict, optional): Mapping from relation names to indices
        text_x (torch.Tensor, optional): Text features for TextAugmentedOntologyGNN
        device (str or torch.device, optional): Device to use for inference

    Returns:
        tuple: (embeddings, used_node_ids, extra_info)
            - embeddings (torch.Tensor): Generated embeddings
            - used_node_ids (list): Node IDs corresponding to embeddings
            - extra_info (dict): Additional information including separate embeddings
    """
    # Get device (use model's device if available, otherwise auto-detect)
    if device is None:
        if hasattr(model, 'device'):
            inference_device = model.device
        else:
            inference_device = get_device('auto', verbose=False)
            model = model.to(inference_device)
    else:
        inference_device = get_device(device, verbose=False)
        model = model.to(inference_device)

    # Move data to device
    x = move_to_device(x, inference_device)
    edge_index = move_to_device(edge_index, inference_device)
    if edge_type is not None:
        edge_type = move_to_device(edge_type, inference_device)
    if text_x is not None:
        text_x = move_to_device(text_x, inference_device)

    logger.info("Generating embeddings...")

    model.eval()
    with torch.no_grad():
        # Forward pass - handle different model types
        from .models import TextAugmentedOntologyGNN

        # Initialize extra info dictionary
        extra_info = {
            'text_embeddings': None,
            'structural_embeddings': None,
            'text_model_info': None,
            'embedding_types': ['fused']  # Always have fused embeddings
        }

        if isinstance(model, TextAugmentedOntologyGNN):
            # Text-augmented model - we can extract separate embeddings
            if text_x is None:
                # Generate zero text features as fallback for inference
                logger.warning("TextAugmentedOntologyGNN: No text features provided, using zero features as fallback")
                text_x = torch.zeros((x.shape[0], model.text_dim), dtype=x.dtype, device=x.device)

            # For text-augmented models, we want to capture intermediate embeddings
            # First, get structural embeddings (without text)
            zero_text = torch.zeros_like(text_x)

            # Handle multi-relation text-augmented models
            if hasattr(model, 'num_relations') and edge_type is not None:
                if hasattr(model, 'relation_types') and relation_to_index is not None:
                    # Heterogeneous text-augmented model
                    all_embeddings = model(x, text_x, edge_index, edge_type, relation_to_index)
                    structural_only = model(x, zero_text, edge_index, edge_type, relation_to_index)
                else:
                    # RGCN or weighted GCN text-augmented model
                    all_embeddings = model(x, text_x, edge_index, edge_type)
                    structural_only = model(x, zero_text, edge_index, edge_type)
            else:
                # Standard text-augmented model
                all_embeddings = model(x, text_x, edge_index)
                structural_only = model(x, zero_text, edge_index)

            # Store separate embeddings
            extra_info['structural_embeddings'] = structural_only
            extra_info['embedding_types'].extend(['structural', 'text'])

            # Generate pure text embeddings - just use the provided text features
            if text_x is not None:
                # The text_x already contains the text embeddings from the text model
                extra_info['text_embeddings'] = text_x
                logger.info(f"Captured separate text embeddings with dimension {text_x.shape[1]}")

            # Store text model information if available (from model attributes or config)
            text_model_info = None
            if hasattr(model, 'text_model_type') and hasattr(model, 'text_model_name'):
                text_model_info = {
                    'model_type': model.text_model_type,
                    'model_name': model.text_model_name,
                    'text_dim': getattr(model, 'text_dim', text_x.shape[1] if text_x is not None else None)
                }

            if text_model_info:
                extra_info['text_model_info'] = text_model_info

        elif hasattr(model, 'num_relations') and edge_type is not None:
            # Multi-relation model
            if hasattr(model, 'relation_types') and relation_to_index is not None:
                # Heterogeneous model needs relation mapping
                all_embeddings = model(x, edge_index, edge_type, relation_to_index)
            else:
                # RGCN or weighted GCN model
                all_embeddings = model(x, edge_index, edge_type)
        else:
            # Standard model
            all_embeddings = model(x, edge_index)

        if new_to_training_idx is not None:
            # Extract only embeddings for nodes that were in training
            aligned_embeddings = []
            aligned_node_ids = []

            for new_idx, training_idx in new_to_training_idx.items():
                aligned_embeddings.append(all_embeddings[new_idx])
                if node_ids and new_idx < len(node_ids):
                    aligned_node_ids.append(node_ids[new_idx])
                else:
                    aligned_node_ids.append(f"node_{new_idx}")

            if aligned_embeddings:
                embeddings_tensor = torch.stack(aligned_embeddings)

                # Also align the extra embeddings if they exist
                if extra_info['text_embeddings'] is not None:
                    aligned_text = []
                    for new_idx, _ in new_to_training_idx.items():
                        aligned_text.append(extra_info['text_embeddings'][new_idx])
                    extra_info['text_embeddings'] = torch.stack(aligned_text) if aligned_text else None

                if extra_info['structural_embeddings'] is not None:
                    aligned_structural = []
                    for new_idx, _ in new_to_training_idx.items():
                        aligned_structural.append(extra_info['structural_embeddings'][new_idx])
                    extra_info['structural_embeddings'] = torch.stack(aligned_structural) if aligned_structural else None

                logger.info(f"Generated {embeddings_tensor.shape[0]} aligned embeddings of dimension {embeddings_tensor.shape[1]}")
                return embeddings_tensor, aligned_node_ids, extra_info
            else:
                logger.error("No aligned embeddings found!")
                return None, [], extra_info
        else:
            # Return all embeddings
            if node_ids is None:
                node_ids = [f"node_{i}" for i in range(all_embeddings.shape[0])]

            logger.info(f"Generated {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
            return all_embeddings, node_ids, extra_info


def embed_ontology_with_model(model_path, owl_file, output_file=None, device=None):
    """
    Generate embeddings for an ontology using a pre-trained model.

    Args:
        model_path (str): Path to trained model checkpoint
        owl_file (str): Path to OWL ontology file
        output_file (str, optional): Path to save embeddings
        device (str or torch.device, optional): Device to use for inference

    Returns:
        dict: Dictionary containing embeddings and metadata
            - embeddings (torch.Tensor): Generated embeddings
            - node_ids (list): Corresponding node IDs
            - model_config (dict): Model configuration
            - alignment_info (dict): Information about ontology alignment
    """
    # Load the trained model
    model, checkpoint = load_model_checkpoint(model_path)

    # Check if model was trained with multi-relation data
    model_config = checkpoint['model_config']
    use_multi_relation = (
        model_config.get('use_multi_relation', False) or
        model_config.get('model_type') in ['rgcn', 'weighted_gcn', 'heterogeneous'] or
        model_config.get('backbone_model') in ['rgcn', 'weighted_gcn', 'heterogeneous']
    )

    # Build graph from the new OWL file
    logger.info(f"Loading OWL ontology from {owl_file}")
    if use_multi_relation:
        logger.info("Using multi-relation graph building for consistency with training")
        graph_data = build_multi_relation_graph_from_owl(owl_file)
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_type = graph_data['edge_types']
        class_to_index = graph_data['class_to_index']
        relation_to_index = graph_data.get('relation_to_index')
    else:
        logger.info("Using standard graph building")
        x, edge_index, class_to_index = build_graph_from_owl(owl_file)
        edge_type = None
        relation_to_index = None

    # Align new ontology with training data
    training_class_to_index = checkpoint['class_to_index']
    new_to_training_idx, training_node_ids = align_ontology_with_training(class_to_index, training_class_to_index)

    alignment_info = {
        'total_new_classes': len(class_to_index),
        'total_training_classes': len(training_class_to_index),
        'aligned_classes': len(new_to_training_idx),
        'alignment_ratio': len(new_to_training_idx) / len(class_to_index) if class_to_index else 0
    }

    if not new_to_training_idx:
        logger.error("No matching classes found between training and target ontology!")
        return {
            'embeddings': None,
            'node_ids': [],
            'model_config': checkpoint['model_config'],
            'alignment_info': alignment_info
        }

    # Generate embeddings
    embeddings, node_ids, extra_info = generate_embeddings_from_model(
        model, x, edge_index, new_to_training_idx, training_node_ids,
        edge_type=edge_type, relation_to_index=relation_to_index, device=device
    )

    # Add text model info from checkpoint if not already present
    if extra_info.get('text_model_info') is None and 'text_model_type' in model_config:
        extra_info['text_model_info'] = {
            'model_type': model_config.get('text_model_type', 'unknown'),
            'model_name': model_config.get('text_model_name', 'unknown'),
            'text_dim': model_config.get('text_dim')
        }

    result = {
        'embeddings': embeddings,
        'node_ids': node_ids,
        'model_config': checkpoint['model_config'],
        'alignment_info': alignment_info
    }

    # Save embeddings if output file is specified
    if output_file and embeddings is not None:
        from .io import save_embeddings_to_parquet, create_embedding_metadata

        # Create metadata for the embeddings
        metadata = create_embedding_metadata(
            owl_file=owl_file,
            model_config=checkpoint['model_config'],
            alignment_info=alignment_info,
            text_model_info=extra_info.get('text_model_info'),
            embedding_types=extra_info.get('embedding_types', ['fused'])
        )

        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata=metadata,
            text_embeddings=extra_info.get('text_embeddings'),
            structural_embeddings=extra_info.get('structural_embeddings')
        )
        result['output_file'] = output_file
        result['metadata'] = metadata

    return result


def embed_same_ontology(model_path, owl_file, output_file=None, device=None):
    """
    Generate embeddings for the same ontology used in training.
    This is more efficient as it doesn't require alignment.

    Args:
        model_path (str): Path to trained model checkpoint
        owl_file (str): Path to OWL ontology file (same as training)
        output_file (str, optional): Path to save embeddings
        device (str or torch.device, optional): Device to use for inference

    Returns:
        dict: Dictionary containing embeddings and metadata
    """
    # Load the trained model
    model, checkpoint = load_model_checkpoint(model_path)

    # Check if model was trained with multi-relation data
    model_config = checkpoint['model_config']
    use_multi_relation = (
        model_config.get('use_multi_relation', False) or
        model_config.get('model_type') in ['rgcn', 'weighted_gcn', 'heterogeneous'] or
        model_config.get('backbone_model') in ['rgcn', 'weighted_gcn', 'heterogeneous']
    )

    # Check if model is text-augmented
    is_text_augmented = model_config.get('model_type') == 'text_augmented'

    # Build graph from the OWL file
    if use_multi_relation:
        logger.info("Using multi-relation graph building for consistency with training")
        graph_data = build_multi_relation_graph_from_owl(owl_file)
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_type = graph_data['edge_types']
        class_to_index = graph_data['class_to_index']
        relation_to_index = graph_data.get('relation_to_index')
    else:
        logger.info("Using standard graph building")
        x, edge_index, class_to_index = build_graph_from_owl(owl_file)
        edge_type = None
        relation_to_index = None

    # Use the node IDs from the checkpoint for consistency
    training_node_ids = checkpoint['node_ids']

    # Generate text features if model is text-augmented
    text_x = None
    if is_text_augmented:
        logger.info("Text-augmented model detected, generating text features...")
        try:
            from .text_features import (
                extract_rich_semantic_features_from_owl,
                create_text_embedding_model,
                create_text_node_features
            )

            # Extract text features
            text_features = extract_rich_semantic_features_from_owl(owl_file)

            # Recreate text embedding model with same config
            text_model_type = model_config.get('text_model_type', 'sentence_transformer')
            text_model_name = model_config.get('text_model_name', 'all-MiniLM-L6-v2')

            text_embedding_model = create_text_embedding_model(
                text_model_type,
                model_name=text_model_name
            )

            # Generate text embeddings
            text_x = create_text_node_features(
                text_features,
                class_to_index,
                text_embedding_model,
                use_combined=True
            )

            logger.info(f"Generated text features with shape: {text_x.shape}")

        except Exception as e:
            logger.error(f"Failed to generate text features: {e}")
            raise ValueError(f"Text-augmented model requires text features, but generation failed: {e}")

    # Generate embeddings (no alignment needed)
    embeddings, node_ids, extra_info = generate_embeddings_from_model(
        model, x, edge_index, node_ids=training_node_ids,
        edge_type=edge_type, relation_to_index=relation_to_index,
        text_x=text_x, device=device
    )

    # Add text model info from checkpoint if not already present
    if extra_info.get('text_model_info') is None and 'text_model_type' in model_config:
        extra_info['text_model_info'] = {
            'model_type': model_config.get('text_model_type', 'unknown'),
            'model_name': model_config.get('text_model_name', 'unknown'),
            'text_dim': model_config.get('text_dim')
        }

    result = {
        'embeddings': embeddings,
        'node_ids': node_ids,
        'model_config': checkpoint['model_config'],
        'alignment_info': {
            'total_classes': len(class_to_index),
            'aligned_classes': len(class_to_index),
            'alignment_ratio': 1.0
        }
    }

    # Save embeddings if output file is specified
    if output_file and embeddings is not None:
        from .io import save_embeddings_to_parquet, create_embedding_metadata

        # Create metadata for the embeddings
        metadata = create_embedding_metadata(
            owl_file=owl_file,
            model_config=checkpoint['model_config'],
            alignment_info=result['alignment_info'],
            text_model_info=extra_info.get('text_model_info'),
            embedding_types=extra_info.get('embedding_types', ['fused'])
        )

        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata=metadata,
            text_embeddings=extra_info.get('text_embeddings'),
            structural_embeddings=extra_info.get('structural_embeddings')
        )
        result['output_file'] = output_file
        result['metadata'] = metadata

    return result