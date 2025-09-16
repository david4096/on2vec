"""
Embedding generation utilities
"""

import torch
import logging
from .training import load_model_checkpoint
from .ontology import build_graph_from_owl, build_multi_relation_graph_from_owl, align_ontology_with_training

logger = logging.getLogger(__name__)


def generate_embeddings_from_model(model, x, edge_index, new_to_training_idx=None, node_ids=None, edge_type=None, relation_to_index=None):
    """
    Generate embeddings using a trained model.

    Args:
        model (torch.nn.Module): Trained model
        x (torch.Tensor): Node features
        edge_index (torch.Tensor): Graph edge indices
        new_to_training_idx (dict, optional): Mapping from new indices to training indices
        node_ids (list, optional): List of node IDs to use
        edge_type (torch.Tensor, optional): Edge types for multi-relation models
        relation_to_index (dict, optional): Mapping from relation names to indices

    Returns:
        tuple: (embeddings, used_node_ids)
            - embeddings (torch.Tensor): Generated embeddings
            - used_node_ids (list): Node IDs corresponding to embeddings
    """
    logger.info("Generating embeddings...")

    model.eval()
    with torch.no_grad():
        # Forward pass - handle multi-relation models
        if hasattr(model, 'num_relations') and edge_type is not None:
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
                logger.info(f"Generated {embeddings_tensor.shape[0]} aligned embeddings of dimension {embeddings_tensor.shape[1]}")
                return embeddings_tensor, aligned_node_ids
            else:
                logger.error("No aligned embeddings found!")
                return None, []
        else:
            # Return all embeddings
            if node_ids is None:
                node_ids = [f"node_{i}" for i in range(all_embeddings.shape[0])]

            logger.info(f"Generated {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
            return all_embeddings, node_ids


def embed_ontology_with_model(model_path, owl_file, output_file=None):
    """
    Generate embeddings for an ontology using a pre-trained model.

    Args:
        model_path (str): Path to trained model checkpoint
        owl_file (str): Path to OWL ontology file
        output_file (str, optional): Path to save embeddings

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
        model_config.get('model_type') in ['rgcn', 'weighted_gcn', 'heterogeneous']
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
    embeddings, node_ids = generate_embeddings_from_model(
        model, x, edge_index, new_to_training_idx, training_node_ids,
        edge_type=edge_type, relation_to_index=relation_to_index
    )

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
            alignment_info=alignment_info
        )

        save_embeddings_to_parquet(embeddings, node_ids, output_file, metadata=metadata)
        result['output_file'] = output_file
        result['metadata'] = metadata

    return result


def embed_same_ontology(model_path, owl_file, output_file=None):
    """
    Generate embeddings for the same ontology used in training.
    This is more efficient as it doesn't require alignment.

    Args:
        model_path (str): Path to trained model checkpoint
        owl_file (str): Path to OWL ontology file (same as training)
        output_file (str, optional): Path to save embeddings

    Returns:
        dict: Dictionary containing embeddings and metadata
    """
    # Load the trained model
    model, checkpoint = load_model_checkpoint(model_path)

    # Check if model was trained with multi-relation data
    model_config = checkpoint['model_config']
    use_multi_relation = (
        model_config.get('use_multi_relation', False) or
        model_config.get('model_type') in ['rgcn', 'weighted_gcn', 'heterogeneous']
    )

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

    # Generate embeddings (no alignment needed)
    embeddings, node_ids = generate_embeddings_from_model(
        model, x, edge_index, node_ids=training_node_ids,
        edge_type=edge_type, relation_to_index=relation_to_index
    )

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
            alignment_info=result['alignment_info']
        )

        save_embeddings_to_parquet(embeddings, node_ids, output_file, metadata=metadata)
        result['output_file'] = output_file
        result['metadata'] = metadata

    return result