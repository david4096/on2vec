"""
Training utilities for ontology embedding models
"""

import torch
from torch.optim import Adam
import time
import logging
from .models import OntologyGNN, MultiRelationOntologyGNN, HeterogeneousOntologyGNN, TextAugmentedOntologyGNN
from .loss_functions import get_loss_function
from .device_utils import get_device, move_to_device, get_memory_usage, optimize_for_device, log_device_performance_tips

logger = logging.getLogger(__name__)


def train_model(model, x, edge_index, optimizer, loss_fn, epochs=100, edge_type=None, text_x=None, device=None):
    """
    Train a GNN model on ontology data.

    Args:
        model (torch.nn.Module): The GNN model to train
        x (torch.Tensor): Node features (structural features)
        edge_index (torch.Tensor): Graph edge indices
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (callable): Loss function
        epochs (int): Number of training epochs
        edge_type (torch.Tensor, optional): Edge types for multi-relation models
        text_x (torch.Tensor, optional): Text features for TextAugmentedOntologyGNN
        device (str or torch.device, optional): Device to use for training

    Returns:
        torch.nn.Module: Trained model
    """
    # Get device (use model's device if available, otherwise auto-detect)
    if device is None:
        if hasattr(model, 'device'):
            training_device = model.device
        else:
            training_device = get_device('auto', verbose=True)
            model = model.to(training_device)
    else:
        training_device = get_device(device, verbose=True)
        model = model.to(training_device)

    # Move data to device
    x = move_to_device(x, training_device)
    edge_index = move_to_device(edge_index, training_device)
    if edge_type is not None:
        edge_type = move_to_device(edge_type, training_device)
    if text_x is not None:
        text_x = move_to_device(text_x, training_device)

    # Log device and performance tips
    logger.info(f"Training on device: {training_device}")
    log_device_performance_tips(training_device)

    # Get device-specific optimization settings
    device_settings = optimize_for_device(training_device)

    model.train()
    start_time = time.time()

    # Report memory usage if on GPU
    if training_device.type in ['cuda']:
        used_mem, total_mem = get_memory_usage(training_device)
        logger.info(f"GPU Memory: {used_mem:.1f}/{total_mem:.1f} GB")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs} starting...")
        epoch_start_time = time.time()
        optimizer.zero_grad()

        # Forward pass - handle different model types
        if isinstance(model, TextAugmentedOntologyGNN):
            # Text-augmented model
            if text_x is None:
                raise ValueError("TextAugmentedOntologyGNN requires text_x features")

            # Handle multi-relation text-augmented models
            if model.model_type in ['rgcn', 'weighted_gcn', 'heterogeneous']:
                if edge_type is None:
                    raise ValueError(f"Text-augmented {model.model_type} model requires edge_type")

                if model.model_type == 'heterogeneous':
                    # Heterogeneous model needs relation mapping
                    relation_to_index = {rel: i for i, rel in enumerate(model.relation_types)}
                    out = model(x, text_x, edge_index, edge_type, relation_to_index)
                else:
                    # RGCN or weighted GCN
                    out = model(x, text_x, edge_index, edge_type)
            else:
                # Standard GCN/GAT text-augmented model
                out = model(x, text_x, edge_index)
        elif hasattr(model, 'num_relations') and edge_type is not None:
            # Multi-relation model
            if hasattr(model, 'relation_types'):
                # Heterogeneous model needs relation mapping
                relation_to_index = {rel: i for i, rel in enumerate(model.relation_types)}
                out = model(x, edge_index, edge_type, relation_to_index)
            else:
                # RGCN or weighted GCN model
                out = model(x, edge_index, edge_type)
        else:
            # Standard model
            out = model(x, edge_index)

        loss = loss_fn(out, edge_index)

        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} complete, Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss {loss.item():.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training complete. Total Time: {total_time:.2f}s")

    return model


def save_model_checkpoint(model, class_to_index, output_path, relation_data=None):
    """
    Save model checkpoint with metadata.

    Args:
        model (torch.nn.Module): Trained model
        class_to_index (dict): Class-to-index mapping from ontology
        output_path (str): Path to save the checkpoint
        relation_data (dict, optional): Multi-relation graph data

    Returns:
        None
    """
    logger.info(f"Saving model checkpoint to {output_path}")

    # Extract node IDs (IRIs) from class_to_index
    node_ids = [cls.iri for cls in class_to_index.keys()]

    # Base model config
    # Detect model type from class name
    if isinstance(model, TextAugmentedOntologyGNN):
        model_type = 'text_augmented'
        backbone_model = getattr(model, 'model_type', 'gcn')  # The backbone GNN type
    elif isinstance(model, HeterogeneousOntologyGNN):
        model_type = 'heterogeneous'
        backbone_model = None
    elif isinstance(model, MultiRelationOntologyGNN):
        model_type = getattr(model, 'model_type', 'rgcn')  # rgcn or weighted_gcn
        backbone_model = None
    else:
        model_type = getattr(model, 'model_type', 'gcn')  # Standard models
        backbone_model = None

    model_config = {
        'model_type': model_type,
        'input_dim': getattr(model, 'input_dim', None),
        'hidden_dim': getattr(model, 'hidden_dim', None),
        'out_dim': getattr(model, 'out_dim', None)
    }

    # Add backbone model for text-augmented models
    if backbone_model is not None:
        model_config['backbone_model'] = backbone_model

    # Add text-augmented specific config
    if isinstance(model, TextAugmentedOntologyGNN):
        model_config['structural_dim'] = getattr(model, 'structural_dim', None)
        model_config['text_dim'] = getattr(model, 'text_dim', None)
        model_config['fusion_method'] = getattr(model, 'fusion_method', 'concat')
        model_config['text_model_type'] = getattr(model, 'text_model_type', 'unknown')
        model_config['text_model_name'] = getattr(model, 'text_model_name', 'unknown')

    # Add multi-relation specific config
    if hasattr(model, 'num_relations'):
        model_config['num_relations'] = model.num_relations
        model_config['dropout'] = getattr(model, 'dropout', 0.0)

        if hasattr(model, 'relation_types'):
            model_config['relation_types'] = model.relation_types

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'class_to_index': {cls.iri: idx for cls, idx in class_to_index.items()},
        'node_ids': node_ids,
        'num_nodes': len(node_ids)
    }

    # Add relation data if provided
    if relation_data:
        checkpoint['relation_data'] = {
            'relation_to_index': relation_data.get('relation_to_index', {}),
            'relation_names': relation_data.get('relation_names', []),
            'edge_type_counts': relation_data.get('edge_type_counts', {})
        }

    torch.save(checkpoint, output_path)
    logger.info(f"Model checkpoint saved to {output_path}")


def load_model_checkpoint(checkpoint_path):
    """
    Load model checkpoint and return model with metadata.

    Args:
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        tuple: (model, checkpoint_data)
            - model (torch.nn.Module): Loaded model in eval mode
            - checkpoint_data (dict): Checkpoint metadata
    """
    logger.info(f"Loading model checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model configuration
    config = checkpoint['model_config']
    model_type = config['model_type']

    # Recreate the appropriate model based on type
    if model_type in ['rgcn', 'weighted_gcn']:
        # Multi-relation model
        if 'num_relations' not in config:
            raise ValueError(f"{model_type} model requires num_relations in checkpoint config")
        model = MultiRelationOntologyGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            num_relations=config['num_relations'],
            model_type=model_type,
            dropout=config.get('dropout', 0.0)
        )
    elif model_type == 'heterogeneous':
        # Heterogeneous model
        model = HeterogeneousOntologyGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            relation_types=config['relation_types'],
            dropout=config.get('dropout', 0.0)
        )
    elif model_type == 'text_augmented':
        # Text-augmented model
        model = TextAugmentedOntologyGNN(
            structural_dim=config['structural_dim'],
            text_dim=config['text_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            model_type=config.get('backbone_model', 'gcn'),
            fusion_method=config.get('fusion_method', 'concat'),
            dropout=config.get('dropout', 0.0),
            num_relations=config.get('num_relations'),
            relation_types=config.get('relation_types'),
            num_bases=config.get('num_bases')
        )
    elif model_type in ['gcn', 'gat']:
        # Standard model (gcn, gat)
        model = OntologyGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            model_type=model_type
        )
    else:
        # Handle legacy or incorrectly named model types
        if model_type.startswith('text_'):
            # Legacy text-augmented models that were incorrectly named
            backbone = model_type.replace('text_', '')
            if backbone in ['gcn', 'gat', 'rgcn', 'weighted_gcn']:
                logger.warning(f"Found legacy model type '{model_type}', interpreting as text_augmented with {backbone} backbone")
                # Try to construct a TextAugmentedOntologyGNN
                model = TextAugmentedOntologyGNN(
                    structural_dim=config.get('structural_dim', config['input_dim']),
                    text_dim=config.get('text_dim', 0),
                    hidden_dim=config['hidden_dim'],
                    out_dim=config['out_dim'],
                    model_type=backbone,
                    fusion_method=config.get('fusion_method', 'concat'),
                    dropout=config.get('dropout', 0.0),
                    num_relations=config.get('num_relations'),
                    relation_types=config.get('relation_types'),
                    num_bases=config.get('num_bases')
                )
            else:
                raise ValueError(f"Unsupported legacy model type: {model_type}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: gcn, gat, rgcn, weighted_gcn, heterogeneous, text_augmented")

    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully: {model_type} with {config['out_dim']} output dimensions")

    return model, checkpoint


def create_training_setup(model_type='gcn', hidden_dim=128, out_dim=64, learning_rate=0.01, loss_fn_name='triplet'):
    """
    Create a complete training setup with model, optimizer, and loss function.

    Args:
        model_type (str): Type of GNN model ('gcn' or 'gat')
        hidden_dim (int): Hidden dimension size
        out_dim (int): Output embedding dimension
        learning_rate (float): Learning rate for optimizer
        loss_fn_name (str): Name of loss function to use

    Returns:
        tuple: (model, optimizer, loss_fn) ready for training
    """
    # Model will be initialized when we know input_dim from data
    optimizer_class = Adam
    loss_fn = get_loss_function(loss_fn_name)

    return {
        'model_type': model_type,
        'hidden_dim': hidden_dim,
        'out_dim': out_dim,
        'learning_rate': learning_rate,
        'optimizer_class': optimizer_class,
        'loss_fn': loss_fn,
        'loss_fn_name': loss_fn_name
    }


def train_ontology_embeddings(owl_file, model_output, model_type='gcn', hidden_dim=128, out_dim=64,
                            epochs=100, loss_fn_name='triplet', learning_rate=0.01, use_multi_relation=False,
                            dropout=0.0, num_bases=None, device=None):
    """
    Complete training pipeline from OWL file to saved model.

    Args:
        owl_file (str): Path to OWL ontology file
        model_output (str): Path to save trained model
        model_type (str): Type of GNN model ('gcn', 'gat', 'rgcn', 'weighted_gcn', 'heterogeneous')
        hidden_dim (int): Hidden dimension size
        out_dim (int): Output embedding dimension
        epochs (int): Number of training epochs
        loss_fn_name (str): Name of loss function
        learning_rate (float): Learning rate
        use_multi_relation (bool): Use multi-relation graph building
        dropout (float): Dropout rate for multi-relation models
        num_bases (int, optional): Number of bases for RGCN decomposition
        device (str or torch.device, optional): Device to use for training

    Returns:
        dict: Training results with model path and metadata
    """
    from .ontology import build_graph_from_owl, build_multi_relation_graph_from_owl

    logger.info(f"Starting training pipeline for {owl_file}")

    # Load ontology and build graph
    if use_multi_relation or model_type in ['rgcn', 'weighted_gcn', 'heterogeneous']:
        logger.info("Building multi-relation graph...")
        graph_data = build_multi_relation_graph_from_owl(owl_file)
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_type = graph_data['edge_types']
        class_to_index = graph_data['class_to_index']
        relation_data = {
            'relation_to_index': graph_data['relation_to_index'],
            'relation_names': graph_data['relation_names'],
            'edge_type_counts': graph_data['edge_type_counts']
        }
        num_relations = len(graph_data['relation_names'])
    else:
        logger.info("Building standard graph...")
        x, edge_index, class_to_index = build_graph_from_owl(owl_file)
        edge_type = None
        relation_data = None
        num_relations = 0

    # Create model based on type
    input_dim = x.size(1)

    if model_type in ['rgcn', 'weighted_gcn']:
        if not use_multi_relation and model_type in ['rgcn', 'weighted_gcn']:
            raise ValueError(f"Model type '{model_type}' requires use_multi_relation=True")

        model = MultiRelationOntologyGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_relations=num_relations,
            model_type=model_type,
            num_bases=num_bases,
            dropout=dropout,
            device=device
        )
    elif model_type == 'heterogeneous':
        if not use_multi_relation:
            raise ValueError("Heterogeneous model requires use_multi_relation=True")

        model = HeterogeneousOntologyGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            relation_types=graph_data['relation_names'],
            dropout=dropout,
            device=device
        )
    else:
        # Standard GNN models
        model = OntologyGNN(input_dim, hidden_dim, out_dim, model_type=model_type, device=device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_fn_name)

    # Train the model
    trained_model = train_model(model, x, edge_index, optimizer, loss_fn, epochs=epochs, edge_type=edge_type, device=device)

    # Save the model
    save_model_checkpoint(trained_model, class_to_index, model_output, relation_data=relation_data)

    return {
        'model_path': model_output,
        'num_nodes': len(class_to_index),
        'num_edges': edge_index.shape[1],
        'num_relations': num_relations,
        'model_config': {
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'epochs': epochs,
            'loss_function': loss_fn_name,
            'use_multi_relation': use_multi_relation,
            'dropout': dropout
        }
    }


def train_text_augmented_ontology_embeddings(owl_file, model_output,
                                           text_model_type='sentence_transformer',
                                           text_model_name='all-MiniLM-L6-v2',
                                           backbone_model='gcn', fusion_method='concat',
                                           hidden_dim=128, out_dim=64,
                                           epochs=100, loss_fn_name='triplet',
                                           learning_rate=0.01, dropout=0.0, device=None):
    """
    Complete training pipeline for text-augmented ontology embeddings.
    Combines structural and semantic text features.

    Args:
        owl_file (str): Path to OWL ontology file
        model_output (str): Path to save trained model
        text_model_type (str): Type of text embedding model
        text_model_name (str): Name/identifier for text model
        backbone_model (str): GNN backbone model ('gcn', 'gat')
        fusion_method (str): How to combine features ('concat', 'add', 'attention')
        hidden_dim (int): Hidden dimension size
        out_dim (int): Output embedding dimension
        epochs (int): Number of training epochs
        loss_fn_name (str): Name of loss function
        learning_rate (float): Learning rate
        dropout (float): Dropout rate
        device (str or torch.device, optional): Device to use for training

    Returns:
        dict: Training results with model path and metadata
    """
    from .ontology import build_graph_from_owl
    from .text_features import (
        extract_rich_semantic_features_from_owl,
        create_text_embedding_model,
        create_text_node_features
    )

    logger.info(f"Starting text-augmented training pipeline for {owl_file}")

    # Build structural graph
    logger.info("Building structural graph...")
    x_structural, edge_index, class_to_index = build_graph_from_owl(owl_file)

    # Extract text features
    logger.info("Extracting semantic text features...")
    text_features = extract_rich_semantic_features_from_owl(owl_file)

    # Create text embedding model
    logger.info(f"Initializing text embedding model: {text_model_type} - {text_model_name}")
    text_embedding_model = create_text_embedding_model(
        text_model_type,
        model_name=text_model_name
    )

    # Generate text embeddings
    logger.info("Generating text node features...")
    x_text = create_text_node_features(
        text_features,
        class_to_index,
        text_embedding_model,
        use_combined=True
    )

    # Get dimensions
    structural_dim = x_structural.size(1)
    text_dim = x_text.size(1)

    logger.info(f"Feature dimensions - Structural: {structural_dim}, Text: {text_dim}")

    # Build graph with multi-relation support if needed
    relation_data = None
    if backbone_model in ['rgcn', 'weighted_gcn', 'heterogeneous']:
        logger.info("Multi-relation model requested, building multi-relation graph...")
        from .ontology import build_multi_relation_graph_from_owl
        graph_data = build_multi_relation_graph_from_owl(owl_file)
        x_structural = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_type = graph_data['edge_types']
        class_to_index = graph_data['class_to_index']
        relation_data = {
            'relation_to_index': graph_data['relation_to_index'],
            'relation_names': graph_data['relation_names'],
            'edge_types': edge_type,
            'num_relations': len(graph_data['relation_names'])
        }
        logger.info(f"Multi-relation graph built with {len(graph_data['relation_names'])} relation types")

    # Create text-augmented model with full multi-relation support
    model_params = {
        'structural_dim': structural_dim,
        'text_dim': text_dim,
        'hidden_dim': hidden_dim,
        'out_dim': out_dim,
        'model_type': backbone_model,
        'fusion_method': fusion_method,
        'dropout': dropout,
        'device': device
    }

    # Add multi-relation parameters if needed
    if backbone_model in ['rgcn', 'weighted_gcn']:
        model_params['num_relations'] = relation_data['num_relations']
    elif backbone_model == 'heterogeneous':
        model_params['relation_types'] = relation_data['relation_names']

    model = TextAugmentedOntologyGNN(**model_params)

    # Add text model information as attributes for later retrieval
    model.text_model_type = text_model_type
    model.text_model_name = text_model_name

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_fn_name)

    # Train the model
    logger.info("Starting training...")
    trained_model = train_model(
        model, x_structural, edge_index, optimizer, loss_fn,
        epochs=epochs,
        edge_type=relation_data['edge_types'] if relation_data else None,
        text_x=x_text,
        device=device
    )

    # Save the model with text-specific metadata
    logger.info(f"Saving text-augmented model to {model_output}")

    # Enhanced model config for text-augmented models
    model_config = {
        'model_type': 'text_augmented',
        'structural_dim': structural_dim,
        'text_dim': text_dim,
        'hidden_dim': hidden_dim,
        'out_dim': out_dim,
        'backbone_model': backbone_model,
        'fusion_method': fusion_method,
        'dropout': dropout,
        'text_model_type': text_model_type,
        'text_model_name': text_model_name
    }

    # Add multi-relation specific config if applicable
    if backbone_model in ['rgcn', 'weighted_gcn', 'heterogeneous'] and relation_data:
        if backbone_model in ['rgcn', 'weighted_gcn']:
            model_config['num_relations'] = relation_data.get('num_relations')
        elif backbone_model == 'heterogeneous':
            model_config['relation_types'] = relation_data.get('relation_names')
            model_config['num_relations'] = len(relation_data.get('relation_names', []))

    # Extract node IDs (IRIs) from class_to_index
    node_ids = [cls.iri for cls in class_to_index.keys()]

    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'model_config': model_config,
        'class_to_index': {cls.iri: idx for cls, idx in class_to_index.items()},
        'node_ids': node_ids,
        'num_nodes': len(node_ids),
        'text_features': text_features  # Save text features for future use
    }

    torch.save(checkpoint, model_output)
    logger.info(f"Text-augmented model saved to {model_output}")

    return {
        'model_path': model_output,
        'num_nodes': len(class_to_index),
        'num_edges': edge_index.shape[1],
        'structural_dim': structural_dim,
        'text_dim': text_dim,
        'text_features_extracted': len(text_features),
        'model_config': {
            'model_type': 'text_augmented',
            'backbone_model': backbone_model,
            'fusion_method': fusion_method,
            'text_model_type': text_model_type,
            'text_model_name': text_model_name,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'epochs': epochs,
            'loss_function': loss_fn_name,
            'dropout': dropout
        }
    }