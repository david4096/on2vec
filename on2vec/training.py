"""
Training utilities for ontology embedding models
"""

import torch
from torch.optim import Adam
import time
import logging
from .models import OntologyGNN, MultiRelationOntologyGNN, HeterogeneousOntologyGNN
from .loss_functions import get_loss_function

logger = logging.getLogger(__name__)


def train_model(model, x, edge_index, optimizer, loss_fn, epochs=100, edge_type=None):
    """
    Train a GNN model on ontology data.

    Args:
        model (torch.nn.Module): The GNN model to train
        x (torch.Tensor): Node features
        edge_index (torch.Tensor): Graph edge indices
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (callable): Loss function
        epochs (int): Number of training epochs
        edge_type (torch.Tensor, optional): Edge types for multi-relation models

    Returns:
        torch.nn.Module: Trained model
    """
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs} starting...")
        epoch_start_time = time.time()
        optimizer.zero_grad()

        # Forward pass - handle multi-relation models
        if hasattr(model, 'num_relations') and edge_type is not None:
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
    # For heterogeneous models, detect from class name
    if isinstance(model, HeterogeneousOntologyGNN):
        model_type = 'heterogeneous'
    else:
        model_type = getattr(model, 'model_type', 'unknown')

    model_config = {
        'model_type': model_type,
        'input_dim': getattr(model, 'input_dim', None),
        'hidden_dim': getattr(model, 'hidden_dim', None),
        'out_dim': getattr(model, 'out_dim', None)
    }

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
    elif model_type in ['gcn', 'gat']:
        # Standard model (gcn, gat)
        model = OntologyGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            model_type=model_type
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: gcn, gat, rgcn, weighted_gcn, heterogeneous")

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
                            dropout=0.0, num_bases=None):
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
            dropout=dropout
        )
    elif model_type == 'heterogeneous':
        if not use_multi_relation:
            raise ValueError("Heterogeneous model requires use_multi_relation=True")

        model = HeterogeneousOntologyGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            relation_types=graph_data['relation_names'],
            dropout=dropout
        )
    else:
        # Standard GNN models
        model = OntologyGNN(input_dim, hidden_dim, out_dim, model_type=model_type)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_fn_name)

    # Train the model
    trained_model = train_model(model, x, edge_index, optimizer, loss_fn, epochs=epochs, edge_type=edge_type)

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