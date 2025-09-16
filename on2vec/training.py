"""
Training utilities for ontology embedding models
"""

import torch
from torch.optim import Adam
import time
import logging
from .models import OntologyGNN
from .loss_functions import get_loss_function

logger = logging.getLogger(__name__)


def train_model(model, x, edge_index, optimizer, loss_fn, epochs=100):
    """
    Train a GNN model on ontology data.

    Args:
        model (torch.nn.Module): The GNN model to train
        x (torch.Tensor): Node features
        edge_index (torch.Tensor): Graph edge indices
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (callable): Loss function
        epochs (int): Number of training epochs

    Returns:
        torch.nn.Module: Trained model
    """
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs} starting...")
        epoch_start_time = time.time()
        optimizer.zero_grad()

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


def save_model_checkpoint(model, class_to_index, output_path):
    """
    Save model checkpoint with metadata.

    Args:
        model (torch.nn.Module): Trained model
        class_to_index (dict): Class-to-index mapping from ontology
        output_path (str): Path to save the checkpoint

    Returns:
        None
    """
    logger.info(f"Saving model checkpoint to {output_path}")

    # Extract node IDs (IRIs) from class_to_index
    node_ids = [cls.iri for cls in class_to_index.keys()]

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model.model_type,
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'out_dim': model.out_dim
        },
        'class_to_index': {cls.iri: idx for cls, idx in class_to_index.items()},
        'node_ids': node_ids,
        'num_nodes': len(node_ids)
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

    # Recreate the model
    model = OntologyGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        out_dim=config['out_dim'],
        model_type=config['model_type']
    )

    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully: {config['model_type']} with {config['out_dim']} output dimensions")

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
                            epochs=100, loss_fn_name='triplet', learning_rate=0.01):
    """
    Complete training pipeline from OWL file to saved model.

    Args:
        owl_file (str): Path to OWL ontology file
        model_output (str): Path to save trained model
        model_type (str): Type of GNN model ('gcn' or 'gat')
        hidden_dim (int): Hidden dimension size
        out_dim (int): Output embedding dimension
        epochs (int): Number of training epochs
        loss_fn_name (str): Name of loss function
        learning_rate (float): Learning rate

    Returns:
        dict: Training results with model path and metadata
    """
    from .ontology import build_graph_from_owl

    logger.info(f"Starting training pipeline for {owl_file}")

    # Load ontology and build graph
    x, edge_index, class_to_index = build_graph_from_owl(owl_file)

    # Create model
    input_dim = x.size(1)
    model = OntologyGNN(input_dim, hidden_dim, out_dim, model_type=model_type)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_fn_name)

    # Train the model
    trained_model = train_model(model, x, edge_index, optimizer, loss_fn, epochs=epochs)

    # Save the model
    save_model_checkpoint(trained_model, class_to_index, model_output)

    return {
        'model_path': model_output,
        'num_nodes': len(class_to_index),
        'num_edges': edge_index.shape[1],
        'model_config': {
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'epochs': epochs,
            'loss_function': loss_fn_name
        }
    }