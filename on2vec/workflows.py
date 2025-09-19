#!/usr/bin/env python3
"""
Core workflows for on2vec - training, embedding generation, and model creation.

This module contains the main workflow functions that were previously in
separate scripts like main.py, embed.py, etc.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .training import train_ontology_embeddings, train_text_augmented_ontology_embeddings
from .embedding import embed_same_ontology, embed_ontology_with_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_and_embed_workflow(
    owl_file: str,
    model_type: str = 'gcn',
    hidden_dim: int = 16,
    out_dim: int = 8,
    epochs: int = 100,
    output: str = 'embeddings.parquet',
    model_output: str = 'model.pt',
    loss_fn: str = 'triplet',
    skip_training: bool = False,
    use_multi_relation: bool = False,
    dropout: float = 0.0,
    num_bases: Optional[int] = None,
    learning_rate: float = 0.01,
    # Text augmentation parameters
    use_text_features: bool = False,
    text_model_type: str = 'sentence_transformer',
    text_model_name: str = 'all-MiniLM-L6-v2',
    fusion_method: str = 'concat',
    # Device parameters
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete workflow for training GNN models and generating embeddings.

    This is the main workflow function that combines training and embedding
    generation in a single call.
    """

    logger.info(f"Starting train and embed workflow for {owl_file}")
    logger.info(f"Model type: {model_type}, Use text features: {use_text_features}")

    # Auto-enable multi-relation for models that require it
    if model_type in ['rgcn', 'weighted_gcn', 'heterogeneous']:
        use_multi_relation = True
        logger.info(f"Auto-enabling multi-relation graph for {model_type} model")

    training_result = None
    if not skip_training:
        # Training phase
        if use_text_features:
            logger.info("Starting text-augmented training phase...")
            logger.info(f"Text model: {text_model_type} - {text_model_name}")
            logger.info(f"Fusion method: {fusion_method}")
            training_result = train_text_augmented_ontology_embeddings(
                owl_file=owl_file,
                model_output=model_output,
                text_model_type=text_model_type,
                text_model_name=text_model_name,
                backbone_model=model_type,
                fusion_method=fusion_method,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                epochs=epochs,
                loss_fn_name=loss_fn,
                learning_rate=learning_rate,
                dropout=dropout,
                device=device
            )
            logger.info(f"Text-augmented training completed. Model saved to {training_result['model_path']}")
            logger.info(f"Structural features: {training_result['structural_dim']}, Text features: {training_result['text_dim']}")
            logger.info(f"Text features extracted for {training_result['text_features_extracted']} classes")
        else:
            logger.info("Starting standard training phase...")
            training_result = train_ontology_embeddings(
                owl_file=owl_file,
                model_output=model_output,
                model_type=model_type,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                epochs=epochs,
                loss_fn_name=loss_fn,
                learning_rate=learning_rate,
                use_multi_relation=use_multi_relation,
                dropout=dropout,
                num_bases=num_bases,
                device=device
            )
            logger.info(f"Training completed. Model saved to {training_result['model_path']}")
            if training_result.get('num_relations', 0) > 0:
                logger.info(f"Multi-relation graph with {training_result['num_relations']} relation types")

    # Embedding phase
    logger.info("Starting embedding generation phase...")
    embedding_result = embed_same_ontology(
        model_path=model_output,
        owl_file=owl_file,
        output_file=output,
        device=device
    )

    logger.info(f"Embeddings generated and saved to {output}")
    logger.info(f"Generated {len(embedding_result['node_ids'])} embeddings")

    return {
        'training_result': training_result,
        'embedding_result': embedding_result,
        'model_path': model_output,
        'embeddings_path': output,
        'num_embeddings': len(embedding_result['node_ids'])
    }


def embed_with_trained_model(
    model_path: str,
    owl_file: str,
    output_file: str = 'embeddings.parquet',
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate embeddings using a pre-trained model.

    This corresponds to the functionality in embed.py.
    """
    logger.info(f"Generating embeddings using model: {model_path}")
    logger.info(f"Target ontology: {owl_file}")
    logger.info(f"Output file: {output_file}")

    # Use the embed_ontology_with_model function for cross-ontology compatibility
    embedding_result = embed_ontology_with_model(
        model_path=model_path,
        owl_file=owl_file,
        output_file=output_file,
        device=device
    )

    logger.info(f"Embeddings generated successfully")
    logger.info(f"Generated {len(embedding_result['node_ids'])} embeddings")
    logger.info(f"Embeddings saved to: {output_file}")

    return {
        'embedding_result': embedding_result,
        'output_file': output_file,
        'num_embeddings': len(embedding_result['node_ids'])
    }


def train_model_only(
    owl_file: str,
    model_output: str,
    model_type: str = 'gcn',
    hidden_dim: int = 16,
    out_dim: int = 8,
    epochs: int = 100,
    loss_fn: str = 'triplet',
    use_multi_relation: bool = False,
    dropout: float = 0.0,
    num_bases: Optional[int] = None,
    learning_rate: float = 0.01,
    # Text augmentation parameters
    use_text_features: bool = False,
    text_model_type: str = 'sentence_transformer',
    text_model_name: str = 'all-MiniLM-L6-v2',
    fusion_method: str = 'concat',
    # Device parameters
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a model without generating embeddings.

    Useful for the two-phase workflow where you train once and embed multiple times.
    """

    logger.info(f"Training model for {owl_file}")
    logger.info(f"Model type: {model_type}, Output: {model_output}")

    # Auto-enable multi-relation for models that require it
    if model_type in ['rgcn', 'weighted_gcn', 'heterogeneous']:
        use_multi_relation = True
        logger.info(f"Auto-enabling multi-relation graph for {model_type} model")

    if use_text_features:
        logger.info("Training text-augmented model...")
        logger.info(f"Text model: {text_model_type} - {text_model_name}")
        logger.info(f"Fusion method: {fusion_method}")
        training_result = train_text_augmented_ontology_embeddings(
            owl_file=owl_file,
            model_output=model_output,
            text_model_type=text_model_type,
            text_model_name=text_model_name,
            backbone_model=model_type,
            fusion_method=fusion_method,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
            loss_fn_name=loss_fn,
            learning_rate=learning_rate,
            dropout=dropout,
            device=device
        )
        logger.info(f"Text-augmented training completed")
        logger.info(f"Structural features: {training_result['structural_dim']}, Text features: {training_result['text_dim']}")
    else:
        logger.info("Training standard GNN model...")
        training_result = train_ontology_embeddings(
            owl_file=owl_file,
            model_output=model_output,
            model_type=model_type,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
            loss_fn_name=loss_fn,
            learning_rate=learning_rate,
            use_multi_relation=use_multi_relation,
            dropout=dropout,
            num_bases=num_bases,
            device=device
        )
        logger.info(f"Training completed")
        if training_result.get('num_relations', 0) > 0:
            logger.info(f"Multi-relation graph with {training_result['num_relations']} relation types")

    logger.info(f"Model saved to {training_result['model_path']}")
    return training_result