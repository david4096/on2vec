"""
OWL ontology processing utilities
"""

import torch
from owlready2 import get_ontology
import logging

logger = logging.getLogger(__name__)


def build_graph_from_owl(owl_file):
    """
    Build a graph representation from an OWL ontology file.

    Args:
        owl_file (str): Path to the OWL ontology file

    Returns:
        tuple: (node_features, edge_index, class_to_index_mapping)
            - node_features (torch.Tensor): Identity matrix for node features
            - edge_index (torch.Tensor): Graph edges as tensor
            - class_to_index (dict): Mapping from ontology classes to indices
    """
    logger.info(f"Loading OWL ontology from {owl_file}")

    try:
        ontology = get_ontology(owl_file).load()
        logger.info(f"Successfully loaded ontology: {owl_file}")
    except Exception as e:
        logger.error(f"Failed to load ontology: {owl_file}. Error: {e}")
        raise

    classes = list(ontology.classes())
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    edges = []

    # Build edges from subclass relationships
    for cls in classes:
        for parent in cls.is_a:
            if hasattr(parent, "iri") and parent in class_to_index:
                # Add bidirectional edges for undirected graph
                edges.append((class_to_index[parent], class_to_index[cls]))
                edges.append((class_to_index[cls], class_to_index[parent]))

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        logger.warning("No edges found in ontology")
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(classes)
    # Use identity matrix as initial node features
    x = torch.eye(num_nodes, dtype=torch.float)

    logger.info(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")
    return x, edge_index, class_to_index


def extract_node_ids_from_ontology(class_to_index):
    """
    Extract node IDs (IRIs) from the class-to-index mapping.

    Args:
        class_to_index (dict): Mapping from ontology classes to indices

    Returns:
        list: List of node IDs (IRIs)
    """
    return [cls.iri for cls in class_to_index.keys()]


def align_ontology_with_training(new_class_to_index, training_class_to_index):
    """
    Align a new ontology with training data.

    Args:
        new_class_to_index (dict): Class-to-index mapping for new ontology
        training_class_to_index (dict): Class-to-index mapping from training (IRI strings)

    Returns:
        tuple: (new_to_training_mapping, aligned_node_ids)
            - new_to_training_mapping (dict): Mapping from new indices to training indices
            - aligned_node_ids (list): Node IDs that exist in both ontologies
    """
    logger.info("Aligning ontology with training data...")

    # Convert training mapping back from IRI strings to match current format
    training_iris = set(training_class_to_index.keys())
    new_iris = {cls.iri for cls in new_class_to_index.keys()}

    # Find intersection
    common_iris = training_iris.intersection(new_iris)
    missing_iris = training_iris - new_iris
    new_iris_only = new_iris - training_iris

    logger.info(f"Common classes: {len(common_iris)}")
    if missing_iris:
        logger.warning(f"Missing from new ontology: {len(missing_iris)} classes")
    if new_iris_only:
        logger.info(f"New classes not in training: {len(new_iris_only)}")

    # Create mapping from new indices to training indices
    new_to_training_idx = {}
    aligned_node_ids = []

    for cls, new_idx in new_class_to_index.items():
        if cls.iri in training_class_to_index:
            training_idx = training_class_to_index[cls.iri]
            new_to_training_idx[new_idx] = training_idx
            aligned_node_ids.append(cls.iri)

    return new_to_training_idx, aligned_node_ids