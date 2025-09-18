"""
OWL ontology processing utilities
"""

import os
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
        # Try different approaches to load the ontology
        ontology_iri = f"file://{os.path.abspath(owl_file)}"

        # First, try the standard approach
        try:
            ontology = get_ontology(owl_file).load()
        except Exception as first_error:
            logger.warning(f"Standard loading failed for {owl_file}: {first_error}")

            # Try loading with file:// IRI
            try:
                ontology = get_ontology(ontology_iri).load()
            except Exception as second_error:
                logger.warning(f"IRI loading failed for {owl_file}: {second_error}")

                # Try loading into a new world to isolate any conflicts
                from owlready2 import World
                world = World()
                try:
                    ontology = world.get_ontology(owl_file).load()
                except Exception as third_error:
                    logger.error(f"All loading attempts failed for {owl_file}")
                    logger.error(f"  Standard: {first_error}")
                    logger.error(f"  IRI: {second_error}")
                    logger.error(f"  New world: {third_error}")
                    raise third_error

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


def build_multi_relation_graph_from_owl(owl_file, include_subclass=True):
    """
    Build a graph representation from an OWL ontology file that includes all ObjectProperty relations,
    not just subclass relationships.

    Args:
        owl_file (str): Path to the OWL ontology file
        include_subclass (bool): Whether to include subclass relations (default: True)

    Returns:
        dict: Dictionary containing:
            - 'node_features' (torch.Tensor): Identity matrix for node features
            - 'edge_index' (torch.Tensor): Graph edges as tensor
            - 'edge_types' (torch.Tensor): Edge type indices
            - 'class_to_index' (dict): Mapping from ontology classes to indices
            - 'relation_to_index' (dict): Mapping from relation IRIs to indices
            - 'relation_names' (list): List of relation names/IRIs
            - 'edge_type_counts' (dict): Count of edges per relation type
    """
    logger.info(f"Loading OWL ontology with all relations from {owl_file}")

    try:
        # Try different approaches to load the ontology
        ontology_iri = f"file://{os.path.abspath(owl_file)}"

        # First, try the standard approach
        try:
            ontology = get_ontology(owl_file).load()
        except Exception as first_error:
            logger.warning(f"Standard loading failed for {owl_file}: {first_error}")

            # Try loading with file:// IRI
            try:
                ontology = get_ontology(ontology_iri).load()
            except Exception as second_error:
                logger.warning(f"IRI loading failed for {owl_file}: {second_error}")

                # Try loading into a new world to isolate any conflicts
                from owlready2 import World
                world = World()
                try:
                    ontology = world.get_ontology(owl_file).load()
                except Exception as third_error:
                    logger.error(f"All loading attempts failed for {owl_file}")
                    logger.error(f"  Standard: {first_error}")
                    logger.error(f"  IRI: {second_error}")
                    logger.error(f"  New world: {third_error}")
                    raise third_error

        logger.info(f"Successfully loaded ontology: {owl_file}")
    except Exception as e:
        logger.error(f"Failed to load ontology: {owl_file}. Error: {e}")
        raise

    classes = list(ontology.classes())
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    # Get all object properties from the ontology
    object_properties = list(ontology.object_properties())
    relation_names = []

    # Add subclass relation if requested
    if include_subclass:
        relation_names.append("rdfs:subClassOf")

    # Add all object properties
    relation_names.extend([prop.iri for prop in object_properties])

    relation_to_index = {rel: i for i, rel in enumerate(relation_names)}

    edges = []
    edge_types = []
    edge_type_counts = {rel: 0 for rel in relation_names}

    logger.info(f"Found {len(object_properties)} object properties")
    logger.info(f"Object properties: {[prop.name for prop in object_properties[:10]]}")  # Show first 10

    # Build edges from subclass relationships
    if include_subclass:
        subclass_rel_idx = relation_to_index["rdfs:subClassOf"]
        for cls in classes:
            for parent in cls.is_a:
                if hasattr(parent, "iri") and parent in class_to_index:
                    # Add bidirectional edges for undirected graph
                    edges.append((class_to_index[parent], class_to_index[cls]))
                    edge_types.append(subclass_rel_idx)
                    edges.append((class_to_index[cls], class_to_index[parent]))
                    edge_types.append(subclass_rel_idx)
                    edge_type_counts["rdfs:subClassOf"] += 2

    # Build edges from object property relationships
    for prop in object_properties:
        prop_rel_idx = relation_to_index[prop.iri]

        try:
            # Get all instances/relationships for this property
            # owlready2 stores property relationships as attributes on classes
            for cls in classes:
                if hasattr(cls, prop.name):
                    # Get the property values for this class
                    prop_values = getattr(cls, prop.name, [])

                    # Handle both single values and lists
                    if not isinstance(prop_values, list):
                        prop_values = [prop_values] if prop_values is not None else []

                    for target in prop_values:
                        if hasattr(target, "iri") and target in class_to_index:
                            # Add bidirectional edges
                            edges.append((class_to_index[cls], class_to_index[target]))
                            edge_types.append(prop_rel_idx)
                            edges.append((class_to_index[target], class_to_index[cls]))
                            edge_types.append(prop_rel_idx)
                            edge_type_counts[prop.iri] += 2

                # Also check if this class appears as an object in any relationships
                # by examining the property's domain and range restrictions
                try:
                    # For restrictions and more complex relationships
                    for restriction in cls.is_a:
                        if hasattr(restriction, 'property') and restriction.property == prop:
                            if hasattr(restriction, 'value') and restriction.value in class_to_index:
                                edges.append((class_to_index[cls], class_to_index[restriction.value]))
                                edge_types.append(prop_rel_idx)
                                edges.append((class_to_index[restriction.value], class_to_index[cls]))
                                edge_types.append(prop_rel_idx)
                                edge_type_counts[prop.iri] += 2
                except AttributeError:
                    # Some restrictions may not have the expected attributes
                    pass

        except Exception as e:
            logger.warning(f"Error processing property {prop.name}: {e}")
            continue

    # Create tensors
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type_tensor = torch.empty((0,), dtype=torch.long)
        logger.warning("No edges found in ontology")
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

    num_nodes = len(classes)
    # Use identity matrix as initial node features
    x = torch.eye(num_nodes, dtype=torch.float)

    # Filter out relations with no edges
    used_relations = {rel: count for rel, count in edge_type_counts.items() if count > 0}

    logger.info(f"Multi-relation graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")
    logger.info(f"Relation types used: {len(used_relations)}")
    logger.info(f"Edge type distribution: {used_relations}")

    return {
        'node_features': x,
        'edge_index': edge_index,
        'edge_types': edge_type_tensor,
        'class_to_index': class_to_index,
        'relation_to_index': relation_to_index,
        'relation_names': relation_names,
        'edge_type_counts': edge_type_counts
    }


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