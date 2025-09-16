"""
on2vec: Generate vector embeddings from OWL ontologies using Graph Neural Networks
"""

from .models import OntologyGNN, MultiRelationOntologyGNN, HeterogeneousOntologyGNN
from .training import train_model, train_ontology_embeddings, load_model_checkpoint, save_model_checkpoint
from .embedding import generate_embeddings_from_model, embed_ontology_with_model
from .ontology import build_graph_from_owl, build_multi_relation_graph_from_owl
from .io import (
    save_embeddings_to_parquet,
    load_embeddings_from_parquet,
    create_embedding_metadata,
    inspect_parquet_metadata,
    convert_parquet_to_csv,
    load_embeddings_as_dataframe,
    add_embedding_vectors,
    subtract_embedding_vectors,
    get_embedding_vector
)

__version__ = "0.1.0"
__all__ = [
    "OntologyGNN",
    "MultiRelationOntologyGNN",
    "HeterogeneousOntologyGNN",
    "train_model",
    "train_ontology_embeddings",
    "load_model_checkpoint",
    "save_model_checkpoint",
    "generate_embeddings_from_model",
    "embed_ontology_with_model",
    "build_graph_from_owl",
    "build_multi_relation_graph_from_owl",
    "save_embeddings_to_parquet",
    "load_embeddings_from_parquet",
    "create_embedding_metadata",
    "inspect_parquet_metadata",
    "convert_parquet_to_csv",
    "load_embeddings_as_dataframe",
    "add_embedding_vectors",
    "subtract_embedding_vectors",
    "get_embedding_vector"
]