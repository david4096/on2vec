"""
on2vec: Generate vector embeddings from OWL ontologies using Graph Neural Networks
"""

from .models import OntologyGNN
from .training import train_model, load_model_checkpoint, save_model_checkpoint
from .embedding import generate_embeddings_from_model
from .ontology import build_graph_from_owl
from .io import save_embeddings_to_parquet, load_embeddings_from_parquet, create_embedding_metadata

__version__ = "0.1.0"
__all__ = [
    "OntologyGNN",
    "train_model",
    "load_model_checkpoint",
    "save_model_checkpoint",
    "generate_embeddings_from_model",
    "build_graph_from_owl",
    "save_embeddings_to_parquet",
    "load_embeddings_from_parquet",
    "create_embedding_metadata"
]