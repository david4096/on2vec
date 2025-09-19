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
from .visualization import (
    plot_pca_2d,
    plot_tsne_2d,
    plot_umap_2d,
    plot_embedding_distribution,
    create_visualization_comparison,
    load_embeddings_for_viz
)
from .evaluation import (
    EmbeddingEvaluator,
    evaluate_embeddings,
    create_evaluation_benchmark
)
from .benchmarks import (
    OntologyBenchmarkDatasets,
    BaselineComparison,
    setup_benchmark_datasets,
    compare_with_baselines
)

__version__ = "0.1.1"
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
    "get_embedding_vector",
    "plot_pca_2d",
    "plot_tsne_2d",
    "plot_umap_2d",
    "plot_embedding_distribution",
    "create_visualization_comparison",
    "load_embeddings_for_viz",
    "EmbeddingEvaluator",
    "evaluate_embeddings",
    "create_evaluation_benchmark",
    "OntologyBenchmarkDatasets",
    "BaselineComparison",
    "setup_benchmark_datasets",
    "compare_with_baselines"
]