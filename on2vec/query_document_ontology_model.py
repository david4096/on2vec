"""
Query/Document architecture with ontology augmentation for Sentence Transformers.
Based on: https://sbert.net/docs/sentence_transformer/usage/custom_models.html

This implements a specialized retrieval model where:
1. Queries are processed with text-only encoding (fast)
2. Documents are augmented with ontology structural information (rich)
3. Different embedding spaces optimized for query vs document
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import logging

logger = logging.getLogger(__name__)


class QueryEncoder(nn.Module):
    """Fast text-only encoder for queries."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.output_dim = self.encoder.get_sentence_embedding_dimension()

    def forward(self, queries: List[str]) -> torch.Tensor:
        """Encode queries using text only."""
        return self.encoder.encode(queries, convert_to_tensor=True)


class DocumentEncoder(nn.Module):
    """Rich encoder for documents with ontology augmentation."""

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        ontology_embeddings_file: str = None,
        fusion_method: str = 'concat',
        top_k_concepts: int = 5,
        concept_weight: float = 0.3
    ):
        super().__init__()
        self.text_encoder = SentenceTransformer(model_name)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.fusion_method = fusion_method
        self.top_k_concepts = top_k_concepts
        self.concept_weight = concept_weight

        # Load ontology data
        self.ontology_data = None
        self.structural_dim = None
        if ontology_embeddings_file:
            self._load_ontology_data(ontology_embeddings_file)
            self._init_fusion_layers()

    def _load_ontology_data(self, embeddings_file: str):
        """Load ontology embeddings with concept descriptions."""
        logger.info(f"Loading ontology data for document encoding: {embeddings_file}")

        df = pl.read_parquet(embeddings_file)

        # Verify multi-embedding format
        required_cols = ['node_id', 'text_embedding', 'structural_embedding']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File must contain {required_cols}, found: {df.columns}")

        # Extract embeddings and metadata
        self.concept_iris = df['node_id'].to_list()
        self.concept_text_embeds = np.stack(df['text_embedding'].to_list())
        self.concept_struct_embeds = np.stack(df['structural_embedding'].to_list())

        self.structural_dim = self.concept_struct_embeds.shape[1]

        # Generate concept descriptions for matching
        self.concept_descriptions = [
            self._iri_to_readable(iri) for iri in self.concept_iris
        ]

        # Pre-compute concept text embeddings for fast matching
        self.concept_text_encodings = self.text_encoder.encode(
            self.concept_descriptions,
            convert_to_tensor=True
        )

        logger.info(f"Loaded {len(self.concept_iris)} ontology concepts")

    def _iri_to_readable(self, iri: str) -> str:
        """Convert IRI to readable text."""
        import re
        # Extract meaningful part
        name = iri.split('#')[-1] if '#' in iri else iri.split('/')[-1]
        # Clean up formatting
        name = re.sub(r'[_-]', ' ', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return name.lower()

    def _init_fusion_layers(self):
        """Initialize layers for text + structural fusion."""
        if self.structural_dim is None:
            # Fallback to text-only if no structural data
            self.output_dim = self.text_dim
            self.fusion_layer = nn.Identity()
            return

        if self.fusion_method == 'concat':
            self.output_dim = self.text_dim + self.structural_dim
            self.fusion_layer = nn.Identity()

        elif self.fusion_method == 'projection':
            # Project both to common space then combine
            hidden_dim = max(self.text_dim, self.structural_dim)
            self.text_proj = nn.Linear(self.text_dim, hidden_dim)
            self.struct_proj = nn.Linear(self.structural_dim, hidden_dim)
            self.combiner = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.output_dim = hidden_dim

        elif self.fusion_method == 'gated':
            # Gated fusion - learn to weight text vs structural
            self.gate = nn.Sequential(
                nn.Linear(self.text_dim + self.structural_dim, 1),
                nn.Sigmoid()
            )
            common_dim = min(self.text_dim, self.structural_dim)
            self.text_proj = nn.Linear(self.text_dim, common_dim)
            self.struct_proj = nn.Linear(self.structural_dim, common_dim)
            self.output_dim = common_dim

    def find_relevant_concepts(self, document: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find ontology concepts most relevant to document.

        Returns:
            Tuple of (structural_embeddings, weights)
        """
        if self.ontology_data is None:
            return np.zeros((1, self.structural_dim)), np.array([1.0])

        # Encode document
        doc_embedding = self.text_encoder.encode([document], convert_to_tensor=True)

        # Compute similarities with all concepts
        similarities = torch.cosine_similarity(
            doc_embedding,
            self.concept_text_encodings,
            dim=1
        )

        # Get top-k most similar concepts
        top_k_indices = torch.topk(similarities, k=self.top_k_concepts).indices
        top_k_scores = similarities[top_k_indices]

        # Normalize scores to weights
        weights = torch.softmax(top_k_scores, dim=0).cpu().numpy()

        # Get corresponding structural embeddings
        structural_embeds = self.concept_struct_embeds[top_k_indices.cpu().numpy()]

        return structural_embeds, weights

    def forward(self, documents: List[str]) -> torch.Tensor:
        """Encode documents with ontology augmentation."""
        # Get text embeddings
        text_embeddings = self.text_encoder.encode(documents, convert_to_tensor=True)

        # Ensure embeddings are on CPU for numpy operations
        if hasattr(text_embeddings, 'device') and text_embeddings.device.type != 'cpu':
            text_embeddings = text_embeddings.cpu()

        if self.ontology_data is None:
            return text_embeddings

        # Get structural information for each document
        batch_structural = []

        for doc in documents:
            struct_embeds, weights = self.find_relevant_concepts(doc)

            # Weighted combination of relevant concepts
            weighted_struct = np.average(struct_embeds, axis=0, weights=weights)
            batch_structural.append(weighted_struct)

        structural_embeddings = torch.tensor(
            np.stack(batch_structural),
            dtype=torch.float32
        )

        # Fuse text and structural embeddings
        return self._fuse_embeddings(text_embeddings, structural_embeddings)

    def _fuse_embeddings(self, text_embeds: torch.Tensor, struct_embeds: torch.Tensor) -> torch.Tensor:
        """Fuse text and structural embeddings."""
        if self.fusion_method == 'concat':
            return torch.cat([text_embeds, struct_embeds], dim=1)

        elif self.fusion_method == 'projection':
            text_proj = self.text_proj(text_embeds)
            struct_proj = self.struct_proj(struct_embeds)
            combined = torch.cat([text_proj, struct_proj], dim=1)
            return self.combiner(combined)

        elif self.fusion_method == 'gated':
            # Learn gate weights
            combined_input = torch.cat([text_embeds, struct_embeds], dim=1)
            gate_weights = self.gate(combined_input)

            # Project to common dimension
            text_proj = self.text_proj(text_embeds)
            struct_proj = self.struct_proj(struct_embeds)

            # Gated combination
            return gate_weights * text_proj + (1 - gate_weights) * struct_proj


class QueryDocumentOntologyModel(nn.Module):
    """
    Complete Query/Document model with ontology augmentation.

    Architecture:
    - Queries: Fast text-only encoding
    - Documents: Rich text + ontology structural encoding
    - Separate optimization paths for different use cases
    """

    def __init__(
        self,
        query_model: str = 'all-MiniLM-L6-v2',
        doc_model: str = 'all-MiniLM-L6-v2',
        ontology_embeddings_file: str = None,
        fusion_method: str = 'concat',
        projection_dim: Optional[int] = None
    ):
        super().__init__()

        # Initialize encoders
        self.query_encoder = QueryEncoder(query_model)
        self.doc_encoder = DocumentEncoder(
            doc_model,
            ontology_embeddings_file,
            fusion_method
        )

        # Optional projection to common space
        self.projection_dim = projection_dim
        if projection_dim:
            self.query_proj = nn.Linear(self.query_encoder.output_dim, projection_dim)
            self.doc_proj = nn.Linear(self.doc_encoder.output_dim, projection_dim)
        else:
            self.query_proj = nn.Identity()
            self.doc_proj = nn.Identity()

        logger.info(f"Initialized QueryDocumentOntologyModel")
        logger.info(f"Query encoder output: {self.query_encoder.output_dim}")
        logger.info(f"Document encoder output: {self.doc_encoder.output_dim}")

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries (fast, text-only)."""
        query_embeds = self.query_encoder(queries)
        # Ensure on CPU and detached for numpy operations
        if hasattr(query_embeds, 'device') and query_embeds.device.type != 'cpu':
            query_embeds = query_embeds.cpu()
        result = self.query_proj(query_embeds)
        return result.detach() if hasattr(result, 'detach') else result

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents (rich, with ontology)."""
        doc_embeds = self.doc_encoder(documents)
        # Ensure on CPU and detached for numpy operations
        if hasattr(doc_embeds, 'device') and doc_embeds.device.type != 'cpu':
            doc_embeds = doc_embeds.cpu()
        result = self.doc_proj(doc_embeds)
        return result.detach() if hasattr(result, 'detach') else result

    def forward(self, queries: List[str], documents: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training/evaluation.

        Returns:
            Dictionary with 'query_embeddings' and 'document_embeddings'
        """
        return {
            'query_embeddings': self.encode_queries(queries),
            'document_embeddings': self.encode_documents(documents)
        }


def create_retrieval_model_with_ontology(
    ontology_embeddings_file: str,
    query_model: str = 'all-MiniLM-L6-v2',
    document_model: str = 'all-MiniLM-L6-v2',
    fusion_method: str = 'concat',
    projection_dim: Optional[int] = None
) -> QueryDocumentOntologyModel:
    """
    Create a query/document retrieval model augmented with ontology knowledge.

    Args:
        ontology_embeddings_file: Path to on2vec generated embeddings (with text + structural)
        query_model: Base model for query encoding
        document_model: Base model for document encoding
        fusion_method: How to fuse text + structural ('concat', 'projection', 'gated')
        projection_dim: Optional common dimension for query/doc embeddings

    Returns:
        QueryDocumentOntologyModel instance
    """
    return QueryDocumentOntologyModel(
        query_model=query_model,
        doc_model=document_model,
        ontology_embeddings_file=ontology_embeddings_file,
        fusion_method=fusion_method,
        projection_dim=projection_dim
    )


# Example usage and evaluation
def demo_retrieval_with_ontology():
    """Demonstrate ontology-augmented retrieval."""
    print("=== Query/Document Retrieval with Ontology Demo ===")

    # Example setup (replace with your actual embeddings file)
    embeddings_file = "path/to/your/text_augmented_embeddings.parquet"

    try:
        # Create the model
        model = create_retrieval_model_with_ontology(
            ontology_embeddings_file=embeddings_file,
            fusion_method='gated',  # Learn optimal text/structure weighting
            projection_dim=256      # Common embedding space
        )

        # Example queries and documents
        queries = [
            "What causes heart disease?",
            "How do genetic mutations lead to cancer?",
            "Protein folding abnormalities"
        ]

        documents = [
            "Cardiovascular disease is caused by atherosclerosis and hypertension leading to cardiac complications.",
            "Oncogenic mutations in tumor suppressor genes like p53 result in uncontrolled cell proliferation.",
            "Misfolded proteins aggregate and cause neurodegenerative diseases through cellular toxicity.",
            "Weather patterns affect agricultural productivity through temperature and rainfall variations.",
            "Machine learning algorithms optimize parameters through gradient descent and backpropagation."
        ]

        # Encode separately (different pathways)
        query_embeds = model.encode_queries(queries)
        doc_embeds = model.encode_documents(documents)

        print(f"Query embeddings shape: {query_embeds.shape}")
        print(f"Document embeddings shape: {doc_embeds.shape}")

        # Compute retrieval scores
        scores = torch.mm(query_embeds, doc_embeds.t())

        print("\\nRetrieval Results (Query -> Most Relevant Documents):")
        print("=" * 60)

        for i, query in enumerate(queries):
            # Get top documents for this query
            query_scores = scores[i]
            top_docs = torch.topk(query_scores, k=3)

            print(f"\\nQuery: '{query}'")
            print("-" * 40)

            for rank, (score, doc_idx) in enumerate(zip(top_docs.values, top_docs.indices)):
                print(f"{rank+1}. Score: {score:.3f}")
                print(f"   Doc: '{documents[doc_idx][:80]}...'")

    except FileNotFoundError:
        print(f"Embeddings file not found: {embeddings_file}")
        print("\\n🔧 To use this model:")
        print("1. Train an ontology model with on2vec using --use_text_features")
        print("2. Ensure the output parquet has text_embedding and structural_embedding columns")
        print("3. Pass the parquet file path to this function")


if __name__ == "__main__":
    demo_retrieval_with_ontology()