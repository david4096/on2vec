"""
Custom Sentence Transformer models that integrate on2vec ontology embeddings.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import logging

logger = logging.getLogger(__name__)


class OntologyAugmentedTextModel(nn.Module):
    """
    Custom model that combines text embeddings with ontology structural embeddings.

    This model:
    1. Encodes input text using a Sentence Transformer
    2. Finds similar ontology concepts using semantic similarity
    3. Retrieves structural embeddings for matched concepts
    4. Fuses text and structural embeddings
    """

    def __init__(
        self,
        text_model_name: str = 'all-MiniLM-L6-v2',
        ontology_embeddings_file: str = None,
        fusion_method: str = 'concat',
        top_k_matches: int = 3,
        structural_weight: float = 0.3
    ):
        """
        Initialize the ontology-augmented text model.

        Args:
            text_model_name: Name of the base Sentence Transformer model
            ontology_embeddings_file: Path to parquet file with ontology embeddings
            fusion_method: How to combine embeddings ('concat', 'weighted_avg', 'attention')
            top_k_matches: Number of top ontology matches to consider
            structural_weight: Weight for structural embeddings in fusion
        """
        super().__init__()

        self.text_model_name = text_model_name
        self.fusion_method = fusion_method
        self.top_k_matches = top_k_matches
        self.structural_weight = structural_weight

        # Load base text model
        self.text_encoder = SentenceTransformer(text_model_name)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()

        # Load ontology embeddings if provided
        self.ontology_data = None
        self.structural_dim = None
        if ontology_embeddings_file:
            self._load_ontology_embeddings(ontology_embeddings_file)

        # Initialize fusion layers
        self._init_fusion_layers()

        logger.info(f"Initialized OntologyAugmentedTextModel with {fusion_method} fusion")

    def _load_ontology_embeddings(self, embeddings_file: str):
        """Load ontology embeddings from parquet file."""
        logger.info(f"Loading ontology embeddings from {embeddings_file}")

        df = pl.read_parquet(embeddings_file)

        # Check if this is a multi-embedding file (text-augmented)
        has_text_embeddings = 'text_embedding' in df.columns
        has_structural_embeddings = 'structural_embedding' in df.columns

        if not has_text_embeddings or not has_structural_embeddings:
            raise ValueError(
                f"Embeddings file must contain both 'text_embedding' and 'structural_embedding' columns. "
                f"Found columns: {df.columns}. Please use a text-augmented model output."
            )

        # Extract data
        self.node_ids = df['node_id'].to_list()
        self.text_embeddings = np.stack(df['text_embedding'].to_list())
        self.structural_embeddings = np.stack(df['structural_embedding'].to_list())

        self.structural_dim = self.structural_embeddings.shape[1]

        # Create concept descriptions for semantic matching
        # Use the IRI's last part as a simple description
        self.concept_descriptions = [
            self._iri_to_description(iri) for iri in self.node_ids
        ]

        logger.info(f"Loaded {len(self.node_ids)} ontology concepts")
        logger.info(f"Text embeddings: {self.text_embeddings.shape}")
        logger.info(f"Structural embeddings: {self.structural_embeddings.shape}")

    def _iri_to_description(self, iri: str) -> str:
        """Convert IRI to a human-readable description."""
        # Extract the last part after # or /
        if '#' in iri:
            name = iri.split('#')[-1]
        elif '/' in iri:
            name = iri.split('/')[-1]
        else:
            name = iri

        # Convert underscores to spaces and handle camelCase
        import re
        name = re.sub(r'_', ' ', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

        return name.lower()

    def _init_fusion_layers(self):
        """Initialize layers for embedding fusion."""
        if self.structural_dim is None:
            # Can't initialize fusion layers yet
            self.fusion_layer = None
            return

        if self.fusion_method == 'concat':
            # Simple concatenation
            self.output_dim = self.text_dim + self.structural_dim
            self.fusion_layer = nn.Identity()

        elif self.fusion_method == 'weighted_avg':
            # Weighted average (same dimensions required)
            if self.text_dim != self.structural_dim:
                # Project to common dimension
                common_dim = min(self.text_dim, self.structural_dim)
                self.text_projector = nn.Linear(self.text_dim, common_dim)
                self.struct_projector = nn.Linear(self.structural_dim, common_dim)
                self.output_dim = common_dim
            else:
                self.text_projector = nn.Identity()
                self.struct_projector = nn.Identity()
                self.output_dim = self.text_dim

        elif self.fusion_method == 'attention':
            # Attention-based fusion
            hidden_dim = max(self.text_dim, self.structural_dim)
            self.text_projector = nn.Linear(self.text_dim, hidden_dim)
            self.struct_projector = nn.Linear(self.structural_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.output_projector = nn.Linear(hidden_dim, hidden_dim)
            self.output_dim = hidden_dim

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def find_similar_concepts(self, query_text: str) -> List[Tuple[int, float]]:
        """
        Find ontology concepts most similar to the query text.

        Returns:
            List of (concept_index, similarity_score) tuples
        """
        if self.ontology_data is None:
            return []

        # Encode query text and concept descriptions
        query_embedding = self.text_encoder.encode([query_text])
        concept_embeddings = self.text_encoder.encode(self.concept_descriptions)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, concept_embeddings)[0]

        # Get top-k matches
        top_indices = np.argsort(similarities)[-self.top_k_matches:][::-1]

        return [(idx, similarities[idx]) for idx in top_indices]

    def get_structural_embeddings_for_concepts(self, concept_indices: List[int]) -> torch.Tensor:
        """Get structural embeddings for given concept indices."""
        if len(concept_indices) == 0:
            return torch.zeros((1, self.structural_dim))

        # Get structural embeddings and weights
        struct_embeds = self.structural_embeddings[concept_indices]
        return torch.tensor(struct_embeds, dtype=torch.float32)

    def forward(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining text and structural embeddings.

        Args:
            sentences: List of input sentences

        Returns:
            Dictionary with 'sentence_embedding' key
        """
        batch_size = len(sentences)

        # Get text embeddings
        text_embeddings = self.text_encoder.encode(sentences, convert_to_tensor=True)

        # Ensure embeddings are on CPU for numpy operations
        if hasattr(text_embeddings, 'device') and text_embeddings.device.type != 'cpu':
            text_embeddings = text_embeddings.cpu()

        if self.ontology_data is None:
            # No ontology data available, return text embeddings only
            return {'sentence_embedding': text_embeddings}

        # Get structural embeddings for each sentence
        structural_embeddings_list = []

        for sentence in sentences:
            # Find similar concepts
            similar_concepts = self.find_similar_concepts(sentence)

            if similar_concepts:
                # Get indices and similarity scores
                indices, scores = zip(*similar_concepts)

                # Get structural embeddings
                struct_embeds = self.get_structural_embeddings_for_concepts(list(indices))

                # Weight by similarity scores
                scores_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
                # Ensure struct_embeds is on CPU
                if hasattr(struct_embeds, 'device') and struct_embeds.device.type != 'cpu':
                    struct_embeds = struct_embeds.cpu()
                weighted_struct = (struct_embeds * scores_tensor).mean(dim=0, keepdim=True)

                structural_embeddings_list.append(weighted_struct)
            else:
                # No matches found, use zero embedding
                structural_embeddings_list.append(
                    torch.zeros((1, self.structural_dim), dtype=torch.float32)
                )

        structural_embeddings = torch.cat(structural_embeddings_list, dim=0)

        # Fuse embeddings
        fused_embeddings = self._fuse_embeddings(text_embeddings, structural_embeddings)

        return {'sentence_embedding': fused_embeddings}

    def _fuse_embeddings(self, text_embeds: torch.Tensor, struct_embeds: torch.Tensor) -> torch.Tensor:
        """Fuse text and structural embeddings."""
        if self.fusion_method == 'concat':
            return torch.cat([text_embeds, struct_embeds], dim=1)

        elif self.fusion_method == 'weighted_avg':
            # Project to common dimension if needed
            text_proj = self.text_projector(text_embeds)
            struct_proj = self.struct_projector(struct_embeds)

            # Weighted average
            return (1 - self.structural_weight) * text_proj + self.structural_weight * struct_proj

        elif self.fusion_method == 'attention':
            # Project both to hidden dimension
            text_proj = self.text_projector(text_embeds).unsqueeze(1)  # [batch, 1, hidden]
            struct_proj = self.struct_projector(struct_embeds).unsqueeze(1)  # [batch, 1, hidden]

            # Concatenate for attention
            combined = torch.cat([text_proj, struct_proj], dim=1)  # [batch, 2, hidden]

            # Apply attention
            attn_output, _ = self.attention(combined, combined, combined)

            # Pool attention output (mean over sequence dimension)
            pooled = attn_output.mean(dim=1)  # [batch, hidden]

            return self.output_projector(pooled)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class OntologyAugmentedSentenceTransformer(SentenceTransformer):
    """
    Custom Sentence Transformer that integrates ontology structural embeddings.

    This extends the standard SentenceTransformer with ontology knowledge.
    """

    def __init__(
        self,
        model_name_or_path: str = 'all-MiniLM-L6-v2',
        ontology_embeddings_file: str = None,
        fusion_method: str = 'concat',
        **kwargs
    ):
        """
        Initialize ontology-augmented Sentence Transformer.

        Args:
            model_name_or_path: Base Sentence Transformer model
            ontology_embeddings_file: Path to ontology embeddings parquet file
            fusion_method: How to fuse embeddings ('concat', 'weighted_avg', 'attention')
            **kwargs: Additional arguments for SentenceTransformer
        """
        # Initialize base model
        super().__init__(model_name_or_path, **kwargs)

        # Replace the last module with our custom fusion module
        self.ontology_fusion = OntologyAugmentedTextModel(
            text_model_name=model_name_or_path,
            ontology_embeddings_file=ontology_embeddings_file,
            fusion_method=fusion_method
        )

        # Update the modules list to include our custom module
        self._modules['ontology_fusion'] = self.ontology_fusion

        logger.info(f"Created OntologyAugmentedSentenceTransformer with {fusion_method} fusion")


def create_ontology_augmented_model(
    base_model: str = 'all-MiniLM-L6-v2',
    ontology_embeddings_file: str = None,
    fusion_method: str = 'concat',
    top_k_matches: int = 3,
    structural_weight: float = 0.3
) -> OntologyAugmentedTextModel:
    """
    Factory function to create an ontology-augmented text model.

    Args:
        base_model: Base Sentence Transformer model name
        ontology_embeddings_file: Path to multi-embedding parquet file from on2vec
        fusion_method: How to combine embeddings ('concat', 'weighted_avg', 'attention')
        top_k_matches: Number of ontology matches to consider per query
        structural_weight: Weight for structural embeddings in fusion

    Returns:
        OntologyAugmentedTextModel instance
    """
    return OntologyAugmentedTextModel(
        text_model_name=base_model,
        ontology_embeddings_file=ontology_embeddings_file,
        fusion_method=fusion_method,
        top_k_matches=top_k_matches,
        structural_weight=structural_weight
    )


# Example usage and demonstration
def demo_ontology_augmented_similarity():
    """Demonstrate the ontology-augmented similarity functionality."""
    print("=== Ontology-Augmented Text Similarity Demo ===")

    # This would use your trained ontology embeddings
    embeddings_file = "path/to/your/ontology_embeddings.parquet"  # Replace with actual file

    try:
        # Create the model
        model = create_ontology_augmented_model(
            base_model='all-MiniLM-L6-v2',
            ontology_embeddings_file=embeddings_file,
            fusion_method='concat',
            top_k_matches=3
        )

        # Example queries
        queries = [
            "heart disease and cardiovascular problems",
            "genetic mutations in cancer cells",
            "protein folding mechanisms"
        ]

        # Get embeddings
        embeddings = model(queries)['sentence_embedding']

        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Text dim + Structural dim = {model.text_dim} + {model.structural_dim}")

        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)

        print("\nSimilarity matrix:")
        for i, query1 in enumerate(queries):
            for j, query2 in enumerate(queries):
                if i <= j:
                    print(f"'{query1[:30]}...' <-> '{query2[:30]}...': {similarities[i,j]:.3f}")

    except FileNotFoundError:
        print(f"Embeddings file not found: {embeddings_file}")
        print("Please train an ontology model first using on2vec with text features enabled.")


if __name__ == "__main__":
    demo_ontology_augmented_similarity()