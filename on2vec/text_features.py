"""
Semantic text feature extraction from OWL ontologies
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
from owlready2 import get_ontology
import re

logger = logging.getLogger(__name__)


def extract_rich_semantic_features_from_owl(owl_file: str) -> Dict[str, Dict[str, str]]:
    """
    Extract rich semantic textual features from an OWL ontology file.
    Includes comprehensive extraction from various semantic properties.

    Args:
        owl_file (str): Path to the OWL ontology file

    Returns:
        dict: Dictionary mapping class IRIs to their rich semantic features:
            {
                'class_iri': {
                    'label': 'Class Label',
                    'comment': 'rdfs:comment content',
                    'definition': 'skos:definition content',
                    'description': 'dc:description or similar',
                    'alternative_labels': 'skos:altLabel values',
                    'preferred_labels': 'skos:prefLabel values',
                    'examples': 'skos:example values',
                    'notes': 'skos:note values',
                    'see_also': 'rdfs:seeAlso references',
                    'annotations': 'other annotation properties',
                    'combined_text': 'all text combined'
                }
            }
    """
    logger.info(f"Extracting rich semantic features from {owl_file}")

    try:
        ontology = get_ontology(owl_file).load()
        logger.info(f"Successfully loaded ontology: {owl_file}")
    except Exception as e:
        logger.error(f"Failed to load ontology: {owl_file}. Error: {e}")
        raise

    classes = list(ontology.classes())
    text_features = {}

    # Common semantic properties to extract
    semantic_properties = [
        # Basic properties
        'label', 'comment', 'definition', 'description',
        # SKOS properties
        'prefLabel', 'altLabel', 'hiddenLabel',
        'definition', 'scopeNote', 'editorialNote', 'historyNote',
        'example', 'note',
        # Dublin Core properties
        'dc_title', 'dc_description', 'dc_subject', 'dc_creator',
        'dct_title', 'dct_description', 'dct_subject',
        # Other common annotation properties
        'seeAlso', 'isDefinedBy', 'versionInfo'
    ]

    for cls in classes:
        class_iri = cls.iri
        features = {prop: '' for prop in [
            'label', 'comment', 'definition', 'description',
            'alternative_labels', 'preferred_labels', 'examples',
            'notes', 'see_also', 'annotations', 'combined_text'
        ]}

        # Extract class name/label
        if hasattr(cls, 'name') and cls.name:
            features['label'] = clean_text(cls.name)

        # Extract semantic properties systematically
        for prop_name in semantic_properties:
            try:
                if hasattr(cls, prop_name):
                    prop_value = getattr(cls, prop_name)
                    if prop_value:
                        # Handle both single values and lists
                        if isinstance(prop_value, list):
                            prop_text = ' '.join([clean_text(str(val)) for val in prop_value])
                        else:
                            prop_text = clean_text(str(prop_value))

                        # Map to appropriate feature category
                        if prop_name in ['label', 'prefLabel']:
                            if not features['label']:
                                features['label'] = prop_text
                            else:
                                features['preferred_labels'] += ' ' + prop_text
                        elif prop_name in ['altLabel', 'hiddenLabel']:
                            features['alternative_labels'] += ' ' + prop_text
                        elif prop_name in ['comment']:
                            features['comment'] += ' ' + prop_text
                        elif prop_name in ['definition', 'scopeNote']:
                            features['definition'] += ' ' + prop_text
                        elif prop_name in ['description', 'dc_description', 'dct_description']:
                            features['description'] += ' ' + prop_text
                        elif prop_name in ['example']:
                            features['examples'] += ' ' + prop_text
                        elif prop_name in ['note', 'editorialNote', 'historyNote']:
                            features['notes'] += ' ' + prop_text
                        elif prop_name in ['seeAlso', 'isDefinedBy']:
                            features['see_also'] += ' ' + prop_text
                        else:
                            features['annotations'] += ' ' + prop_text

            except (AttributeError, Exception) as e:
                # Some properties might not be available or accessible
                logger.debug(f"Could not extract {prop_name} from {class_iri}: {e}")
                continue

        # Clean up accumulated text
        for key in features:
            if key != 'combined_text':
                features[key] = clean_text(features[key])

        # Try to extract additional annotation properties dynamically
        try:
            # Get all properties that might contain textual information
            for prop in dir(cls):
                if not prop.startswith('_') and prop not in semantic_properties:
                    try:
                        value = getattr(cls, prop)
                        if value and isinstance(value, (str, list)):
                            if isinstance(value, list):
                                text_value = ' '.join([str(v) for v in value if isinstance(v, str)])
                            else:
                                text_value = str(value)

                            if text_value and len(text_value.strip()) > 0:
                                # Only include if it looks like meaningful text
                                if any(c.isalpha() for c in text_value):
                                    features['annotations'] += ' ' + clean_text(text_value)
                    except:
                        continue
        except:
            pass

        # Clean annotations
        features['annotations'] = clean_text(features['annotations'])

        # Combine all text features with priorities
        text_parts = []

        # Primary text (higher weight)
        if features['label']:
            text_parts.append(features['label'])
        if features['definition']:
            text_parts.append(features['definition'])
        if features['comment']:
            text_parts.append(features['comment'])

        # Secondary text
        if features['description']:
            text_parts.append(features['description'])
        if features['preferred_labels']:
            text_parts.append(features['preferred_labels'])

        # Tertiary text
        if features['alternative_labels']:
            text_parts.append(features['alternative_labels'])
        if features['examples']:
            text_parts.append(features['examples'])
        if features['notes']:
            text_parts.append(features['notes'])

        # Additional context
        if features['annotations']:
            # Limit annotations to avoid noise
            annotations_text = features['annotations'][:500]  # Truncate if too long
            text_parts.append(annotations_text)

        features['combined_text'] = ' '.join([part for part in text_parts if part.strip()])

        # Only include classes that have meaningful text content
        if features['combined_text'].strip() and len(features['combined_text'].strip()) > 3:
            text_features[class_iri] = features

    logger.info(f"Extracted rich semantic features for {len(text_features)} classes out of {len(classes)} total")

    # Log some statistics about feature richness
    feature_stats = {}
    for feature_type in ['label', 'comment', 'definition', 'description', 'alternative_labels', 'examples']:
        count = sum(1 for features in text_features.values() if features[feature_type].strip())
        feature_stats[feature_type] = count

    logger.info(f"Feature richness: {feature_stats}")
    return text_features


def extract_text_features_from_owl(owl_file: str) -> Dict[str, Dict[str, str]]:
    """
    Extract textual features from an OWL ontology file.
    This is a simplified version that delegates to the rich semantic extraction.

    Args:
        owl_file (str): Path to the OWL ontology file

    Returns:
        dict: Dictionary mapping class IRIs to their text features
    """
    return extract_rich_semantic_features_from_owl(owl_file)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text (str): Raw text content

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\-\.,;:()\[\]{}]', ' ', text)

    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def create_text_node_features(text_features: Dict[str, Dict[str, str]],
                            class_to_index: Dict,
                            text_embedding_model,
                            use_combined: bool = True) -> torch.Tensor:
    """
    Create node features from text embeddings.

    Args:
        text_features (dict): Text features extracted from ontology
        class_to_index (dict): Mapping from ontology classes to indices
        text_embedding_model: Model to generate text embeddings
        use_combined (bool): Whether to use combined text or individual components

    Returns:
        torch.Tensor: Text-based node features matrix [num_nodes, embedding_dim]
    """
    num_nodes = len(class_to_index)

    # Get sample embedding to determine dimensionality
    sample_text = "sample text"
    sample_embedding = text_embedding_model.encode([sample_text])
    embedding_dim = sample_embedding.shape[1] if len(sample_embedding.shape) > 1 else len(sample_embedding[0])

    # Initialize feature matrix
    text_node_features = torch.zeros(num_nodes, embedding_dim, dtype=torch.float)

    # Generate embeddings for each class
    texts_to_embed = []
    indices_to_update = []

    for cls, idx in class_to_index.items():
        class_iri = cls.iri if hasattr(cls, 'iri') else str(cls)

        if class_iri in text_features:
            if use_combined:
                text_content = text_features[class_iri]['combined_text']
            else:
                # Use label as primary, fall back to other features
                text_content = (text_features[class_iri]['label'] or
                              text_features[class_iri]['comment'] or
                              text_features[class_iri]['definition'] or
                              text_features[class_iri]['description'])
        else:
            # Fallback: use class name if available
            class_name = cls.name if hasattr(cls, 'name') else class_iri.split('/')[-1].split('#')[-1]
            text_content = clean_text(class_name)

        if text_content.strip():
            texts_to_embed.append(text_content)
            indices_to_update.append(idx)

    if texts_to_embed:
        # Generate embeddings in batch for efficiency
        embeddings = text_embedding_model.encode(texts_to_embed)

        # Convert to tensor if needed
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float)

        # Update node features
        for i, idx in enumerate(indices_to_update):
            text_node_features[idx] = embeddings[i]

    logger.info(f"Created text node features: {text_node_features.shape}")
    logger.info(f"Non-zero feature vectors: {(text_node_features.norm(dim=1) > 0).sum().item()}")

    return text_node_features


class TextEmbeddingModel:
    """
    Base class for text embedding models.
    """

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings.

        Args:
            texts (list): List of text strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        raise NotImplementedError


class SentenceTransformerModel(TextEmbeddingModel):
    """
    Sentence Transformer based text embedding model.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package required. Install with: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using SentenceTransformer.

        Args:
            texts (list): List of text strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings


class HuggingFaceTransformerModel(TextEmbeddingModel):
    """
    Hugging Face Transformer based text embedding model.
    """

    def __init__(self, model_name: str = 'bert-base-uncased', pooling: str = 'mean'):
        """
        Initialize HuggingFace Transformer model.

        Args:
            model_name (str): Name of the HuggingFace model
            pooling (str): Pooling strategy ('mean', 'cls', 'max')
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        self.model_name = model_name
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        logger.info(f"Loaded HuggingFace model: {model_name}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using HuggingFace Transformer.

        Args:
            texts (list): List of text strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        embeddings = []
        batch_size = 32  # Process in batches to manage memory

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                # Forward pass
                outputs = self.model(**inputs)

                # Pool embeddings
                if self.pooling == 'mean':
                    # Mean pooling over sequence length
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state * attention_mask
                    batch_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                elif self.pooling == 'cls':
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling == 'max':
                    # Max pooling over sequence length
                    batch_embeddings = outputs.last_hidden_state.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")

                embeddings.append(batch_embeddings)

        # Concatenate all batches
        return torch.cat(embeddings, dim=0)


class OpenAIEmbeddingModel(TextEmbeddingModel):
    """
    OpenAI embedding model (requires API key).
    """

    def __init__(self, model_name: str = 'text-embedding-ada-002'):
        """
        Initialize OpenAI embedding model.

        Args:
            model_name (str): Name of the OpenAI embedding model
        """
        try:
            import openai
            import os
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.model_name = model_name

        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = openai.OpenAI()
        logger.info(f"Initialized OpenAI embedding model: {model_name}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using OpenAI embedding API.

        Args:
            texts (list): List of text strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        try:
            # OpenAI API accepts batch requests
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            return torch.tensor(embeddings, dtype=torch.float)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class SimpleTextModel(TextEmbeddingModel):
    """
    Simple text embedding model using TF-IDF or basic approaches.
    """

    def __init__(self, method: str = 'tfidf', max_features: int = 300):
        """
        Initialize simple text model.

        Args:
            method (str): Method to use ('tfidf', 'count')
            max_features (int): Maximum number of features
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.is_fitted = False

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using simple vectorization.

        Args:
            texts (list): List of text strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        except ImportError:
            raise ImportError("scikit-learn package required. Install with: pip install scikit-learn")

        if not self.is_fitted:
            if self.method == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    lowercase=True
                )
            else:  # count
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    lowercase=True
                )

            # Fit on the provided texts
            self.vectorizer.fit(texts)
            self.is_fitted = True
            logger.info(f"Fitted {self.method} vectorizer with {len(texts)} texts")

        # Transform texts to embeddings
        embeddings = self.vectorizer.transform(texts).toarray()
        return torch.tensor(embeddings, dtype=torch.float)


def create_text_embedding_model(model_type: str, **kwargs) -> TextEmbeddingModel:
    """
    Factory function to create text embedding models.

    Args:
        model_type (str): Type of model ('sentence_transformer', 'huggingface', 'openai', 'tfidf')
        **kwargs: Model-specific arguments

    Returns:
        TextEmbeddingModel: Configured text embedding model
    """
    if model_type == 'sentence_transformer':
        model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
        return SentenceTransformerModel(model_name)

    elif model_type == 'huggingface':
        model_name = kwargs.get('model_name', 'bert-base-uncased')
        pooling = kwargs.get('pooling', 'mean')
        return HuggingFaceTransformerModel(model_name, pooling)

    elif model_type == 'openai':
        model_name = kwargs.get('model_name', 'text-embedding-ada-002')
        return OpenAIEmbeddingModel(model_name)

    elif model_type == 'tfidf':
        method = kwargs.get('method', 'tfidf')
        max_features = kwargs.get('max_features', 300)
        return SimpleTextModel(method, max_features)

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: 'sentence_transformer', 'huggingface', 'openai', 'tfidf'")


def combine_structural_and_text_features(structural_features: torch.Tensor,
                                       text_features: torch.Tensor,
                                       combination_method: str = 'concat') -> torch.Tensor:
    """
    Combine structural (graph-based) and text-based node features.

    Args:
        structural_features (torch.Tensor): Structural node features
        text_features (torch.Tensor): Text-based node features
        combination_method (str): How to combine features ('concat', 'add', 'weighted_sum')

    Returns:
        torch.Tensor: Combined node features
    """
    if combination_method == 'concat':
        # Concatenate features along feature dimension
        combined_features = torch.cat([structural_features, text_features], dim=1)
    elif combination_method == 'add':
        # Element-wise addition (requires same dimensions)
        if structural_features.shape[1] != text_features.shape[1]:
            raise ValueError(f"Feature dimensions must match for addition: {structural_features.shape[1]} vs {text_features.shape[1]}")
        combined_features = structural_features + text_features
    elif combination_method == 'weighted_sum':
        # Weighted sum with learnable weights
        if structural_features.shape[1] != text_features.shape[1]:
            raise ValueError(f"Feature dimensions must match for weighted sum: {structural_features.shape[1]} vs {text_features.shape[1]}")
        # Simple equal weighting for now - could be made learnable
        combined_features = 0.5 * structural_features + 0.5 * text_features
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    logger.info(f"Combined features shape: {combined_features.shape}")
    return combined_features