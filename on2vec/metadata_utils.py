"""
Utilities for extracting metadata from on2vec parquet files.
"""

import json
import logging
from typing import Dict, Any, Optional
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def extract_parquet_metadata(parquet_file: str) -> Dict[str, Any]:
    """
    Extract on2vec metadata from parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dictionary containing metadata
    """
    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        arrow_table = parquet_file_obj.read()

        metadata = {}
        if arrow_table.schema.metadata:
            for key, value in arrow_table.schema.metadata.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value

                if key_str.startswith('on2vec.'):
                    metadata_key = key_str[7:]  # Remove 'on2vec.' prefix
                    try:
                        metadata[metadata_key] = json.loads(value_str)
                    except json.JSONDecodeError:
                        metadata[metadata_key] = value_str

        return metadata

    except Exception as e:
        logger.warning(f"Could not extract metadata from {parquet_file}: {e}")
        return {}


def get_text_model_info(parquet_file: str) -> Optional[Dict[str, Any]]:
    """
    Extract text model information from parquet metadata.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dictionary with text model info or None if not found
    """
    metadata = extract_parquet_metadata(parquet_file)

    # Try different possible keys for text model info
    text_model_info = None

    if 'text_model_info' in metadata:
        text_model_info = metadata['text_model_info']
    elif 'model_config' in metadata and isinstance(metadata['model_config'], dict):
        model_config = metadata['model_config']
        if 'text_model_name' in model_config:
            text_model_info = {
                'model_type': model_config.get('text_model_type', 'sentence_transformer'),
                'model_name': model_config['text_model_name'],
                'text_dim': model_config.get('text_dim')
            }

    return text_model_info


def get_base_model_from_embeddings(parquet_file: str) -> Optional[str]:
    """
    Extract the base text model name used to generate the embeddings.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Base model name (e.g., 'all-MiniLM-L6-v2') or None if not found
    """
    text_model_info = get_text_model_info(parquet_file)

    if text_model_info and 'model_name' in text_model_info:
        return text_model_info['model_name']

    return None


def validate_text_embeddings_compatibility(parquet_file: str, requested_base_model: str) -> bool:
    """
    Check if the requested base model is compatible with the embeddings file.

    Args:
        parquet_file: Path to parquet file
        requested_base_model: Base model name requested by user

    Returns:
        True if compatible, False otherwise
    """
    stored_base_model = get_base_model_from_embeddings(parquet_file)

    if stored_base_model is None:
        logger.warning(f"Could not determine base model from {parquet_file}")
        return True  # Allow if we can't determine

    if stored_base_model != requested_base_model:
        logger.warning(f"Base model mismatch: embeddings use '{stored_base_model}' but '{requested_base_model}' was requested")
        return False

    return True


def get_embedding_info(parquet_file: str) -> Dict[str, Any]:
    """
    Get comprehensive embedding information from parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dictionary with embedding information
    """
    metadata = extract_parquet_metadata(parquet_file)
    text_model_info = get_text_model_info(parquet_file)

    # Load basic info from the file
    import polars as pl
    df = pl.read_parquet(parquet_file)

    info = {
        'num_embeddings': len(df),
        'has_text_embeddings': 'text_embedding' in df.columns,
        'has_structural_embeddings': 'structural_embedding' in df.columns,
        'base_model': get_base_model_from_embeddings(parquet_file),
        'text_model_info': text_model_info,
        'metadata': metadata
    }

    if info['has_text_embeddings']:
        info['text_dim'] = len(df['text_embedding'][0])

    if info['has_structural_embeddings']:
        info['structural_dim'] = len(df['structural_embedding'][0])

    return info