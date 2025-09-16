"""
Input/Output utilities for embeddings and data
"""

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_embeddings_to_parquet(embeddings, node_ids, output_file, metadata=None):
    """
    Save embeddings to a Parquet file with optional metadata.

    Args:
        embeddings (torch.Tensor): Tensor containing embeddings
        node_ids (list): List of node identifiers
        output_file (str): Path to output Parquet file
        metadata (dict, optional): Metadata about the ontology and model

    Returns:
        None
    """
    logger.info(f"Saving embeddings to Parquet file: {output_file}")

    # Convert embeddings tensor to a Python list
    embeddings_list = embeddings.cpu().tolist()

    # Ensure we have the right number of node IDs
    if len(node_ids) != len(embeddings_list):
        logger.warning(f"Mismatch between node_ids ({len(node_ids)}) and embeddings ({len(embeddings_list)})")
        # Truncate or pad as needed
        if len(node_ids) > len(embeddings_list):
            node_ids = node_ids[:len(embeddings_list)]
        else:
            # Add generic node IDs if needed
            for i in range(len(embeddings_list) - len(node_ids)):
                node_ids.append(f"node_{len(node_ids) + i}")

    # Create a Polars DataFrame
    data = {'node_id': node_ids, 'embedding': embeddings_list}
    df = pl.DataFrame(data)

    # Convert to Arrow Table
    arrow_table = df.to_arrow()

    # Add metadata to the Arrow Table
    if metadata:
        # Convert metadata to bytes for Arrow metadata
        import json
        from datetime import datetime

        # Add timestamp if not provided
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()

        # Add embedding dimension info
        if embeddings_list:
            metadata['embedding_dimension'] = len(embeddings_list[0])
            metadata['num_embeddings'] = len(embeddings_list)

        # Convert all metadata values to strings for Arrow compatibility
        arrow_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                arrow_metadata[f"on2vec.{key}"] = json.dumps(value)
            else:
                arrow_metadata[f"on2vec.{key}"] = str(value)

        # Apply metadata to the table
        arrow_table = arrow_table.replace_schema_metadata(arrow_metadata)

    # Write to Parquet
    pq.write_table(arrow_table, output_file)

    logger.info(f"Embeddings saved to {output_file}")
    if metadata:
        logger.info(f"Metadata included: {list(metadata.keys())}")


def load_embeddings_from_parquet(parquet_file, return_metadata=False):
    """
    Load embeddings from a Parquet file with optional metadata.

    Args:
        parquet_file (str): Path to the Parquet file
        return_metadata (bool): Whether to return metadata along with embeddings

    Returns:
        tuple: (node_ids, embeddings) or (node_ids, embeddings, metadata)
            - node_ids (list): List of node identifiers
            - embeddings (np.ndarray): 2D array of embeddings
            - metadata (dict, optional): Metadata from the Parquet file
    """
    logger.info(f"Loading embeddings from Parquet file: {parquet_file}")

    # Read Parquet file
    parquet_file_obj = pq.ParquetFile(parquet_file)
    arrow_table = parquet_file_obj.read()

    # Extract metadata if present
    metadata = {}
    if arrow_table.schema.metadata:
        import json
        for key, value in arrow_table.schema.metadata.items():
            key_str = key.decode() if isinstance(key, bytes) else key
            value_str = value.decode() if isinstance(value, bytes) else value

            if key_str.startswith('on2vec.'):
                metadata_key = key_str[7:]  # Remove 'on2vec.' prefix
                try:
                    # Try to parse as JSON first
                    metadata[metadata_key] = json.loads(value_str)
                except json.JSONDecodeError:
                    # If not JSON, store as string
                    metadata[metadata_key] = value_str

    # Convert to Polars DataFrame
    df = pl.from_arrow(arrow_table)

    # Extract node IDs and embeddings
    node_ids = df['node_id'].to_list()
    embeddings = np.vstack(df['embedding'].to_list())

    logger.info(f"Loaded {len(node_ids)} embeddings of dimension {embeddings.shape[1]}")
    if metadata:
        logger.info(f"Loaded metadata: {list(metadata.keys())}")

    if return_metadata:
        return node_ids, embeddings, metadata
    else:
        return node_ids, embeddings


def save_embeddings_to_csv(embeddings, node_ids, output_file):
    """
    Save embeddings to a CSV file (alternative format).

    Args:
        embeddings (torch.Tensor): Tensor containing embeddings
        node_ids (list): List of node identifiers
        output_file (str): Path to output CSV file

    Returns:
        None
    """
    logger.info(f"Saving embeddings to CSV file: {output_file}")

    # Convert embeddings to numpy
    embeddings_np = embeddings.cpu().numpy()

    # Create column names for embedding dimensions
    embedding_dim = embeddings_np.shape[1]
    embedding_cols = [f'dim_{i}' for i in range(embedding_dim)]

    # Create DataFrame
    data = {'node_id': node_ids}
    for i, col in enumerate(embedding_cols):
        data[col] = embeddings_np[:, i]

    df = pl.DataFrame(data)
    df.write_csv(output_file)

    logger.info(f"Embeddings saved to {output_file}")


def load_embeddings_from_csv(csv_file):
    """
    Load embeddings from a CSV file.

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        tuple: (node_ids, embeddings)
            - node_ids (list): List of node identifiers
            - embeddings (np.ndarray): 2D array of embeddings
    """
    logger.info(f"Loading embeddings from CSV file: {csv_file}")

    # Read CSV file
    df = pl.read_csv(csv_file)

    # Extract node IDs
    node_ids = df['node_id'].to_list()

    # Extract embedding columns (all columns except node_id)
    embedding_cols = [col for col in df.columns if col != 'node_id']
    embeddings = df.select(embedding_cols).to_numpy()

    logger.info(f"Loaded {len(node_ids)} embeddings of dimension {embeddings.shape[1]}")

    return node_ids, embeddings


def export_embeddings(embeddings, node_ids, output_file, format='parquet'):
    """
    Export embeddings in the specified format.

    Args:
        embeddings (torch.Tensor): Tensor containing embeddings
        node_ids (list): List of node identifiers
        output_file (str): Path to output file
        format (str): Output format ('parquet' or 'csv')

    Returns:
        None
    """
    if format.lower() == 'parquet':
        save_embeddings_to_parquet(embeddings, node_ids, output_file)
    elif format.lower() == 'csv':
        save_embeddings_to_csv(embeddings, node_ids, output_file)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")


def load_embeddings(input_file, format=None):
    """
    Load embeddings from file, auto-detecting format if not specified.

    Args:
        input_file (str): Path to input file
        format (str, optional): Input format ('parquet' or 'csv')

    Returns:
        tuple: (node_ids, embeddings)
            - node_ids (list): List of node identifiers
            - embeddings (np.ndarray): 2D array of embeddings
    """
    if format is None:
        # Auto-detect format from file extension
        if input_file.endswith('.parquet'):
            format = 'parquet'
        elif input_file.endswith('.csv'):
            format = 'csv'
        else:
            raise ValueError(f"Cannot determine format for {input_file}. Specify format parameter.")

    if format.lower() == 'parquet':
        return load_embeddings_from_parquet(input_file)
    elif format.lower() == 'csv':
        return load_embeddings_from_csv(input_file)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")




def create_embedding_metadata(owl_file, model_config=None, alignment_info=None, additional_info=None):
    """
    Create metadata dictionary for embedding files.

    Args:
        owl_file (str): Path to source OWL file
        model_config (dict, optional): Model configuration information
        alignment_info (dict, optional): Ontology alignment information
        additional_info (dict, optional): Additional metadata

    Returns:
        dict: Metadata dictionary
    """
    import os
    from datetime import datetime

    metadata = {
        'source_ontology_file': os.path.basename(owl_file),
        'source_ontology_path': os.path.abspath(owl_file),
        'generation_timestamp': datetime.now().isoformat(),
        'on2vec_version': '0.1.0'
    }

    if model_config:
        metadata['model_config'] = model_config

    if alignment_info:
        metadata['alignment_info'] = alignment_info

    if additional_info:
        metadata.update(additional_info)

    # Add file stats if file exists
    if os.path.exists(owl_file):
        stat = os.stat(owl_file)
        metadata['source_file_size'] = stat.st_size
        metadata['source_file_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return metadata