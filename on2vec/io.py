"""
Input/Output utilities for embeddings and data
"""

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_embeddings_to_parquet(embeddings, node_ids, output_file, metadata=None,
                               text_embeddings=None, structural_embeddings=None,
                               additional_vectors=None):
    """
    Save embeddings to a Parquet file with optional metadata and additional vectors.

    Args:
        embeddings (torch.Tensor): Tensor containing main/fused embeddings
        node_ids (list): List of node identifiers
        output_file (str): Path to output Parquet file
        metadata (dict, optional): Metadata about the ontology and model
        text_embeddings (torch.Tensor, optional): Text-only embeddings
        structural_embeddings (torch.Tensor, optional): Structure-only embeddings
        additional_vectors (dict, optional): Dict of {name: tensor} for additional vectors

    Returns:
        None
    """
    logger.info(f"Saving embeddings to Parquet file: {output_file}")

    # Convert main embeddings tensor to a Python list
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

    # Create a Polars DataFrame with main embedding
    data = {'node_id': node_ids, 'embedding': embeddings_list}

    # Add text embeddings if provided
    if text_embeddings is not None:
        text_embeddings_list = text_embeddings.cpu().tolist()
        if len(text_embeddings_list) == len(node_ids):
            data['text_embedding'] = text_embeddings_list
            logger.info(f"Added text embeddings with dimension {len(text_embeddings_list[0]) if text_embeddings_list else 0}")
        else:
            logger.warning(f"Text embeddings length mismatch: {len(text_embeddings_list)} != {len(node_ids)}")

    # Add structural embeddings if provided
    if structural_embeddings is not None:
        structural_embeddings_list = structural_embeddings.cpu().tolist()
        if len(structural_embeddings_list) == len(node_ids):
            data['structural_embedding'] = structural_embeddings_list
            logger.info(f"Added structural embeddings with dimension {len(structural_embeddings_list[0]) if structural_embeddings_list else 0}")
        else:
            logger.warning(f"Structural embeddings length mismatch: {len(structural_embeddings_list)} != {len(node_ids)}")

    # Add any additional vectors
    if additional_vectors:
        for vector_name, vector_tensor in additional_vectors.items():
            if vector_tensor is not None:
                vector_list = vector_tensor.cpu().tolist()
                if len(vector_list) == len(node_ids):
                    data[f'{vector_name}_embedding'] = vector_list
                    logger.info(f"Added {vector_name} embeddings with dimension {len(vector_list[0]) if vector_list else 0}")
                else:
                    logger.warning(f"{vector_name} embeddings length mismatch: {len(vector_list)} != {len(node_ids)}")

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




def create_embedding_metadata(owl_file, model_config=None, alignment_info=None,
                            additional_info=None, text_model_info=None,
                            embedding_types=None):
    """
    Create metadata dictionary for embedding files.

    Args:
        owl_file (str): Path to source OWL file
        model_config (dict, optional): Model configuration information
        alignment_info (dict, optional): Ontology alignment information
        additional_info (dict, optional): Additional metadata
        text_model_info (dict, optional): Text embedding model information
        embedding_types (list, optional): List of embedding types stored in file

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

    if text_model_info:
        metadata['text_model_info'] = text_model_info
        logger.info(f"Added text model info: {text_model_info}")

    if embedding_types:
        metadata['embedding_types'] = embedding_types
        logger.info(f"Recorded embedding types: {embedding_types}")

    if additional_info:
        metadata.update(additional_info)

    # Add file stats if file exists
    if os.path.exists(owl_file):
        stat = os.stat(owl_file)
        metadata['source_file_size'] = stat.st_size
        metadata['source_file_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return metadata


def inspect_parquet_metadata(parquet_file):
    """
    Display metadata from a Parquet file in a nicely formatted way.

    Args:
        parquet_file (str): Path to the Parquet file

    Returns:
        dict: The metadata dictionary
    """
    logger.info(f"Inspecting metadata from: {parquet_file}")

    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        arrow_table = parquet_file_obj.read()

        print(f"\n📊 Embedding File: {parquet_file}")
        print("=" * 60)

        # Basic file info
        df = pl.from_arrow(arrow_table)
        num_embeddings = len(df)
        embedding_dim = len(df['embedding'][0]) if num_embeddings > 0 else 0

        print(f"📈 Embeddings: {num_embeddings:,} vectors")
        print(f"📐 Dimensions: {embedding_dim}")

        # Extract and display metadata
        metadata = {}
        if arrow_table.schema.metadata:
            import json
            for key, value in arrow_table.schema.metadata.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value

                if key_str.startswith('on2vec.'):
                    metadata_key = key_str[7:]  # Remove 'on2vec.' prefix
                    try:
                        metadata[metadata_key] = json.loads(value_str)
                    except json.JSONDecodeError:
                        metadata[metadata_key] = value_str

        if metadata:
            print(f"\n🏷️  Metadata:")
            print("-" * 30)

            # Display key metadata fields in a nice format
            if 'source_ontology_file' in metadata:
                print(f"📄 Source Ontology: {metadata['source_ontology_file']}")

            if 'generation_timestamp' in metadata:
                from datetime import datetime
                try:
                    timestamp = datetime.fromisoformat(metadata['generation_timestamp'])
                    print(f"⏰ Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    print(f"⏰ Generated: {metadata['generation_timestamp']}")

            if 'model_config' in metadata:
                config = metadata['model_config']
                print(f"🤖 Model: {config.get('model_type', 'unknown').upper()}")
                print(f"   └─ Hidden: {config.get('hidden_dim', 'unknown')}, Output: {config.get('out_dim', 'unknown')}")
                if 'loss_function' in config:
                    print(f"   └─ Loss: {config['loss_function']}")

            if 'alignment_info' in metadata:
                align = metadata['alignment_info']
                print(f"🔗 Alignment: {align.get('aligned_classes', 0)}/{align.get('total_classes', 0)} classes")
                if 'alignment_ratio' in align:
                    print(f"   └─ Ratio: {align['alignment_ratio']:.1%}")

            if 'source_file_size' in metadata:
                size_mb = int(metadata['source_file_size']) / (1024 * 1024)
                print(f"📊 Source Size: {size_mb:.1f} MB")

            # Show any additional metadata
            shown_keys = {'source_ontology_file', 'generation_timestamp', 'model_config',
                         'alignment_info', 'source_file_size', 'source_file_modified',
                         'source_ontology_path', 'on2vec_version'}

            additional = {k: v for k, v in metadata.items() if k not in shown_keys}
            if additional:
                print(f"\n📋 Additional Metadata:")
                for key, value in additional.items():
                    if isinstance(value, (dict, list)):
                        print(f"   {key}: {json.dumps(value, indent=2)}")
                    else:
                        print(f"   {key}: {value}")

        print("\n" + "=" * 60)

        return metadata

    except Exception as e:
        logger.error(f"Failed to inspect metadata: {e}")
        raise


def convert_parquet_to_csv(parquet_file, csv_file=None):
    """
    Convert a Parquet embedding file to CSV format.

    Args:
        parquet_file (str): Path to input Parquet file
        csv_file (str, optional): Path to output CSV file. If None, uses same name with .csv extension

    Returns:
        str: Path to the created CSV file
    """
    if csv_file is None:
        csv_file = parquet_file.replace('.parquet', '.csv')

    logger.info(f"Converting {parquet_file} to {csv_file}")

    try:
        # Load embeddings from Parquet
        node_ids, embeddings = load_embeddings_from_parquet(parquet_file)

        # Create DataFrame with embedding columns
        embedding_dim = embeddings.shape[1]
        embedding_cols = [f'dim_{i}' for i in range(embedding_dim)]

        data = {'node_id': node_ids}
        for i, col in enumerate(embedding_cols):
            data[col] = embeddings[:, i]

        df = pl.DataFrame(data)
        df.write_csv(csv_file)

        logger.info(f"CSV file created: {csv_file}")
        logger.info(f"Converted {len(node_ids)} embeddings with {embedding_dim} dimensions")

        return csv_file

    except Exception as e:
        logger.error(f"Failed to convert to CSV: {e}")
        raise


def load_embeddings_as_dataframe(parquet_file, return_metadata=False):
    """
    Load embeddings from Parquet file as a Polars DataFrame.

    Args:
        parquet_file (str): Path to the Parquet file
        return_metadata (bool): Whether to return metadata along with DataFrame

    Returns:
        pl.DataFrame or tuple: DataFrame with 'node_id' and 'embedding' columns,
                              optionally with metadata dict
    """
    logger.info(f"Loading embeddings as DataFrame from: {parquet_file}")

    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        arrow_table = parquet_file_obj.read()
        df = pl.from_arrow(arrow_table)

        logger.info(f"Loaded DataFrame with {len(df)} rows")

        if return_metadata:
            metadata = {}
            if arrow_table.schema.metadata:
                import json
                for key, value in arrow_table.schema.metadata.items():
                    key_str = key.decode() if isinstance(key, bytes) else key
                    value_str = value.decode() if isinstance(value, bytes) else value

                    if key_str.startswith('on2vec.'):
                        metadata_key = key_str[7:]  # Remove 'on2vec.' prefix
                        try:
                            metadata[metadata_key] = json.loads(value_str)
                        except json.JSONDecodeError:
                            metadata[metadata_key] = value_str

            return df, metadata
        else:
            return df

    except Exception as e:
        logger.error(f"Failed to load DataFrame: {e}")
        raise


def add_embedding_vectors(parquet_file1, node_id1, parquet_file2, node_id2):
    """
    Add two embedding vectors from potentially different Parquet files.

    Args:
        parquet_file1 (str): Path to first Parquet file
        node_id1 (str): Node ID in first file
        parquet_file2 (str): Path to second Parquet file
        node_id2 (str): Node ID in second file

    Returns:
        np.ndarray: Sum of the two embedding vectors
    """
    logger.info(f"Adding vectors: {node_id1} + {node_id2}")

    try:
        # Load first embedding
        df1 = load_embeddings_as_dataframe(parquet_file1)
        row1 = df1.filter(pl.col('node_id') == node_id1)
        if len(row1) == 0:
            raise ValueError(f"Node ID '{node_id1}' not found in {parquet_file1}")
        embedding1 = np.array(row1['embedding'][0])

        # Load second embedding
        df2 = load_embeddings_as_dataframe(parquet_file2)
        row2 = df2.filter(pl.col('node_id') == node_id2)
        if len(row2) == 0:
            raise ValueError(f"Node ID '{node_id2}' not found in {parquet_file2}")
        embedding2 = np.array(row2['embedding'][0])

        # Check dimensions match
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")

        result = embedding1 + embedding2
        logger.info(f"Vector addition successful, result dimension: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"Failed to add vectors: {e}")
        raise


def subtract_embedding_vectors(parquet_file1, node_id1, parquet_file2, node_id2):
    """
    Subtract two embedding vectors from potentially different Parquet files.

    Args:
        parquet_file1 (str): Path to first Parquet file (minuend)
        node_id1 (str): Node ID in first file
        parquet_file2 (str): Path to second Parquet file (subtrahend)
        node_id2 (str): Node ID in second file

    Returns:
        np.ndarray: Difference of the two embedding vectors (vector1 - vector2)
    """
    logger.info(f"Subtracting vectors: {node_id1} - {node_id2}")

    try:
        # Load first embedding
        df1 = load_embeddings_as_dataframe(parquet_file1)
        row1 = df1.filter(pl.col('node_id') == node_id1)
        if len(row1) == 0:
            raise ValueError(f"Node ID '{node_id1}' not found in {parquet_file1}")
        embedding1 = np.array(row1['embedding'][0])

        # Load second embedding
        df2 = load_embeddings_as_dataframe(parquet_file2)
        row2 = df2.filter(pl.col('node_id') == node_id2)
        if len(row2) == 0:
            raise ValueError(f"Node ID '{node_id2}' not found in {parquet_file2}")
        embedding2 = np.array(row2['embedding'][0])

        # Check dimensions match
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")

        result = embedding1 - embedding2
        logger.info(f"Vector subtraction successful, result dimension: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"Failed to subtract vectors: {e}")
        raise


def get_embedding_vector(parquet_file, node_id):
    """
    Get a single embedding vector from a Parquet file.

    Args:
        parquet_file (str): Path to Parquet file
        node_id (str): Node ID to retrieve

    Returns:
        np.ndarray: The embedding vector
    """
    try:
        df = load_embeddings_as_dataframe(parquet_file)
        row = df.filter(pl.col('node_id') == node_id)
        if len(row) == 0:
            raise ValueError(f"Node ID '{node_id}' not found in {parquet_file}")

        return np.array(row['embedding'][0])

    except Exception as e:
        logger.error(f"Failed to get embedding vector: {e}")
        raise