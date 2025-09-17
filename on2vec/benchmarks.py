"""
Benchmark datasets and baseline comparison utilities for ontology embeddings
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


class OntologyBenchmarkDatasets:
    """
    Collection of benchmark ontology datasets for evaluation.
    """

    # Known ontology datasets with download URLs
    DATASETS = {
        'go': {
            'name': 'Gene Ontology',
            'url': 'http://purl.obolibrary.org/obo/go.owl',
            'description': 'Gene Ontology - structured vocabulary of gene and gene product attributes',
            'domain': 'biology',
            'size': 'large',
            'relations': ['subclass', 'part_of', 'regulates']
        },
        'chebi': {
            'name': 'Chemical Entities of Biological Interest',
            'url': 'http://purl.obolibrary.org/obo/chebi.owl',
            'description': 'Chemical ontology for biological and chemical entities',
            'domain': 'chemistry',
            'size': 'large',
            'relations': ['subclass', 'has_part', 'is_conjugate_base_of']
        },
        'hp': {
            'name': 'Human Phenotype Ontology',
            'url': 'http://purl.obolibrary.org/obo/hp.owl',
            'description': 'Comprehensive logical standard to describe phenotypic abnormalities in humans',
            'domain': 'medicine',
            'size': 'medium',
            'relations': ['subclass', 'part_of', 'has_modifier']
        },
        'mondo': {
            'name': 'Mondo Disease Ontology',
            'url': 'http://purl.obolibrary.org/obo/mondo.owl',
            'description': 'Ontology of diseases and disorders',
            'domain': 'medicine',
            'size': 'medium',
            'relations': ['subclass', 'has_material_basis_in', 'disease_has_location']
        },
        'cl': {
            'name': 'Cell Ontology',
            'url': 'http://purl.obolibrary.org/obo/cl.owl',
            'description': 'Ontology for cell types',
            'domain': 'biology',
            'size': 'medium',
            'relations': ['subclass', 'develops_from', 'part_of']
        },
        'uberon': {
            'name': 'Uberon Anatomy Ontology',
            'url': 'http://purl.obolibrary.org/obo/uberon.owl',
            'description': 'Cross-species anatomy ontology',
            'domain': 'biology',
            'size': 'large',
            'relations': ['subclass', 'part_of', 'develops_from']
        },
        'foodon': {
            'name': 'Food Ontology',
            'url': 'http://purl.obolibrary.org/obo/foodon.owl',
            'description': 'Ontology for food, food products, and food-related concepts',
            'domain': 'food_science',
            'size': 'medium',
            'relations': ['subclass', 'has_ingredient', 'derives_from']
        },
        'edam': {
            'name': 'EDAM Ontology',
            'url': 'http://edamontology.org/EDAM.owl',
            'description': 'Ontology for bioinformatics operations, types, formats, and identifiers',
            'domain': 'bioinformatics',
            'size': 'small',
            'relations': ['subclass', 'has_input', 'has_output']
        },
        'cvdo': {
            'name': 'Cardiovascular Disease Ontology',
            'url': 'https://raw.githubusercontent.com/OpenLHS/CVDO/master/CVDO.owl',
            'description': 'Ontology for cardiovascular diseases',
            'domain': 'medicine',
            'size': 'small',
            'relations': ['subclass', 'has_location', 'has_symptom']
        },
        'ncit': {
            'name': 'NCI Thesaurus',
            'url': 'https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/22.03d/Thesaurus.owl',
            'description': 'Cancer research vocabulary',
            'domain': 'oncology',
            'size': 'very_large',
            'relations': ['subclass', 'Anatomic_Structure_Has_Location', 'Gene_Product_Has_Biochemical_Function']
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize benchmark dataset manager.

        Args:
            cache_dir (str, optional): Directory to cache downloaded ontologies
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.on2vec_cache' / 'benchmark_datasets'

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")

    def list_available_datasets(self, domain: Optional[str] = None,
                              size: Optional[str] = None) -> Dict[str, Dict]:
        """
        List available benchmark datasets.

        Args:
            domain (str, optional): Filter by domain
            size (str, optional): Filter by size (small, medium, large, very_large)

        Returns:
            dict: Filtered dataset information
        """
        datasets = {}

        for key, info in self.DATASETS.items():
            include = True

            if domain and info['domain'] != domain:
                include = False

            if size and info['size'] != size:
                include = False

            if include:
                datasets[key] = info

        return datasets

    def download_dataset(self, dataset_key: str, force_redownload: bool = False) -> str:
        """
        Download a benchmark dataset.

        Args:
            dataset_key (str): Key identifying the dataset
            force_redownload (bool): Whether to redownload existing files

        Returns:
            str: Path to the downloaded ontology file
        """
        if dataset_key not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(self.DATASETS.keys())}")

        dataset_info = self.DATASETS[dataset_key]
        url = dataset_info['url']

        # Create filename from URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        if not filename.endswith(('.owl', '.obo', '.xml')):
            filename = f"{dataset_key}.owl"

        filepath = self.cache_dir / filename

        # Check if file already exists
        if filepath.exists() and not force_redownload:
            logger.info(f"Dataset {dataset_key} already cached at {filepath}")
            return str(filepath)

        logger.info(f"Downloading {dataset_info['name']} from {url}")

        try:
            # Download with progress
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({percent:.1f}%)")

            logger.info(f"Successfully downloaded {dataset_key} to {filepath}")

            # Verify the file is valid
            file_size = filepath.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                logger.warning(f"Downloaded file seems too small ({file_size} bytes)")

            return str(filepath)

        except requests.RequestException as e:
            logger.error(f"Failed to download {dataset_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing download for {dataset_key}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            raise

    def download_benchmark_suite(self, domains: Optional[List[str]] = None,
                                sizes: Optional[List[str]] = None,
                                force_redownload: bool = False) -> Dict[str, str]:
        """
        Download a suite of benchmark datasets.

        Args:
            domains (list, optional): List of domains to include
            sizes (list, optional): List of sizes to include
            force_redownload (bool): Whether to redownload existing files

        Returns:
            dict: Mapping of dataset keys to file paths
        """
        # Get filtered datasets
        available_datasets = self.list_available_datasets()

        if domains:
            available_datasets = {
                k: v for k, v in available_datasets.items()
                if v['domain'] in domains
            }

        if sizes:
            available_datasets = {
                k: v for k, v in available_datasets.items()
                if v['size'] in sizes
            }

        logger.info(f"Downloading benchmark suite: {len(available_datasets)} datasets")

        downloaded_paths = {}
        failed_downloads = []

        for dataset_key in available_datasets:
            try:
                path = self.download_dataset(dataset_key, force_redownload)
                downloaded_paths[dataset_key] = path
            except Exception as e:
                logger.error(f"Failed to download {dataset_key}: {e}")
                failed_downloads.append(dataset_key)

        if failed_downloads:
            logger.warning(f"Failed to download {len(failed_downloads)} datasets: {failed_downloads}")

        logger.info(f"Successfully downloaded {len(downloaded_paths)} datasets")
        return downloaded_paths

    def get_dataset_info(self, dataset_key: str) -> Dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset_key (str): Dataset identifier

        Returns:
            dict: Dataset information
        """
        if dataset_key not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")

        return self.DATASETS[dataset_key].copy()

    def create_cross_domain_evaluation_set(self,
                                         train_domains: List[str],
                                         test_domains: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Create training and test sets for cross-domain evaluation.

        Args:
            train_domains (list): List of domains for training
            test_domains (list): List of domains for testing

        Returns:
            dict: Training and test dataset paths
        """
        train_datasets = {}
        test_datasets = {}

        # Download training datasets
        for domain in train_domains:
            domain_datasets = self.list_available_datasets(domain=domain)
            for key in domain_datasets:
                try:
                    path = self.download_dataset(key)
                    train_datasets[key] = path
                except Exception as e:
                    logger.warning(f"Failed to download training dataset {key}: {e}")

        # Download test datasets
        for domain in test_domains:
            domain_datasets = self.list_available_datasets(domain=domain)
            for key in domain_datasets:
                try:
                    path = self.download_dataset(key)
                    test_datasets[key] = path
                except Exception as e:
                    logger.warning(f"Failed to download test dataset {key}: {e}")

        return {
            'train': train_datasets,
            'test': test_datasets
        }


class BaselineComparison:
    """
    Framework for comparing against baseline ontology embedding methods.
    """

    def __init__(self):
        """Initialize baseline comparison framework."""
        self.baselines = {
            'random': self._create_random_baseline,
            'structural': self._create_structural_baseline,
            'text_only': self._create_text_only_baseline,
            'node2vec': self._create_node2vec_baseline,
            'deepwalk': self._create_deepwalk_baseline
        }

    def list_available_baselines(self) -> List[str]:
        """
        List available baseline methods.

        Returns:
            list: Available baseline method names
        """
        return list(self.baselines.keys())

    def create_baseline_embeddings(self,
                                  baseline_method: str,
                                  ontology_file: str,
                                  embedding_dim: int = 64,
                                  **kwargs) -> str:
        """
        Create baseline embeddings for comparison.

        Args:
            baseline_method (str): Name of baseline method
            ontology_file (str): Path to ontology file
            embedding_dim (int): Embedding dimension
            **kwargs: Method-specific arguments

        Returns:
            str: Path to generated baseline embeddings
        """
        if baseline_method not in self.baselines:
            raise ValueError(f"Unknown baseline method: {baseline_method}")

        logger.info(f"Creating {baseline_method} baseline embeddings for {ontology_file}")

        return self.baselines[baseline_method](
            ontology_file, embedding_dim, **kwargs
        )

    def _create_random_baseline(self, ontology_file: str, embedding_dim: int, **kwargs) -> str:
        """Create random embeddings baseline."""
        from .ontology import build_graph_from_owl
        from .io import save_embeddings_to_parquet, create_embedding_metadata
        import numpy as np
        import tempfile

        # Build graph to get node count
        x, edge_index, class_mapping = build_graph_from_owl(ontology_file)
        n_nodes = len(class_mapping)

        # Create random embeddings
        np.random.seed(42)  # For reproducibility
        embeddings = np.random.normal(0, 1, (n_nodes, embedding_dim))

        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create node IDs
        node_ids = []
        for ont_class in class_mapping.keys():
            node_id = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            node_ids.append(node_id)

        # Create metadata
        metadata = create_embedding_metadata(
            ontology_file=ontology_file,
            model_type='random_baseline',
            embedding_dim=embedding_dim,
            method_info={'baseline': 'random', 'distribution': 'normal'}
        )

        # Save to temporary file
        output_file = tempfile.mktemp(suffix='_random_baseline.parquet')
        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata
        )

        logger.info(f"Created random baseline embeddings: {output_file}")
        return output_file

    def _create_structural_baseline(self, ontology_file: str, embedding_dim: int, **kwargs) -> str:
        """Create structural features baseline (degree, centrality, etc.)."""
        from .ontology import build_graph_from_owl
        from .io import save_embeddings_to_parquet, create_embedding_metadata
        import numpy as np
        import networkx as nx
        import tempfile
        from sklearn.preprocessing import StandardScaler

        # Build graph
        x, edge_index, class_mapping = build_graph_from_owl(ontology_file)

        # Create NetworkX graph
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

        # Calculate structural features
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
        closeness_centrality = nx.closeness_centrality(G)

        try:
            clustering_coefficient = nx.clustering(G)
        except:
            clustering_coefficient = {node: 0 for node in G.nodes()}

        # Create feature matrix
        features = []
        node_ids = []

        for ont_class, ont_idx in class_mapping.items():
            node_id = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            node_ids.append(node_id)

            # Collect structural features
            feature_vector = [
                degree_centrality.get(ont_idx, 0),
                betweenness_centrality.get(ont_idx, 0),
                closeness_centrality.get(ont_idx, 0),
                clustering_coefficient.get(ont_idx, 0),
                G.degree(ont_idx) if ont_idx in G else 0  # Raw degree
            ]

            # Pad or truncate to desired dimension
            if len(feature_vector) < embedding_dim:
                feature_vector.extend([0] * (embedding_dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:embedding_dim]

            features.append(feature_vector)

        embeddings = np.array(features, dtype=np.float32)

        # Standardize features
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        # Create metadata
        metadata = create_embedding_metadata(
            ontology_file=ontology_file,
            model_type='structural_baseline',
            embedding_dim=embedding_dim,
            method_info={'baseline': 'structural', 'features': ['degree', 'betweenness', 'closeness', 'clustering']}
        )

        # Save to temporary file
        output_file = tempfile.mktemp(suffix='_structural_baseline.parquet')
        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata
        )

        logger.info(f"Created structural baseline embeddings: {output_file}")
        return output_file

    def _create_text_only_baseline(self, ontology_file: str, embedding_dim: int, **kwargs) -> str:
        """Create text-only baseline using TF-IDF or simple text features."""
        from .text_features import extract_text_features_from_owl, create_text_embedding_model
        from .ontology import build_graph_from_owl
        from .io import save_embeddings_to_parquet, create_embedding_metadata
        import tempfile

        # Extract text features
        text_features = extract_text_features_from_owl(ontology_file)

        # Build graph to get class mapping
        x, edge_index, class_mapping = build_graph_from_owl(ontology_file)

        # Create text embedding model
        text_model = create_text_embedding_model('tfidf', max_features=embedding_dim)

        # Prepare texts and node IDs
        texts = []
        node_ids = []

        for ont_class in class_mapping.keys():
            node_id = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            node_ids.append(node_id)

            if node_id in text_features:
                text_content = text_features[node_id]['combined_text']
            else:
                # Fallback to class name
                text_content = ont_class.name if hasattr(ont_class, 'name') else node_id.split('/')[-1]

            texts.append(text_content if text_content.strip() else 'unknown')

        # Generate embeddings
        embeddings = text_model.encode(texts)

        # Convert to numpy if needed
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        elif not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.toarray()

        # Create metadata
        metadata = create_embedding_metadata(
            ontology_file=ontology_file,
            model_type='text_only_baseline',
            embedding_dim=embeddings.shape[1],
            method_info={'baseline': 'text_only', 'text_model': 'tfidf'}
        )

        # Save to temporary file
        output_file = tempfile.mktemp(suffix='_text_baseline.parquet')
        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata
        )

        logger.info(f"Created text-only baseline embeddings: {output_file}")
        return output_file

    def _create_node2vec_baseline(self, ontology_file: str, embedding_dim: int, **kwargs) -> str:
        """Create Node2Vec baseline."""
        try:
            from node2vec import Node2Vec
        except ImportError:
            logger.error("node2vec package required. Install with: pip install node2vec")
            raise

        from .ontology import build_graph_from_owl
        from .io import save_embeddings_to_parquet, create_embedding_metadata
        import networkx as nx
        import tempfile
        import numpy as np

        # Build graph
        x, edge_index, class_mapping = build_graph_from_owl(ontology_file)

        # Create NetworkX graph
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

        # Create Node2Vec model
        node2vec = Node2Vec(
            G,
            dimensions=embedding_dim,
            walk_length=kwargs.get('walk_length', 30),
            num_walks=kwargs.get('num_walks', 200),
            workers=kwargs.get('workers', 4)
        )

        # Train model
        model = node2vec.fit(
            window=kwargs.get('window', 10),
            min_count=1,
            batch_words=4
        )

        # Extract embeddings
        embeddings = []
        node_ids = []

        for ont_class, ont_idx in class_mapping.items():
            node_id = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            node_ids.append(node_id)

            if str(ont_idx) in model.wv:
                embedding = model.wv[str(ont_idx)]
            else:
                # Fallback to random vector
                embedding = np.random.normal(0, 0.1, embedding_dim)

            embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)

        # Create metadata
        metadata = create_embedding_metadata(
            ontology_file=ontology_file,
            model_type='node2vec_baseline',
            embedding_dim=embedding_dim,
            method_info={
                'baseline': 'node2vec',
                'walk_length': kwargs.get('walk_length', 30),
                'num_walks': kwargs.get('num_walks', 200)
            }
        )

        # Save to temporary file
        output_file = tempfile.mktemp(suffix='_node2vec_baseline.parquet')
        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata
        )

        logger.info(f"Created Node2Vec baseline embeddings: {output_file}")
        return output_file

    def _create_deepwalk_baseline(self, ontology_file: str, embedding_dim: int, **kwargs) -> str:
        """Create DeepWalk baseline."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            logger.error("gensim package required. Install with: pip install gensim")
            raise

        from .ontology import build_graph_from_owl
        from .io import save_embeddings_to_parquet, create_embedding_metadata
        import networkx as nx
        import tempfile
        import numpy as np
        import random

        # Build graph
        x, edge_index, class_mapping = build_graph_from_owl(ontology_file)

        # Create NetworkX graph
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

        # Generate random walks
        def random_walk(graph, start_node, walk_length):
            walk = [start_node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            return walk

        # Generate walks
        walks = []
        num_walks = kwargs.get('num_walks', 10)
        walk_length = kwargs.get('walk_length', 40)

        for _ in range(num_walks):
            for node in G.nodes():
                walk = random_walk(G, node, walk_length)
                walks.append([str(n) for n in walk])

        # Train Word2Vec model
        model = Word2Vec(
            walks,
            vector_size=embedding_dim,
            window=kwargs.get('window', 5),
            min_count=1,
            sg=1,  # Skip-gram
            workers=kwargs.get('workers', 4)
        )

        # Extract embeddings
        embeddings = []
        node_ids = []

        for ont_class, ont_idx in class_mapping.items():
            node_id = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            node_ids.append(node_id)

            if str(ont_idx) in model.wv:
                embedding = model.wv[str(ont_idx)]
            else:
                # Fallback to random vector
                embedding = np.random.normal(0, 0.1, embedding_dim)

            embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)

        # Create metadata
        metadata = create_embedding_metadata(
            ontology_file=ontology_file,
            model_type='deepwalk_baseline',
            embedding_dim=embedding_dim,
            method_info={
                'baseline': 'deepwalk',
                'num_walks': num_walks,
                'walk_length': walk_length
            }
        )

        # Save to temporary file
        output_file = tempfile.mktemp(suffix='_deepwalk_baseline.parquet')
        save_embeddings_to_parquet(
            embeddings, node_ids, output_file, metadata
        )

        logger.info(f"Created DeepWalk baseline embeddings: {output_file}")
        return output_file


# Convenience functions
def setup_benchmark_datasets(cache_dir: Optional[str] = None) -> OntologyBenchmarkDatasets:
    """
    Set up benchmark dataset manager.

    Args:
        cache_dir (str, optional): Directory to cache datasets

    Returns:
        OntologyBenchmarkDatasets: Configured dataset manager
    """
    return OntologyBenchmarkDatasets(cache_dir)


def compare_with_baselines(embeddings_file: str,
                          ontology_file: str,
                          baseline_methods: List[str] = ['random', 'structural'],
                          embedding_dim: Optional[int] = None,
                          output_dir: str = 'baseline_comparison') -> Dict[str, Any]:
    """
    Compare embeddings against baseline methods.

    Args:
        embeddings_file (str): Path to on2vec embeddings
        ontology_file (str): Path to ontology file
        baseline_methods (list): List of baseline methods to compare against
        embedding_dim (int, optional): Embedding dimension (inferred if not provided)
        output_dir (str): Directory to save comparison results

    Returns:
        dict: Comparison results
    """
    from .evaluation import EmbeddingEvaluator
    from .io import load_embeddings_as_dataframe
    import json

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get embedding dimension if not provided
    if embedding_dim is None:
        df, _ = load_embeddings_as_dataframe(embeddings_file)
        embedding_dim = len(df['embedding'].iloc[0])

    # Create baseline comparison framework
    baseline_comp = BaselineComparison()

    # Evaluate original embeddings
    logger.info("Evaluating original embeddings")
    original_evaluator = EmbeddingEvaluator(embeddings_file, ontology_file)
    original_results = original_evaluator.create_evaluation_report()

    # Create and evaluate baselines
    baseline_results = {}

    for method in baseline_methods:
        logger.info(f"Creating and evaluating {method} baseline")

        try:
            # Create baseline embeddings
            baseline_file = baseline_comp.create_baseline_embeddings(
                method, ontology_file, embedding_dim
            )

            # Evaluate baseline
            baseline_evaluator = EmbeddingEvaluator(baseline_file, ontology_file)
            baseline_evaluation = baseline_evaluator.create_evaluation_report()

            baseline_results[method] = {
                'embeddings_file': baseline_file,
                'evaluation': baseline_evaluation
            }

        except Exception as e:
            logger.error(f"Failed to create/evaluate {method} baseline: {e}")
            baseline_results[method] = {'error': str(e)}

    # Compile comparison results
    comparison_results = {
        'original': {
            'embeddings_file': embeddings_file,
            'evaluation': original_results
        },
        'baselines': baseline_results
    }

    # Save results
    results_file = Path(output_dir) / 'baseline_comparison.json'

    def convert_numpy(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    comparison_serializable = convert_numpy(comparison_results)

    with open(results_file, 'w') as f:
        json.dump(comparison_serializable, f, indent=2)

    logger.info(f"Baseline comparison results saved to {output_dir}")
    return comparison_results