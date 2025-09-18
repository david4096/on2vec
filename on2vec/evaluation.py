"""
Comprehensive evaluation framework for ontology embeddings
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

from .io import load_embeddings_as_dataframe
from .ontology import build_graph_from_owl, build_multi_relation_graph_from_owl

logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    """
    Comprehensive evaluation framework for ontology embeddings.

    Supports both intrinsic and extrinsic evaluation methods:
    - Intrinsic: clustering quality, visualization coherence, structural preservation
    - Extrinsic: downstream classification, link prediction, semantic similarity
    """

    def __init__(self, embeddings_file: str, ontology_file: Optional[str] = None):
        """
        Initialize evaluator with embedding data.

        Args:
            embeddings_file (str): Path to embeddings parquet file
            ontology_file (str, optional): Path to original OWL ontology file
        """
        self.embeddings_file = embeddings_file
        self.ontology_file = ontology_file

        # Load embeddings
        self.df, self.metadata = load_embeddings_as_dataframe(
            embeddings_file, return_metadata=True
        )

        # Extract embeddings matrix
        self.embeddings = np.stack(self.df['embedding'].to_numpy())
        self.node_ids = self.df['node_id'].to_numpy()

        # Create index mapping
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}

        # Load ontology structure if provided
        self.ontology_graph = None
        self.multi_relation_graph = None
        if ontology_file:
            try:
                from owlready2 import get_ontology
                self.ontology = get_ontology(ontology_file).load()

                # Build graph structures for evaluation
                x, edge_index, class_mapping = build_graph_from_owl(ontology_file)
                self.ontology_graph = {
                    'edge_index': edge_index,
                    'class_mapping': class_mapping
                }

                # Try multi-relation graph
                try:
                    multi_data = build_multi_relation_graph_from_owl(ontology_file)
                    self.multi_relation_graph = multi_data
                except:
                    logger.warning("Could not build multi-relation graph")

            except Exception as e:
                logger.warning(f"Could not load ontology structure: {e}")
                self.ontology = None

        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        logger.info(f"Metadata keys: {list(self.metadata.keys()) if self.metadata else 'None'}")

    def evaluate_intrinsic(self,
                          clustering_methods: List[str] = ['kmeans', 'dbscan', 'hierarchical'],
                          n_clusters_range: List[int] = [5, 10, 15, 20],
                          random_state: int = 42) -> Dict[str, Any]:
        """
        Perform intrinsic evaluation of embeddings.

        Args:
            clustering_methods (list): Clustering algorithms to evaluate
            n_clusters_range (list): Range of cluster numbers to test
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Dictionary of intrinsic evaluation results
        """
        logger.info("Starting intrinsic evaluation")
        results = {}

        # Adjust cluster range based on number of samples
        max_clusters = max(2, min(self.embeddings.shape[0] - 1, max(n_clusters_range)))
        adjusted_n_clusters_range = [k for k in n_clusters_range if k < self.embeddings.shape[0]]

        if not adjusted_n_clusters_range:
            # If no clusters from the range are valid, create a reasonable range
            max_reasonable_clusters = min(max_clusters, 10)
            adjusted_n_clusters_range = list(range(2, max_reasonable_clusters + 1))

        logger.info(f"Original cluster range: {n_clusters_range}")
        logger.info(f"Adjusted cluster range for {self.embeddings.shape[0]} samples: {adjusted_n_clusters_range}")

        # 1. Clustering Quality Metrics
        results['clustering'] = self._evaluate_clustering(
            clustering_methods, adjusted_n_clusters_range, random_state
        )

        # 2. Embedding Distribution Analysis
        results['distribution'] = self._analyze_embedding_distribution()

        # 3. Dimensionality and Structure
        results['dimensionality'] = self._analyze_dimensionality()

        # 4. Neighborhood Preservation (if ontology available)
        if self.ontology_graph:
            results['neighborhood'] = self._evaluate_neighborhood_preservation()

        # 5. Distance Distribution Analysis
        results['distances'] = self._analyze_distance_distribution()

        logger.info("Completed intrinsic evaluation")
        return results

    def evaluate_extrinsic(self,
                          classification_tasks: Optional[List[str]] = None,
                          link_prediction: bool = True,
                          similarity_tasks: Optional[List[str]] = None,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Any]:
        """
        Perform extrinsic evaluation of embeddings.

        Args:
            classification_tasks (list, optional): Classification tasks to evaluate
            link_prediction (bool): Whether to perform link prediction evaluation
            similarity_tasks (list, optional): Similarity tasks to evaluate
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Dictionary of extrinsic evaluation results
        """
        logger.info("Starting extrinsic evaluation")
        results = {}

        # 1. Link Prediction (if ontology structure available)
        if link_prediction and self.ontology_graph:
            results['link_prediction'] = self._evaluate_link_prediction(test_size, random_state)

        # 2. Classification Tasks
        if classification_tasks:
            results['classification'] = self._evaluate_classification_tasks(
                classification_tasks, test_size, random_state
            )

        # 3. Semantic Similarity Tasks
        if similarity_tasks:
            results['similarity'] = self._evaluate_similarity_tasks(similarity_tasks)

        # 4. Hierarchy Preservation (if ontology available)
        if self.ontology_graph:
            results['hierarchy'] = self._evaluate_hierarchy_preservation()

        logger.info("Completed extrinsic evaluation")
        return results

    def evaluate_ontology_specific(self) -> Dict[str, Any]:
        """
        Perform ontology-specific evaluation metrics.

        Returns:
            dict: Dictionary of ontology-specific evaluation results
        """
        logger.info("Starting ontology-specific evaluation")
        results = {}

        if not self.ontology_graph:
            logger.warning("No ontology structure available for ontology-specific evaluation")
            return results

        # 1. Hierarchy Preservation Metrics
        results['hierarchy_preservation'] = self._evaluate_hierarchy_preservation()

        # 2. Semantic Coherence Analysis
        results['semantic_coherence'] = self._evaluate_semantic_coherence()

        # 3. Structural Consistency
        results['structural_consistency'] = self._evaluate_structural_consistency()

        # 4. Multi-relation Analysis (if available)
        if self.multi_relation_graph:
            results['multi_relation'] = self._evaluate_multi_relation_preservation()

        logger.info("Completed ontology-specific evaluation")
        return results

    def _evaluate_clustering(self, methods: List[str], n_clusters_range: List[int],
                           random_state: int) -> Dict[str, Any]:
        """Evaluate clustering quality of embeddings."""
        clustering_results = {}

        # Safety check: ensure we have enough samples for clustering
        if self.embeddings.shape[0] < 2:
            logger.warning("Not enough samples for clustering evaluation")
            return {'error': 'Insufficient samples for clustering (need at least 2)'}

        # Filter out invalid cluster numbers
        valid_n_clusters = [k for k in n_clusters_range if 2 <= k < self.embeddings.shape[0]]
        if not valid_n_clusters:
            logger.warning(f"No valid cluster numbers for {self.embeddings.shape[0]} samples")
            return {'error': f'No valid cluster numbers for {self.embeddings.shape[0]} samples'}

        for method in methods:
            method_results = {}

            if method == 'kmeans':
                for n_clusters in valid_n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    labels = kmeans.fit_predict(self.embeddings)

                    # Calculate metrics
                    silhouette = silhouette_score(self.embeddings, labels)
                    inertia = kmeans.inertia_

                    method_results[f'n_clusters_{n_clusters}'] = {
                        'silhouette_score': silhouette,
                        'inertia': inertia,
                        'labels': labels
                    }

            elif method == 'dbscan':
                # Try different eps values
                eps_values = np.arange(0.1, 2.0, 0.2)
                for eps in eps_values:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    labels = dbscan.fit_predict(self.embeddings)

                    if len(set(labels)) > 1:  # Valid clustering
                        silhouette = silhouette_score(self.embeddings, labels)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)

                        method_results[f'eps_{eps:.1f}'] = {
                            'silhouette_score': silhouette,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'labels': labels
                        }

            elif method == 'hierarchical':
                for n_clusters in valid_n_clusters:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = hierarchical.fit_predict(self.embeddings)

                    silhouette = silhouette_score(self.embeddings, labels)

                    method_results[f'n_clusters_{n_clusters}'] = {
                        'silhouette_score': silhouette,
                        'labels': labels
                    }

            clustering_results[method] = method_results

        return clustering_results

    def _analyze_embedding_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of embeddings."""
        # Calculate basic statistics
        means = np.mean(self.embeddings, axis=0)
        stds = np.std(self.embeddings, axis=0)

        # Calculate norms
        norms = np.linalg.norm(self.embeddings, axis=1)

        # Pairwise similarities
        dot_products = np.dot(self.embeddings, self.embeddings.T)
        similarities = dot_products / (norms[:, None] * norms[None, :])

        return {
            'mean_values': {
                'overall_mean': np.mean(means),
                'std_of_means': np.std(means),
                'dimension_means': means.tolist()[:10]  # First 10 dimensions
            },
            'std_values': {
                'overall_std': np.mean(stds),
                'std_of_stds': np.std(stds),
                'dimension_stds': stds.tolist()[:10]  # First 10 dimensions
            },
            'norms': {
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'min_norm': np.min(norms),
                'max_norm': np.max(norms)
            },
            'similarities': {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities)
            }
        }

    def _analyze_dimensionality(self) -> Dict[str, Any]:
        """Analyze embedding dimensionality and effective dimensions."""
        # PCA analysis
        from sklearn.decomposition import PCA

        pca = PCA()
        pca.fit(self.embeddings)

        # Find number of components for different variance levels
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_80 = np.argmax(cumsum >= 0.8) + 1
        n_components_90 = np.argmax(cumsum >= 0.9) + 1
        n_components_95 = np.argmax(cumsum >= 0.95) + 1

        return {
            'original_dimensions': self.embeddings.shape[1],
            'effective_dimensions': {
                '80_percent_variance': n_components_80,
                '90_percent_variance': n_components_90,
                '95_percent_variance': n_components_95
            },
            'explained_variance_ratio': pca.explained_variance_ratio_[:20].tolist(),  # First 20
            'cumulative_variance_ratio': cumsum[:20].tolist()  # First 20
        }

    def _analyze_distance_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of pairwise distances."""
        # Sample subset for efficiency if too large
        n_samples = min(1000, len(self.embeddings))
        indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
        sample_embeddings = self.embeddings[indices]

        # Calculate pairwise distances
        distances = pdist(sample_embeddings, metric='euclidean')
        cosine_distances = pdist(sample_embeddings, metric='cosine')

        return {
            'euclidean_distances': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'percentiles': np.percentile(distances, [10, 25, 50, 75, 90]).tolist()
            },
            'cosine_distances': {
                'mean': np.mean(cosine_distances),
                'std': np.std(cosine_distances),
                'min': np.min(cosine_distances),
                'max': np.max(cosine_distances),
                'percentiles': np.percentile(cosine_distances, [10, 25, 50, 75, 90]).tolist()
            },
            'sample_size': n_samples
        }

    def _evaluate_neighborhood_preservation(self) -> Dict[str, Any]:
        """Evaluate how well local neighborhoods are preserved."""
        if not self.ontology_graph:
            return {}

        # Build adjacency matrix from ontology graph
        edge_index = self.ontology_graph['edge_index']
        class_mapping = self.ontology_graph['class_mapping']

        # Create adjacency matrix
        n_nodes = len(class_mapping)
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_matrix[src, dst] = 1

        # Make symmetric (undirected)
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix = (adj_matrix > 0).astype(int)

        # Find embedding indices for ontology nodes
        embedding_indices = []
        ontology_indices = []

        for ont_class, ont_idx in class_mapping.items():
            class_iri = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
            if class_iri in self.node_to_idx:
                embedding_indices.append(self.node_to_idx[class_iri])
                ontology_indices.append(ont_idx)

        if len(embedding_indices) < 10:  # Need enough overlap
            return {'error': 'Insufficient overlap between ontology and embeddings'}

        # Get corresponding embeddings
        subset_embeddings = self.embeddings[embedding_indices]
        subset_adj = adj_matrix[np.ix_(ontology_indices, ontology_indices)]

        # Calculate embedding similarities
        similarities = np.dot(subset_embeddings, subset_embeddings.T)
        norms = np.linalg.norm(subset_embeddings, axis=1)
        similarities = similarities / (norms[:, None] * norms[None, :])

        # Evaluate neighborhood preservation
        k_values = [1, 3, 5, 10]
        preservation_scores = {}

        for k in k_values:
            if k >= len(embedding_indices):
                continue

            # Find k-nearest neighbors in embedding space
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            nbrs.fit(subset_embeddings)
            _, embedding_neighbors = nbrs.kneighbors(subset_embeddings)

            # Find k-nearest neighbors in ontology space
            ontology_neighbors = []
            for i in range(len(ontology_indices)):
                neighbors = np.argsort(-subset_adj[i])[1:k+1]  # Exclude self
                ontology_neighbors.append(neighbors)

            # Calculate preservation score
            overlaps = []
            for i in range(len(embedding_indices)):
                emb_neighs = set(embedding_neighbors[i][1:])  # Exclude self
                ont_neighs = set(ontology_neighbors[i])
                if len(ont_neighs) > 0:
                    overlap = len(emb_neighs.intersection(ont_neighs)) / len(ont_neighs)
                    overlaps.append(overlap)

            if overlaps:
                preservation_scores[f'k_{k}'] = {
                    'mean_preservation': np.mean(overlaps),
                    'std_preservation': np.std(overlaps),
                    'min_preservation': np.min(overlaps),
                    'max_preservation': np.max(overlaps)
                }

        return {
            'preservation_scores': preservation_scores,
            'n_evaluated_nodes': len(embedding_indices),
            'n_total_ontology_nodes': len(class_mapping),
            'coverage_ratio': len(embedding_indices) / len(class_mapping)
        }

    def _evaluate_link_prediction(self, test_size: float, random_state: int) -> Dict[str, Any]:
        """Evaluate embeddings on link prediction task."""
        if not self.ontology_graph:
            return {}

        edge_index = self.ontology_graph['edge_index']
        class_mapping = self.ontology_graph['class_mapping']

        # Create positive and negative edge samples
        edges = []
        edge_labels = []

        # Positive edges
        for i in range(edge_index.shape[1]):
            src_class = None
            dst_class = None

            # Find classes corresponding to indices
            for ont_class, idx in class_mapping.items():
                if idx == edge_index[0, i]:
                    src_class = ont_class
                if idx == edge_index[1, i]:
                    dst_class = ont_class

            if src_class and dst_class:
                src_iri = src_class.iri if hasattr(src_class, 'iri') else str(src_class)
                dst_iri = dst_class.iri if hasattr(dst_class, 'iri') else str(dst_class)

                if src_iri in self.node_to_idx and dst_iri in self.node_to_idx:
                    edges.append((self.node_to_idx[src_iri], self.node_to_idx[dst_iri]))
                    edge_labels.append(1)

        # Generate negative edges (random non-connected pairs)
        n_positive = len(edges)
        negative_edges = []

        existing_edges = set(edges)
        node_indices = list(range(len(self.embeddings)))

        attempts = 0
        while len(negative_edges) < n_positive and attempts < n_positive * 10:
            src = np.random.choice(node_indices)
            dst = np.random.choice(node_indices)

            if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
                negative_edges.append((src, dst))
                edge_labels.append(0)

            attempts += 1

        if len(negative_edges) == 0:
            return {'error': 'Could not generate negative edges'}

        # Combine positive and negative edges
        all_edges = edges + negative_edges

        # Create features: concatenate and element-wise product of embeddings
        edge_features = []
        for src, dst in all_edges:
            src_emb = self.embeddings[src]
            dst_emb = self.embeddings[dst]

            # Concatenate embeddings
            concat_features = np.concatenate([src_emb, dst_emb])

            # Element-wise product (Hadamard product)
            hadamard_features = src_emb * dst_emb

            # Combine both representations
            combined_features = np.concatenate([concat_features, hadamard_features])
            edge_features.append(combined_features)

        edge_features = np.array(edge_features)
        edge_labels = np.array(edge_labels)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            edge_features, edge_labels, test_size=test_size, random_state=random_state, stratify=edge_labels
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifiers
        classifiers = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'mlp': MLPClassifier(random_state=random_state, max_iter=500, early_stopping=True)
        }

        results = {}
        for clf_name, clf in classifiers.items():
            try:
                # Train classifier
                clf.fit(X_train_scaled, y_train)

                # Predictions
                y_pred = clf.predict(X_test_scaled)
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

                # Metrics
                results[clf_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'average_precision': average_precision_score(y_test, y_pred_proba)
                }
            except Exception as e:
                logger.warning(f"Failed to train {clf_name}: {e}")
                results[clf_name] = {'error': str(e)}

        return {
            'classifiers': results,
            'n_positive_edges': n_positive,
            'n_negative_edges': len(negative_edges),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

    def _evaluate_hierarchy_preservation(self) -> Dict[str, Any]:
        """Evaluate how well the embedding space preserves ontological hierarchy."""
        if not self.ontology_graph:
            return {}

        edge_index = self.ontology_graph['edge_index']
        class_mapping = self.ontology_graph['class_mapping']

        # Build hierarchy relationships
        hierarchy_pairs = []

        for i in range(edge_index.shape[1]):
            src_class = None
            dst_class = None

            # Find classes corresponding to indices
            for ont_class, idx in class_mapping.items():
                if idx == edge_index[0, i]:
                    src_class = ont_class
                if idx == edge_index[1, i]:
                    dst_class = ont_class

            if src_class and dst_class:
                src_iri = src_class.iri if hasattr(src_class, 'iri') else str(src_class)
                dst_iri = dst_class.iri if hasattr(dst_class, 'iri') else str(dst_class)

                if src_iri in self.node_to_idx and dst_iri in self.node_to_idx:
                    src_idx = self.node_to_idx[src_iri]
                    dst_idx = self.node_to_idx[dst_iri]
                    hierarchy_pairs.append((src_idx, dst_idx))

        if len(hierarchy_pairs) < 10:
            return {'error': 'Insufficient hierarchy relationships found'}

        # Calculate embedding similarities for hierarchy pairs
        hierarchy_similarities = []
        for src_idx, dst_idx in hierarchy_pairs:
            src_emb = self.embeddings[src_idx]
            dst_emb = self.embeddings[dst_idx]

            # Cosine similarity
            similarity = np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb))
            hierarchy_similarities.append(similarity)

        # Generate random pairs for comparison
        n_random = len(hierarchy_pairs)
        random_similarities = []
        node_indices = list(range(len(self.embeddings)))

        for _ in range(n_random):
            src_idx = np.random.choice(node_indices)
            dst_idx = np.random.choice(node_indices)

            if src_idx != dst_idx:
                src_emb = self.embeddings[src_idx]
                dst_emb = self.embeddings[dst_idx]

                similarity = np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb))
                random_similarities.append(similarity)

        # Statistical comparison
        from scipy.stats import mannwhitneyu

        hierarchy_similarities = np.array(hierarchy_similarities)
        random_similarities = np.array(random_similarities)

        try:
            statistic, p_value = mannwhitneyu(hierarchy_similarities, random_similarities, alternative='greater')
            significant = p_value < 0.05
        except:
            statistic, p_value, significant = None, None, None

        return {
            'hierarchy_similarity_stats': {
                'mean': np.mean(hierarchy_similarities),
                'std': np.std(hierarchy_similarities),
                'median': np.median(hierarchy_similarities),
                'min': np.min(hierarchy_similarities),
                'max': np.max(hierarchy_similarities)
            },
            'random_similarity_stats': {
                'mean': np.mean(random_similarities),
                'std': np.std(random_similarities),
                'median': np.median(random_similarities),
                'min': np.min(random_similarities),
                'max': np.max(random_similarities)
            },
            'statistical_test': {
                'statistic': statistic,
                'p_value': p_value,
                'significantly_higher': significant
            },
            'n_hierarchy_pairs': len(hierarchy_pairs),
            'similarity_difference': np.mean(hierarchy_similarities) - np.mean(random_similarities)
        }

    def _evaluate_semantic_coherence(self) -> Dict[str, Any]:
        """Evaluate semantic coherence of embeddings."""
        # This is a placeholder for semantic coherence evaluation
        # In a full implementation, you might:
        # 1. Use text features to evaluate semantic similarity
        # 2. Compare embedding clusters with known semantic groups
        # 3. Evaluate consistency with external knowledge bases

        results = {
            'placeholder': True,
            'note': 'Semantic coherence evaluation requires domain-specific semantic resources'
        }

        # If text features are available in metadata, we can do some basic analysis
        if self.metadata and 'text_features' in str(self.metadata):
            results['text_features_available'] = True
        else:
            results['text_features_available'] = False

        return results

    def _evaluate_structural_consistency(self) -> Dict[str, Any]:
        """Evaluate structural consistency of embeddings with ontology structure."""
        if not self.ontology_graph:
            return {}

        # Calculate graph-theoretic properties and embedding properties
        edge_index = self.ontology_graph['edge_index']

        # Build networkx graph
        import networkx as nx
        G = nx.Graph()

        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
            closeness_centrality = nx.closeness_centrality(G)
        except:
            return {'error': 'Could not calculate graph centrality measures'}

        # Map to embedding space
        centrality_embedding_corr = {}

        for centrality_name, centrality_dict in [
            ('degree', degree_centrality),
            ('betweenness', betweenness_centrality),
            ('closeness', closeness_centrality)
        ]:
            # Get centrality values for nodes that have embeddings
            centrality_values = []
            embedding_norms = []

            class_mapping = self.ontology_graph['class_mapping']
            for ont_class, ont_idx in class_mapping.items():
                if ont_idx in centrality_dict:
                    class_iri = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)
                    if class_iri in self.node_to_idx:
                        centrality_values.append(centrality_dict[ont_idx])
                        emb_idx = self.node_to_idx[class_iri]
                        embedding_norm = np.linalg.norm(self.embeddings[emb_idx])
                        embedding_norms.append(embedding_norm)

            if len(centrality_values) >= 10:
                # Calculate correlation between centrality and embedding norm
                try:
                    corr_pearson, p_pearson = pearsonr(centrality_values, embedding_norms)
                    corr_spearman, p_spearman = spearmanr(centrality_values, embedding_norms)

                    centrality_embedding_corr[centrality_name] = {
                        'pearson_correlation': corr_pearson,
                        'pearson_p_value': p_pearson,
                        'spearman_correlation': corr_spearman,
                        'spearman_p_value': p_spearman,
                        'n_samples': len(centrality_values)
                    }
                except:
                    centrality_embedding_corr[centrality_name] = {'error': 'Could not calculate correlation'}

        return {
            'centrality_correlations': centrality_embedding_corr,
            'graph_properties': {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G),
                'n_connected_components': nx.number_connected_components(G)
            }
        }

    def _evaluate_multi_relation_preservation(self) -> Dict[str, Any]:
        """Evaluate how well multi-relation structure is preserved."""
        if not self.multi_relation_graph:
            return {}

        edge_index = self.multi_relation_graph['edge_index']
        edge_types = self.multi_relation_graph['edge_types']
        relation_names = self.multi_relation_graph['relation_names']

        # Analyze embedding similarities by relation type
        relation_similarities = {}

        for rel_idx, rel_name in enumerate(relation_names):
            # Find edges of this relation type
            rel_edges = edge_index[:, edge_types == rel_idx]

            if rel_edges.shape[1] == 0:
                continue

            # Calculate similarities for edges of this type
            similarities = []
            class_mapping = self.multi_relation_graph.get('class_mapping', {})

            for i in range(rel_edges.shape[1]):
                src_ont_idx = rel_edges[0, i].item()
                dst_ont_idx = rel_edges[1, i].item()

                # Find corresponding embedding indices
                src_emb_idx = None
                dst_emb_idx = None

                for ont_class, ont_idx in class_mapping.items():
                    class_iri = ont_class.iri if hasattr(ont_class, 'iri') else str(ont_class)

                    if ont_idx == src_ont_idx and class_iri in self.node_to_idx:
                        src_emb_idx = self.node_to_idx[class_iri]
                    elif ont_idx == dst_ont_idx and class_iri in self.node_to_idx:
                        dst_emb_idx = self.node_to_idx[class_iri]

                if src_emb_idx is not None and dst_emb_idx is not None:
                    src_emb = self.embeddings[src_emb_idx]
                    dst_emb = self.embeddings[dst_emb_idx]

                    similarity = np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb))
                    similarities.append(similarity)

            if similarities:
                relation_similarities[rel_name] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'n_edges': len(similarities),
                    'similarities': similarities[:100]  # Store first 100 for analysis
                }

        return {
            'relation_similarities': relation_similarities,
            'n_relation_types': len(relation_names),
            'relation_names': relation_names,
            'total_edges': edge_index.shape[1]
        }

    def _evaluate_classification_tasks(self, tasks: List[str], test_size: float,
                                     random_state: int) -> Dict[str, Any]:
        """Evaluate embeddings on classification tasks."""
        # This is a placeholder for custom classification tasks
        # In practice, you would load specific datasets or create classification
        # tasks based on ontology properties

        return {
            'placeholder': True,
            'note': 'Classification task evaluation requires task-specific datasets'
        }

    def _evaluate_similarity_tasks(self, tasks: List[str]) -> Dict[str, Any]:
        """Evaluate embeddings on similarity tasks."""
        # This is a placeholder for similarity evaluation tasks
        # In practice, you would use gold-standard similarity datasets

        return {
            'placeholder': True,
            'note': 'Similarity task evaluation requires gold-standard similarity datasets'
        }

    def create_evaluation_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.

        Args:
            save_path (str, optional): Path to save the report

        Returns:
            dict: Complete evaluation report
        """
        logger.info("Creating comprehensive evaluation report")

        report = {
            'metadata': {
                'embeddings_file': self.embeddings_file,
                'ontology_file': self.ontology_file,
                'embedding_shape': self.embeddings.shape,
                'embedding_metadata': self.metadata
            },
            'intrinsic_evaluation': self.evaluate_intrinsic(),
            'extrinsic_evaluation': self.evaluate_extrinsic(),
            'ontology_specific_evaluation': self.evaluate_ontology_specific()
        }

        # Save report if path provided
        if save_path:
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            report_serializable = convert_numpy(report)

            with open(save_path, 'w') as f:
                json.dump(report_serializable, f, indent=2)

            logger.info(f"Evaluation report saved to {save_path}")

        return report

    def visualize_evaluation_results(self, results: Dict[str, Any],
                                   save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations of evaluation results.

        Args:
            results (dict): Evaluation results from create_evaluation_report()
            save_dir (str, optional): Directory to save visualizations

        Returns:
            dict: Mapping of visualization names to file paths
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        plots = {}

        # 1. Embedding distribution plot
        plt.figure(figsize=(12, 8))

        # Plot embedding norms
        norms = np.linalg.norm(self.embeddings, axis=1)
        plt.subplot(2, 2, 1)
        plt.hist(norms, bins=50, alpha=0.7)
        plt.title('Distribution of Embedding Norms')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')

        # Plot dimension means
        plt.subplot(2, 2, 2)
        dim_means = np.mean(self.embeddings, axis=0)
        plt.plot(dim_means)
        plt.title('Mean Values by Dimension')
        plt.xlabel('Dimension')
        plt.ylabel('Mean Value')

        # Plot dimension stds
        plt.subplot(2, 2, 3)
        dim_stds = np.std(self.embeddings, axis=0)
        plt.plot(dim_stds)
        plt.title('Standard Deviation by Dimension')
        plt.xlabel('Dimension')
        plt.ylabel('Standard Deviation')

        # Plot PCA explained variance
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, self.embeddings.shape[1]))
        pca.fit(self.embeddings)

        plt.subplot(2, 2, 4)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Cumulative Explained Variance (PCA)')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')

        plt.tight_layout()

        if save_dir:
            plot_path = Path(save_dir) / 'embedding_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots['embedding_distribution'] = str(plot_path)
        else:
            plt.show()

        plt.close()

        # 2. Clustering evaluation plot (if available)
        if 'intrinsic_evaluation' in results and 'clustering' in results['intrinsic_evaluation']:
            clustering_results = results['intrinsic_evaluation']['clustering']

            plt.figure(figsize=(15, 5))

            # K-means results
            if 'kmeans' in clustering_results:
                kmeans_results = clustering_results['kmeans']
                n_clusters = []
                silhouette_scores = []

                for key, values in kmeans_results.items():
                    if key.startswith('n_clusters_'):
                        n_clusters.append(int(key.split('_')[2]))
                        silhouette_scores.append(values['silhouette_score'])

                if n_clusters:
                    plt.subplot(1, 3, 1)
                    plt.plot(n_clusters, silhouette_scores, 'o-')
                    plt.title('K-means Silhouette Scores')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('Silhouette Score')
                    plt.grid(True)

            # DBSCAN results
            if 'dbscan' in clustering_results:
                dbscan_results = clustering_results['dbscan']
                eps_values = []
                silhouette_scores = []

                for key, values in dbscan_results.items():
                    if key.startswith('eps_'):
                        eps_values.append(float(key.split('_')[1]))
                        silhouette_scores.append(values['silhouette_score'])

                if eps_values:
                    plt.subplot(1, 3, 2)
                    plt.plot(eps_values, silhouette_scores, 'o-')
                    plt.title('DBSCAN Silhouette Scores')
                    plt.xlabel('Epsilon')
                    plt.ylabel('Silhouette Score')
                    plt.grid(True)

            # Hierarchical results
            if 'hierarchical' in clustering_results:
                hierarchical_results = clustering_results['hierarchical']
                n_clusters = []
                silhouette_scores = []

                for key, values in hierarchical_results.items():
                    if key.startswith('n_clusters_'):
                        n_clusters.append(int(key.split('_')[2]))
                        silhouette_scores.append(values['silhouette_score'])

                if n_clusters:
                    plt.subplot(1, 3, 3)
                    plt.plot(n_clusters, silhouette_scores, 'o-')
                    plt.title('Hierarchical Silhouette Scores')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('Silhouette Score')
                    plt.grid(True)

            plt.tight_layout()

            if save_dir:
                plot_path = Path(save_dir) / 'clustering_evaluation.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots['clustering_evaluation'] = str(plot_path)
            else:
                plt.show()

            plt.close()

        return plots


# Convenience functions for easy access
def evaluate_embeddings(embeddings_file: str, ontology_file: Optional[str] = None,
                       save_report: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to perform comprehensive evaluation of embeddings.

    Args:
        embeddings_file (str): Path to embeddings parquet file
        ontology_file (str, optional): Path to original OWL ontology file
        save_report (str, optional): Path to save evaluation report

    Returns:
        dict: Complete evaluation report
    """
    evaluator = EmbeddingEvaluator(embeddings_file, ontology_file)
    return evaluator.create_evaluation_report(save_report)


def create_evaluation_benchmark(embeddings_files: List[str],
                              ontology_files: Optional[List[str]] = None,
                              output_dir: str = "evaluation_benchmark") -> Dict[str, Any]:
    """
    Create a benchmark comparison across multiple embedding files.

    Args:
        embeddings_files (list): List of embedding file paths
        ontology_files (list, optional): Corresponding ontology files
        output_dir (str): Directory to save benchmark results

    Returns:
        dict: Benchmark comparison results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if ontology_files is None:
        ontology_files = [None] * len(embeddings_files)

    benchmark_results = {}

    for i, (emb_file, ont_file) in enumerate(zip(embeddings_files, ontology_files)):
        logger.info(f"Evaluating {emb_file}")

        try:
            evaluator = EmbeddingEvaluator(emb_file, ont_file)

            # Create report
            report_path = Path(output_dir) / f"evaluation_report_{i:02d}.json"
            results = evaluator.create_evaluation_report(str(report_path))

            # Create visualizations
            viz_dir = Path(output_dir) / f"visualizations_{i:02d}"
            viz_paths = evaluator.visualize_evaluation_results(results, str(viz_dir))

            benchmark_results[emb_file] = {
                'results': results,
                'report_path': str(report_path),
                'visualizations': viz_paths
            }

        except Exception as e:
            logger.error(f"Failed to evaluate {emb_file}: {e}")
            benchmark_results[emb_file] = {'error': str(e)}

    # Save benchmark summary
    summary_path = Path(output_dir) / "benchmark_summary.json"

    # Extract key metrics for comparison
    comparison_metrics = {}
    for emb_file, data in benchmark_results.items():
        if 'error' not in data:
            results = data['results']
            comparison_metrics[emb_file] = {
                'embedding_shape': results['metadata']['embedding_shape'],
                # Add key metrics as they become available
            }

    with open(summary_path, 'w') as f:
        import json
        json.dump({
            'comparison_metrics': comparison_metrics,
            'full_results': {k: v for k, v in benchmark_results.items() if 'error' not in v}
        }, f, indent=2)

    logger.info(f"Benchmark results saved to {output_dir}")
    return benchmark_results