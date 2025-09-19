"""
Neural network models for ontology embedding
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
import logging

from .device_utils import get_device, move_to_device

logger = logging.getLogger(__name__)


class OntologyGNN(torch.nn.Module):
    """Graph Neural Network for learning ontology embeddings."""

    def __init__(self, input_dim, hidden_dim, out_dim, model_type='gcn', device=None):
        """
        Initialize the GNN model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            model_type (str): Type of GNN ('gcn' or 'gat')
            device (str or torch.device, optional): Device to place model on
        """
        super(OntologyGNN, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Set up device
        self.device = get_device(device, verbose=False)

        if model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, out_dim)
        else:
            raise ValueError("Unsupported model type. Use 'gcn' or 'gat'.")

        # Move model to device
        self.to(self.device)

    def forward(self, x, edge_index):
        """
        Forward pass of the GNN.

        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph edge indices

        Returns:
            torch.Tensor: Node embeddings
        """
        # Ensure inputs are on the same device as model
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class MultiRelationOntologyGNN(torch.nn.Module):
    """
    Multi-Relation Graph Neural Network for learning ontology embeddings.
    Uses Relational Graph Convolutional Networks (RGCN) to handle different edge types.
    """

    def __init__(self, input_dim, hidden_dim, out_dim, num_relations, model_type='rgcn',
                 num_bases=None, dropout=0.0, device=None):
        """
        Initialize the Multi-Relation GNN model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            num_relations (int): Number of different relation types
            model_type (str): Type of multi-relation GNN ('rgcn' or 'weighted_gcn')
            num_bases (int, optional): Number of bases for RGCN decomposition
            dropout (float): Dropout rate
            device (str or torch.device, optional): Device to place model on
        """
        super(MultiRelationOntologyGNN, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.dropout = dropout

        # Set up device
        self.device = get_device(device, verbose=False)

        logger.info(f"Initializing {model_type} model with {num_relations} relation types")

        if model_type == 'rgcn':
            # Use RGCN for handling multiple relation types
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)

        elif model_type == 'weighted_gcn':
            # Alternative: Use standard GCN with learnable relation weights
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

            # Learnable weights for different relation types
            self.relation_weights = torch.nn.Parameter(torch.ones(num_relations))

        else:
            raise ValueError("Unsupported model type. Use 'rgcn' or 'weighted_gcn'.")

        # Move model to device
        self.to(self.device)

    def forward(self, x, edge_index, edge_type=None):
        """
        Forward pass of the Multi-Relation GNN.

        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]

        Returns:
            torch.Tensor: Node embeddings [num_nodes, out_dim]
        """
        # Ensure inputs are on the same device as model
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_type is not None:
            edge_type = edge_type.to(self.device)

        if self.model_type == 'rgcn':
            # RGCN can handle edge types directly
            if edge_type is None:
                raise ValueError("RGCN requires edge_type to be specified")

            x = self.conv1(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_type)

        elif self.model_type == 'weighted_gcn':
            # Weighted GCN: Apply relation-specific weights to edges
            if edge_type is not None:
                # Create edge weights based on relation types
                edge_weights = self.relation_weights[edge_type]
            else:
                # If no edge types provided, use uniform weights
                edge_weights = None

            # Apply weighted GCN layers
            x = self.conv1(x, edge_index, edge_weights)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weights)

        return x

    def get_relation_weights(self):
        """
        Get the learned relation weights (for weighted_gcn model).

        Returns:
            torch.Tensor: Relation weights if using weighted_gcn, None otherwise
        """
        if self.model_type == 'weighted_gcn':
            return self.relation_weights.detach()
        return None


class HeterogeneousOntologyGNN(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network that can handle different types of relations
    with separate message passing for each relation type.
    """

    def __init__(self, input_dim, hidden_dim, out_dim, relation_types, dropout=0.0, device=None):
        """
        Initialize the Heterogeneous GNN model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            relation_types (list): List of relation type names/identifiers
            dropout (float): Dropout rate
            device (str or torch.device, optional): Device to place model on
        """
        super(HeterogeneousOntologyGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.relation_types = relation_types
        self.num_relations = len(relation_types)
        self.dropout = dropout

        # Set up device
        self.device = get_device(device, verbose=False)

        logger.info(f"Initializing heterogeneous model with {self.num_relations} relation types")

        # Separate GCN layers for each relation type
        self.relation_convs1 = torch.nn.ModuleDict()
        self.relation_convs2 = torch.nn.ModuleDict()

        for i, rel_type in enumerate(relation_types):
            # Use index-based keys to avoid issues with special characters in IRIs
            rel_key = f"rel_{i}"
            self.relation_convs1[rel_key] = GCNConv(input_dim, hidden_dim)
            self.relation_convs2[rel_key] = GCNConv(hidden_dim, out_dim)

        # Attention mechanism to combine different relation embeddings
        self.attention = torch.nn.Linear(out_dim, 1)

        # Move model to device
        self.to(self.device)

    def forward(self, x, edge_index, edge_type, relation_to_index):
        """
        Forward pass of the Heterogeneous GNN.

        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph edge indices
            edge_type (torch.Tensor): Edge type indices
            relation_to_index (dict): Mapping from relation names to indices

        Returns:
            torch.Tensor: Node embeddings
        """
        # Ensure inputs are on the same device as model
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)

        # Separate edges by relation type
        relation_embeddings = []
        index_to_relation = {v: k for k, v in relation_to_index.items()}

        for rel_idx in range(self.num_relations):
            if rel_idx in index_to_relation:
                rel_key = f"rel_{rel_idx}"  # Use index-based key

                # Get edges for this relation type
                rel_mask = (edge_type == rel_idx)
                if rel_mask.sum() > 0:
                    rel_edge_index = edge_index[:, rel_mask]

                    # Apply relation-specific convolutions
                    rel_x = self.relation_convs1[rel_key](x, rel_edge_index)
                    rel_x = F.relu(rel_x)
                    rel_x = F.dropout(rel_x, p=self.dropout, training=self.training)
                    rel_x = self.relation_convs2[rel_key](rel_x, rel_edge_index)

                    relation_embeddings.append(rel_x)

        if not relation_embeddings:
            # Fallback: return zeros if no relations found
            return torch.zeros(x.shape[0], self.out_dim, device=x.device)

        # Stack and combine relation-specific embeddings
        if len(relation_embeddings) == 1:
            return relation_embeddings[0]

        # Use attention to combine multiple relation embeddings
        stacked_embeddings = torch.stack(relation_embeddings, dim=0)  # [num_relations, num_nodes, out_dim]
        attention_weights = torch.softmax(
            self.attention(stacked_embeddings).squeeze(-1), dim=0
        )  # [num_relations, num_nodes]

        # Weighted combination
        combined = (stacked_embeddings * attention_weights.unsqueeze(-1)).sum(dim=0)

        return combined


class TextAugmentedOntologyGNN(torch.nn.Module):
    """
    Graph Neural Network that combines structural and textual features for ontology embedding.
    """

    def __init__(self, structural_dim, text_dim, hidden_dim, out_dim,
                 model_type='gcn', fusion_method='concat', dropout=0.0,
                 num_relations=None, relation_types=None, num_bases=None, device=None):
        """
        Initialize the Text-Augmented GNN model.

        Args:
            structural_dim (int): Dimension of structural features
            text_dim (int): Dimension of text features
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            model_type (str): Type of GNN ('gcn', 'gat', 'rgcn', 'weighted_gcn', 'heterogeneous')
            fusion_method (str): How to combine features ('concat', 'add', 'attention')
            dropout (float): Dropout rate
            num_relations (int, optional): Number of relations (for rgcn, weighted_gcn)
            relation_types (list, optional): List of relation types (for heterogeneous)
            num_bases (int, optional): Number of bases for RGCN decomposition
            device (str or torch.device, optional): Device to place model on
        """
        super(TextAugmentedOntologyGNN, self).__init__()
        self.structural_dim = structural_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.model_type = model_type
        self.fusion_method = fusion_method
        self.dropout = dropout
        self.num_relations = num_relations
        self.relation_types = relation_types

        # Feature fusion layer
        if fusion_method == 'concat':
            self.input_dim = structural_dim + text_dim
            self.fusion_layer = None
        elif fusion_method == 'add':
            if structural_dim != text_dim:
                # Project to same dimension for addition
                self.structural_proj = torch.nn.Linear(structural_dim, text_dim)
                self.input_dim = text_dim
            else:
                self.structural_proj = None
                self.input_dim = structural_dim
            self.fusion_layer = None
        elif fusion_method == 'attention':
            self.input_dim = hidden_dim
            self.fusion_layer = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.structural_proj = torch.nn.Linear(structural_dim, hidden_dim)
            self.text_proj = torch.nn.Linear(text_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # GNN layers - support all model types
        if model_type == 'gcn':
            self.conv1 = GCNConv(self.input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        elif model_type == 'gat':
            self.conv1 = GATConv(self.input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, out_dim)

        elif model_type == 'rgcn':
            if num_relations is None:
                raise ValueError("RGCN model requires num_relations to be specified")
            self.conv1 = RGCNConv(self.input_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)

        elif model_type == 'weighted_gcn':
            if num_relations is None:
                raise ValueError("Weighted GCN model requires num_relations to be specified")
            self.conv1 = GCNConv(self.input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
            # Learnable weights for different relation types
            self.relation_weights = torch.nn.Parameter(torch.ones(num_relations))

        elif model_type == 'heterogeneous':
            if relation_types is None:
                raise ValueError("Heterogeneous model requires relation_types to be specified")
            self.num_relations = len(relation_types)
            self.relation_types = relation_types
            # Separate GCN layers for each relation type
            self.relation_convs1 = torch.nn.ModuleDict()
            self.relation_convs2 = torch.nn.ModuleDict()

            for i, rel_type in enumerate(relation_types):
                # Use index-based key for PyTorch module names (they can't contain dots or special chars)
                rel_key = f"rel_{i}"
                self.relation_convs1[rel_key] = GCNConv(self.input_dim, hidden_dim)
                self.relation_convs2[rel_key] = GCNConv(hidden_dim, out_dim)

            # Aggregation layer to combine outputs from different relation types
            self.aggregation_weights = torch.nn.Parameter(torch.ones(self.num_relations))

        else:
            raise ValueError(
                f"Unsupported model type '{model_type}' for TextAugmentedOntologyGNN. "
                f"Supported types: ['gcn', 'gat', 'rgcn', 'weighted_gcn', 'heterogeneous']"
            )

        logger.info(f"Initialized TextAugmentedOntologyGNN: {fusion_method} fusion, {model_type} backbone")

        # Set up device and move model
        self.device = get_device(device, verbose=False)
        self.to(self.device)

    def forward(self, structural_x, text_x, edge_index, edge_type=None, relation_to_index=None):
        """
        Forward pass combining structural and textual features.

        Args:
            structural_x (torch.Tensor): Structural node features
            text_x (torch.Tensor): Text node features
            edge_index (torch.Tensor): Graph edge indices
            edge_type (torch.Tensor, optional): Edge type indices (for multi-relation models)
            relation_to_index (dict, optional): Mapping from relation types to indices

        Returns:
            torch.Tensor: Node embeddings
        """
        # Ensure inputs are on the same device as model
        structural_x = structural_x.to(self.device)
        text_x = text_x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_type is not None:
            edge_type = edge_type.to(self.device)

        # Fuse structural and text features
        if self.fusion_method == 'concat':
            x = torch.cat([structural_x, text_x], dim=1)
        elif self.fusion_method == 'add':
            if self.structural_proj is not None:
                structural_x = self.structural_proj(structural_x)
            x = structural_x + text_x
        elif self.fusion_method == 'attention':
            # Project both feature types to same dimension
            structural_proj = self.structural_proj(structural_x)
            text_proj = self.text_proj(text_x)

            # Stack for attention mechanism [batch_size, seq_len=2, hidden_dim]
            combined = torch.stack([structural_proj, text_proj], dim=1)

            # Apply self-attention
            attended, _ = self.fusion_layer(combined, combined, combined)
            # Average or sum the attended representations
            x = attended.mean(dim=1)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply GNN layers based on model type
        if self.model_type in ['gcn', 'gat']:
            # Standard GCN/GAT forward pass
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        elif self.model_type == 'rgcn':
            # RGCN requires edge_type
            if edge_type is None:
                raise ValueError("RGCN requires edge_type to be specified")
            x = self.conv1(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_type)

        elif self.model_type == 'weighted_gcn':
            # Weighted GCN applies relation weights to edge features
            if edge_type is None:
                raise ValueError("Weighted GCN requires edge_type to be specified")

            # Apply relation weights
            edge_weights = self.relation_weights[edge_type]

            # Use weighted edges
            x = self.conv1(x, edge_index, edge_weights)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weights)

        elif self.model_type == 'heterogeneous':
            # Heterogeneous model processes each relation type separately
            if edge_type is None or relation_to_index is None:
                raise ValueError("Heterogeneous model requires both edge_type and relation_to_index")

            # Group edges by relation type
            relation_outputs = []

            for rel_idx, rel_type in enumerate(self.relation_types):
                # Use index-based key for module access
                rel_key = f"rel_{rel_idx}"

                # Find edges of this relation type
                mask = edge_type == rel_idx
                if torch.sum(mask) == 0:
                    continue  # Skip if no edges of this type

                rel_edge_index = edge_index[:, mask]

                # Apply relation-specific convolutions
                rel_x = self.relation_convs1[rel_key](x, rel_edge_index)
                rel_x = F.relu(rel_x)
                rel_x = F.dropout(rel_x, p=self.dropout, training=self.training)
                rel_x = self.relation_convs2[rel_key](rel_x, rel_edge_index)

                relation_outputs.append(rel_x)

            if relation_outputs:
                # Combine outputs from different relation types
                stacked_outputs = torch.stack(relation_outputs, dim=0)  # [num_relations, num_nodes, out_dim]

                # Apply learnable aggregation weights
                aggregation_weights = F.softmax(self.aggregation_weights[:len(relation_outputs)], dim=0)
                x = (stacked_outputs * aggregation_weights.view(-1, 1, 1)).sum(dim=0)
            else:
                # Fallback if no relation edges found
                logger.warning("No relation edges found, using identity mapping")
                x = torch.zeros(x.size(0), self.out_dim, device=x.device)

        return x