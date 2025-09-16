"""
Neural network models for ontology embedding
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
import logging

logger = logging.getLogger(__name__)


class OntologyGNN(torch.nn.Module):
    """Graph Neural Network for learning ontology embeddings."""

    def __init__(self, input_dim, hidden_dim, out_dim, model_type='gcn'):
        """
        Initialize the GNN model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            model_type (str): Type of GNN ('gcn' or 'gat')
        """
        super(OntologyGNN, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, out_dim)
        else:
            raise ValueError("Unsupported model type. Use 'gcn' or 'gat'.")

    def forward(self, x, edge_index):
        """
        Forward pass of the GNN.

        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph edge indices

        Returns:
            torch.Tensor: Node embeddings
        """
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
                 num_bases=None, dropout=0.0):
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
        """
        super(MultiRelationOntologyGNN, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.dropout = dropout

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

    def __init__(self, input_dim, hidden_dim, out_dim, relation_types, dropout=0.0):
        """
        Initialize the Heterogeneous GNN model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output embedding dimension
            relation_types (list): List of relation type names/identifiers
            dropout (float): Dropout rate
        """
        super(HeterogeneousOntologyGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.relation_types = relation_types
        self.num_relations = len(relation_types)
        self.dropout = dropout

        logger.info(f"Initializing heterogeneous model with {self.num_relations} relation types")

        # Separate GCN layers for each relation type
        self.relation_convs1 = torch.nn.ModuleDict()
        self.relation_convs2 = torch.nn.ModuleDict()

        for rel_type in relation_types:
            rel_key = str(rel_type)  # Ensure string keys for ModuleDict
            self.relation_convs1[rel_key] = GCNConv(input_dim, hidden_dim)
            self.relation_convs2[rel_key] = GCNConv(hidden_dim, out_dim)

        # Attention mechanism to combine different relation embeddings
        self.attention = torch.nn.Linear(out_dim, 1)

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
        # Separate edges by relation type
        relation_embeddings = []
        index_to_relation = {v: k for k, v in relation_to_index.items()}

        for rel_idx in range(self.num_relations):
            if rel_idx in index_to_relation:
                rel_name = str(index_to_relation[rel_idx])

                # Get edges for this relation type
                rel_mask = (edge_type == rel_idx)
                if rel_mask.sum() > 0:
                    rel_edge_index = edge_index[:, rel_mask]

                    # Apply relation-specific convolutions
                    rel_x = self.relation_convs1[rel_name](x, rel_edge_index)
                    rel_x = F.relu(rel_x)
                    rel_x = F.dropout(rel_x, p=self.dropout, training=self.training)
                    rel_x = self.relation_convs2[rel_name](rel_x, rel_edge_index)

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