"""
Neural network models for ontology embedding
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


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