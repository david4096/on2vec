"""
Loss functions for training ontology embeddings
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def cross_entropy_loss(embeddings, edge_index, num_neg_samples=1000):
    """
    Binary cross-entropy loss for node embeddings.

    Optimized version that uses set-based edge checking for faster negative sampling.

    Args:
        embeddings (torch.Tensor): Node embeddings
        edge_index (torch.Tensor): Graph edge indices
        num_neg_samples (int): Number of negative samples

    Returns:
        torch.Tensor: Loss value
    """
    logger.debug("Calculating cross entropy loss...")
    num_nodes = embeddings.size(0)

    pos_src = edge_index[0]
    pos_dst = edge_index[1]

    # Create a set of existing edges for fast lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0][i].item(), edge_index[1][i].item()))

    # Generate negative samples more efficiently
    neg_src = []
    neg_dst = []
    attempts = 0
    max_attempts = num_neg_samples * 10  # Prevent infinite loops

    while len(neg_src) < num_neg_samples and attempts < max_attempts:
        # Generate batch of candidates
        batch_size = min(num_neg_samples * 2, num_neg_samples - len(neg_src))
        candidate_src = torch.randint(0, num_nodes, (batch_size,))
        candidate_dst = torch.randint(0, num_nodes, (batch_size,))

        # Filter out existing edges
        for i in range(batch_size):
            src_item = candidate_src[i].item()
            dst_item = candidate_dst[i].item()
            if (src_item, dst_item) not in edge_set and src_item != dst_item:
                neg_src.append(src_item)
                neg_dst.append(dst_item)
                if len(neg_src) >= num_neg_samples:
                    break

        attempts += batch_size

    # Convert to tensors
    neg_indices_src = torch.tensor(neg_src[:num_neg_samples], dtype=torch.long, device=embeddings.device)
    neg_indices_dst = torch.tensor(neg_dst[:num_neg_samples], dtype=torch.long, device=embeddings.device)

    if len(neg_indices_src) == 0:
        logger.warning("No valid negative samples found, using random samples")
        neg_indices_src = torch.randint(0, num_nodes, (min(100, num_neg_samples),), device=embeddings.device)
        neg_indices_dst = torch.randint(0, num_nodes, (min(100, num_neg_samples),), device=embeddings.device)

    # Prepare labels: 1 for positive pairs, 0 for negative pairs
    labels = torch.cat([torch.ones(pos_src.size(0), device=embeddings.device),
                       torch.zeros(neg_indices_src.size(0), device=embeddings.device)])

    # Calculate scores for positive and negative pairs
    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)
    neg_scores = (embeddings[neg_indices_src] * embeddings[neg_indices_dst]).sum(dim=1)

    # Concatenate the positive and negative scores
    scores = torch.cat([pos_scores, neg_scores])

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    return loss


def contrastive_loss(embeddings, edge_index, num_neg_samples=1000):
    """
    Contrastive loss for node embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings
        edge_index (torch.Tensor): Graph edge indices
        num_neg_samples (int): Number of negative samples

    Returns:
        torch.Tensor: Loss value
    """
    logger.debug("Calculating contrastive loss...")

    # Positive pairs (connected nodes) - minimize distance
    pos_loss = torch.norm(embeddings[edge_index[0]] - embeddings[edge_index[1]], dim=1).pow(2).sum()

    # Negative sampling
    num_nodes = embeddings.size(0)
    negative_samples_found = 0
    neg_indices_src = []
    neg_indices_dst = []

    while negative_samples_found < num_neg_samples:
        candidate_src = torch.randint(0, num_nodes, (num_neg_samples,))
        candidate_dst = torch.randint(0, num_nodes, (num_neg_samples,))

        # Check if the pairs are actual edges
        mask = ~torch.isin(torch.stack([candidate_src, candidate_dst], dim=0).t(), edge_index.t()).any(dim=1)

        valid_src = candidate_src[mask]
        valid_dst = candidate_dst[mask]

        neg_indices_src.append(valid_src)
        neg_indices_dst.append(valid_dst)

        negative_samples_found += valid_src.size(0)

    neg_indices_src = torch.cat(neg_indices_src)[:num_neg_samples]
    neg_indices_dst = torch.cat(neg_indices_dst)[:num_neg_samples]

    if len(neg_indices_src) == 0:
        logger.warning("No valid negative samples found")
        return pos_loss

    # Negative pairs (non-connected nodes) - maximize distance (margin = 1)
    neg_loss = torch.relu(1 - torch.norm(embeddings[neg_indices_src] - embeddings[neg_indices_dst], dim=1).pow(2)).sum()

    return pos_loss + neg_loss


def triplet_loss(embeddings, edge_index, margin=1.0):
    """
    Triplet margin loss for node embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings
        edge_index (torch.Tensor): Graph edge indices
        margin (float): Margin for triplet loss

    Returns:
        torch.Tensor: Loss value
    """
    logger.debug("Calculating triplet loss...")

    anchor = embeddings[edge_index[0]]
    positive = embeddings[edge_index[1]]

    # Generate random negative samples
    num_nodes = embeddings.size(0)
    neg_indices = torch.randint(0, num_nodes, (edge_index.size(1),))
    negative = embeddings[neg_indices]

    loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin)

    return loss


def cosine_embedding_loss(embeddings, edge_index):
    """
    Cosine embedding loss for node embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings
        edge_index (torch.Tensor): Graph edge indices

    Returns:
        torch.Tensor: Loss value
    """
    logger.debug("Calculating cosine embedding loss...")

    positive = embeddings[edge_index[0]]
    negative = embeddings[edge_index[1]]

    # Cosine similarity for positive pairs (connected nodes) - target = 1
    pos_loss = F.cosine_embedding_loss(positive, negative, torch.ones_like(edge_index[0], dtype=torch.float))

    # Random negative pairs - target = -1
    num_nodes = embeddings.size(0)
    neg_indices = torch.randint(0, num_nodes, (edge_index.size(1),))
    neg_embeds = embeddings[neg_indices]

    neg_loss = F.cosine_embedding_loss(positive, neg_embeds, -torch.ones_like(edge_index[0], dtype=torch.float))

    return pos_loss + neg_loss


# Dictionary mapping loss function names to functions
LOSS_FUNCTIONS = {
    'contrastive': contrastive_loss,
    'triplet': triplet_loss,
    'cosine': cosine_embedding_loss,
    'cross_entropy': cross_entropy_loss
}


def get_loss_function(name):
    """
    Get loss function by name.

    Args:
        name (str): Name of the loss function

    Returns:
        callable: Loss function

    Raises:
        ValueError: If loss function name is not supported
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unsupported loss function: {name}. Choose from: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[name]