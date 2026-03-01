```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np


class UserEncoder(nn.Module):
    """Encodes user history and demographics into a user representation."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, embedding_dim, padding_idx=0)
        
        # Transformer encoder for history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_positions: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,)
            history_item_ids: (batch_size, seq_len)
            history_positions: (batch_size, seq_len)
            history_mask: (batch_size, seq_len), True where padded
        Returns:
            user_repr: (batch_size, hidden_dim)
        """
        # User embedding
        user_emb = self.user_embedding(user_ids)  # (B, D)
        
        # Item history embeddings + positional
        item_emb = self.item_embedding(history_item_ids)  # (B, S, D)
        pos_emb = self.position_embedding(history_positions)  # (B, S, D)
        seq_emb = item_emb + pos_emb  # (B, S, D)
        
        # Transformer over history
        seq_out = self.transformer(seq_emb, src_key_padding_mask=history_mask)  # (B, S, D)
        
        # Aggregate: mean pooling over non-padded positions
        mask_expanded = (~history_mask).float().unsqueeze(-1)  # (B, S, 1)
        seq_agg = (seq_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)  # (B, D)
        
        # Combine user embedding with sequence aggregation
        combined = torch.cat([user_emb, seq_agg], dim=-1)  # (B, 2D)
        user_repr = self.layer_norm(F.gelu(self.output_proj(combined)))  # (B, H)
        user_repr = self.dropout(user_repr)
        
        return user_repr


class ItemEncoder(nn.Module):
    """Encodes item features (genre, metadata, content) into item representations."""
    
    def __init__(
        self,
        num_items: int,
        num_genres: int,
        num_content_types: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4, padding_idx=0)
        self.content_type_embedding = nn.Embedding(num_content_types, embedding_dim // 4, padding_idx=0)
        
        # Feature fusion
        genre_dim = embedding_dim // 4
        content_type_dim = embedding_dim // 4
        input_dim = embedding_dim + genre_dim + content_type_dim
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(
        self,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        content_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (batch_size,) or (batch_size, num_candidates)
            genre_ids: same shape as item_ids
            content_type_ids: same shape as item_ids
        Returns:
            item_repr: (..., hidden_dim)
        """
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genre_ids)
        content_type_emb = self.content_type_embedding(content_type_ids)
        
        combined = torch.cat([item_emb, genre_emb, content_type_emb], dim=-1)
        item_repr = self.feature_net(combined)
        
        return item_repr


class RewardHead(nn.Module):
    """Scalar reward prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize last layer with small weights for stable training
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim)
        Returns:
            reward: (..., 1)
        """
        return self.net(x)


class NetflixRewardModel(nn.Module):
    """
    Reward model for Netflix RLHF recommender system.
    
    Learns to predict human preference scores for recommended items
    given user context. Trained using Bradley-Terry preference model
    with comparison data (chosen vs rejected recommendations).
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        num_content_types: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_transformer_heads: int = 