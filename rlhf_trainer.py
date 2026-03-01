```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    # Model dimensions
    num_users: int = 100000
    num_items: int = 50000
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    rollout_length: int = 2048
    max_recommendations: int = 10
    
    # Reward
    watch_reward: float = 1.0
    rating_reward_scale: float = 0.5
    completion_bonus: float = 0.5
    skip_penalty: float = -0.3
    diversity_bonus: float = 0.1
    
    # RLHF specific
    reward_model_lr: float = 1e-4
    preference_batch_size: int = 32
    reward_model_epochs: int = 3
    kl_penalty_coef: float = 0.1


class UserEncoder(nn.Module):
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.watch_history_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.feature_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        user_ids: torch.Tensor,
        watch_history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_embs = self.user_embedding.weight[watch_history] if watch_history.dim() > 1 else watch_history
        
        # Encode watch history
        gru_out, hidden = self.watch_history_encoder(item_embs)
        history_repr = hidden[-1]
        
        # Combine user embedding with history
        combined = torch.cat([user_emb, history_repr], dim=-1)
        user_state = self.feature_projection(combined)
        
        return user_state


class ItemEncoder(nn.Module):
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Item feature encoder (genre, duration, release year, etc.)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim + 16, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        
        if item_features is not None:
            combined = torch.cat([item_emb, item_features], dim=-1)
        else:
            # Pad with zeros if no additional features
            padding = torch.zeros(
                *item_emb.shape[:-1], 16,
                device=item_emb.device
            )
            combined = torch.cat([item_emb, padding], dim=-1)
        
        item_repr = self.feature_encoder(combined)
        return item_repr


class RecommendationPolicy(nn.Module):
    """Actor network for recommendation policy"""
    
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        
        self.user_encoder = UserEncoder(config)
        self.item_encoder = ItemEncoder(config)
        
        # Attention-based item scoring
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Policy head
        self.policy_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Temperature for exploration
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        user_ids: torch.Tensor,
        watch_history: torch.Tensor,
        candidate_items: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Encode user state
        user_state = self.user_encoder(user_ids, watch_history)
        
        # Encode candidate items
        batch_size = candidate_items.size(0)
        num_candidates = candidate_items.size(1)
        
        flat_items = candidate_items.view(-1)
        flat_features = None
        if item_features is not None:
            flat_features = item_features.view(-1, item_features.size(-1))
        
        item_reprs = self.item_encoder(flat_items, flat_features)
        item_reprs = item_reprs.view(batch_size, num_candidates, -1)
        
        # User-item attention
        user_query = user_state.unsqueeze(1)
        attended_items, attention_weights = self.attention(
            user_query, item_reprs, item_reprs
        )
        
        # Score each item
        user_expanded = user_state.unsqueeze(1).expand(-1, num_candidates, -1)
        combined = torch.cat([user_expanded, item_reprs], dim=-1)
        
        scores = self.policy_network(combined).squeeze(-1)
        
        # Apply temperature scaling
        temperature = self.log_temperature.exp()
        logits = scores / temperature
        
        return logits, attention_weights
    
    def get_action(
        self,
        user_ids: torch.Tensor,
        watch_history: torch.Tensor,
        candidate_