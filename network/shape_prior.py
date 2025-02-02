import torch
import torch.nn as nn
import math
from . import net_configs as config


class RefinePrior(nn.Module):
    """
    Refines shape prior tokens by leveraging the Attention class to learn 
    inter-class relationships and capture long-range dependencies.
    """
    def __init__(self, config):
        super(RefinePrior, self).__init__()
        self.attention = Attention(config)

    def forward(self, shape_tokens):
        """
        Args:
            shape_tokens: Tensor of shape [B, num_classes, embed_dim]
                - B: Batch size
                - num_classes: Number of shape tokens (classes)
                - embed_dim: Embedding dimension
        Returns:
            refined_shape_tokens: Tensor of shape [B, num_classes, embed_dim]
        """
        refined_shape_tokens = self.attention(shape_tokens)
        return refined_shape_tokens


class CrossAttention(nn.Module):
    """
    Cross-attention layer to compute attention between image and shape token.
    """
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # linear projection
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # output projection
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.dropout_rate)

        # Softmax for attention probabilities
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_tokens, key_value_tokens):
        """
        Args:
            query_tokens: Tensor of shape [B, query_len, embed_dim]
            key_value_tokens: Tensor of shape [B, key_value_len, embed_dim]

        Returns:
            cross_attended_features: Tensor of shape [B, query_len, embed_dim]
        """
        # Q, K, V projections
        query_layer = self.transpose_for_scores(self.query(query_tokens))
        key_layer = self.transpose_for_scores(self.key(key_value_tokens))
        value_layer = self.transpose_for_scores(self.value(key_value_tokens))

        # scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention probabilities
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # aggregate values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)

        # output projection
        cross_attended_features = self.out(context_layer)
        cross_attended_features = self.proj_dropout(cross_attended_features)

        return cross_attended_features


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.dropout_rate)

        # positional embeddings
        if config.get("n_classes", None) is not None:
            self.position_embeddings = nn.Parameter(
                torch.randn(1, self.num_attention_heads, config.n_classes, config.n_classes)
            )
        else:
            self.position_embeddings = None

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # relative positional embedding
        if self.position_embeddings is not None:
            attention_scores = attention_scores + self.position_embeddings

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # context vectors (weighted sum of value vectors)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output

# need to revise    
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x