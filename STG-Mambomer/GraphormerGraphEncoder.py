from typing import Optional, Tuple, Union, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm

from GraphormerGraphEncoderLayer import *


class GraphormerGraphEncoder(nn.Module):
    def __init__(
            self,
            # < graph
            static_graph: bool = True,
            graph_token=False,
            # >
            # transformer
            num_encoder_layers: int = 12,
            embedding_dim: int = 600,
            ffn_embedding_dim: int = 600,
            num_attention_heads: int = 30,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            encoder_normalize_before: bool = True,
            pre_layernorm: bool = True,
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
    ) -> None:

        super().__init__()
        self.static_graph = static_graph
        self.graph_token = graph_token

        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.embed_scale = embed_scale
        self.traceable = traceable
        self.apply_graphormer_init = apply_graphormer_init

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, eps=1e-8)
        else:
            self.emb_layer_norm = None

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False


        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def compute_attn_bias(self, batched_data):
        attn_bias = self.graph_attn_bias(batched_data)
        return attn_bias


    def forward_transformer_layers(
            self,
            x,
            padding_mask,
            attn_bias=None,
            attn_mask=None,
            last_state_only=True,
            get_attn_scores=False,
    ):
        # B x T x C -> T x B x C
        x = x.contiguous().transpose(0, 1)

        inner_states, attn_scores = [], []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                get_attn_scores=get_attn_scores,
            )
            if not last_state_only:
                inner_states.append(x)
                attn_scores.append(attn)

        if last_state_only:
            inner_states = [x]
            attn_scores = [attn]
        return inner_states, attn_scores

    def forward(
            self,
            x,
            attn_bias=None,
            # perturb=None,
            last_state_only: bool = True,
            # token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            get_attn_scores=False,
    ) -> Union[Tensor, list[torch.Tensor]]:
        if get_attn_scores:
            last_state_only = False
        B, T, D = x.shape

        padding_mask = None

        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        inner_states, attn_scores = self.forward_transformer_layers(
            x=x,
            padding_mask=padding_mask,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            last_state_only=last_state_only,
            get_attn_scores=get_attn_scores,
        )

        inner_states = np.array([t.detach().cpu().numpy() for t in inner_states])
        attn_scores = np.array(attn_scores)


        if not last_state_only:
            return torch.stack(inner_states), torch.stack(attn_scores)
        else:
            return inner_states, attn_scores
