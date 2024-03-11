from typing import Any, AnyStr, Dict, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GATConv
from torch.nn.modules.module import Module

class GAT(Module):
    """backbone"""
    def __init__(self, args: Dict, activation=F.elu, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        self.args = args
        self.activation = activation
        self.negative_slope = negative_slope
        self.residual = residual
        self.gat_layers = self.build_gnn_layers()

    def build_gnn_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(GATConv(self.args.input_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        for _ in range(1, self.args.num_gnn_layers - 1):
            layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        
        return layers

    def forward(self, inputs: torch.Tensor, g: dgl.graph):
        h = inputs
        for l in range(self.args.num_gnn_layers-1):
            h = self.gat_layers[l](g, h).flatten(1)
        h = torch.mean(self.gat_layers[-1](g, h), dim=1)
        return h


class GATBatch(Module):
    """backbone"""
    def __init__(self, args: Dict, activation=F.elu, negative_slope=0.2, residual=False):
        super(GATBatch, self).__init__()
        self.args = args
        self.activation = activation
        self.negative_slope = negative_slope
        self.residual = residual
        self.gat_layers = self.build_gnn_layers()

    def build_gnn_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(GATConv(self.args.input_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        for _ in range(1, self.args.num_gnn_layers - 1):
            layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim, self.args.hidden_dim, self.args.num_gnn_heads, \
                    self.args.feat_drop_rate, self.args.attn_drop_rate, self.negative_slope, self.residual, self.activation))
        
        return layers

    def forward(self, inputs: torch.Tensor, blocks):
        h = inputs
        for l in range(self.args.num_gnn_layers-1):
            h = self.gat_layers[l](blocks[l], h).flatten(1)
        h = torch.mean(self.gat_layers[-1](blocks[-1], h), dim=1)

        return h