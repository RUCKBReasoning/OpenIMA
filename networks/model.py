from typing import Any, AnyStr, Dict, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
sys.path.append("networks/")
from torch.nn.modules.module import Module
from .gat import GAT

class GNNModel(Module):
    """backbone + projection head"""
    def __init__(self, args: Dict):
        super(GNNModel, self).__init__()
        # gnn backbone
        if args.encoder_name == "gat":
            self.encoder = GAT(args)
        else:
            raise NotImplementedError(
                'encoder not supported: {}'.format(args.encoder_name))

    def forward(self, feats: torch.Tensor, adj: dgl.graph):
        emb = self.encoder(feats, adj)
        return F.normalize(emb, dim=1)

