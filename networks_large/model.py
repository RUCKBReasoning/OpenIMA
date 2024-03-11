from typing import Any, AnyStr, Dict, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import sys
sys.path.append("networks/")

from torch.nn.modules.module import Module
from .gat import GATBatch

class GNNModel(Module):
    """backbone + projection head"""
    def __init__(self, args: Dict):
        super(GNNModel, self).__init__()
        # gnn backbone
        if args.encoder_name == "gat":
            self.encoder = GATBatch(args)
        else:
            raise NotImplementedError(
                'encoder not supported: {}'.format(args.encoder_name))

        # projection head
        # self.head = nn.Sequential(
        #     nn.Linear(args.hidden_dim, args.hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.hidden_dim, args.hidden_dim)
        # )

    def forward(self, feats: torch.Tensor, adj: dgl.graph):
        emb = self.encoder(feats, adj)
        return F.normalize(emb, dim=1)

