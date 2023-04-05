# -*- coding: utf-8 -*-
import torch
from torch import nn

#@save
class BPRLoss(torch.nn.Module):
    def __init__(self, gamma, **kwargs):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, positive, negative):
        # L2正则化可通过torch的优化器来实现
        distances = positive - negative
        loss = - torch.log(self.gamma + torch.sigmoid(distances)).sum()
        return loss

#@save
class HingeLossbRec(nn.Module):
    def __init__(self, **kwargs):
        super(HingeLossbRec, self).__init__()

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = torch.max(- distances + margin, 0, keepdim=True).sum()
        return loss