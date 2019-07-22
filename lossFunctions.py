# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:15:28 2019

@author: Mirac
"""

import torch

class BCEwithIgnore(torch.nn.Module):
    def _init_(self):
        super(BCEwithIgnore, self)._init_()
    
    def forward(self, score, y):
        zeros = torch.zeros_like(y)
        ones  = torch.ones_like(y)
        num_uncertain = torch.sum(torch.where(y==-1, ones, zeros))
        positive = torch.where(y==-1, zeros, y)
        negative = torch.where(y==-1, ones, y)
        p = torch.log(score)
        one_minus_p = torch.log(1 - score)
        loss = -1 * torch.sum(p * positive + (1-negative) * one_minus_p) / ( y.numel() - num_uncertain)
        return loss
      
      
class WeightedBCE(torch.nn.Module):
    def __init__(self, weight):
        super(WeightedBCE, self).__init__()
        self.w = weight
    
    def forward(self, score, y):
        loss = -1 * torch.mean(y * torch.log(score) * self.w +\
                               (1-y) * torch.log(1 - score) * (2 - self.w)) 
        
        return loss
      

class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, weight):
        super(WeightedCrossEntropy, self).__init__()
        self.w = weight
    
    def forward(self, output, y):
        scores = torch.softmax(output.view(-1, 3), dim=1)
        y = y.view(-1, 1)
        y_onehot = torch.zeros((y.shape[0], 3), device=y.device)
        y_onehot.scatter_(1, y, 1)
        weights = self.w.repeat(len(y)// 14 , 1)
        loss = - torch.mean(torch.log(scores).type(torch.double) * y_onehot.type(torch.double) * weights.type(torch.double)) 
        
        return loss