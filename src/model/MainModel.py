""" 
Write your main Model here
"""
import torch
import torch.nn as nn


class MainModel(nn.Module):
    def __init__(self, hparams):
        super(MainModel, self).__init__()

        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)
