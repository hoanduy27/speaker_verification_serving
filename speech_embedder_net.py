#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""
import torch
import torch.nn as nn


class SpeechEmbedder(nn.Module):
    """Implementation of https://github.com/HarryVolek/PyTorch_Speaker_Verification.git"""
    def __init__(
        self, n_mels, 
        lstm_hidden, 
        lstm_layers,
        emb_dim
    ):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(n_mels, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(lstm_hidden, emb_dim)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
