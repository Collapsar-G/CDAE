#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $models.py

@Time    :   $2021.4.21 $8：50

@Desc    :   模型


"""

import torch
import torch.nn as nn


class CDAE(nn.Module):
    def __init__(self, config):
        self.num_item = config.num_item
        self.num_user = config.num_user
        self.num_hidden = config.num_hidden
        self.drop_rate = config.dropout
        super(CDAE, self).__init__()
        self.item2hidden = nn.Sequential(
            nn.Linear(self.num_item, self.num_hidden),
            nn.Dropout(self.drop_rate)
        )
        self.id2hidden = nn.Embedding(self.num_user, self.num_hidden)
        self.hidden2out = nn.Linear(self.num_hidden, self.num_item)
        self.sigmoid = nn.Sigmoid()

    def forward(self, uid, purchase_vec):
        # print(uid,purchase_vec)
        hidden = self.sigmoid(self.id2hidden(uid).squeeze(dim=1) + self.item2hidden(purchase_vec))
        out = self.hidden2out(hidden)
        out = self.sigmoid(out)
        return out
