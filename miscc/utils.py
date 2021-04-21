#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $utils.py

@Time    :   $2021.4.21 $8：50

@Desc    :   永远的工具人


"""
import numpy as np
import torch

def data2matrix(data, num_user=1, num_item=1):
    data_matrix = np.zeros(shape=(num_user, num_item))
    for u in data.keys():
        for i in data[u].keys():
            data_matrix[u - 1][i - 1] = 1
    data_matrix = torch.tensor(data_matrix)
    return data_matrix

def save_model(model, path):
    return
