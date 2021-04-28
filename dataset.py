#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $dataset.py

@Time    :   $2021.4.21 $8：50

@Desc    :   dataset


"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from miscc.config import cfg


def getdata_ml1m(filepath, sep='::', header='infer'):
    """

    :param filepath: 文件路径
    :param sep:
    :param header:
    :return: 训练数据和测试数据
    """
    train_data, test_data = {}, {}
    df = pd.read_csv(filepath, sep=sep, header=header, engine='python').iloc[:, :3] - 1
    df = df.values.tolist()
    print("成功从%s读取数据" % filepath)
    train, test = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train:
        train_data.setdefault(uid, {}).setdefault(iid, score)
    for uid, iid, score in test:
        test_data.setdefault(uid, {}).setdefault(iid, score)
    print("测试集和训练集分割完成")
    # print(test_data)
    return train_data, test_data


def getdata_ml_learn(train_path, test_path):
    print("#############")
    train_data, test_data = {}, {}
    train_df = pd.read_csv(train_path).iloc[:, :3] - 1
    train_df = train_df.values.tolist()
    for uid, iid, score in train_df:
        train_data.setdefault(uid, {}).setdefault(iid, score)
    test_df = pd.read_csv(test_path).iloc[:, :3] - 1
    test_df = test_df.values.tolist()
    for uid, iid, score in test_df:
        test_data.setdefault(uid, {}).setdefault(iid, score)
    print("测试集和训练集分割完成")
    # print(test_data)
    return train_data, test_data


class ml_Dataset(Dataset):
    def __init__(self, data_matrix, data_score):
        self.data_matrix = data_matrix
        self.data_score = data_score

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.data_matrix[idx], dtype=torch.float)
        score_vec = torch.tensor(self.data_score[idx], dtype=torch.float)
        uid = torch.tensor([idx, ], dtype=torch.long)
        if cfg.GPU_ID != "":
            purchase_vec = purchase_vec.cuda()
            score_vec = score_vec.cuda()
            uid = uid.cuda()
        return purchase_vec, uid, score_vec

    def __len__(self):
        return len(self.data_score)
