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

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
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
        train_data.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test:
        test_data.setdefault(uid, {}).setdefault(iid, 1)
    print("测试集和训练集分割完成")
    return train_data, test_data


class ml_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.data[idx], dtype=torch.float)

        uid = torch.tensor([idx, ], dtype=torch.long)
        if cfg.GPU_ID != "":
            purchase_vec = purchase_vec.cuda()
            uid = uid.cuda()
        return purchase_vec, uid

    def __len__(self):
        return len(self.data)
