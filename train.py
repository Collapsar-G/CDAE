#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $train.py

@Time    :   $2021.4.21 $8：50

@Desc    :   训练入口


"""
import os

import numpy as np

from dataset import getdata_ml1m, ml_Dataset
from models import CDAE
from miscc.config import cfg
from miscc.utils import data2matrix

import torch
from torch.utils.data import Dataset, DataLoader
import random
from matplotlib import pyplot as plt
import datetime
import dateutil.tz


def add_negative_items(purchase_vec, num_mask):
    """

    :param purchase_vec: batch_size x num_item
    :param num_mask:
    :return:
    """
    batch_size, num_item = purchase_vec.size()
    # print(batch_size,num_item)
    idx = torch.full((batch_size, num_item), 0.)

    idx = idx.view(-1)
    items = (idx == 0).nonzero().squeeze(-1)
    items = items.cpu().numpy().tolist()
    tmp_zr = random.sample(items, num_mask)
    # print(tmp_zr)
    idx = idx.numpy()
    for i in tmp_zr:
        idx[i] = 1
    idx = torch.from_numpy(idx)
    idx = idx.view((batch_size, num_item))
    if cfg.GPU_ID != "":
        idx = idx.cuda()
    # print(idx)
    return idx


def test(model, test_data, train_matrix, top_k=5):
    model.eval()
    users = list(test_data.keys())
    input_data = torch.tensor(train_matrix[users], dtype=torch.float)
    uids = torch.tensor(users, dtype=torch.long).view(-1, 1)
    if cfg.GPU_ID != "":
        input_data = input_data.cuda()
        uids = uids.cuda()
    out = model(uids, input_data)
    out = (out - 999 * input_data).detach().numpy()
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)[:top_k]
        for k, v in tmp_list:
            if k in test_data[u]:
                hit += 1
        recalls += hit / len(test_data[u])
        precisions += hit / top_k
        hits += hit
        total_purchase_nb += len(test_data[u])
    recall = recalls / len(users)
    precision = precisions / len(users)
    print('recall:{}, precision:{}'.format(recall, precision))
    return precision, recall


def plot_precision(epoche_list, precision_list):
    plt.title('precision')
    plt.plot(epoche_list, precision_list, marker='o')
    plt.savefig('precision_ml1m.png')
    plt.show()


if __name__ == "__main__":
    dataset = "ml_1m"

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/%s_%s' % \
                 (dataset, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/terminal.txt", 'w') as f:
        f.write('dataset:%s \n' % dataset)
    config = cfg[dataset]
    with open(output_dir + "/terminal.txt", 'a') as f:
        f.write('config:%s \n' % str(config))

    if cfg.GPU_ID != "":
        torch.cuda.set_device(cfg.GPU_ID)

    print("数据集：%s" % dataset)
    train_data, test_data = getdata_ml1m(config.path)
    train_matrix = data2matrix(train_data, num_user=config.num_user, num_item=config.num_item)
    test_matrix = data2matrix(test_data, num_user=config.num_user, num_item=config.num_item)
    if cfg.GPU_ID != "":
        train_matrix = train_matrix.cuda()
        test_matrix = test_matrix.cuda()
    print("成功构建评分矩阵")
    print("=" * 15)

    train_dataset = ml_Dataset(train_matrix)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = ml_Dataset(test_matrix)
    test_dataload = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True)
    print("DataLoader加载完成")

    model = CDAE(config)
    if cfg.GPU_ID != "":
        model.cuda()
    print("模型构建完成")

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    epoche_list = []
    precision_list = []

    for epoch in range(config.epochs):
        model.train()
        loss = torch.tensor(0.).cuda()
        for purchase_vec, uid in train_dataloader:
            mask_vec = add_negative_items(purchase_vec, config.num_mask) + purchase_vec

            out = model(uid, purchase_vec) * mask_vec
            loss += torch.sum((out - purchase_vec).square()) / mask_vec.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            str = '| epoch {:3d} /{:5d}  | loss {:5.7f} | learning rate {:5.7f}'.format(epoch, config.epochs,
                                                                                        loss,
                                                                                        opt.state_dict()[
                                                                                            'param_groups'][
                                                                                            0]['lr'])
            with open(output_dir + "/terminal.txt", 'a') as f:
                f.write(str+' \n')
                f.write('-' * 55 +' \n')
            print(str)
            print('-' * 55)
        if epoch + 1 % 100 == 0:
            print('=' * 55)
            print('*' * 55)
            print('=' * 55)
            torch.save(model.state_dict(), output_dir + "/models_%d.pth" % (epoch + 1))
            print("模型已存储至:%s" % (output_dir + "/models_%d.pth" % (epoch + 1)))
            print('-' * 55)
    #         print("开始测试")
    #         precision, _ = test(model, test_data, train_matrix, top_k=config.top_n)
    #         epoche_list.append(epoch + 1)
    #         precision_list.append(precision)
    #         print("测试完成，继续训练")
    # plot_precision(epoche_list, precision_list)
