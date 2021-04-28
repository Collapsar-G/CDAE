#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $config.py

@Time    :   $2021.4.21 $8：50

@Desc    :   参数文件


"""
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

__c = edict()
cfg = __c

__c.batch_size = 128
__c.GPU_ID = 0
__c.random_seed = 200

__c.ml_1m = edict()
__c.ml_1m.num_user = 6040
__c.ml_1m.num_item = 3952
__c.ml_1m.num_hidden = 100
__c.ml_1m.num_mask = 5
__c.ml_1m.epochs = 300
__c.ml_1m.lr = 0.005
__c.ml_1m.top_n = 10
__c.ml_1m.path = 'data/ml-1m/ratings.dat'
__c.ml_1m.dropout = 0.1
__c.ml_1m.lam = 0.2
__c.ml_1m.a_fun = "tanh"
__c.ml_1m.b_fun = "softmax"
__c.ml_1m.optim = "Adam"
__c.ml_1m.patience = 3

__c.ml_learn = edict()
__c.ml_learn.num_user = 2504
__c.ml_learn.num_item = 2999
__c.ml_learn.num_hidden = 100
__c.ml_learn.num_mask = 5
__c.ml_learn.epochs = 300
__c.ml_learn.lr = 0.01
__c.ml_learn.top_n = 10
__c.ml_learn.train_path = 'data/ml_learn/dvd_sparse_train.csv'
__c.ml_learn.test_path = 'data/ml_learn/dvd_sparse_test.csv'
__c.ml_learn.dropout = 0.1
__c.ml_learn.lam = 0.2
__c.ml_learn.a_fun = "sigmoid"
__c.ml_learn.b_fun = "softmax"
__c.ml_learn.optim = "Adam"
__c.ml_learn.type = "dvd"
__c.ml_learn.patience = 7
