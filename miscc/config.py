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

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__c = edict()
cfg = __c

__c.batch_size = 32
__c.GPU_ID = 0

__c.ml_1m = edict()
__c.ml_1m.num_user = 6040
__c.ml_1m.num_item = 3952
__c.ml_1m.num_hidden = 48
__c.ml_1m.num_mask = 128
__c.ml_1m.epochs = 200
__c.ml_1m.lr = 0.001
__c.ml_1m.top_n = 10
__c.ml_1m.path = 'data/ml-1m/ratings.dat'
__c.ml_1m.dropout = 0.3
