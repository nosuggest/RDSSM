#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: Config.py
@time: 2020/9/14 19:33
@desc:
'''
import os
import sys

getModulePath = lambda p: os.path.join(os.path.dirname(__file__), p)

sys.path.append(getModulePath("."))


class Config(object):
    batchSize = 128
    sequenceLength = 20  # 取了所有序列长度的均值
    embeddingSize = 128
    epoch = 40
    learningRate = 0.001
    evaluateEvery = 1000
    checkpointEvery = 6000

    file_train = './data/oppo_round1_test_A_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    vocab_path = "./data/vocab.txt"
    rate = 0.95  # 正樣本訓練比率

    unk = '[UNK]'
    pad = '[PAD]'

    norm, epsilon = False, 0.001

    L1_N = 400
    L2_N = 120

    hidden_size_rnn = 100