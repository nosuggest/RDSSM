#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: Config.py
@time: 2020/9/14 19:33
@desc:
'''
from utils import load_vocab


class TrainingConfig(object):
    epoches = 50
    evaluateEvery = 500
    checkpointEvery = 500
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 128
    numFilters = 128

    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0


class Config(object):
    batchSize = 128
    sequenceLength = 10  # 取了所有序列长度的均值

    file_train = './data/oppo_round1_test_A_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    vocab_path = "./data/vocab.txt"

    unk = '[UNK]'
    pad = '[PAD]'

    vocab_map = load_vocab(vocab_path)
    vocab_size = len(vocab_map)
    # training = TrainingConfig()
    #
    # model = ModelConfig()
