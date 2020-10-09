#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: DataSet.py
@time: 2020/9/14 19:58
@desc:
'''
import pandas as pd
from collections import Counter
import numpy as np
import json
import gensim
from utils import gen_word_set, convert_word2id, convert_seq2bow, load_vocab
from Config import Config

from random import shuffle, choice
from tqdm import tqdm


class Dataset(object):
    def __init__(self):
        self.config = Config

        self._sequenceLength = Config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = Config.embeddingSize
        self._batchSize = Config.batchSize

        self._vocab_map = load_vocab(Config.vocab_path)
        self._vocab_size = len(self._vocab_map)

    def _genTrainEvalData(self, data_map, rate):
        """
        生成训练集和验证集
        """
        title, title_len, review_pos, review_pos_len, review_neg, review_neg_len = data_map['title'], data_map[
            'title_len'], \
                                                                                   data_map['review_pos'], data_map[
                                                                                       'review_pos_len'], \
                                                                                   data_map['review_neg'], data_map[
                                                                                       'review_neg_len']
        index = list(range(len(title)))
        unzip_all_data = []
        for idx in tqdm(index):
            unzip_all_data.append([title[idx], review_pos[idx], 1])
            for item in review_neg[idx]:
                unzip_all_data.append([title[idx], item, 0])

        trainIndex = int(len(unzip_all_data) * rate)
        index = list(range(len(unzip_all_data)))
        shuffle(index)

        trainData = np.array(unzip_all_data)[index[:trainIndex]]

        evalData = np.array(unzip_all_data)[index[trainIndex:]]

        return trainData, evalData

    def dataGen(self, file_path="./data/title_reviews.txt"):
        """
        初始化训练集和验证集
        """

        data_map = {'title': [], 'title_len': [], 'review_pos': [], 'review_pos_len': [], 'review_neg': [],
                    'review_neg_len': []}
        with open(file_path, encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                spline = line.strip().split('\t')
                if len(spline) != 6:
                    continue
                title, pos, neg1, neg2, neg3, neg4 = spline
                negs = [neg1, neg2, neg3, neg4]
                cur_arr, cur_len = [], []
                # only 4 negative sample
                # for each in negs:
                #     cur_arr.append(convert_word2id(each, self._vocab_map))
                #     each_len = len(each) if len(each) < Config.sequenceLength else Config.sequenceLength
                #     cur_len.append(each_len)
                each = choice(negs)
                cur_arr.append(convert_word2id(each, self._vocab_map))
                each_len = len(each) if len(each) < Config.sequenceLength else Config.sequenceLength
                cur_len.append(each_len)
                data_map['title'].append(convert_word2id(title, self._vocab_map))
                data_map['title_len'].append(
                    len(title) if len(title) < Config.sequenceLength else Config.sequenceLength)
                data_map['review_pos'].append(convert_word2id(pos, self._vocab_map))
                data_map['review_pos_len'].append(
                    len(pos) if len(pos) < Config.sequenceLength else Config.sequenceLength)
                data_map['review_neg'].append(cur_arr)
                data_map['review_neg_len'].append(cur_len)

        return self._genTrainEvalData(data_map, Config.rate)
    def _multiGenTrainEvalData(self, data_map, rate):
        """
        生成训练集和验证集
        """
        title, title_len, review_pos, review_pos_len, review_neg, review_neg_len = data_map['title'], data_map[
            'title_len'], \
                                                                                   data_map['review_pos'], data_map[
                                                                                       'review_pos_len'], \
                                                                                   data_map['review_neg'], data_map[
                                                                                       'review_neg_len']
        index = list(range(len(title)))
        unzip_all_data = []
        for idx in tqdm(index):
            unzip_all_data.append([title[idx], review_pos[idx], 1])
            for item in review_neg[idx]:
                unzip_all_data.append([title[idx], item, 0])

        trainIndex = int(len(unzip_all_data) * rate)
        index = list(range(len(unzip_all_data)))
        shuffle(index)

        trainData = np.array(unzip_all_data)[index[:trainIndex]]

        evalData = np.array(unzip_all_data)[index[trainIndex:]]

        return trainData, evalData

    def multiDataGen(self, file_path="./data/title_reviews.txt"):
        """
        初始化训练集和验证集
        """

        data_map = {'title': [], 'title_len': [], 'review_pos': [], 'review_pos_len': [], 'review_neg': [],
                    'review_neg_len': []}
        with open(file_path, encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                spline = line.strip().split('\t')
                if len(spline) != 6:
                    continue
                title, pos, neg1, neg2, neg3, neg4 = spline
                negs = [neg1, neg2, neg3, neg4]
                cur_arr, cur_len = [], []
                # only 4 negative sample
                for each in negs:
                    cur_arr.append(convert_word2id(each, self._vocab_map))
                    each_len = len(each) if len(each) < Config.sequenceLength else Config.sequenceLength
                    cur_len.append(each_len)
                each = choice(negs)
                cur_arr.append(convert_word2id(each, self._vocab_map))
                each_len = len(each) if len(each) < Config.sequenceLength else Config.sequenceLength
                cur_len.append(each_len)
                data_map['title'].append(convert_word2id(title, self._vocab_map))
                data_map['title_len'].append(
                    len(title) if len(title) < Config.sequenceLength else Config.sequenceLength)
                data_map['review_pos'].append(convert_word2id(pos, self._vocab_map))
                data_map['review_pos_len'].append(
                    len(pos) if len(pos) < Config.sequenceLength else Config.sequenceLength)
                data_map['review_neg'].append(cur_arr)
                data_map['review_neg_len'].append(cur_len)

        return self._genTrainEvalData(data_map, Config.rate)
