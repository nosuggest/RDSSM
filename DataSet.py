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
from utils import gen_word_set, convert_word2id, convert_seq2bow


class Dataset(object):
    def __init__(self, config):
        self.config = config

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize

        self._vocab_map = config.vocab_map
        self._vocab_size = config.vocab_size

    def get_data(self, file_path):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
        """
        data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, query_pred, title, tag, label = spline
                if label == '0':
                    continue
                cur_arr, cur_len = [], []
                query_pred = json.loads(query_pred)
                # only 4 negative sample
                for each in query_pred:
                    if each == title:
                        continue
                    cur_arr.append(convert_word2id(each, conf.vocab_map))
                    each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                    cur_len.append(each_len)
                if len(cur_arr) >= 4:
                    data_map['query'].append(convert_word2id(prefix, conf.vocab_map))
                    data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                    data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map))
                    data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                    data_map['doc_neg'].extend(cur_arr[:4])
                    data_map['doc_neg_len'].extend(cur_len[:4])
                pass
        return data_map

    def get_data_siamese_rnn(file_path):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
        """
        data_arr = []
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, _, title, tag, label = spline
                prefix_seq = convert_word2id(prefix, conf.vocab_map)
                title_seq = convert_word2id(title, conf.vocab_map)
                data_arr.append([prefix_seq, title_seq, int(label)])
        return data_arr

    def get_data_bow(file_path):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, prefix, label]], shape = [n, 3]
        """
        data_arr = []
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, _, title, tag, label = spline
                prefix_ids = convert_seq2bow(prefix, conf.vocab_map)
                title_ids = convert_seq2bow(title, conf.vocab_map)
                data_arr.append([prefix_ids, title_ids, int(label)])
        return data_arr

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        return labelIds



    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)

        with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)

        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("/home/11112877/word2vec/w2v.bin", binary=True,
                                                                  unicode_errors='ignore')
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(list(np.zeros(self._embeddingSize)))
        wordEmbedding.append(list(np.random.randn(self._embeddingSize)))

        for word in words:
            try:
                vector = list(wordVec.wv[word])
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)

        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx,
                                                                                    self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
