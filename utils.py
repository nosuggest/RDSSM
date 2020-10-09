#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: utils.py
@time: 2020/9/26 16:41
@desc:
'''
import sys
import os

getModulePath = lambda p: os.path.join(os.path.dirname(__file__), p)

sys.path.append(getModulePath("."))
import json
from Config import Config
import numpy as np
from random import choice
import re

rule = re.compile("[^\u4e00-\u9fa5a-zA-Z0-9.,，。!?！？]<>《》\'\"")


def load_vocab(file_path):
    '''字映射'''
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


def gen_word_set(file_path="title_reviews.txt", out_path='./data/chars.txt'):
    '''扩充字映射表'''
    char_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) != 6:
                continue
            char_set.update(set("".join(spline)))
    with open(out_path, 'w', encoding='utf8') as out:
        for w in char_set:
            out.write(w + '\n')
    pass


def convert_word2id(query, vocab_map):
    '''query转换为id'''
    ids = []
    for w in query:
        ids.append(vocab_map.get(w, vocab_map[Config.unk]))

    while len(ids) < Config.sequenceLength:
        ids.append(vocab_map[Config.pad])
    return ids[:Config.sequenceLength]


def convert_seq2bow(query, vocab_map):
    '''query做onehot'''
    bow_ids = np.zeros(len(vocab_map))
    for w in query:
        bow_ids[vocab_map.get(w, vocab_map[Config.unk])] += 1
    return bow_ids


def convert_word2bow(wordid, vocab_map):
    '''query做onehot'''
    bow_ids = np.zeros(len(vocab_map))
    for w in wordid:
        if w == 0:
            continue
        bow_ids[w] += 1
    return bow_ids


def is_num_by_except(num):
    try:
        int(num.replace(" ", ""))
        return True
    except ValueError:
        #        print "%s ValueError" % num
        return False


def _reshape_input(file_path):
    formated_data = {}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) != 4:
                continue
            news_id, title, reviews, scores = spline
            if formated_data.get(title):
                formated_data[title] = list(set(formated_data[title] + eval(reviews)))
            else:
                formated_data[title] = list(set(eval(reviews)))
    formated_data_final = {}
    for k, v in formated_data.items():
        tmp = []
        for review in v:
            review = review.replace("\u200b", "").replace(" ", "").strip()

            # 过短
            if len(review) < 2:
                continue
            # 网址广告
            if "http" in review or "www." in review:
                continue
            # 过长
            if len(review) > 20:
                continue
            # 純數字
            if is_num_by_except(review):
                continue

            tmp.append(review)
        if len(tmp) > 0:
            formated_data_final[k] = tmp
    return formated_data_final


def genSamples(file_path, out_path="title_reviews.txt"):
    formated_data_final = _reshape_input(file_path)
    neg_reviews = list(formated_data_final.values())

    out = open(out_path, "w")
    for k, v in formated_data_final.items():
        for review in v:
            out.write(k)
            out.write("\t")
            out.write(review)
            out.write("\t")
            for i in range(4):
                neg_temp_match = choice(neg_reviews)
                neg_r = choice(neg_temp_match)
                while neg_r in v:
                    neg_temp_match = choice(neg_reviews)
                    neg_r = choice(neg_temp_match)
                out.write(neg_r)
                out.write("\t")
            out.write("\n")

    out.close()



def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta

def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res