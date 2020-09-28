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
