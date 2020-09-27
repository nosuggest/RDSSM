#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: utils.py
@time: 2020/9/26 16:41
@desc:
'''
import json
from Config import Config
import numpy as np


def load_vocab(file_path):
    '''字映射'''
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


def gen_word_set(file_path, out_path='./data/words.txt'):
    '''扩充字映射表'''
    word_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            query_pred = json.loads(query_pred)
            word_set.update(set(prefix))
            _ = [word_set.update(set(word)) for word in query_pred]
    with open(out_path, 'w', encoding='utf8') as out:
        for w in word_set:
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
    bow_ids = np.zeros(Config.vocab_size)
    for w in query:
        bow_ids[vocab_map.get(w, vocab_map[Config.unk])] += 1
    return bow_ids

