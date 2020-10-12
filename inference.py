#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: inference.py
@time: 2020/10/09 10:17
@desc:
'''
from Config import Config
import tensorflow as tf
from utils import gen_word_set, convert_word2id, convert_word2bow, load_vocab
import numpy as np

# config = Config()
_vocab_map = load_vocab(Config.vocab_path)
query = "香芋味酸奶卷，好想吃一口#不可辜负的美食##美食分享# @微博热视频 @微博美食"
doc = "你的美食我的胖"

query_list = convert_word2id(query, _vocab_map)
doc_list = convert_word2id(doc, _vocab_map)
query_in = np.array([convert_word2bow(query_list, _vocab_map)])
doc_in = np.array([convert_word2bow(doc_list, _vocab_map)])

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint("../model/DSSM/model/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        query_batch = graph.get_operation_by_name("input/title_batch").outputs[0]
        doc_batch = graph.get_operation_by_name("input/review_batch").outputs[0]
        on_train = graph.get_operation_by_name("input/on_train").outputs[0]
        keep_prob = graph.get_operation_by_name("input/drop_out_prob").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("Loss/predictions:0")
        cos_scores = graph.get_tensor_by_name("Cosine_Similarity/cos_scores:0")

        pred, prob = sess.run([predictions, cos_scores],
                              feed_dict={query_batch: query_in, doc_batch: doc_in, on_train: False, keep_prob: 1.0})
        print("pred：{}".format(pred))
        print("prob：{}".format(prob))
