#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: dssm_rnn.py
@time: 2020/9/26 19:59
@desc:
'''
import tensorflow as tf
import numpy as np
from Config import Config


class RDSSM():
    def __init__(self, config, nwords):
        with tf.name_scope('input'):
            self.title_batch = tf.placeholder(tf.int32, shape=[None, None], name='title_batch')
            self.review_batch = tf.placeholder(tf.int32, shape=[None, None], name='review_batch')
            self.title_seq_length = tf.placeholder(tf.int32, shape=[None], name='title_sequence_length')
            self.seq_length = tf.placeholder(tf.int32, shape=[None], name='review_length')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.on_train = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('char_embedding_layer'):
            _char_embedding = tf.get_variable(name="char_embedding", dtype=tf.float32,
                                              shape=[nwords, config.embeddingSize])
            title_embed = tf.nn.embedding_lookup(_char_embedding, self.title_batch, name='title_embed')
            review_embed = tf.nn.embedding_lookup(_char_embedding, self.review_batch,
                                                  name='review_embed')

        with tf.name_scope('RNN'):
            cell_fw = tf.contrib.rnn.GRUCell(config.hidden_size_rnn, reuse=tf.AUTO_REUSE)
            cell_bw = tf.contrib.rnn.GRUCell(config.hidden_size_rnn, reuse=tf.AUTO_REUSE)
            # title
            (_, _), (title_output_fw, title_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                         title_embed,
                                                                                         sequence_length=self.title_seq_length,
                                                                                         dtype=tf.float32)
            title_rnn_output = tf.concat([title_output_fw, title_output_bw], axis=-1)
            title_rnn_output = tf.nn.dropout(title_rnn_output, self.keep_prob)
            # review_pos
            (_, _), (review_output_fw, review_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                           review_embed,
                                                                                           sequence_length=self.seq_length,
                                                                                           dtype=tf.float32)
            review_rnn_output = tf.concat([review_output_fw, review_output_bw], axis=-1)
            review_rnn_output = tf.nn.dropout(review_rnn_output, self.keep_prob)

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            # title的模
            title_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(title_rnn_output), 1, True)), [5, 1])
            # review的模
            review_norm = tf.sqrt(tf.reduce_sum(tf.square(review_rnn_output), 1, True))
            # 點積
            prod = tf.reduce_sum(tf.multiply(tf.tile(title_rnn_output, [5, 1]), review_rnn_output), 1, True)
            norm_prod = tf.multiply(title_norm, review_norm)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            self.cos_sim = tf.reshape(cos_sim_raw, [-1, 5])

        with tf.name_scope('Loss'):
            # Train Loss
            self.losses = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cos_sim, labels=self.label))
