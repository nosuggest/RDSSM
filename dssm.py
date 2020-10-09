#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: dssm.py
@time: 2020/9/26 15:59
@desc:
'''
import tensorflow as tf
import numpy as np


class DSSM():
    def __init__(self, config, nwords):
        with tf.name_scope('input'):
            self.query_batch = tf.placeholder(tf.float32, shape=[None, None], name='title_batch')
            self.doc_batch = tf.placeholder(tf.float32, shape=[None, None], name='review_batch')
            self.doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='review_label_batch')
            self.on_train = tf.placeholder(tf.bool, name='on_train')
            self.keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')
            self.weighted_loss = 1

        # embedding
        with tf.name_scope('FC1'):
            query_l1 = self.add_layer(self.query_batch, nwords, config.L1_N, activation_function=None)
            doc_l1 = self.add_layer(self.doc_batch, nwords, config.L1_N, activation_function=None)

        # dense
        with tf.name_scope('BN1'):
            query_l1 = self.batch_normalization(query_l1, self.on_train, config.L1_N)
            doc_l1 = self.batch_normalization(doc_l1, self.on_train, config.L1_N)

        with tf.name_scope('ACT1'):
            query_l1 = tf.nn.leaky_relu(query_l1)
            doc_l1 = tf.nn.leaky_relu(doc_l1)

        with tf.name_scope('Drop_out'):
            query_l1 = tf.nn.dropout(query_l1, self.keep_prob)
            doc_l1 = tf.nn.dropout(doc_l1, self.keep_prob)

        # dense
        with tf.name_scope('FC2'):
            query_l2 = self.add_layer(query_l1, config.L1_N, config.L2_N, activation_function=None)
            doc_l2 = self.add_layer(doc_l1, config.L1_N, config.L2_N, activation_function=None)

        with tf.name_scope('BN2'):
            query_l2 = self.batch_normalization(query_l2, self.on_train, config.L2_N)
            doc_l2 = self.batch_normalization(doc_l2, self.on_train, config.L2_N)

        with tf.name_scope('ACT2'):
            query_l2 = tf.nn.leaky_relu(query_l2)
            doc_l2 = tf.nn.leaky_relu(doc_l2)
            self.query_pred = query_l2
            self.doc_pred = doc_l2

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            self.cos_sim = self.get_cosine_score(self.query_pred, self.doc_pred)
            # cos_sim_prob = tf.clip_by_value(self.cos_sim, 1e-8, 1.0)

        with tf.name_scope('Loss'):
            # Train Loss
            self.predictions = tf.cast(tf.greater_equal(self.cos_sim, 0.0), tf.int32, name="predictions")
            self.labels = self.doc_label_batch

            # weighted loss
            self.losses = -tf.reduce_mean(
                self.weighted_loss * self.doc_label_batch * tf.log(
                    tf.clip_by_value(tf.nn.sigmoid(self.cos_sim), 1e-8, 1.0)) + (
                        1 - self.doc_label_batch) * tf.log(
                    tf.clip_by_value(1 - tf.nn.sigmoid(self.cos_sim), 1e-8, 1.0)))

            # self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits=tf.cast(tf.reshape(self.cos_sim, [-1, 1]), dtype=tf.float32),
            #     labels=tf.cast(tf.reshape(self.doc_label_batch, [-1, 1]), dtype=tf.float32)))

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        wlimit = np.sqrt(6.0 / (in_size + out_size))
        Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit))
        biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit))
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def mean_var_with_update(self, ema, fc_mean, fc_var):
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    def batch_normalization(self, x, phase_train, out_size):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2 + reg

    def get_cosine_score(self, query_arr, doc_arr):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
        cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
        return cos_scores
