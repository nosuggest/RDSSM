#!/usr/bin/env python
# encoding: utf-8
'''
@author: slade
@file: main.py
@time: 2020/9/28 16:31
@desc:
'''

import os
import datetime
from DataSet import Dataset
from utils import *
import tensorflow as tf
from dssm import DSSM

config = Config()
dataset = Dataset()
nwords = dataset._vocab_size
trainData, evalData = dataset.dataGen()

train_epoch_steps = int(len(trainData) / Config.batchSize) - 1
eval_epoch_steps = int(len(evalData) / Config.batchSize) - 1

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={"CPU": 78})
    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        dssm = DSSM(config, nwords)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(dssm.losses)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        tf.summary.scalar("loss", dssm.losses)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        savedModelPath = "../model/DSSM/savedModel"
        # if os.path.exists(savedModelPath):
        #     os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())


        def pull_batch(data_map, batch_id):
            cur_data = data_map[batch_id * config.batchSize:(batch_id + 1) * config.batchSize]
            query_in = [convert_word2bow(x[0], dataset._vocab_map) for x in cur_data]
            doc_in = [convert_word2bow(x[1], dataset._vocab_map) for x in cur_data]
            label = [x[2] for x in cur_data]

            # query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
            return query_in, doc_in, label


        def feed_dict(on_training, data_set, batch_id, drop_prob):
            query_in, doc_in, label = pull_batch(data_set, batch_id)
            query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
            return {dssm.query_batch: query_in, dssm.doc_batch: doc_in, dssm.doc_label_batch: label,
                    dssm.on_train: on_training, dssm.keep_prob: drop_prob}


        for ep in range(config.epoch):
            # 训练模型
            print("start training model")
            for batch_id in range(train_epoch_steps):
                _, summary, step, loss = sess.run(
                    [trainOp, summaryOp, globalStep, dssm.losses], feed_dict=feed_dict(True, trainData, batch_id, 0.5))

                currentStep = tf.train.global_step(sess, globalStep)
                trainSummaryWriter.add_summary(summary, step)
                print("train: epoch: {}, step: {}, loss: {}".format(
                    ep, currentStep, loss))

                if currentStep % config.evaluateEvery == 0:
                    print("\nEvaluation:\n")
                    eval_loss = 0
                    for batchEval in range(eval_epoch_steps):
                        loss_v = sess.run(dssm.losses, feed_dict=feed_dict(False, evalData, batchEval, 1))
                        eval_loss += loss_v
                    eval_loss /= (eval_epoch_steps)

                    time_str = datetime.datetime.now().isoformat()
                    print(
                        "eval: epoch: {}, {}, step: {}, loss: {}".format(
                            ep, time_str,
                            currentStep,
                            eval_loss))
                    evalSummaryWriter.add_summary(summary, step)

                if currentStep % config.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/DSSM/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))

        # 保存模型
        inputs = {"query_batch": tf.saved_model.utils.build_tensor_info(dssm.query_batch),
                  "doc_batch": tf.saved_model.utils.build_tensor_info(dssm.query_batch),
                  "doc_label_batch": tf.saved_model.utils.build_tensor_info(dssm.query_batch),
                  "on_train": tf.saved_model.utils.build_tensor_info(dssm.query_batch),
                  "keep_prob": tf.saved_model.utils.build_tensor_info(dssm.query_batch)}

        outputs = {"query_pred": tf.saved_model.utils.build_tensor_info(dssm.query_pred),
                   "doc_pred": tf.saved_model.utils.build_tensor_info(dssm.doc_pred),
                   "cos_sim": tf.saved_model.utils.build_tensor_info(dssm.cos_sim)
                   }

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()
