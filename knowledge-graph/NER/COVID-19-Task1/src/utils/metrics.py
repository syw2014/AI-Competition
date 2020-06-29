#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : metrics.py
# PythonVersion: python3.6
# Date    : 2020/6/11 15:56
# Software: PyCharm
"""Different metrics to evaluate model."""
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SparsePrecisionScore(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, axis=-1), [-1]).numpy()
        precision = precision_score(y_true, y_predict, average=self.average)
        return precision


class SparseRecallScore(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, axis=-1), [-1]).numpy()
        recall = recall_score(y_true, y_predict, average=self.average)
        return recall


class SparseAccuracyScore(object):
    def __init__(self, predict_sparse=False):
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, axis=-1), [-1]).numpy()
        accuracy = accuracy_score(y_true, y_predict)
        return accuracy


class SparseF1Score(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(tf.argmax(y_predict, axis=-1), [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, axis=-1), [-1]).numpy()
        f1 = f1_score(y_true, y_predict, average=self.average)
        return f1
