#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : losses.py
# PythonVersion: python3.6
# Date    : 2020/6/11 16:25
# Software: PyCharm
"""Define losses for models with tf2.0"""

import tensorflow as tf

class MaskSparseCategoricalCrossentropy(object):
    def __init__(self, from_logits=False, use_mask=False):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        """
        :param y_true:
        :param y_predict:
        :param input_mask:
        :return:
        """
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict)
        # mask loss
        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask

        return tf.reduce_mean(cross_entropy)
