#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model.py
# PythonVersion: python3.6
# Date    : 2020/6/22 8:49
# Software: PyCharm
"""Create a Bi-LSTM+CRF as baseline for NER."""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class BiLSTMCRF(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_dim,
                 max_seq_len,
                 weights=None,
                 weights_trainable=False):
        super(BiLSTMCRF, self).__init__()
        self.hidden_num = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # define layers
        if weights is not None:
            weights = np.array(weights)  # use pre-trained w2v
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                       weights=[weights],
                                                       trainable=weights_trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       input_length=max_seq_len)
        self.lstm_cell = tf.keras.layers.LSTM(hidden_num, return_sequences=True)
        self.biLSTM = tf.keras.layers.Bidirectional(self.lstm_cell, merge_mode='concat')
        self.dense = tf.keras.layers.Dense(label_size)  # lstm output space -> tag space

        # define transition
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))

        self.dropout = tf.keras.layers.Dropout(0.3)

    @tf.function
    def call(self, inputs, labels, training=None):
        # calculate the real sequence length, 0 is the padding element
        seq_len = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        embedding = self.embedding(inputs)  # inputs:[batch_size, max_seq], output: [batch_size, max_seq, embedding_dim]
        dropout = self.dropout(embedding, training)
        # input:[batch_size, max_seq, embedding_dim],
        # output:[batch_size, max_seq,embedding_dim*2]
        encoding = self.biLSTM(dropout)
        # lstm space -> tag space, output:[batch_size, max_seq, num_tags]
        output = self.dense(encoding)

        log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(output,
                                                                             labels,
                                                                             seq_len,
                                                                             transition_params=self.transition_params)

        return output, seq_len, log_likelihood

    def _viterbi_decode(self, feats):
        """
        Viterbi decode to find the best path and score
        Args:
            feats:

        Returns:

        """
        pass

    @tf.function
    def predict(self, inputs, training=None):
        # calculate the real sequence length, 0 is the padding element
        seq_len = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        embedding = self.embedding(inputs)
        dropout = self.dropout(embedding, training)
        encoding = self.biLSTM(dropout)
        output = self.dense(encoding)

        return output, seq_len
