#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : bilstm_crf.py
# PythonVersion: python3.6
# Date    : 2020/7/1 7:50
# Software: PyCharm
"""An implementation of Bi-LSTM+CRF reference torch tutorial.
ref: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""
import tensorflow as tf
import numpy as np


class BiLSTMCRF(tf.keras.Model):
    def __init__(self, vocab_size, tag_to_idx, max_seq_len,
                 embedding_dim,
                 hidden_dim,
                 weights=None,
                 weights_trainable=False):
        """

        Args:
            vocab_size:
            tag_to_idx:
            max_seq_len:
            embedding_dim:
            hidden_dim:
            weights:
            weights_trainable:
        """
        super(BiLSTMCRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        # define layers used in model
        if weights is not None:
            weights = np.array(weights)
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       weights=[weights],
                                                       trainable=weights_trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       input_length=max_seq_len)
        self.lstm = tf.keras.layers.LSTM(self.hidden_dim // 2, return_sequences=True)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm, merge_mode='concat')

        # maps the output of LSTM into tag space
        self.hidden2tag = tf.keras.layers.Dense(self.tagset_size)

        # Matrix of transition parameters. Entry i,j is the score of transitioning i to j
        # this the parameters of CRF layer, need to be learning from data
        self.transitions = tf.Variable(tf.random.uniform(shape=(self.tagset_size, self.tagset_size)))
        self.dropout = tf.keras.layers.Dropout(0.3)
        