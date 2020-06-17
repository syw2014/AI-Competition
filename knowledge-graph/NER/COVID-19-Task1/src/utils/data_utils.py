#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : data_utils.py
# PythonVersion: python3.6
# Date    : 2020/6/9 14:59
# Software: PyCharm
"""Design tools to create input for model.
    1> split dataset into train/dev/test=0.7:0.1:0.2
    2> padding function for token sequence
    3> create input dataset or iterator for model
"""
import random
import json
from tqdm import tqdm
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import re

def doc_to_sentences(doc):
    """Cut doc into sentences with !.;?"""
    pat = u'[!.;?]'
    sents = re.split(pat,doc)
    seg_words = re.findall(pat, doc)
    seg_words.insert(len(doc)-1, "")
    # keep seg words in each sentence
    results = [s+w for s,w in zip(sents, seg_words)]
    return results

def train_test_split(samples, n_samples, train_proportion=0.7,
                     dev_proportion=0.1, shuffle=True):
    """
    Split N samples into train/dev/test, the proportion=0.7:0.1:0.2
    :param samples: input samples array
    :param n_samples: total samples
    :param train_proportion: input train proportion, default 0.7
    :param dev_proportion: input train proportion, default 0.1
    :param test_proportion: input train proportion, default 0.2
    :return: n_train, n_dev, n_test
    """
    train_size = math.ceil(train_proportion * n_samples)
    dev_size = math.ceil(dev_proportion * n_samples)
    test_size = n_samples - train_size - dev_size
    # prepare sample index
    train = np.arange(train_size)
    dev = np.arange(train_size, train_size + dev_size)
    test = np.arange(train_size + dev_size, n_samples)
    if shuffle:
        random.shuffle(samples)
    train = [samples[idx] for idx in train]
    dev = [samples[idx] for idx in dev]
    test = [samples[idx] for idx in test]

    return train, dev, test


def data_split(filename, output_dir):
    """
    Split data into train/dev/test,
    :param filename: input data file name
    :param output_dir: output directory
    :return:
    """
    dataset = {}
    # Here we cut text into sentences, mean while convert entity to BIO format
    pass


def padding(token_ids, max_seq_len, padding_id, seq_front=False):
    """
    Padding input token id sequence to the maximum length.
    :param token_ids: input token id(int32) list
    :param max_seq_len: the maximum sequence length
    :param padding_id: padding id
    :param seq_front: weather padding in the front or backend
    :return:
    """
    length = len(token_ids)
    if length < max_seq_len:
        if seq_front:
            token_ids = [padding_id] * (max_seq_len - length) + token_ids
        else:
            token_ids = token_ids + [padding_id] * (max_seq_len - length)
    else:
        token_ids = token_ids[:max_seq_len]

    return token_ids


def create_dataset_with_tf(filename, vocab, epochs, batch_size, max_seq_len, mode):
    """
    Create input dataset for model with tf.data.Dataset api.
    Here we use tf.data.Dataset.from_tensor_slices to create tf.dataset, so firstly it will read all data into memory
    and may occupy large memory if your corpus was large.
    :param filename: input data file, format <label, text>
    :param vocab: Vocab object, which contain the whole vocabulary
    :param epochs: how man epochs will run
    :param batch_size: batch size
    :param max_seq_len: the maximum sequence length
    :param mode: which mode will run in train/evaluate/test
    :return: dataset, num_samples
    """
    label_dict = {'task': 0, 'qa': 1, 'chat': 2}

    def get_label_id(row):
        """Convert label text to label id."""
        return label_dict[row.strip()]

    header_names = ["label", "text"]
    data = pd.read_csv(filename, sep='\t', header=None, names=header_names, encoding='utf-8')
    data["label"] = data['label'].apply(get_label_id)
    data = data.dropna()
    # sequence segment and convert token to ids
    data["split"] = data['text'].apply(lambda line: vocab.text_to_ids(line))

    # sequence padding
    padding_id = vocab.vocab['UNK']
    data['inputs'] = data['split'].apply(lambda tokens: padding(tokens, max_seq_len, padding_id, seq_front=False))

    num_samples = data.shape[0]
    # tf.data.dataset
    dataset = tf.data.Dataset.from_tensor_slices((data['inputs'], data['label']))

    if mode == "train":
        dataset = dataset.repeat(epochs)
        dataset.shuffle(num_samples)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, num_samples


def create_single_input(text, vocab, max_seq_len):
    """
    Convert an input text to model input(padded int id sequence)
    :param text: input text
    :param vocab: Vocab object
    :param max_seq_len: the maximum sequence lenght
    :return: padded id(int32) list
    """
    token_ids = vocab.doc_to_ids(text)
    padded_ids = padding(token_ids, max_seq_len, seq_front=False)
    return padded_ids


if __name__ == '__main__':
    filename = "../data/dataset/dialog_all.txt"
    data_dir = "../data/dataset/bot_cli/"
    # data_split(filename, data_dir)
