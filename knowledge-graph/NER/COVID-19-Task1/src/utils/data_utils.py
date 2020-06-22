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
import spacy
from spacy.gold import biluo_tags_from_offsets

spacy_model = spacy.load('en_core_web_sm')


def doc_to_sentences(doc):
    """Cut doc into sentences with !.;?"""
    pat = u'[!.;?]'
    sents = re.split(pat, doc)
    seg_words = re.findall(pat, doc)
    seg_words.insert(len(doc) - 1, "")
    # keep seg words in each sentence
    results = [s + w for s, w in zip(sents, seg_words)]
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


def text_to_bio(tags, tokens):
    """Convert tokens and tags sequence to bio format. From BILUO(begin,in,last,unit,out) to BIO(begin,in,out)"""
    sentence_delimiter = ['?', '\t', '.', '!']
    start_idx = 0
    sents_tags_seq = []
    for idx, e in enumerate(tokens):
        if e in sentence_delimiter:
            t_tags = tags[start_idx: idx + 1]
            new_tags = []
            for x in t_tags:
                if x.startswith('L'):
                    x = 'I-' + x.split('-')[-1]
                elif x.startswith('U'):
                    x = 'B-' + x.split('-')[-1]
                new_tags.append(x)
            t_sent = tokens[start_idx:idx + 1]
            sents_tags_seq.append([t_sent, new_tags])
            start_idx = idx + 1
    return sents_tags_seq


def clean_entity_types(entities):
    starts = {}
    ends = {}
    for e in entities:
        ent_length = e['end'] - e['start']
        # check from entity start index
        if e['start'] not in starts:
            starts[e['start']] = e['entity']
        else:
            if len(starts[e['start']]) < ent_length:
                starts[e['start']] = e['entity']

        # check from entity end index
        if e['end'] not in ends:
            ends[e['end']] = e['entity']
        else:
            if len(ends[e['end']]) < ent_length:
                ends[e['end']] = e['entity']
    temp = list(set(starts.values()).intersection(set(ends.values())))
    res = []
    for x in entities:
        if x['entity'] in temp:
            res.append((x['start'], x['end'], x['type']))
    return res


def data_split(filename, output_dir):
    """
    Split data into train/dev/test,
    :param filename: input data file name
    :param output_dir: output directory
    :return:
    """
    dataset = []
    # Here we cut text into sentences, mean while convert entity to BIO format
    with open(filename, 'r', encoding='utf-8') as f, open('error.txt', 'w', encoding='utf-8') as f2:
        for line in tqdm(f.readlines()):
            jdata = json.loads(line.strip())
            doc = spacy_model(jdata['text'])
            tokens = [w.text for w in doc]
            entities = clean_entity_types(jdata['entities'])
            try:
                tags = biluo_tags_from_offsets(doc, entities)
            except ValueError:
                f2.write(line)
            data = text_to_bio(tags, tokens)
            dataset.extend(data)

    # split data
    train, dev, test = train_test_split(dataset, len(dataset))
    seq_len = {}
    with open(output_dir + 'train.txt', 'w', encoding='utf-8') as f:
        for x in train:
            tokens_str = ' '.join(x[0])
            tag_str = ' '.join(x[1])
            f.write(tokens_str + '==' + tag_str + '\n')
            if len(x[1]) not in seq_len:
                seq_len[len(x[1])] = 1
            else:
                seq_len[len(x[1])] += 1

    with open(output_dir + 'dev.txt', 'w', encoding='utf-8') as f:
        for x in test:
            tokens_str = ' '.join(x[0])
            tag_str = ' '.join(x[1])
            f.write(tokens_str + '\t' + tag_str + '\n')
            if len(x[1]) not in seq_len:
                seq_len[len(x[1])] = 1
            else:
                seq_len[len(x[1])] += 1

    with open(output_dir + 'test.txt', 'w', encoding='utf-8') as f:
        for x in dev:
            tokens_str = ' '.join(x[0])
            tag_str = ' '.join(x[1])
            f.write(tokens_str + '\t' + tag_str + '\n')
            if len(x[1]) not in seq_len:
                seq_len[len(x[1])] = 1
            else:
                seq_len[len(x[1])] += 1
    with open(output_dir+'seq_len.txt', 'w', encoding='utf-8') as f:
        for k,v in seq_len.items():
            f.write(str(k) + '\t' + str(v) + '\n')


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
    # label_dict = {'task': 0, 'qa': 1, 'chat': 2}

    def get_label_id(row):
        """Convert label text to label id."""
        return vocab.get_seq_labels(row)

    header_names = ["text", "cate"]
    data = pd.read_csv(filename, sep='==', header=None, names=header_names, encoding='utf-8')
    data["cate"] = data['cate'].apply(get_label_id)
    data = data.dropna()
    # sequence segment and convert token to ids
    data["split"] = data['text'].apply(lambda line: vocab.text_to_ids(line))

    # sequence padding
    padding_id = vocab.vocab['UNK']
    data['inputs'] = data['split'].apply(lambda tokens: padding(tokens, max_seq_len, padding_id, seq_front=False))

    # label sequence padding
    padding_id = vocab.labels['O']
    data['labels'] = data['cate'].apply(lambda label_ids: padding(label_ids, max_seq_len, padding_id, seq_front=False))

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
    :param max_seq_len: the maximum sequence length
    :return: padded id(int32) list
    """
    token_ids = vocab.doc_to_ids(text)
    padded_ids = padding(token_ids, max_seq_len, "", seq_front=False)
    return padded_ids


if __name__ == '__main__':
    data_dir = '../../data/task_1/'
    filename = data_dir + 'task1_train_correct.json'
    data_split(filename, data_dir)
