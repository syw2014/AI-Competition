#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : create_vocab.py
# PythonVersion: python3.6
# Date    : 2020/6/9 16:17
# Software: PyCharm
"""Create vocabulary with text data for domain classification."""
import json
import os
from collections import Counter

from tqdm import tqdm

try:
    from opencc import OpenCC
except ImportError:
    print("Should run `pip install opencc-python-reimplemented` to install opencc package")
import spacy


def is_digit(word):
    word = word.replace('.', '').replace('-', '').replace('%', '')
    return word.isdigit()


class Vocab(object):
    def __init__(self, stopwords_file=None, vocab_size=None, vocab_dir=None):
        """
        Vocab class to create vocabulary and store vocab in vocab.json file
        :param stopwords_file: input stopwords file name
        :param vocab_size: the size of vocabulary will be keep
        :param vocab_dir: the directory to store vocab.json file
        """
        self.stopwords_file = stopwords_file
        self.vocab_size = vocab_size
        self.vocab_dir = vocab_dir
        self.vocab = {}  # word -> id
        self.reverse_vocab = None  # id->word
        self.counter = Counter()  # counter for word
        self.label_to_id = {'O': 0}  # label->id
        self.id_to_label = {}  # id->label
        self.seqs_label_prefix = ['B', 'I']  # use BIO as sequence label
        self.spacy_model = spacy.load('en_core_web_sm')

        # load stop words
        if self.stopwords_file is not None:
            self.stopwords = [line.strip() for line in
                              open(self.stopwords_file, 'r', encoding='utf-8').readlines()
                              if not line.startswith('#')]
        else:
            self.stopwords = []

    def add_line(self, text, remove_stopwords=True):
        """
        Load input data file and
        :param text: input text
        :return:
        """
        # TODO, here we spacy not jieba as spacy supply more tools for english processing
        # tokens = list(jieba.cut(text))
        doc = self.spacy_model(text)
        tokens = [w.text for w in doc]
        if remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]
        # # filter all digit
        tokens = [w for w in tokens if not is_digit(w)]
        # remove empty
        if len(tokens) != 0:
            self.counter.update(tokens)

    def add_label(self, label):
        for pre in self.seqs_label_prefix:
            if pre + '-' + label not in self.label_to_id:
                id = len(self.label_to_id)
                self.label_to_id[pre + '-' + label] = id

    def create_vocab(self):
        """
        Create vocabulary.
        :return:
        """
        words = self.counter.most_common(self.vocab_size)
        # TODO, here can filter words with term frequency
        # add unknown and number char
        words = ['UNK'] + [w for (w, c) in words] + ['NUM']
        self.vocab = dict(zip(words, range(len(words))))
        self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
        print("Create {} words in vocabulary".format(len(self.vocab)))
        self.id_to_label = dict(zip(self.label_to_id.values(), self.label_to_id.keys()))
        print("Create {} labels from given dataset".format(len(self.label_to_id)))

    def save_vocab(self):
        """
        Write vocab.json file to self.vocab_dir
        :return:
        """
        if len(self.vocab) != 0 and os.path.exists(self.vocab_dir):
            with open(self.vocab_dir + 'vocab.json', 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, indent=4, ensure_ascii=False)
        else:
            raise Exception("No vocabulary generation or vocab_dir not exists")

        if len(self.label_to_id) != 0 and os.path.exists(self.vocab_dir):
            with open(self.vocab_dir + 'labels.json', 'w', encoding='utf-8') as f:
                json.dump(self.label_to_id, f, indent=4, ensure_ascii=False)
        else:
            raise Exception("No labels generation or vocab_dir not exists")

    def load_vocab_label(self):
        """
        Load vocabulary from vocab.json in self.vocab_dir
        :return:
        """
        with open(self.vocab_dir + 'vocab.json', 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
            print("Load {} words from {}.".format(len(self.vocab), self.vocab_dir + 'vocab.json'))

        with open(self.vocab_dir + 'labels.json', 'r', encoding='utf-8') as f:
            self.label_to_id = json.load(f)
            self.id_to_label = dict(zip(self.label_to_id.values(), self.label_to_id.keys()))
            print("Load {} words from {}.".format(len(self.label_to_id), self.vocab_dir + 'labels.json'))

    def seq_to_ids(self, token_seq):
        """
        Convert token sequence to id sequence
        :param token_seq: input token list
        :return: int32 id list
        """
        # TODO, can use id of UNK instead of word not in vocab
        token_ids = [self.vocab[w] for w in token_seq if w in self.vocab]
        return token_ids

    def text_to_ids(self, text, isSegment=True, remove_stopwords=False):
        """
        Convert text to int32 id list
        :param text: input text.
        :return: int32 id list
        """

        if isSegment:
            # tokens = list(jieba.cut(text))
            doc = self.spacy_model(text)
            tokens = [w.text for w in doc]
        else:
            tokens = text.split(' ')
        # filter all digit
        tokens = [w for w in tokens if not is_digit(w)]
        if remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]
        token_ids = self.seq_to_ids(tokens)

        return token_ids

    def get_vocab_size(self):
        return len(self.vocab)

    def get_label_size(self):
        return len(self.label_to_id)

    def label_to_id(self, label):
        return self.label_to_id[label]

    def get_label(self, id):
        return self.id_to_label[id]

    def get_seq_labels(self, label_str):
        print(label_str)
        seq_label_ids = [self.label_to_id[x] for x in label_str.split(' ')]
        return seq_label_ids


if __name__ == '__main__':
    data_dir = '../../data/task1_public/'
    stopwords_file = data_dir + 'stopwords.txt'
    vocab = Vocab(stopwords_file, vocab_dir=data_dir)
    data = data_dir + 'new_train.json'
    with open(data, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            jdata = json.loads(line.strip())
            text = jdata['text']
            arr = text.split('\t')
            vocab.add_line(arr[0], False)
            vocab.add_line(arr[1], False)
            for e in jdata['entities']:
                vocab.add_label(e['type'])
    vocab.create_vocab()
    vocab.save_vocab()
