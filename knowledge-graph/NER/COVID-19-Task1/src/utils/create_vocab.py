#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : create_vocab.py
# PythonVersion: python3.6
# Date    : 2020/6/9 16:17
# Software: PyCharm
"""Create vocabulary with text data for domain classification."""
import json, os
from collections import Counter
import jieba
from tqdm import tqdm
try:
    from opencc import OpenCC
except ImportError:
    print("Should run `pip install opencc-python-reimplemented` to install opencc package")


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
        self.vocab = {} # word -> id
        self.reverse_vocab = None # id->word
        self.counter = Counter() # counter for word

        # load stop words
        if self.stopwords_file is not None:
            self.stopwords = [line.strip().lstrip('#')for line in
                          open(self.stopwords_file, 'r', encoding='utf-8').readlines()]
        else:
            self.stopwords = []

    def create_count(self, filename, remove_stopwords=True):
        """
        Load input data file and
        :param filename: input text file name, here assign each line in file was a json like {"query":text,..}
                        we only process query
        :return:
        """
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                jdata = json.loads(line.strip())
                if "query" not in jdata:
                    continue
                text = jdata['query']
                # tokenize
                tokens = list(jieba.cut(text.lower()))
                if remove_stopwords:
                    tokens = [w for w in tokens if w not in self.stopwords]
                # remove empty
                if len(tokens) == 0:
                    continue
                self.counter.update(tokens)
        print("Found {} tokens input file {}".format(len(self.counter), filename))

    def create_vocab(self):
        """
        Create vocabulary.
        :return:
        """
        words = self.counter.most_common(self.vocab_size)
        # TODO, here can filter words with term frequency
        words = ['UNK'] + [w for (w, c) in words]
        self.vocab = dict(zip(words, range(len(words))))
        # self.vocab['UNK'] = len(self.vocab)
        self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
        print("Create {} words in vocabulary".format(len(self.vocab)))

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

    def load_vocab(self):
        """
        Load vocabulary from vocab.json in self.vocab_dir
        :return:
        """
        with open(self.vocab_dir+'vocab.json', 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
            print("Load {} words from {}.".format(len(self.vocab), self.vocab_dir+'vocab.json'))

    def seq_to_ids(self, token_seq):
        """
        Convert token sequence to id sequence
        :param token_seq: input token list
        :return: int32 id list
        """
        # TODO, can use id of UNK instead of word not in vocab
        token_ids = [self.vocab[w] for w in token_seq if w in self.vocab]
        return token_ids

    def text_to_ids(self, text, isSegment=True, remove_stopwords=True):
        """
        Convert text to int32 id list
        :param text: input text.
        :return: int32 id list
        """
        tokens = text
        if isSegment:
            tokens = list(jieba.cut(text.lower()))
        if remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]
        token_ids = self.seq_to_ids(tokens)

        return token_ids
    
    def get_vocab_size(self):
        return len(self.vocab)


if __name__ == '__main__':
    data_dir = '../data/dataset/'
    stopwords_file = data_dir + 'stopwords.txt'
    vocab = Vocab(stopwords_file, vocab_dir=data_dir)
    data = data_dir + 'text_for_vocab.txt'
    vocab.create_count(data)
    vocab.create_vocab()
    vocab.save_vocab()
