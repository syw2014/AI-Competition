#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : data_analysis.py
# PythonVersion: python3.6
# Date    : 2020/6/14 18:38
# Software: PyCharm
"""Files to analyze the data information like entity types, entities, documents length,
entity distribution, and split train file into train and dev."""

import json
from tqdm import tqdm


def data_dist():
    data_dir = "./data/task_1/"
    train_file = data_dir + "task1_train.json"
    test_file = data_dir + "task1_valid_noAnswer.json"

    entities = {}
    doc_types_cnt = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            jdata = json.loads(line.strip())
            for e in jdata['entities']:
                if e['type'] in entities:
                    entities[e['type']].append(e['entity'])
                else:
                    entities[e['type']] = [e['entity']]
                # count type document
                if e['type'] in doc_types_cnt:
                    doc_types_cnt[e['type']] += 1
                else:
                    doc_types_cnt[e['type']] = 1
    print("doc types cnt->{}".format(doc_types_cnt))
    # print(entities)
    with open(data_dir+'entity.txt', 'w', encoding='utf-8') as f:
        for t, e in entities.items():
            for ee in list(set(e)):
                f.write(t + '\t' + ee +'\n')


if __name__ == '__main__':
    data_dist()