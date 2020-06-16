#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : entity_search.py
# PythonVersion: python3.6
# Date    : 2020/6/15 10:59
# Software: PyCharm
"""Use string matching to extract entities"""

import ahocorasick as ahc
import os, json, pickle
import jieba

class EntityExtract(object):
    def __init__(self, entity_file=None):
        self.ahc = ahc.Automaton()
        self.type_to_id = {} # store entity type -> type id
        self.entity_file = entity_file

    def build_dict(self):
        """Create Trie and build automaton"""
        with open(self.entity_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split('\t')
                # add word to jieba
                jieba.add_word(arr[1])
                # assign idx for type
                if arr[0] in self.type_to_id:
                    idx = self.type_to_id[arr[0]]
                else:
                    idx = len(self.type_to_id)
                    self.type_to_id[arr[0]] = idx
                self.ahc.add_word(arr[1], (idx, arr[1]))
        # build
        self.ahc.make_automaton()
        print("Build automaton completed!")

    def search(self, text):
        entities = []
        if len(text) == 0:
            return entities
        all = {}
        for idx, (end_index, word) in self.ahc.iter(text.strip()):
            all[word] = [end_index, idx]
        # the longest matching
        print(all)
        res = set()
        for e in all.keys():
            tmp = ""
            for x in all.keys():
                if x.find(e) != -1 and x != e:
                    tmp = x
                    break
            if tmp != "":
                continue
            else:
                res.add(e)
        # process results
        for e in list(res):
            if e in all:
                tmp = {"entity":e, "start": all[e][0]-len(e), "end": all[e][0]}
                entities.append(tmp)

        return entities

def predict():
    pass

if __name__ == '__main__':
    filename = "../data/task_1/entity.txt"
    text = "Health security in 2014: building on preparedness knowledge for emerging health threats\tIdeas, information," \
           " and microbes are shared worldwide more easily than ever before. New infections, such as the novel influenza" \
           " A H7N9 or Middle East respiratory syndrome coronavirus, pay little heed to political boundaries as they spread; " \
           "nature pays little heed to destruction wrought by increasingly frequent natural disasters. Hospital-acquired " \
           "infections are hard to prevent and contain, because the bacteria are developing resistance to the therapeutic " \
           "advances of the 20th century. Indeed, threats come in ever-complicated combinations: a combined earthquake, " \
           "tsunami, and radiation disaster; blackouts in skyscrapers that require new thinking about evacuations and " \
           "medically fragile populations; or bombings that require as much psychological profiling as chemical profiling."
    ahc = EntityExtract(filename)
    ahc.build_dict()
    res = ahc.search(text)
    print(res)