#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : demo.py
# PythonVersion: python3.5
# Date    : 2019/3/8 14:51
# Software: PyCharm

import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr import VarLenFeat, SingleFeat


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


data = pd.read_csv("/data/research/data/short_video/icme2019/movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
target = ['rating']

# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# preprocess the sequence feature

key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
print('->', len(key2index))
print(type(genres_list))

genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post',)
print(type(genres_list))

# 2.count #unique features for each sparse field and generate feature config for sequence feature

sparse_feat_list = [SingleFeat(feat, data[feat].nunique())
                    for feat in sparse_features]
print('->', len(key2index))

sequence_feature = [VarLenFeat('genres', len(
    key2index) + 1, max_len, 'mean')]  # Notice : value 0 is for padding for sequence input feature

# 3.generate input data for model
sparse_input = [data[feat.name].values for feat in sparse_feat_list]
dense_input = []
sequence_input = [genres_list]
print(sequence_feature)

model_input = sparse_input + dense_input + \
    sequence_input  # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM({"sparse": sparse_feat_list,
                "sequence": sequence_feature}, final_activation='linear')

model.compile("adam", "mse", metrics=['mse'],)
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2,)