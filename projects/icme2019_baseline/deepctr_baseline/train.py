import pandas as pd
from deepctr import SingleFeat, VarLenFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import xDeepFM_MTL
from tqdm import tqdm
import json
import numpy as np

ONLINE_FLAG = True
loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

DATA_DIR = "/data/research/data/short_video/icme2019/"
train_file = DATA_DIR + "/final_track2_train.txt.test"
test_file = DATA_DIR + "/final_track2_test_no_anwser.txt.test"
title_file = DATA_DIR + "/track2_title.txt"

if __name__ == "__main__":
    data = pd.read_csv(train_file, sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did',
        'create_time', 'video_duration'])

    # TODO, use video title
    _NUM_WORDS = 0
    _MAXLEN = 10
    title_len_list = []
    title_list = []
    with open(title_file) as f:
        for line in f.readlines():
            jdata = json.loads(line)
            token_seq = []
            for k, v in jdata['title_features'].items():
                token_seq += [int(k)] * v
            if len(token_seq) == 0:
                continue
            jdata['title_features'] = token_seq
            title_list.append(jdata)
            _NUM_WORDS += len(jdata['title_features'])
            title_len_list.append(len(token_seq))
    # _MAX_LEN = np.array(title_len_list).mean()
    # title data frame
    title_df = pd.DataFrame.from_dict(title_list)
    print("Load title data completed!")
    print(title_df[:3])
    print(title_df.dtypes)

    if ONLINE_FLAG:
        test_data = pd.read_csv(test_file, sep='\t', names=[
                                'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like',
            'music_id', 'did', 'create_time', 'video_duration'])
        train_size = data.shape[0]
        data = data.append(test_data)
    else:
        train_size = int(data.shape[0]*(1-VALIDATION_FRAC))

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did',]
    dense_features = ['video_duration',  'create_time']  # 'creat_time'

    # TODO, combine with original dataframe, sequence
    data = pd.merge(data, title_df, how='left', on='item_id')
    sequence_features = ['title_features']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    # process nan
    for row in data.loc[data.title_features.isnull(), 'title_features'].index:
        data.at[row, 'title_features'] = [0]

    # Padding
    # for feat in sequence_features:
    #     # print(feat)
    #     # print(data[feat])
    #     data[feat] = data[feat].apply(lambda x: x + [0]*(_MAXLEN - len(x)) if _MAXLEN > len(x) else x[:_MAXLEN])

    target = ['finish', 'like']

    print(data[:3])

    for feat in sparse_features:
        # 将value转为id
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 统计feature，和feature维度，对于稀疏feature，维度就是独立元素个数， 对于dense feature， 维度是0
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    # sequence feature, dimension was the vocabulary size,
    sequence_feature_list = [VarLenFeat(feat, _NUM_WORDS+1, _MAXLEN, 'mean') for feat in sequence_features]

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    print(train['title_features'].values.tolist())

    # Padding
    train_seq_padded = pad_sequences(train['title_features'].values, maxlen=_MAXLEN, padding='post')
    # print(train_seq_padded)
    test_seq_padded = pad_sequences(test['title_features'].values, maxlen=_MAXLEN, padding='post')
    # print(test_seq_padded)

    # 构造训练数据
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list ] + \
        [train_seq_padded]

    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list] + \
                       [test_seq_padded]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    # Todo ,add text sequence features
    print("Feed data into model.")

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list,
                         "sequence": sequence_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights,)

    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2**14)

    else:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1, validation_data=(test_model_input, test_labels))

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            DATA_DIR + '/ouput/xdeepfm/result.csv', index=None, float_format='%.6f')
