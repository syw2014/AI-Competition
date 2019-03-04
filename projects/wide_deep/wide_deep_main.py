#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : wide_deep_main.py
# PythonVersion: python3.5
# Date    : 2019/3/1 17:07
# Software: PyCharm

"""Wide and Deep demo module based on tensorflow models. Remove more dependency."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import urllib

import tensorflow as tf


flags = tf.app.flags

# First, download data and clean
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' %(DATA_URL, TRAINING_FILE)

EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' %(DATA_URL, EVAL_FILE)

# set header
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',  'race',
    'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

# define default value,and indicate the type of each columns
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281}


def _download_and_clean_file(filename, url):
    """Download data from url, and makes changes to match the CSV format."""
    tempfile, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(tempfile, 'r') as temp_eval_file, tf.gfile.Open(filename, 'w') as eval_file:
        for line in temp_eval_file:
            line = line.strip()
            line = line.replace(', ', ',')
            if not line or ',' not in line:
                continue
            if line[-1] == '.':
                line = line[:-1]
            line += '\n'
            eval_file.write(line)
        tf.gfile.Remove(tempfile)


def download(data_dir):
    """Download census data if it is not already present."""
    tf.gfile.MakeDirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.gfile.Exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.gfile.Exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)


def build_model_columns():
    """Builds a set of wide and deep feature columns."""

    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns
    base_columns = [education, marital_status, relationship, workclass, occupation, age_buckets]

    # cross feature
    cross_columns = [
        tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'],
                                         hash_bucket_size=_HASH_BUCKET_SIZE)]

    wide_columns = base_columns + cross_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),

        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8)
    ]

    return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """
    Generate an input function for the Estimator.
    :param data_file: string ,input file name
    :param num_epochs: int, number of train/test epochs
    :param shuffle: bool, to indicate shuffle or not
    :param batch_size: int ,batch size
    :return: dataset
    """
    # check file
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have data.' % data_file)

    def parse_csv(value):
        """Parse each line in csv file."""
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels, '>50k') # binary classification
        return features, classes

    # extract lines from input files using dataset api
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # we call repeat after shuffling, rather than before, to prevent separate epochs from blending together
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def define_census_flags():
    flags.DEFINE_string('data_dir', '/data/research/data/tutorial_data/census/', 'data path')
    flags.DEFINE_string('model_dir', '/data/research/data/tutorial_data/census/output/', 'model path')
    flags.DEFINE_string('export_dir', '/data/research/data/tutorial_data/census/export/', 'model path')
    flags.DEFINE_string('model_type', 'wide_deep', 'select model topology, values=[wide, deep, wide_deep]')
    flags.DEFINE_integer('train_epochs', 5, 'number of run epochs')
    flags.DEFINE_integer('batch_size', 100, 'batch size')
    flags.DEFINE_integer('eval_pre_epochs', 1, 'evaluate model after number epochs')


def build_estimator(model_dir, model_type, model_column_fn):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = model_column_fn()

    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which trains faster than GPU for this model
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU':0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


def export_model(model, model_type, export_dir, model_column_fn):
    """Export to SaveModel format.
    """
    wide_columns, deep_columns = model_column_fn()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn,
                            strip_default_attrs=True)


def run_estimator(args, model_column_fn):
    """Run estimator."""
    model = build_estimator(args.model_dir, args.model_type, model_column_fn)
    for n in range(args.train_epochs // args.eval_pre_epochs):
        input_fn_la = lambda: input_fn(args.data_dir+TRAINING_FILE, args.train_epochs, True, args.batch_size)
        model.train(input_fn=input_fn_la)

        input_fn_la = lambda : input_fn(args.data_dir+EVAL_FILE, 1, False, args.batch_size)
        results = model.evaluate(input_fn=input_fn_la)

        # display evaluation metrics
        tf.logging.info('Results at epoch %d / %d', (n+1) * args.eval_pre_epochs, args.train_epochs)
        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

    # Export the model
    if args.export_dir is not None:
        export_model(model, args.model_type, args.export_dir,
                     model_column_fn)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    define_census_flags()
    FLAGS = flags.FLAGS

    # download dataset
    download(FLAGS.data_dir)
    run_estimator(FLAGS, build_model_columns)


if __name__ == "__main__":
    tf.app.run(main)