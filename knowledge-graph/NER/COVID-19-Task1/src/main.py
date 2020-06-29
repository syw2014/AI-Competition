#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : main.py
# PythonVersion: python3.6
# Date    : 2020/6/22 11:32
# Software: PyCharm
"""Program main entrance, can run in train/evaluate/predict model."""

import argparse
import time

from utils.create_vocab import *
from utils.data_utils import *
from net.metrics import *
from utils.metrics import *
from model import BiLSTMCRF

import tensorflow_addons as tfa

parser = argparse.ArgumentParser()

# define files or file directory
parser.add_argument("--train_data", default='../data/task_1/dev.txt', type=str,
                    help="input training file")
parser.add_argument("--dev_data", default='../data/task_1/dev.txt', type=str,
                    help="input evaluate file")
parser.add_argument("--test_data", default='../data/task_1/test.txt', type=str,
                    help="input predict file")
parser.add_argument("--vocab_dir", default='../data/task_1/', type=str,
                    help="input vocabulary file directory")
parser.add_argument("--model_dir", default='../result/model/', type=str,
                    help="output model directory")
parser.add_argument("--summary_dir", default='../result/model/', type=str,
                    help="output summary directory")

# hyper-parameters
parser.add_argument("--max_seq_len", default=100,
                    help="output summary directory")
parser.add_argument("--embedding_dim", default=100, type=int,
                    help="word embedding dimension")
parser.add_argument("--hidden_num", default=512, type=int,
                    help="hidden size")
parser.add_argument("--epochs", default=1, type=int,
                    help="run epochs")
parser.add_argument("--batch_size", default=16, type=int,
                    help="train/dev batch size")
parser.add_argument("--learning_rate", default=0.01, type=float,
                    help="train learning rate")
parser.add_argument("--mode", default="train", type=str,
                    help="which mode will be run in train/evaluate/predict")


args = parser.parse_args()


def get_timestamp():
    now = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    return timestamp


@tf.function
def train_step(inputs, labels, model, run_loss, optimizer, is_training=False):
    """
    Run step for model
    """
    with tf.GradientTape() as tape:
        y_pred, seq_len, log_likelihood = model(inputs, labels, training=is_training)
        loss = -tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # run_loss.update_state(loss)

    return loss, y_pred, seq_len


@tf.function
def test_step(inputs, labels, model, run_loss, is_training=False):
    """
    Run test/evaluate step
    """
    with tf.GradientTape() as tape:
        y_pred, seq_len, log_likelihood = model(inputs, labels, training=is_training)
        loss = -tf.reduce_mean(log_likelihood)

    # run_loss.update_state(loss)
    return loss, y_pred, seq_len


def cal_acc_one_step(model, logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy


def train_and_eval(model, train_dataset, num_train_samples, eval_dataset, num_eval_samples,
                   optimizer,
                   loss_object,
                   params,
                   manager,
                   summary_writer):
    """
    Main entrance to train and evaluate Bi-LSTM+CRF model
    """
    # train model with tf.GradientTape
    train_loss = tf.metrics.Mean()
    # train_accuracy = tf.metrics.SparseCategoricalAccuracy()
    # val_loss = tf.metrics.Mean()
    # val_accuracy = tf.metrics.SparseCategoricalAccuracy()
    #
    # sparse_pre_score = SparsePrecisionScore(average="macro")
    # sparse_recall = SparseRecallScore(average="macro")
    # sparse_acc_score = SparseAccuracyScore()
    # sparse_f1 = SparseF1Score(average="macro")

    steps_per_epoch = num_train_samples // args.batch_size
    # if last checkpoint was exist, then get evaluate first
    if manager.latest_checkpoint:
        best_acc = evaluate(model, eval_dataset, num_eval_samples, optimizer, loss_object, params)
    else:
        best_acc = 0.0

    with summary_writer.as_default():
        for epoch in range(args.epochs):
            train_loss.reset_states()
            # train_accuracy.reset_states()

            for (batch_idx, (inputs, labels)) in enumerate(train_dataset.take(steps_per_epoch)):
                time_s = time.time()
                train_loss, pred, seq_len = train_step(inputs, labels, model, train_loss, optimizer, True)
                time_e = time.time()
                train_accuracy = cal_acc_one_step(model, pred, seq_len, labels)
                # write to summary file
                # tf.summary.scalar("train_loss", train_loss.result().numpy(), step=batch_idx)
                # tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch_idx)
                # tf.summary.scalar("learning_rate", params.learning_rate, step=batch_idx)
                # summary_writer.flush()

                print(
                    "{} INFO: Train batch:{}/{}\tloss:{:.4f}\tacc:{:.4f} time:{:.4f}s"
                        .format(get_timestamp(), batch_idx + epoch * steps_per_epoch, params.epochs * steps_per_epoch,
                                train_loss.result().numpy(),
                                train_accuracy,
                                (time_e - time_s)))
                # evaluate model after steps
                # if batch_idx % 100 == 0:
                #     val_loss.reset_states()
                #     val_accuracy.reset_states()
                #     steps_in_eval = num_eval_samples // args.batch_size
                #     # define metrics
                #     acc_score = []
                #     prec_score = []
                #     recall = []
                #     f1 = []
                #     for (eval_batch_idx, (eval_inputs, eval_labels)) in enumerate(eval_dataset.take(steps_in_eval)):
                #         _, prediction = test_step(eval_inputs, eval_labels, model, val_loss,
                #                                   val_accuracy, loss_object)
                #         tf.summary.scalar("evaluate_loss", val_loss.result().numpy(), step=eval_batch_idx)
                #         tf.summary.scalar("evaluate_accuracy", val_accuracy.result().numpy(), step=eval_batch_idx)
                #         summary_writer.flush()
                #         acc_score.append(sparse_acc_score(eval_labels, prediction))
                #         prec_score.append(sparse_pre_score(eval_labels, prediction))
                #         recall.append(sparse_recall(eval_labels, prediction))
                #         f1.append(sparse_f1(eval_labels, prediction))
                #     print("{} INFO: Evaluate loss:{:.4f}\t accuracy:{:.4f}".format(get_timestamp(),
                #                                                                    val_loss.result().numpy(),
                #                                                                    val_accuracy.result().numpy()))
                #     print("{} INFO: Evaluate f1:{:.4f}\tacc:{:.4f}\tprecision:{:.4f}\trecall:{:.4f}".format(
                #         get_timestamp(), np.mean(f1), np.mean(acc_score), np.mean(prec_score), np.mean(recall)))
                #
                #     if val_accuracy.result().numpy() > best_acc:
                #         manager.save(checkpoint_number=epoch * steps_per_epoch + batch_idx)
                #         best_acc = val_accuracy.result().numpy()
                #         print("Model saved!")
                #         print(
                #             "{} INFO: Found best metrics f1:{:.4f}\tacc:{:.4f}\tprecision:{:.4f}\trecall:{:.4f}"
                #                 .format(get_timestamp(), np.mean(f1), np.mean(acc_score), np.mean(prec_score),
                #                         np.mean(recall)))


def evaluate(model, dev_dataset, num_dev_samples, loss_object, args):
    """
    Evaluate the trained model.
    :param model:
    :param dev_dataset:
    :param num_dev_samples:
    :param loss_object:
    :param args:
    :return:
    """
    val_loss = tf.metrics.Mean()
    val_accuracy = tf.metrics.SparseCategoricalAccuracy()
    steps_in_eval = num_dev_samples // args.batch_size
    # define metrics
    sparse_pre_score = SparsePrecisionScore(average="macro")
    sparse_recall = SparseRecallScore(average="macro")
    sparse_acc_score = SparseAccuracyScore()
    sparse_f1 = SparseF1Score(average="macro")
    acc_score = []
    prec_score = []
    recall = []
    f1 = []
    for (batch_idx, (inputs, labels)) in enumerate(dev_dataset.take(steps_in_eval)):
        _, prediction = test_step(inputs, labels, model, val_loss, val_accuracy, loss_object)
        acc_score.append(sparse_acc_score(labels, prediction))
        prec_score.append(sparse_pre_score(labels, prediction))
        recall.append(sparse_recall(labels, prediction))
        f1.append(sparse_f1(labels, prediction))
    print("{} INFO: Evaluate loss:{:.4f}\t accuracy:{:.4f}".format(get_timestamp(), val_loss.result().numpy(),
                                                                   val_accuracy.result().numpy()))
    print("{} INFO: Evaluate f1:{:.4f}\tacc:{:.4f}\tprecision:{:.4f}\trecall:{:.4f}".format(
        get_timestamp(), np.mean(f1), np.mean(acc_score), np.mean(prec_score), np.mean(recall)))
    return np.mean(acc_score)


def predict_doc(model, vocab, text, args):
    """
    Evaluate the trained model.
    :param model:
    :param vocab:
    :param text:
    :param args:
    :return:
    """
    # convert text to input id list
    token_ids = vocab.text_to_ids(text)
    input = padding(token_ids, args.max_seq_len, vocab.vocab['UNK'])
    with tf.GradientTape() as tape:
        output = model.predict(np.array([input]), False)
    # parse probability and label id
    prediction = output.numpy()
    print(prediction)
    id_to_label = {0: "task", 1: "qa", 2: "chat"}
    idx = np.argmax(prediction)
    score = round(prediction[-1][idx], 4)
    result = {"label": id_to_label[idx], "score": score}
    print(result)
    return result


def predict_batch(model, vocab, filename, outfile, args):
    """
    Evaluate the trained model.
    :param model:
    :param vocab:
    :param filename:
    :param outfile:
    :param args:
    :return:
    """
    # convert text to input id list
    # create predict dataset
    # Note: in order to keep the same process, predict file format must be <sample_id\t text>, sample_id was
    # placeholder, in train/evaluate mode it's the real label
    pred_dataset, num_pred_data = create_dataset_with_tf(filename, vocab,
                                                         args.epochs, args.batch_size, args.max_seq_len,
                                                         "predict")
    # get prediction
    steps_pre_epoch = num_pred_data // args.batch_size
    pred_ids = []
    probs = []
    ids = []
    for batch_ids, (inputs, sample_ids) in enumerate(pred_dataset.take(steps_pre_epoch)):
        with tf.GradientTape() as tape:
            output = model.predict(inputs, False)
        pred_ids.extend(np.argmax(output.numpy(), axis=1))
        probs.extend(np.max(output.numpy(), axis=1))
        ids.extend(sample_ids.numpy())

    # parse predictions
    id_to_label = {0: "task", 1: "qa", 2: "chat"}

    # sample_id, predict id, score
    assert len(ids) == len(pred_ids) == len(probs)
    with open(outfile, 'w', encoding='utf-8') as f:
        for id, e in enumerate(ids):
            f.write(str(e) + '\t' + str(pred_ids[id]) + '\t' + id_to_label[pred_ids[id]] + '\t' + str(
                round(probs[id], 4)) + '\n')


def main():
    # load vocab
    vocab = Vocab(stopwords_file=args.vocab_dir + 'stopwords.txt', vocab_dir=args.vocab_dir)
    vocab.load_vocab_label()
    vocab_size = vocab.get_vocab_size()
    label_size = vocab.get_label_size()

    # load pre-trained word embedding
    # embeddings_index = {}
    # embedding_matrix = {}
    # if args.w2v_file is not None:
    #     with open(args.w2v_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             arrs = line.split()
    #             if len(arrs) == 2:
    #                 continue
    #             w = arrs[0]
    #             vec = np.asarray(arrs[1:], dtype='float32')
    #             embeddings_index[w] = vec
    #     print('{} INFO: Use pre-train word embedding , Found {} word vectors'.format(
    #         get_timestamp(), len(embeddings_index)))
    #
    #     # convert embedding to weights
    #     embedding_matrix = np.zeros((vocab_size, args.embedding_dim))
    #     for word, idx in vocab.vocab.items():
    #         if word in embeddings_index:
    #             embedding_matrix[idx] = embeddings_index[word]
    #
    # # define tf train summary writer
    # summary_writer = tf.summary.create_file_writer(args.summary_dir)

    # load data
    train_dataset, num_train_samples = create_dataset_with_tf(args.train_data, vocab,
                                                              args.epochs,
                                                              args.batch_size,
                                                              args.max_seq_len,
                                                              args.mode)
    dev_dataset, num_dev_samples = create_dataset_with_tf(args.test_data, vocab, 1,
                                                          args.batch_size,
                                                          args.max_seq_len,
                                                          "evaluate")

    model = BiLSTMCRF(args.hidden_num, vocab_size, label_size, args.embedding_dim,
                      args.max_seq_len,
                      weights=None,
                      weights_trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, decay=0.0)
    steps_per_epoch = num_train_samples // args.batch_size
    train_loss = tf.metrics.Mean()
    for epoch in range(args.epochs):
        train_loss.reset_states()
        # train_accuracy.reset_states()

        for (batch_idx, (inputs, labels)) in enumerate(train_dataset.take(steps_per_epoch)):
            time_s = time.time()
            train_loss, pred, seq_len = train_step(inputs, labels, model, train_loss, optimizer, True)
            time_e = time.time()
            train_accuracy = cal_acc_one_step(model, pred, seq_len, labels)
            # write to summary file
            # tf.summary.scalar("train_loss", train_loss.result().numpy(), step=batch_idx)
            # tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch_idx)
            # tf.summary.scalar("learning_rate", params.learning_rate, step=batch_idx)
            # summary_writer.flush()

            print(
                "{} INFO: Train batch:{}/{}\tloss:{:.4f}\tacc:{:.4f} time:{:.4f}s"
                    .format(get_timestamp(), batch_idx + epoch * steps_per_epoch, args.epochs * steps_per_epoch,
                            train_loss,
                            train_accuracy,
                            (time_e - time_s)))

    # define loss
    # TODO user defined loss and metrics
    # sparse_loss = MaskSparseCategoricalCrossentropy()

    # train model with tf.GradientTape
    # define checkpoint manager
    # ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # manager = tf.train.CheckpointManager(
    #     ckpt, directory=args.model_dir, max_to_keep=3)
    # # load last checkpoint
    # ckpt.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    #     print("Restore checkpoint from {}".format(manager.latest_checkpoint))
    # elif args.mode in ["evaluate", "predict"]:
    #     raise Exception("No checkpoint found in {}".format(manager.latest_checkpoint))
    # else:
    #     print("No checkpoint found, initializing from scratch.")

    # if args.mode == 'train':
    #     train_and_eval(model, train_dataset, num_train_samples, dev_dataset, num_dev_samples,
    #                    optimizer,
    #                    sparse_loss,
    #                    args,
    #                    manager,
    #                    summary_writer)
    # elif args.mode == 'evaluate':
    #     evaluate(model, dev_dataset, num_dev_samples, optimizer, sparse_loss, args)
    # elif args.mode == "predict":
    #     # predict
    #     text = "我的女友好象不开窍！不懂男女之欢。不和我配合"
    #     # predict_doc(model, vocab, text, args)
    #     # predict_doc(model, vocab, "播放让我们荡起双桨", args)
    #     predict_batch(model, vocab, args.test_data, args.result_dir + "/result.txt", args)
    # else:
    #     raise Exception("Unknown mode {} found, only run in train/evaluate/predict".format(args.mode))

    # # TODO, use tf.keras.Model.fit, compile
    # # training model with tf.keras.Model.fit
    # model.compile(optimizer=optimizer,
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.summary()
    # model.fit(train_dataset, verbose=1, batch_size=args.batch_size,
    #           epochs=args.epochs,
    #           validation_data=dev_dataset,
    #           shuffle=True,
    #           workers=4,
    #           use_multiprocessing=False)
    # model.evaluate(dev_dataset, verbose=1)


if __name__ == "__main__":
    main()
