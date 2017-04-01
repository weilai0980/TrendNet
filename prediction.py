# -*- coding: utf-8 -*-
"""use various model to do the prediction."""
import os
from os.path import join
import datetime

import numpy as np
import tensorflow as tf

import settings.parameters as para
import utils.formData as form_data
from utils.myprint import myprint
from utils.visualizeLabels import visualize_histogram

from model.linearRegression import LinearRegression
from model.cnn import CNN
from model.lstm import LSTM
from model.mixNN import MixNN


def evaluate_data(train_labels, test_labels, mapping, out_path):
    """evaluate the dataset."""
    train_labels = train_labels * (
        mapping["max_labels"] - mapping["min_labels"]) + mapping["min_labels"]
    scope = train_labels[:, 0]
    min_scope = np.min(scope)
    max_scope = np.max(scope)
    length = train_labels[:, 1]
    min_length = np.min(length)
    max_length = np.max(length)
    myprint(
        "Train: max scope={}, min scope={}, max length={}, min length={}".
        format(max_scope, min_scope, max_length, min_length))
    visualize_histogram(train_labels, join(out_path, "histogram_train"))

    test_labels = test_labels * (
        mapping["max_labels"] - mapping["min_labels"]) + mapping["min_labels"]
    scope = test_labels[:, 0]
    min_scope = np.min(scope)
    max_scope = np.max(scope)
    length = test_labels[:, 1]
    min_length = np.min(length)
    max_length = np.max(length)
    myprint(
        "Test: max scope={}, min scope={}, max length={}, min length={}".
        format(max_scope, min_scope, max_length, min_length))
    visualize_histogram(test_labels, join(out_path, "histogram_test"))


def run(MODEL, data):
    """setup the model and train the model."""
    train_data, train_labels, \
        val_data, val_labels, \
        test_data, test_labels, mapping = form_data.normalize_data(data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=para.ALLOW_SOFT_PLACEMENT,
            log_device_placement=para.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # init model
            model = MODEL()
            # build the model.
            model.inference()
            model.loss()
            # Define Training procedure
            model.training(decay_steps=data["train_data"].shape[0])
            # Keep track of gradient values and sparsity (optional)
            model.keep_tracking(sess)
            # Apply some statistics on the train and test labels.
            evaluate_data(train_labels, test_labels, mapping, model.out_dir)
            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            # run epochs
            best_val_loss = float('inf')
            best_val_epoch = 0

            for epoch in range(para.MAX_EPOCHS):
                myprint("Epoch {}" . format(epoch))

                tr_loss, tr_rmse_unnorm, tr_rmse_norm, _ = model.run_epoch(
                    sess, model.train_step,
                    train_data, train_labels, mapping, train=True)
                myprint(
                    "\ntrain loss: {}, train rmse(slope, length): {}, {}" .
                    format(tr_loss, tr_rmse_unnorm, tr_rmse_norm))

                val_loss, val_rmse_unnorm, val_rmse_norm, _ = model.run_epoch(
                    sess, model.dev_step, val_data, val_labels, mapping)
                myprint(
                    "val loss: {}, val rmse(slope, length): {}, {}\n".
                    format(val_loss, val_rmse_unnorm, val_rmse_norm))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    myprint("save best model.\n")
                    model.saver.save(sess, model.best_model)
                if epoch - best_val_epoch > para.EARLY_STOPPING:
                    break

        myprint("\ntest...")
        myprint("restore from path {}".format(model.best_model))
        model.saver.restore(sess, model.best_model)
        te_loss, te_rmse_unnorm, te_rmse_norm, te_prediction = model.run_epoch(
            sess, model.predict_step, test_data, test_labels, mapping)
        myprint(
            "test loss: {}, test rmse(slope, length): {}, {}".
            format(te_loss, te_rmse_unnorm, te_rmse_norm))
        # mv the record and parameter file to the information path
        os.system(
            "mv {o} {d}".format(
                o=para.RECORD_DIRECTORY, d=model.out_dir))
        os.system(
            "mv {o} {d}".format(
                o=para.PARAMETER_DIRECTORY, d=model.out_dir))

if __name__ == '__main__':
    dataset = "household_power_consumption.pickle"
    model = CNN
    data = form_data.init_data(dataset, model)
    start_time = datetime.datetime.now()
    run(model, data)
    exection_time = (datetime.datetime.now() - start_time).total_seconds()
    myprint("execution time: {t:.3f} seconds" . format(t=exection_time))
