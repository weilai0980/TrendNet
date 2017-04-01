# -*- coding: utf-8 -*-
"""baseline method: Support Vector Regression with different kernel."""
import time
from os.path import join

import numpy as np
from sklearn.svm import SVR

from utils.myprint import myprint
import settings.parameters as para
import utils.opfiles as opfile
import utils.errorMetric as error_metric


def define_model(kernel, c, gamma, degree):
    """define the parameter/model of the svr."""
    return SVR(kernel=kernel, C=c, degree=degree, gamma=gamma)


def train_predict(
        train_data, train_labels, test_data, kernel, c, gamma, degree=3):
    """train and then predict."""
    myprint("kernel: {}; c: {}; gamma: {}; degree: {}".format(
        kernel, c, gamma, degree))

    start_time = time.time()
    model = define_model(kernel, c, gamma, degree)
    myprint("train on slope.")

    model.fit(train_data, train_labels[:, 0])
    prediction_1 = np.expand_dims(model.predict(test_data), axis=1)
    myprint("train on length.")

    model.fit(train_data, train_labels[:, 1])
    prediction_2 = np.expand_dims(model.predict(test_data), axis=1)
    prediction = np.hstack((prediction_1, prediction_2))
    svr_fitduration = time.time() - start_time
    myprint(
        "SVR complexity and bandwidth selected and model fitted in %.3f s"
        % svr_fitduration)
    return model, prediction


def train_predict_batches(
        kernel, train_data, train_labels, test_data, test_labels,
        mapping, out_root_path):
    """using the model to do the prediciton for batch of parameters."""
    for c in para.SVR_C:
        for gamma in para.SVR_GAMMA:
            out_path = join(
                out_root_path, "c-{}-gamma-{}".format(c, gamma))
            myprint("build dir {}".format(out_path))
            opfile.build_dir(out_path, para.FORCE_RM_RECORD)

            model, prediction = train_predict(
                train_data, train_labels, test_data, kernel, c, gamma)
            rmse_unnorm, rmse_norm = error_metric.compute_loss(
                (test_labels, prediction), mapping, out_path)
            myprint("the loss of baseline method (SVR+{}):{}, {}\n".format(
                kernel, rmse_unnorm, rmse_norm))
