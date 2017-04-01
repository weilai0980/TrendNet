# -*- coding: utf-8 -*-
"""baseline method: Kernel Ridge Regression with different kernel."""
import time
from os.path import join

from sklearn.kernel_ridge import KernelRidge

import settings.parameters as para
import utils.opfiles as opfile
from utils.myprint import myprint
import utils.errorMetric as error_metric


def define_model(kernel, alpha, gamma, degree):
    """define the parameter/model of the svr."""
    return KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma, degree=degree)


def train_predict(
        train_data, train_labels, test_data, kernel, alpha, gamma, degree=3):
    """train and then predict."""
    myprint("kernel: {}; alpha: {}; gamma: {}; degree: {}".format(
        kernel, alpha, gamma, degree))

    start_time = time.time()
    model = define_model(kernel, alpha, gamma, degree)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)

    kr_fitduration = time.time() - start_time
    myprint(
        "KR complexity and bandwidth selected and model fitted in %.3f s"
        % kr_fitduration)
    return model, prediction


def train_predict_batches(
        kernel, train_data, train_labels, test_data, test_labels,
        mapping, out_root_path):
    """using the model to do the prediciton for batch of parameters."""
    for alpha in para.KR_ALPHA:
        for gamma in para.KR_GAMMA:
            out_path = join(
                out_root_path, "alpha-{}-gamma-{}".format(alpha, gamma))
            myprint("build dir {}".format(out_path))
            opfile.build_dir(out_path, para.FORCE_RM_RECORD)

            model, prediction = train_predict(
                train_data, train_labels, test_data, kernel, alpha, gamma)
            rmse_unnorm, rmse_norm = error_metric.compute_loss(
                (test_labels, prediction), mapping, out_path)
            myprint("the loss of baseline method (KR+{}):{}, {}\n".format(
                kernel, rmse_unnorm, rmse_norm))
