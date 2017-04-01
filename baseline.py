# -*- coding: utf-8 -*-
"""use different baseline methods."""
import os
from os.path import join

from optparse import OptionParser

from utils.logger import Logger
import baselines.kr as kr
import baselines.svr as svr
import utils.opfiles as opfile
import settings.parameters as para
import utils.formData as form_data


log = Logger.get_logger("baseline")


def save_baseline_to_path(method):
    """build a directory to save the prediction of baseline method to files."""
    out_path = join(para.BASELINE_DIRECTORY, method)
    opfile.build_dir(out_path, force=para.FORCE_RM_RECORD)
    return out_path


def run(data, method):
    """run different baseline method."""
    log.info("build dir for final prediction.")
    output_path = save_baseline_to_path(method)

    log.info("start training...")
    train_data, train_labels, \
        val_data, val_labels, \
        test_data, test_labels, mapping = form_data.normalize_data(data)

    if "SVR" in method:
        kernel = method.split("_")[1]
        log.info("run the baseline method (SVR+{})".format(kernel))
        svr.train_predict_batches(
            kernel, train_data, train_labels, test_data, test_labels,
            mapping, output_path)
    elif "KR" in method:
        kernel = method.split("_")[1]
        log.info("run the baseline method (KR+{})".format(kernel))
        kr.train_predict_batches(
            kernel, train_data, train_labels, test_data, test_labels,
            mapping, output_path)

    os.system(
        "mv {o} {d}".format(
            o=para.RECORD_DIRECTORY, d=output_path))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option(
        "-d",
        "--dataset",
        dest="dataset",
        default="household_power_consumption.pickle",
        type="string",
        help="the dataset to use")
    parser.add_option(
        "-m",
        "--method",
        dest="method",
        default="SVR_rbf",
        type="string",
        help="the baseline method.")
    (option, args) = parser.parse_args()

    data = form_data.init_data(option.dataset, option.method)
    run(data, option.method)
