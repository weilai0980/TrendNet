# -*- coding: utf-8 -*-
"""Form the dataset."""
from os.path import join

import numpy as np
import settings.parameters as para
import opfiles as opfile
import splitData as split_data
import inspect


def load_data(path):
    """load the raw data from the directory and adjust the data for a model."""
    data = opfile.load_pickle(path)
    return prepare_data(data)


def prepare_data(data):
    """prepare data for continous model."""
    sequence = data["data"]
    segmentations = data["segmentations"]
    incremental_lengths = np.cumsum(map(lambda x: x[1], segmentations))[:-1]
    continous_X, continous_y = [], []
    discrete_X, discrete_y = [], []
    for index, segmentation in enumerate(incremental_lengths):
        if segmentation > para.CONTINUOUS_WINDOW \
                and index > para.DISCRETE_WINDOW:
            continous_X.append(
                sequence[segmentation - para.CONTINUOUS_WINDOW:
                         segmentation])
            continous_y.append(segmentations[index + 1])
            discrete_X.append(
                segmentations[index + 1 - para.DISCRETE_WINDOW:
                              index + 1])
            discrete_y.append(segmentations[index + 1])
    return continous_X, continous_y, discrete_X, discrete_y


def normalize_labels(labels):
    """normalize the labels.

    the labels is a [N * 2] matrix,
    since the first column already in the range of -1 and 1,
    we only normalize the second column here.
    and we choose the normalization method mentioned below:
    (x - x_min) / (x_max - x_min) + x_min
    """
    max_labels = np.max(labels, axis=0)
    min_labels = np.min(labels, axis=0)
    labels = 1.0 * (labels - min_labels) / (max_labels - min_labels)
    return labels, {"max_labels": max_labels, "min_labels": min_labels}


def normalize_data(data):
    """normalize the dataset."""
    train_data, train_labels = data["train_data"], data["train_labels"]
    val_data, val_labels = data["validation_data"], data["validation_labels"]
    test_data, test_labels = data["test_data"], data["test_labels"]

    train_labels, mapping = normalize_labels(train_labels)
    val_labels = (val_labels - mapping["min_labels"]) / (
        mapping["max_labels"] - mapping["min_labels"])
    test_labels = (test_labels - mapping["min_labels"]) / (
        mapping["max_labels"] - mapping["min_labels"])
    return train_data, train_labels, \
        val_data, val_labels, \
        test_data, test_labels, mapping


def init_data(dataset, model):
    """define parameters and prepare data."""
    # define path.
    path_data = join(para.SEGMENTATION_DIRECTORY, dataset)

    # form the data.
    print("load the dataset and form the dataset...")
    continous_X, continous_y, discrete_X, discrete_y = load_data(path_data)

    if inspect.isclass(model):
        if model.__name__ not in para.MIXTURE_MODELS:
            if model.__name__ in para.CONTINUOUS_MODELS:
                print("load data for continous model.")
                X, y = continous_X, continous_y
            else:
                print("load data for discrete model.")
                X, y = discrete_X, discrete_y
        else:
            print("load data for mixture model.")
            X = np.array(zip(continous_X, discrete_X))
            y = np.array(continous_y)
    else:
        print("load data for baseline model.")
        discrete_X = np.array(discrete_X).reshape((len(discrete_X), -1))
        continous_X = np.array(continous_X)
        X = np.hstack((continous_X, discrete_X))
        y = continous_y

    # debug mode.
    if para.DEBUG:
        X = X[:1024]
        y = y[:1024]
    # split the dataset to train, validation, and test.
    print("split the dataset into train, validation, and test...")
    data = split_data.split_data(X, y)
    print("stat:: train/validation/test split: {}/{}/{}".format(
        data["train_data"].shape,
        data["validation_data"].shape,
        data["test_data"].shape))
    return data
