# -*- coding: utf-8 -*-
"""Split the dataset to train data, validation data and test data."""

import numpy as np
import settings.parameters as para


def split_data(X, y):
    """split the dataset."""
    # set seed
    np.random.seed(para.SEED)
    # convert data to numpy array.
    X = np.array(X)
    y = np.array(y)
    # permutation indices.
    dim_y_row = len(y)
    shuffle_indices = np.random.permutation(np.arange(dim_y_row))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # split the data to train, validation and test.
    X_train = X_shuffled[: int(dim_y_row * para.TRAIN_RATIO)]
    y_train = y_shuffled[: int(dim_y_row * para.TRAIN_RATIO)]
    X_validation = X_shuffled[
        int(dim_y_row * para.TRAIN_RATIO):
        int(dim_y_row * para.TRAIN_RATIO + dim_y_row * para.VALIDATION_RATIO)]
    y_validation = y_shuffled[
        int(dim_y_row * para.TRAIN_RATIO):
        int(dim_y_row * para.TRAIN_RATIO + dim_y_row * para.VALIDATION_RATIO)]
    X_test = X_shuffled[
        int(dim_y_row * para.TRAIN_RATIO + dim_y_row * para.VALIDATION_RATIO):]
    y_test = y_shuffled[
        int(dim_y_row * para.TRAIN_RATIO + dim_y_row * para.VALIDATION_RATIO):]
    return {
        "train_data": X_train,
        "train_labels": y_train,
        "validation_data": X_validation,
        "validation_labels": y_validation,
        "test_data": X_test,
        "test_labels": y_test
    }
