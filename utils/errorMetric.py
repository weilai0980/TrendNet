# -*- coding: utf-8 -*-
"""some helper functions."""
from os.path import join
import numpy as np


def compute_rmse(y, pred):
    """compute the rmse."""
    mse = np.mean((y - pred) ** 2, axis=0)
    rmse = np.array(mse).tolist()
    return np.sqrt(rmse)


def compute_loss(y_pred, mapping, save_to_path):
    """compute the loss."""
    y, pred = y_pred
    rmse_norm = compute_rmse(y, pred)
    pred = pred * (
        mapping["max_labels"] - mapping["min_labels"]) + mapping["min_labels"]
    y = y * (
        mapping["max_labels"] - mapping["min_labels"]) + mapping["min_labels"]
    np.savetxt(join(save_to_path, "y_test"), y)
    np.savetxt(join(save_to_path, "y_pred"), pred)
    rmse_unnorm = compute_rmse(y, pred)
    return rmse_unnorm, rmse_norm


def format_data(y_pred):
    """reshape a list of scores to a vector."""
    return reduce(
        lambda a, b: (
            np.vstack((a[0], b[0])),
            np.vstack((a[1], b[1]))),
        y_pred)
