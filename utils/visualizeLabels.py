# -*- coding: utf-8 -*-
"""visualize the data."""

import numpy as np
import matplotlib
matplotlib.use('Agg')


def visualize_histogram(labels, save_to_path):
    """visualize the distribution of the labels."""
    import matplotlib.pyplot as plt
    slope = labels[:, 0]
    length = filter(lambda x: x <= 1000, labels[:, 1])
    fig, axs = plt.subplots(1, 2)

    axs[0].hist(slope, bins=100)
    axs[0].set_title("Histogram of slope.")

    axs[1].hist(length, bins=100)
    axs[1].set_title("Histogram of length.")

    plt.savefig(save_to_path)
