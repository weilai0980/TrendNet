# -*- coding: utf-8 -*-
"""Generate the data for batch iteration."""
import numpy as np


def batch_iter(data, batch_size, num_epochs, num_steps=1, shuffle=True):
    """Generate a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    batch_size = batch_size * num_steps
    num_batches_per_epoch = len(data) // batch_size
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_data[start_index:end_index]
