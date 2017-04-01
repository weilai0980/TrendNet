# -*- coding: utf-8 -*-
"""generate random segmentation by seed and other pre-defined parameters."""

import numpy as np
import matplotlib.pyplot as plt

import settings.parameters as para
import utils.auxiliary as auxi
from utils.logger import Logger


class RandomSegmentation(object):
    """generate random segmentation based on seed, linear model."""

    def __init__(self):
        """init."""
        super(RandomSegmentation, self).__init__()
        self.log = Logger.get_logger(auxi.get_fullname(self))
        np.random.seed(para.SEED)

    def generate_random_number(self, low, high):
        """generate a random number for a given lower bound and upper bound."""
        return np.random.uniform(low, high)

    def generate_slope(self):
        """generate random slope."""
        while True:
            slope = self.generate_random_number(
                -para.SLOPE_ALLOWANCE, para.SLOPE_ALLOWANCE)
            if slope != 0:
                break
        return slope

    def generate_noise(self):
        """generate random noise for the dataset."""
        return self.generate_random_number(
            -para.NOISE_ALLOWANCE, para.NOISE_ALLOWANCE)

    def generate_sizeof_interval(self):
        """generate a random size for the interval."""
        return int(self.generate_random_number(
            para.NUM_OBSERVATIONS_PER_INTERVAL_ALLOWANCE_LOWER,
            para.NUM_OBSERVATIONS_PER_INTERVAL_ALLOWANCE_UPPER))

    def generate_interval(self, init_point):
        """generate random points for each interval under a linear model."""
        slope = self.generate_slope()
        num_intervals = self.generate_sizeof_interval()
        interval = [obs * slope + init_point + self.generate_noise()
                    for obs in xrange(1, num_intervals + 1)]
        return slope, num_intervals, interval

    def generate(self):
        """generate the random segmentation."""
        self.log.info("start to generate random segmentation.")
        init_point = np.random.rand(1)[0] * \
            para.NUM_OBSERVATIONS_PER_INTERVAL_ALLOWANCE_UPPER
        data = []
        intervals = []
        segmentations = []
        self.log.info("start: generate random segmentation.")
        while len(intervals) <= para.NUM_OBSERVATIONS:
            slope, num_intervals, interval\
                = self.generate_interval(init_point)
            init_point = interval[-1]
            data += intervals
            intervals.append(interval)
            segmentations.append([slope, num_intervals])
        self.log.info("end:   generate random segmentation.")
        return {"data": data,
                "intervals": intervals,
                "segmentations": segmentations}

    def visualize(self, intervals, path_tostore):
        """visualize the time series."""
        fig = plt.figure(figsize=(16, 6))
        plt.scatter(xrange(len(intervals)), intervals, s=8, c='b')
        plt.xlim(0, len(intervals))
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Random Generated Time Series")
        plt.savefig(path_tostore)
