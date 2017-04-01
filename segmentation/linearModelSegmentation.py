# -*- coding: utf-8 -*-
"""use linear regression to build segments."""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import utils.opfiles as opfile
import utils.auxiliary as auxi
from utils.logger import Logger
import settings.parameters as para


class LinearModelSegmentation(object):
    """Time series segmentation through linear regression model."""

    def __init__(self, path_data):
        """init."""
        super(LinearModelSegmentation, self).__init__()
        self.log = Logger.get_logger(auxi.get_fullname(self))
        self.path_data = path_data

    def read_data(self):
        """read the dataset and do some basic preprocessing."""
        self.log.info("read data.")
        data = opfile.read_txt(self.path_data)
        data = [d.split(';') for d in data]
        return data

    def str2time(self, data):
        """convert the string to time for whole dataset."""
        self.log.info("convert string to datetime.")
        pattern = '%d/%m/%Y %H:%M:%S'
        return [[auxi.str2time(d[0] + ' ' + d[1], pattern), ] + d[2:]
                for d in data]

    def check_missing(self, data, index, missing_pattern):
        """check the condition of the missing number."""
        self.log.info("check the missing values.")
        missing = [d for d in data if d[index] == missing_pattern]
        num_missing = len(missing)
        return missing, num_missing

    def rebuild_data(self, line):
        """rebuild each line of the dataset."""
        return [line[0]] + map(float, line[1:])

    def remove_missing(self, data, index, missing_pattern):
        """remove the line that misses the value from the dataset."""
        self.log.info("remove the missing value from the dataset.")
        return [self.rebuild_data(line) for line in data
                if line[index] != missing_pattern]

    def plot_timeseries(self, times, values, path_tostore):
        """plot the time series whose xlabel is in the form of datetime."""
        # define data format.
        self.log.info("plot the time series.")
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y')
        monthFmt = mdates.DateFormatter('%M')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, values)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthFmt)
        ax.xaxis.set_minor_locator(months)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(min(values), max(values))
        plt.tight_layout()
        plt.savefig(path_tostore)

    def visualize_data(self, data, index, is_subdata=True, size_subdata=1000):
        """visualize the dataset."""
        self.log.info("visualize the dataset.")
        if is_subdata:
            sub_data = data[: size_subdata]
        else:
            sub_data = data
        times = map(lambda x: x[0], sub_data)
        values = map(lambda x: x[index], sub_data)
        self.plot_timeseries(times, values)

    def check_time_interval(self, data):
        """check if the interval of the time series is equivalent."""
        self.log.info("check time intervals.")
        times = map(lambda x: x[0], data)
        times1 = times[: -1]
        times2 = times[1:]
        times_zip = zip(times1, times2)
        intervals = map(lambda x: x[1] - x[0], times_zip)
        minimal_interval = min(intervals)
        return minimal_interval, intervals, \
            all([t == minimal_interval for t in intervals])

    def split_imbalance_intervals(self, intervals, data):
        """split the imbalance intervals to balance intervals.

        More precisely, cut off the original intervals.
        """
        self.log.info("split the inbalanced intervals to several balanced.")
        splited_points = [
            (ind, interval) for ind, interval in enumerate(intervals)
            if interval != intervals[0]]
        interval_index = [-1] + [point[0] for point in splited_points]
        interval_index_zip = zip(interval_index[:-1], interval_index[1:])
        return [data[start + 1:end] for start, end in interval_index_zip]

    def num_of_missing_record(self, interval, minimal_interval):
        """get the number of missing record for a given time delta."""
        return int(interval.total_seconds() / minimal_interval.total_seconds())

    def fillin_missing_records(self, minimal_interval, intervals, data):
        """fill in the missing value for the given records."""
        gaps = [
            ind for ind, interval in enumerate(intervals)
            if interval != minimal_interval]

        newdata = []
        self.log.info("fill in the missing records.")
        for index, gap in enumerate(gaps):
            start = data[gap]
            end = data[gap + 1]
            interval = end[0] - start[0]
            num_of_missing = self.num_of_missing_record(
                end[0] - start[0], minimal_interval)
            scale_of_missing = end[1] - start[1]
            slope = scale_of_missing / num_of_missing
            insert_points = [
                (start[0] + minimal_interval * ind, start[1] + slope * ind)
                for ind in xrange(1, num_of_missing)]

            if index == 0:
                newdata += data[: gaps[index] + 1]
            else:
                newdata += data[gaps[index-1] + 1: gaps[index] + 1]
            newdata += insert_points
        newdata += data[gaps[index] + 1:]
        return newdata

    def remove_short_intervals(self, data, threshold):
        """remove those intervals are too short."""
        self.log.info("remove the periods that are too short.")
        return [period for period in data if len(period) > threshold]

    def least_square(self, y, tx):
        """calculate the least square."""
        a = tx.T.dot(tx)
        b = tx.T.dot(y)
        return np.linalg.solve(a, b)

    def compute_rmse(self, y, tx, beta):
        """compute the cost by rmse."""
        e = y - tx.dot(beta)
        mse = e.dot(e) / (2 * len(e))
        return np.sqrt(mse)

    def compute_cost(self, y):
        """calculate the cost for one period."""
        tx = np.c_[np.ones(len(y)).T, np.arange(len(y)).T]
        y = np.array(y).T
        beta = self.least_square(y, tx)
        rmse = self.compute_rmse(y, tx, beta)
        return beta, rmse

    def linear_segmentation(self, serie):
        """apply linear regression on the data to do the segmentation."""
        self.log.info("Use linear regression to do the segmentation.")
        intervals = []
        segmentations = []
        points = []
        total_len = len(serie)
        for ind, point in enumerate(serie):
            if len(points) < 2:
                points.append(point)
                continue
            else:
                tmp_points = copy.copy(points)
                tmp_points.append(point)
                beta, rmse = self.compute_cost(tmp_points)
                if rmse > para.THRESHOLD_SEGMENTATION_ERROR:
                    intervals.append(points)
                    segmentations.append([beta[1], len(points)])
                    points = tmp_points[-1:]
                else:
                    points.append(point)
            if ind % 10000 == 0:
                self.log.info(
                    "processed {p} points, existing {tp} points in total {t}"
                    .format(p=ind, tp=total_len, t=1.0 * ind / total_len))
        return {"data": serie,
                "intervals": intervals,
                "segmentations": segmentations}
