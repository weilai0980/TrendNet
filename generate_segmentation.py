# -*- coding: utf-8 -*-
"""generate segmentation."""
import sys
from os.path import join
import numpy as np


import utils.opfiles as opfile
import settings.parameters as para
from segmentation.randomSegmentation import RandomSegmentation
from segmentation.linearModelSegmentation import LinearModelSegmentation


def random_segmentation(force):
    """generate random segmentation."""
    # define path
    path_out_seg = join(opfile.build_dir(
                        join(para.DATA_DIRECTORY,
                             "output",
                             "segmentation"),
                        force),
                        "random.pickle")
    path_out_fig = join(para.DATA_DIRECTORY,
                        "output",
                        "segmentation",
                        "random.png")

    # generate random time series
    rs = RandomSegmentation()
    result = rs.generate()

    # write to output folder
    opfile.write_pickle(result, path_out_seg)
    # visualize data
    # rs.visualize(result["intervals"], path_out_fig)


def household_power_consumption_linear_segmentation(force):
    """preprocess the dataset and generate segmentation from linear model."""
    # define path
    path_data = join(para.DATA_DIRECTORY,
                     "input",
                     "household_power_consumption.txt")
    path_out_segmentations = join(para.DATA_DIRECTORY,
                                  "output",
                                  "segmentation",
                                  "household_power_consumption.pickle")
    path_out_figure = join(para.DATA_DIRECTORY,
                           "output",
                           "segmentation",
                           "household_power_consumption_plot.png")
    lms = LinearModelSegmentation(path_data)
    # read data and remove header
    data = lms.read_data()
    data = data[1:]
    # convert the string to datetime
    data = lms.str2time(data)
    # remove entries that have missing values
    valid_data = lms.remove_missing(data, index=1, missing_pattern="?")
    # check time interval
    minimal_interval, intervals, equal_intervals \
        = lms.check_time_interval(valid_data)

    # define the index of column to extract
    extract_index = 3
    # only extract (3+1)-th column.
    good_serie = map(lambda x: (x[0], x[extract_index]), valid_data)

    # plot the corresponding curve.
    # sub_data = good_serie[: 1000]
    # times = map(lambda x: x[0], sub_data)
    # values = map(lambda x: x[1], sub_data)
    # lms.plot_timeseries(times, values, path_out_figure)

    # fill in the missing data.
    good_serie = lms.fillin_missing_records(
        minimal_interval, intervals, good_serie)
    # use linear regression to get the intervals and segmentations
    result = lms.linear_segmentation(map(lambda x: x[1], good_serie))
    # pickle the result to the file
    opfile.write_pickle(result, path_out_segmentations)


def test():
    """test function."""
    """test 1."""
    # rs = RandomSegmentation()
    # rs.generate()
    """ test 2."""
    household_power_consumption_linear_segmentation(force=False)


if __name__ == '__main__':
    # define path.
    if True:
        test()
        sys.exit(0)
    else:
        random_segmentation(force=False)
        household_power_consumption_linear_segmentation(force=False)
