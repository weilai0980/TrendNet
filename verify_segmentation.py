# -*- coding: utf-8 -*-
"""verify the correctness of the segmentation."""
from os.path import join
from os import listdir

import utils.opfiles as opfile
import settings.parameters as para


def verify(path_file):
    """verify the segmentation."""
    data = opfile.load_pickle(path_file)
    intervals = data["intervals"]
    segmentations = data["segmentations"]
    print "file:", path_file
    print "number of segmentations:", len(segmentations)
    print "an example 1:", intervals[0]
    print "corresponding slope and lenght:", segmentations[0]
    print "an example 2:", intervals[1]
    print "corresponding slope and lenght:", segmentations[1]
    print "-----------------------------------------------------------------"


def list_all_segmentations(path_segmentation):
    """list all pickle file that relates to segmentation."""
    return [f for f in listdir(path_segmentation) if 'pickle' in f]


if __name__ == '__main__':
    # define path
    path_segmentation = join(para.DATA_DIRECTORY, "output", "segmentation")

    # list all segmentations file.
    files = list_all_segmentations(path_segmentation)
    # verify
    for file in files:
        file_to_verify = join(path_segmentation, file)
        verify(file_to_verify)
