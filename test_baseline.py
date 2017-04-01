# -*- coding: utf-8 -*-
"""run with different baseline."""

import os

if __name__ == '__main__':
    # build parameters.
    os.system("cp parameters_template.py settings/parameters.py")
    # run different method.
    dataset = "household_power_consumption.pickle"

    method = "SVR_rbf"
    os.system("python baseline.py -d {} -m {}".format(dataset, method))

    method = "SVR_sigmoid"
    os.system("python baseline.py -d {} -m {}".format(dataset, method))

    method = "KR_rbf"
    os.system("python baseline.py -d {} -m {}".format(dataset, method))

    method = "KR_sigmoid"
    os.system("python baseline.py -d {} -m {}".format(dataset, method))
