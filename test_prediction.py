# -*- coding: utf-8 -*-
"""run the prediction through different method."""
import os
import copy

import numpy as np

import parameter_template as para


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def write_txt(data, out_path, type="w"):
    """write the data to the txt file."""
    with open(out_path, type) as f:
        f.write(data.encode("utf-8"))


def build_parameters_from_template(to_replace_list):
    """read parameters from the template."""
    template = read_txt("parameters_template.py")
    parameters = []

    for to_replace in to_replace_list:
        parameter = []
        attributes_to_replace = to_replace.keys()
        for line in template:
            attribute = line.split("=")[0].strip()
            if attribute in attributes_to_replace:
                parameter.append(attribute + " = " + to_replace[attribute])
            else:
                parameter.append(line)
        parameters.append(parameter)
    return parameters


def design_CNN_parameters():
    """design parameters for CNN."""
    want_to_change = []
    template = {"SEED": str(np.random.randint(para.SEED_MAX))}

    for cont_window in [500, 1000, 1500, 2000]:
        for lr in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:
            for l2_regu in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:
                tmp = copy.copy(template)
                tmp["CONTINUOUS_WINDOW"] = str(cont_window)
                tmp["LEARNING_RATE"] = str(lr)
                tmp["L2_REGULARIZATION_LAMBDA"] = str(l2_regu)
                want_to_change.append(tmp)
    return want_to_change


def design_RNN_parameters():
    """design parameters for RNN."""
    want_to_change = []
    template = {"SEED": str(np.random.randint(para.SEED_MAX))}

    for disc_window in [100, 300, 500, 700, 900, 1000]:
        for lr in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:
            for l2_regu in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:
                tmp = copy.copy(template)
                tmp["DISCRETE_WINDOW"] = str(disc_window)
                tmp["LEARNING_RATE"] = str(lr)
                tmp["L2_REGULARIZATION_LAMBDA"] = str(l2_regu)
                want_to_change.append(tmp)
    return want_to_change


if __name__ == '__main__':
    to_replace_list = design_CNN_parameters()
    parameters = build_parameters_from_template(to_replace_list)
    for ind, parameter in enumerate(parameters):
        print("parameter iteration:{}\n, parameter:{}!" .
              format(ind, "\n".join(parameter)))
        write_txt("\n".join(parameter), "settings/parameters.py")
        os.system("python prediction.py")
