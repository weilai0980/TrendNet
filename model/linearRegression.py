# -*- coding: utf-8 -*-
"""Define a class to use linear regression to do the trend prediction."""

import tensorflow as tf

from basisModel import BasicModel
import settings.parameters as para


class LinearRegression(BasicModel):
    """use linear regression for trend prediction."""

    def __init__(self):
        """init."""
        super(LinearRegression, self).__init__()
        # define
        self.num_steps = 1
        self.define_placeholder(
            shape_x=[None, para.CONTINUOUS_WINDOW],
            shape_y=[None, para.NUM_CLASSES])
        self.define_parameters_totrack()

    def inference(self, input):
        """use linear regression to output the result."""
        with tf.name_scope("output"):
            W = self.weight_variable(
                [para.CONTINUOUS_WINDOW, para.NUM_CLASSES])
            b = self.bias_variable([para.NUM_CLASSES])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                input,
                W,
                b,
                name="scores")
        return self
