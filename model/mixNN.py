# -*- coding: utf-8 -*-
"""Define a class to use a model that mix the CNN and RNN."""
import tensorflow as tf

from basisModel import BasicModel
import settings.parameters as para


class MixNN(BasicModel):
    """"mix the cnn and rnn to do the trend prediction."""

    def __init__(self):
        """init."""
        super(MixNN, self).__init__()
        # define.
        self.rnn_hidden_dim = para.HIDDEN_DIM
        self.num_steps = para.DISCRETE_WINDOW
        self.define_placeholder(
            shape_x_con=[None, para.CONTINUOUS_WINDOW],
            shape_x_dis=[None, self.num_steps, para.NUM_CLASSES],
            shape_y=[None, para.NUM_CLASSES])
        self.define_parameters_totrack()

    def define_placeholder(self, shape_x_con, shape_x_dis, shape_y):
        """define the placeholders."""
        self.input_x_continuous = tf.placeholder(
            tf.float32, shape_x_con, name="input_x_continuous")
        self.input_x_discrete = tf.placeholder(
            tf.float32, shape_x_dis, name="input_x_discrete")
        self.input_y = tf.placeholder(
            tf.float32, shape_y, name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def fill_feed_dict(
            self, batch_x, batch_y, dropout=para.DROPOUT_RATE):
        """fill the feed_dict for training/validation/test the given step."""
        cont_x, disc_x = zip(*batch_x)
        return {
            self.input_x_continuous: cont_x,
            self.input_x_discrete: disc_x,
            self.input_y: batch_y,
            self.dropout_keep_prob: dropout}

    def inference(self):
        """use cnn + rnn to do the prediction."""
        # *********************************************************************
        # use cnn to extract features.
        cnn_input = tf.expand_dims(self.input_x_continuous, -1)
        cnn_input = tf.expand_dims(cnn_input, -1)

        pooled_outputs = []
        filters = zip(para.FILTER_SIZES, para.NUM_FILTERS)
        for i, (filter_size, num_filters) in enumerate(filters):
            with tf.name_scope("conv-relu-maxpool-%s" % filter_size):
                W = self.weight_variable(
                    shape=[filter_size, 1, 1, num_filters])
                b = self.bias_variable(
                    shape=[num_filters])
                conv = self.conv2d(
                    cnn_input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                h = self.relu(conv, b)
                pooled = self.max_pool(
                    h,
                    ksize=[1, para.CONTINUOUS_WINDOW-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                pooled_outputs.append(pooled)

        num_filters_total = reduce(lambda a, b: a + b, para.NUM_FILTERS)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.cnn_output = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # *********************************************************************
        # use rnn to extract features.
        rnn_input = tf.nn.dropout(
            self.input_x_discrete, self.dropout_keep_prob)
        # create lstm cell.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.rnn_hidden_dim,
            forget_bias=para.FORGET_GATE_BIASES,
            state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=self.dropout_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * para.NUM_LAYERS, state_is_tuple=True)
        # define init state
        self._initial_state = cell.zero_state(para.BATCH_SIZE, tf.float32)
        # start the inference.
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in xrange(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(
                    rnn_input[:, time_step, :], state)
                outputs.append(cell_output)
        self.rnn_output = cell_output

        # *********************************************************************
        # use the feature of cnn and rnn together.
        with tf.name_scope("projection"):
            CNN_W = tf.get_variable(
                "CNN_W",
                shape=[num_filters_total, para.PROJECTION_DIM],
                initializer=tf.contrib.layers.xavier_initializer())

            RNN_W = tf.get_variable(
                "RNN_W",
                shape=[self.rnn_hidden_dim, para.PROJECTION_DIM],
                initializer=tf.contrib.layers.xavier_initializer())
            self.projection = tf.tanh(tf.add(
                tf.matmul(self.cnn_output, CNN_W),
                tf.matmul(self.rnn_output, RNN_W)))
            self.l2_loss += tf.nn.l2_loss(CNN_W)
            self.l2_loss += tf.nn.l2_loss(RNN_W)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[para.PROJECTION_DIM, para.NUM_CLASSES],
                initializer=tf.contrib.layers.xavier_initializer())
            b = self.bias_variable([para.NUM_CLASSES])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.projection,
                W,
                b,
                name="scores")
