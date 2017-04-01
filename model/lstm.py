# -*- coding: utf-8 -*-
"""Define a class to use basic lstm to predict the trend."""
import tensorflow as tf

from basisModel import BasicModel
import settings.parameters as para


class LSTM(BasicModel):
    """use the standard LSTM to do the trend prediction."""

    def __init__(self):
        """init."""
        super(LSTM, self).__init__()
        # define
        self.hidden_size = para.HIDDEN_DIM
        self.num_steps = para.DISCRETE_WINDOW
        self.define_placeholder(
            shape_x=[None, self.num_steps, para.NUM_CLASSES],
            shape_y=[None, para.NUM_CLASSES])
        self.define_parameters_totrack()

    def inference(self):
        """use the lstm to output the result."""
        # refine input
        # reshape_inputs = tf.reshape(
        #     input, [-1, self.num_steps, para.NUM_CLASSES])
        inputs = tf.nn.dropout(self.input_x, self.dropout_keep_prob)

        # create lstm cell.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.hidden_size,
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
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        self.rnn_output = tf.reshape(
            tf.concat(1, outputs), [-1, self.hidden_size])
        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.hidden_size, para.NUM_CLASSES],
                initializer=tf.contrib.layers.xavier_initializer())
            b = self.bias_variable([para.NUM_CLASSES])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                cell_output,
                W,
                b,
                name="scores")
