# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import sys
import os
import time
import shutil
import numpy as np
import tensorflow as tf

import settings.parameters as para
import utils.auxiliary as auxi
from utils.logger import Logger
import utils.batchData as batch_data
import utils.errorMetric as error_metric


class BasicModel(object):
    """a base model for any subsequent implementation."""

    def __init__(self):
        """init."""
        np.random.seed(para.SEED)
        self.log = Logger.get_logger(auxi.get_fullname(self))
        self.force = para.FORCE_RM_RECORD
        self.build_batch = 1

    def weight_variable(self, shape, name="W"):
        """init weight."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name="b"):
        """init bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W, strides, padding='SAME', name="conv"):
        """do convolution."""
        return tf.nn.conv2d(
            x, W,
            strides=strides,
            padding=padding, name=name)

    def relu(self, conv, b, name="relu"):
        """do convolution."""
        return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)

    def max_pool(self, x, ksize, strides, padding='SAME', name="pool"):
        """do max pooling."""
        return tf.nn.max_pool(
            x, ksize=ksize, strides=strides, padding=padding, name=name)

    def define_placeholder(self, shape_x, shape_y):
        """define the placeholders."""
        self.input_x = tf.placeholder(tf.float32, shape_x, name="input_x")
        self.input_y = tf.placeholder(tf.float32, shape_y, name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def define_parameters_totrack(self):
        """define some parameters that will be used in the future."""
        self.l2_loss = tf.constant(0.0)

    def keep_tracking(self, sess):
        """keep track the status."""
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.join(
            para.TRAINING_DIRECTORY, "runs",
            auxi.get_fullname(self))
        if self.force:
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir = os.path.join(out_dir, timestamp)
        self.out_dir = out_dir
        self.log.info("writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", self.loss)

        # Train Summaries
        self.train_summary_op = tf.merge_summary(
            [loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.train.SummaryWriter(
            train_summary_dir, sess.graph_def)

        # dev summaries
        self.dev_summary_op = tf.merge_summary([loss_summary])
        dev_summary_dir = os.path.join(
            out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.train.SummaryWriter(
            dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        self.checkpoint_comparison = os.path.join(checkpoint_dir, "comparison")
        self.best_model = os.path.join(checkpoint_dir, "best_model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            os.makedirs(self.checkpoint_comparison)
        self.saver = tf.train.Saver(tf.all_variables())

    def fill_feed_dict(
            self, batch_x, batch_y, dropout=para.DROPOUT_RATE):
        """fill the feed_dict for training/validation/test the given step."""
        return {
            self.input_x: batch_x,
            self.input_y: batch_y,
            self.dropout_keep_prob: dropout}

    def loss(self):
        """calculate the rmse."""
        with tf.name_scope("loss"):
            losses = tf.reduce_mean(
                tf.square(tf.sub(self.scores, self.input_y)))
            self.loss = tf.sqrt(losses) + \
                para.L2_REGULARIZATION_LAMBDA * self.l2_loss
        return self

    def training(
            self, decay_steps, starter_learning_rate=para.LEARNING_RATE):
        """train the model with learning rate decay."""
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=starter_learning_rate,
            global_step=self.global_step,
            decay_steps=decay_steps,
            decay_rate=para.DECAY_RATE,
            staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(
            self.grads_and_vars,
            global_step=self.global_step)

    def get_batches(self, input_data, input_labels,
                    num_epochs=1, num_steps=1, shuffle=True):
        """get batch data."""
        batch_size = para.BATCH_SIZE
        batches = batch_data.batch_iter(
            list(zip(input_data, input_labels)),
            batch_size=batch_size,
            num_epochs=1,
            num_steps=num_steps,
            shuffle=shuffle)
        num_batch = len(input_data) // (batch_size * num_steps)
        return batches, num_batch

    def train_step(self, sess, batch_x, batch_y):
        """evaluate the training step."""
        feed_dict = self.fill_feed_dict(batch_x, batch_y)
        _, step, summaries, loss, scores = sess.run(
            [self.train_op, self.global_step, self.train_summary_op,
             self.loss, self.scores],
            feed_dict)
        self.train_summary_writer.add_summary(summaries, step)
        return loss, scores

    def dev_step(self, sess, batch_x, batch_y):
        """evaluate the dev step."""
        feed_dict = self.fill_feed_dict(batch_x, batch_y, dropout=1.0)
        _, step, summaries, loss, scores = sess.run(
            [self.train_op, self.global_step, self.dev_summary_op,
             self.loss, self.scores],
            feed_dict)
        self.dev_summary_writer.add_summary(summaries, step)
        return loss, scores

    def predict_step(self, sess, batch_x, batch_y):
        """use the test dataset to do the prediction."""
        feed_dict = self.fill_feed_dict(batch_x, batch_y, dropout=1.0)
        loss, scores = sess.run(
            [self.loss, self.scores],
            feed_dict)
        return loss, scores

    def run_epoch(self, sess, stage,
                  data, labels, mapping,
                  num_steps=1, train=False, verbose=True):
        """the script for each epoch."""
        batches, num_batch = self.get_batches(
            data, labels, num_steps=self.build_batch)

        losses = []
        predictions = []
        y_predictions = []
        for step, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            loss, prediction = stage(sess, x_batch, y_batch)
            losses.append(loss)
            predictions.append(prediction)
            y_predictions.append((y_batch, prediction))
            current_step = tf.train.global_step(sess, self.global_step)

            if verbose and step % verbose == 0 and train:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step + 1, num_batch, np.mean(losses)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        # save model
        if train:
            self.saver.save(
                sess, self.checkpoint_prefix, global_step=current_step)
        # self.log.info("Saved model checkpoint to {}".format(path))
        # calculate the rmse between y and prediction.
        rmse_unnorm, rmse_norm = error_metric.compute_loss(
            error_metric.format_data(y_predictions),
            mapping,
            self.checkpoint_comparison)
        return np.mean(losses), rmse_unnorm, rmse_norm, predictions
