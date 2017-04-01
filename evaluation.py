# -*- coding: utf-8 -*-
"""use the checkpoint to do the valuation."""

from os.path import join

import numpy as np
import tensorflow as tf

from utils.logger import Logger
import utils.batchData as batch_data
import settings.parameters as para
import prediction

log = Logger.get_logger("Main")


def eval(data_dir, model_dir):
    """evaluate the checkpoint."""
    log.info("Evaluating...")
    checkpoint_dir = join(para.CHECKPOINT_DIRECTORY, model_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=para.ALLOW_SOFT_PLACEMENT,
            log_device_placement=para.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph(
            "{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(
            "output/scores").outputs[0]

        # Generate batches
        data = prediction.load_data(data_dir)
        train_data, train_labels, \
            val_data, val_labels, \
            test_data, test_labels = prediction.normalize_data(data)

        batches = batch_data.batch_iter(
            test_data,
            batch_size=para.BATCH_SIZE, num_epochs=1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(
                predictions,
                {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions.append(batch_predictions)
        all_predictions = np.vstack(all_predictions)

        # Print accuracy
        y_test = data["test_labels"]
        error = (all_predictions - y_test).reshape(-1)
        loss = np.sqrt(np.mean(np.square(error)))
        log.info("loss is {}".format(loss))


if __name__ == '__main__':
    data_dir = "household_power_consumption.pickle"
    model_dir = "runs/model.cnn.CNN/1473622972/checkpoints/"
    eval(data_dir, model_dir)
