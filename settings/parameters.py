# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join

import numpy as np

"""system part."""
WORK_DIRECTORY = "/home/tlin/notebooks/code"
DATA_DIRECTORY = join(WORK_DIRECTORY, "data")
SEGMENTATION_DIRECTORY = join(DATA_DIRECTORY, "output", "segmentation")
TRAINING_DIRECTORY = join(DATA_DIRECTORY, "output", "training")
BASELINE_DIRECTORY = join(DATA_DIRECTORY, "output", "baseline")
RECORD_DIRECTORY = join(WORK_DIRECTORY, "record")
PARAMETER_DIRECTORY = join(WORK_DIRECTORY, "settings", "parameters.py")

"""tensorflow configuration."""
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

"""segmentation generation."""
# random segmentation generation.
NUM_OBSERVATIONS = 10000
NUM_OBSERVATIONS_PER_INTERVAL_ALLOWANCE_UPPER = 20
NUM_OBSERVATIONS_PER_INTERVAL_ALLOWANCE_LOWER = 5
SLOPE_ALLOWANCE = 1.0
NOISE_ALLOWANCE = SLOPE_ALLOWANCE / 10
# linear regression segmentation
THRESHOLD_SERIES_PERIOD = 1e4
THRESHOLD_SEGMENTATION_ERROR = 0.6

"""machine learning model."""
DEBUG = True
SEED = 42
SEED_MAX = 1000
NUM_CHANNELS = 1
NUM_CLASSES = 2
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 100

LEARNING_RATE = 0.0001
DECAY_RATE = 0.95
L2_REGULARIZATION_LAMBDA = 5e-4
DROPOUT_RATE = 0.5

EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100
CHECKPOINT_DIRECTORY = TRAINING_DIRECTORY
EARLY_STOPPING = 2

FORCE_RM_RECORD = False

"""time series."""
CONTINUOUS_WINDOW = 100
DISCRETE_WINDOW = 100
CONTINUOUS_MODELS = ["LinearRegression", "CNN"]
DISCRETE_MODELS = ["LSTM"]
MIXTURE_MODELS = ["MixNN"]

"""cnn."""
FILTER_SIZES = [2, 3, 4, 5]
NUM_FILTERS = [128, 128, 128, 128]

"""rnn."""
FORGET_GATE_BIASES = 0.0
NUM_LAYERS = 1
HIDDEN_DIM = 100

"""cnn + rnn."""
PROJECTION_DIM = 10

"""baseline model"""
SVR_C = [1e0, 1e2, 1e4, 1e6]
SVR_GAMMA = np.logspace(-5, 5, 6)
SVR_DEGREE = [1, 2, 3]

KR_ALPHA = [1e0, 1e-1, 1e-2, 1e-3]
KR_GAMMA = np.logspace(-5, 5, 6)
KR_DEGREE = [1, 2, 3]

# KR_ALPHA = [1e0]
# KR_GAMMA = [0]
# KR_DEGREE = [1]
