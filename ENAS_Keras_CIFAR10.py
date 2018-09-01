import numpy as np
import os
import sys
import shutil
import gc
from copy import deepcopy

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LearningRateScheduler

import tensorflow as tf

from ENAS import EfficientNeuralArchitectureSearch
from src.utils import sgdr_learning_rate, get_random_eraser, MixupGenerator

# Load Cifar10 dataset


child_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if child_classes != 10:
    train_part = np.where(y_train < child_classes)[0]
    test_part = np.where(y_test < child_classes)[0]
    x_train = x_train[train_part]
    y_train = y_train[train_part]
    x_test = x_test[test_part]
    y_test = y_test[test_part]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, child_classes)
y_test = to_categorical(y_test, child_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

child_data_gen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(v_l=0, v_h=255))

data_flow_gen = MixupGenerator(
    x_train, y_train, batch_size=128, alpha=0.2, datagen=child_data_gen)()

nt = sgdr_learning_rate(n_Max=0.05, n_min=0.001, ranges=5, init_cycle=10)

ENAS = EfficientNeuralArchitectureSearch(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    child_network_name="cifar10_cnn",
    child_classes=child_classes,
    child_input_shape=(32, 32, 3),
    num_nodes=7,
    num_opers=5,
    controller_lstm_cell_units=32,
    controller_baseline_decay=0.99,
    controller_opt=Adam(lr=0.00035, decay=1e-3, amsgrad=True),
    controller_batch_size=1,
    controller_epochs=50,
    controller_callbacks=[
        EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    ],
    controller_temperature=5.0,
    controller_tanh_constant=2.5,
    controller_normal_model_file="cifar10_normal_controller.hdf5",
    controller_reduction_model_file="cifar10_reduction_controller.hdf5",
    child_init_filters=64,
    child_network_definition=["N", "N", "R", "N", "N", "R"],
    child_weight_directory="./cifar10_weights",
    child_opt_loss='categorical_crossentropy',
    child_opt=SGD(lr=0.05, nesterov=True),
    child_opt_metrics=['accuracy'],
    child_val_batch_size=128,
    child_batch_size=128,
    child_epochs=len(nt),
    child_lr_scedule=nt,
    start_from_record=True,
    run_on_jupyter=False,
    initialize_child_weight_directory=False,
    save_to_disk=True,
    set_from_dict=True,
    data_gen=child_data_gen,
    data_flow_gen=data_flow_gen)
ENAS.search_neural_architecture()

print(ENAS.best_normal_cell)
print(ENAS.best_reduction_cell)

ENAS.train_best_cells()
