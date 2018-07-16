# coding: utf-8


import numpy as np
import os
import sys
import shutil
import gc

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import tensorflow as tf



from src.child_network_micro_search import NetworkOperation
from src.child_network_micro_search import NetworkOperationController
from src.child_network_micro_search import CellGenerator
from src.child_network_micro_search import ChildNetworkGenerator
from src.child_network_micro_search import ChildNetworkManager

from src.controller_network import ControllerRNNGenerator
from src.controller_network import ControllerRNNManager





config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=False
    )
)
sess = tf.Session(config=config)
K.set_session(sess)




# Load Cifar10 dataset


child_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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




# Common parameters
num_nodes=6
num_opers=5
search_epochs = 100
sample_nums = 10
_sep = "-"*10

# Parameters for controller RNN
reward = 0
controller_input_x = np.array([[[num_opers+num_nodes]]])
controller_lstm_cell_units = 32
controller_baseline_decay = 0.99
controller_opt = Adam(lr=0.0001, decay=1e-6, amsgrad=True)

controller_batch_size = 1
controller_epochs = 50
controller_callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')]

controller_temperature = 5.0
controller_tanh_constant = 2.5

# Parameters for child network
child_network_name="cifar10_cnn"
child_input_shape=(32,32,3)
child_init_filters=64
child_network_definition=["N","N","R","N","N","R"]
child_weight_directory="./weights"
initialize_child_weight_directory=False

child_opt_loss='categorical_crossentropy'
child_opt=Adam(lr=0.0001, decay=1e-6, amsgrad=True)
child_opt_metrics=['accuracy']

child_val_index = [i for i in range(len(y_test))]
child_val_batch_size = 256

child_batch_size = 32
child_epochs = 50
child_callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')]

child_data_gen = ImageDataGenerator(
    rotation_range=90, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

child_train_records = []

if initialize_child_weight_directory:
  if os.path.exists(child_weight_directory):
    shutil.rmtree(child_weight_directory)




# Controller RNN

NCR = ControllerRNNGenerator(controller_network_name="normalcontroller",
                             num_nodes=num_nodes,
                             num_opers=num_opers,
                             lstm_cell_units=controller_lstm_cell_units,
                             baseline_decay=controller_baseline_decay,
                             opt=controller_opt)
RCR = ControllerRNNGenerator(controller_network_name="reductioncontroller",
                             num_nodes=num_nodes,
                             num_opers=num_opers,
                             lstm_cell_units=controller_lstm_cell_units,
                             baseline_decay=controller_baseline_decay,
                             opt=controller_opt)

NCRM = ControllerRNNManager(controller_rnn_instance = NCR,
                            input_x = controller_input_x,
                            reward = reward,
                            temperature = controller_temperature,
                            tanh_constant = controller_tanh_constant)
RCRM = ControllerRNNManager(controller_rnn_instance = RCR,
                            input_x = controller_input_x,
                            reward = reward,
                            temperature = controller_temperature,
                            tanh_constant = controller_tanh_constant)




# Efficient neural architecture search
## Micro search for CNN cells




for e in range(search_epochs):
  print("SEARCH EPOCH: {0} / {1}".format(e, search_epochs))
  print("{0} sampling cells {0}".format(_sep))
  normal_controller_pred = NCRM.softmax_predict()
  reduction_controller_pred = RCRM.softmax_predict()

  normal_pred_dict = NCRM.convert_pred_to_ydict(normal_controller_pred)
  reduction_pred_dict = RCRM.convert_pred_to_ydict(reduction_controller_pred)

  sample_cells = []
  for _ in range(sample_nums):
    sample_cell = {}
    random_normal_pred = NCRM.random_sample_softmax(normal_controller_pred)
    random_reduction_pred = RCRM.random_sample_softmax(reduction_controller_pred)

    sample_cell["normal_cell"] = NCRM.convert_pred_to_cell(random_normal_pred)
    sample_cell["reduction_cell"] = RCRM.convert_pred_to_cell(random_reduction_pred)
    sample_cells.append(sample_cell)

  val_batch = np.random.choice(child_val_index,
                               child_val_batch_size, 
                               replace=False)
  x_val_batch = x_test[val_batch]
  y_val_batch = y_test[val_batch]
  best_val_acc = 0
  best_cell_index = 0

  for i in range(len(sample_cells)):
    print("{0} evaluating sample: {1} {0}\ncell: ".format(_sep, 
                                                          i))
    for k,v in sample_cells[i].items():
      print("{0}: {1}".format(k,v))
    NO = NetworkOperation()
    NOC = NetworkOperationController(network_name=child_network_name,
                                     classes=child_classes,
                                     input_shape=child_input_shape,
                                     init_filters=child_init_filters,
                                     NetworkOperationInstance=NO)
    CG = CellGenerator(num_nodes=num_nodes,                   
                       normal_cell=sample_cells[i]["normal_cell"],
                       reduction_cell=sample_cells[i]["reduction_cell"],
                       NetworkOperationControllerInstance=NOC)
    CNG = ChildNetworkGenerator(child_network_definition=child_network_definition,
                                CellGeneratorInstance=CG,
                                opt_loss=child_opt_loss,
                                opt=child_opt,
                                opt_metrics=child_opt_metrics)

    CNM = ChildNetworkManager(weight_directory=child_weight_directory)
    CNM.set_model(CNG.generate_child_network())    
    CNM.set_weight_to_layer()
    val_acc = CNM.evaluate_child_network(x_val_batch, y_val_batch)
    print(val_acc)
    if best_val_acc < val_acc[1]:
      best_val_acc = val_acc[1]
      best_cell_index = i
    del CNM.model
    del CNM
    del CNG
    del CG
    del NOC
    del NO
    for j in range(30):
      gc.collect()

  print("best val accuracy: {0}\nthe current best cell:".format(best_val_acc))
  for k,v in sample_cells[best_cell_index].items():
    print("{0}: {1}".format(k,v))
  print("{0} train child network with the current best cell {0}".format(_sep))
    
  NO = NetworkOperation()
  NOC = NetworkOperationController(network_name=child_network_name,
                                   classes=child_classes,
                                   input_shape=child_input_shape,
                                   init_filters=child_init_filters,
                                   NetworkOperationInstance=NO)
  CG = CellGenerator(num_nodes=num_nodes,                   
                     normal_cell=sample_cells[best_cell_index]["normal_cell"],
                     reduction_cell=sample_cells[best_cell_index]["reduction_cell"],
                     NetworkOperationControllerInstance=NOC)
  CNG = ChildNetworkGenerator(child_network_definition=child_network_definition,
                              CellGeneratorInstance=CG,
                              opt_loss=child_opt_loss,
                              opt=child_opt,
                              opt_metrics=child_opt_metrics)

  CNM = ChildNetworkManager(weight_directory=child_weight_directory)
  CNM.set_model(CNG.generate_child_network())
  print("MODEL SUMMARY: {0}\n".format(CNM.model.summary()))
  CNM.set_weight_to_layer()
  CNM.train_child_network(x_train=x_train, y_train=y_train,
                          validation_data=(x_test, y_test),
                          batch_size = child_batch_size,
                          epochs = child_epochs,
                          callbacks=child_callbacks,
                          data_gen=child_data_gen)
  CNM.save_layer_weight()
  print("{0} training finished {0}".format(_sep))

  val_acc = CNM.evaluate_child_network(x_test, y_test)
  reward = val_acc[1]
  print("evaluation loss: {0}\nevaluation acc: {1}".format(val_acc[0],
                                                           val_acc[1]))

  child_train_record = {}
  child_train_record["normal_cell"] = sample_cells[best_cell_index]["normal_cell"]
  child_train_record["reduction_cell"] = sample_cells[best_cell_index]["reduction_cell"]
  child_train_record["val_loss"] = val_acc[0]
  child_train_record["reward"] = val_acc[1]
  print("epoch: {0}\nrecord: ".format(e))
  for k,v in child_train_record.items():
    print("{0}: {1}".format(k,v))
  child_train_records.append(child_train_record)
    
  if e == search_epochs - 1:
    break
    
  del CNM.model
  del CNM
  del CNG
  del CG
  del NOC
  del NO
  for j in range(30):
    gc.collect()

    
  print("{0} train controller rnn {0}".format(_sep))

  NCRM.reward = reward
  RCRM.reward = reward
  NCRM.train_controller_rnn(targets=normal_pred_dict,
                            batch_size = controller_batch_size,
                            epochs = controller_epochs,
                            callbacks=controller_callbacks)
  RCRM.train_controller_rnn(targets=reduction_pred_dict,
                            batch_size = controller_batch_size,
                            epochs = controller_epochs,
                            callbacks=controller_callbacks)
  print("{0} training finished {0}".format(_sep))
  print("{0} FINISHED SEARCH EPOCH {1} / {2} {0}".format(_sep,
                                                         e, 
                                                         search_epochs))
    
print("{0} FINISHED NEURAL ARCHITECTURE SEARCH {0}".format(_sep))
print("training records:\n{0}".format(child_train_records))
print("final child network:\n{0}".format(CNM.model.summary()))

