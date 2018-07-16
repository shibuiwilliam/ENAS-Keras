import numpy as np
import os
import sys
import random

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras import Model
from keras.layers import Add, Concatenate, Embedding, LSTM, LSTMCell, RNN, Reshape
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import losses, metrics

import tensorflow as tf

from src.keras_utils import get_weight_initializer
from src.keras_utils import get_weight_regularizer
from src.utils import get_random_str
from src.utils import get_size_str
from src.utils import get_int_list_in_str
from src.utils import generate_random_cell


class ControllerRNNGenerator(object):
  def __init__(self,
               controller_network_name,
               num_nodes,
               num_opers,
               lstm_cell_units=32,
               baseline_decay=0.999,
               opt=Adam(lr=0.00035, decay=1e-3, amsgrad=True)):
    self.controller_network_name = controller_network_name
    self.num_nodes = num_nodes
    self.num_opers = num_opers
    self.lstm_cell_units = lstm_cell_units
    self.opt = opt
    self.controller_rnn = self.generate_controller_rnn()
    self.baseline = None
    self.baseline_decay = baseline_decay    
    
  def lstm_reshape(self, inputs, name_prefix, index, 
                   reshaped_inputs=None, initial=False):
    name_prefix = "{0}_{1}_{2}".format(self.controller_network_name, 
                                       name_prefix, index)
    cell = LSTMCell(self.lstm_cell_units, 
                    kernel_initializer=get_weight_initializer(initializer="lstm"),
                    recurrent_initializer=get_weight_initializer(initializer="lstm"))
    if initial:
      x = RNN(cell, return_state=True,
              name="{0}_{1}".format(name_prefix, "lstm"))(inputs)
    else:
      x = RNN(cell, return_state=True,
              name="{0}_{1}".format(name_prefix, "lstm"))(reshaped_inputs,
                                                          initial_state=inputs[1:])   
    rx = Reshape((-1,self.lstm_cell_units),
                 name="{0}_{1}".format(name_prefix, "reshape"))(x[0])
    return x, rx
  
  def dense_softmax(self, inputs, num_classes, name_prefix, index):
    name_prefix = "{0}_{1}_{2}".format(self.controller_network_name, 
                                       name_prefix, index)
    y = Dense(num_classes,
              name="{0}_{1}".format(name_prefix, "dense"))(inputs)
    y = Activation(activation="softmax",
                   name="{0}_{1}".format(name_prefix, "softmax"))(y)
    return y
  
  def generate_controller_rnn(self):
    outputs = []
    controller_input = Input(shape=(1,1,) ,
                             name="{0}_{1}".format(self.controller_network_name, 
                                                   "input"))

    for i in range(2, self.num_nodes):
      for o in ["inputL", "inputR", "operL", "operR"]:
        if i == 2 and o == "inputL":
          _x, _rx, _initial = controller_input, None, True
        else:
          _x, _rx, _initial = x, rx, False
        
        if o in ["inputL", "inputR"]:
          _num_classes = i
        else:
          _num_classes = self.num_opers
          
        x, rx = self.lstm_reshape(inputs=_x, name_prefix=o, index=i, 
                                  reshaped_inputs=_rx, 
                                  initial=_initial)
        y = self.dense_softmax(inputs=x[0], num_classes=_num_classes, 
                               name_prefix=o, index=i)
        outputs.append(y)

    controller_rnn = Model(inputs=controller_input,
                           outputs=outputs)
    return controller_rnn
    
  def compile_controller_rnn(self, reward):
    def _controller_loss(y_true, y_pred):
      if self.baseline is None:
        self.baseline = 0
      else:
        baseline = (1 - self.baseline_decay) * (self.baseline - reward)
      return y_pred * (reward - self.baseline)
    
    def _define_loss(controller_loss):
      outputs_loss = {}
      for i in range(2, self.num_nodes):
        outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "inputL", 
                                          i, "softmax")] = controller_loss
        outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "inputR",
                                          i, "softmax")] = controller_loss
        outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "operL",
                                          i, "softmax")] = controller_loss
        outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "operR",
                                          i, "softmax")] = controller_loss
      return outputs_loss
    
    self.controller_rnn.compile(loss=_define_loss(_controller_loss),
                                optimizer=self.opt)


class ControllerRNNManager(object):
  def __init__(self,
               controller_rnn_instance,
               input_x,
               reward = 0,
               temperature = 5.0,
               tanh_constant = 2.5):
    self.CR = controller_rnn_instance
    self.controller_rnn = self.CR.controller_rnn
    self.reward = reward
    self.input_x = input_x
    self.num_nodes = self.CR.num_nodes
    self.temperature = temperature
    self.tanh_constant = tanh_constant
  
  def compile_with_newest_reward(self):
    self.CR.compile_controller_rnn(self.reward)
    
  def train_controller_rnn(self,
                           targets,
                           batch_size = 1,
                           epochs = 50,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')]):
    self.compile_with_newest_reward()
    self.controller_rnn.fit(self.input_x, targets,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)    
    
  def softmax_predict(self):
    self.compile_with_newest_reward()
    return self.controller_rnn.predict(self.input_x)
    
  def random_sample_softmax(self, controller_pred):
    sample_softmax = []
    for cp in controller_pred:
      cp /=self.temperature
      cp = self.tanh_constant * np.tanh(cp)
      cp = np.exp(cp) / np.sum(np.exp(cp))
      cp = np.array([np.random.multinomial(1, cp[0])])
      sample_softmax.append(cp)
    return sample_softmax
  
  def convert_pred_to_cell(self, controller_pred):
    cell_pred = {}
    for p in range(2, self.num_nodes):
      pos = list(range((p-2)*4, ((p-2)*4)+4))
      cell_pred[p] = {"L" :[np.argmax(controller_pred[pos[0]]), 
                            np.argmax(controller_pred[pos[2]])],
                      "R" :[np.argmax(controller_pred[pos[1]]), 
                            np.argmax(controller_pred[pos[3]])]}
    return cell_pred
  
  def convert_pred_to_ydict(self, controller_pred):
    ydict = {}
    name_prefix = self.CR.controller_network_name
    for i in range(2, self.num_nodes):
      pos = list(range((i-2)*4, ((i-2)*4)+4))
      ydict["{0}_{1}_{2}_{3}".format(name_prefix, "inputL", i, "softmax")] = controller_pred[pos[0]] 
      ydict["{0}_{1}_{2}_{3}".format(name_prefix, "inputR", i, "softmax")] = controller_pred[pos[1]]
      ydict["{0}_{1}_{2}_{3}".format(name_prefix, "operL", i, "softmax")] = controller_pred[pos[2]]
      ydict["{0}_{1}_{2}_{3}".format(name_prefix, "operR", i, "softmax")] = controller_pred[pos[3]]
    return ydict
