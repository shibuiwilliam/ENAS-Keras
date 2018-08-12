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
from keras.callbacks import EarlyStopping, LearningRateScheduler

import tensorflow as tf

from src.child_network_micro_search import NetworkOperation
from src.child_network_micro_search import NetworkOperationController
from src.child_network_micro_search import CellGenerator
from src.child_network_micro_search import ChildNetworkController

from src.controller_network import ControllerRNNController


class EfficientNeuralArchitectureSearch(object):
    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 child_network_name,
                 child_classes,
                 child_input_shape,
                 num_nodes=6,
                 num_opers=5,
                 search_epochs = 10,
                 sample_nums = 5,
                 controller_lstm_cell_units = 32,
                 controller_baseline_decay = 0.99,
                 controller_opt = Adam(lr=0.00035, decay=1e-3, amsgrad=True),
                 controller_batch_size = 1,
                 controller_epochs = 50,
                 controller_callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')],
                 controller_temperature = 5.0,
                 controller_tanh_constant = 2.5,
                 controller_normal_model_file = "normal_controller.hdf5",
                 controller_reduction_model_file = "reduction_controller.hdf5",
                 child_init_filters=64,
                 child_network_definition=["N","N","R"],
                 child_weight_directory="./weights",
                 child_opt_loss='categorical_crossentropy',
                 child_sample_opt=SGD(lr=0.001, decay=1e-6, nesterov=True),
                 child_opt=SGD(lr=0.05, decay=1e-6, nesterov=True),
                 child_opt_metrics=['accuracy'],
                 child_val_batch_size = 128,
                 child_batch_size = 128,
                 child_epochs = 150,
                 child_callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto'),
                                    LearningRateScheduler(lambda ep: float(np.linspace(0.05, 0.001, 20)[ep]))],
                 run_on_jupyter = True,
                 initialize_child_weight_directory=True,
                 save_to_disk=False,
                 set_from_dict=True,
                 data_gen=None,
                 data_flow_gen=None):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.num_nodes=num_nodes
        self.num_opers=num_opers
        self.search_epochs = search_epochs
        self.sample_nums = sample_nums
        
        self.controller_lstm_cell_units = controller_lstm_cell_units
        self.controller_baseline_decay = controller_baseline_decay
        self.controller_opt = controller_opt
        self.controller_batch_size = controller_batch_size
        self.controller_epochs = controller_epochs
        self.controller_callbacks = controller_callbacks
        self.controller_temperature = controller_temperature
        self.controller_tanh_constant = controller_tanh_constant
        self.controller_input_x = np.array([[[self.num_opers+self.num_nodes]]])
        self.controller_normal_model_file = controller_normal_model_file
        self.controller_reduction_model_file = controller_reduction_model_file
        
        self.child_network_name=child_network_name
        self.child_classes=child_classes
        self.child_input_shape=child_input_shape
        self.child_init_filters=child_init_filters
        self.child_network_definition=child_network_definition
        self.child_weight_directory=child_weight_directory
        self.child_opt_loss=child_opt_loss
        self.child_sample_opt = child_sample_opt
        self.child_opt=child_opt
        self.child_opt_metrics=child_opt_metrics
        self.child_batch_size = child_batch_size
        self.child_epochs = child_epochs
        self.child_callbacks = child_callbacks
        self.child_train_records = []
        self.child_val_batch_size = child_val_batch_size
        self.child_train_index = self.get_child_index(self.y_train)
        self.child_val_index = self.get_child_index(self.y_test)
        
        self.run_on_jupyter = run_on_jupyter
        self.save_to_disk=save_to_disk
        self.set_from_dict=set_from_dict
        self.data_gen=data_gen
        self.data_flow_gen=data_flow_gen
        self.initialize_child_weight_directory=initialize_child_weight_directory
        
        self.reward = 0
        
        self.NCRC = self.define_controller_rnn(controller_network_name="normalcontroller", 
                                               model_file=self.controller_normal_model_file)
        self.RCRC = self.define_controller_rnn(controller_network_name="reductioncontroller", 
                                               model_file=self.controller_reduction_model_file)
      
        self.weight_dict = {}
        
        self._sep = "-"*10
        
        self._initialize_child_weight_directory()
                    
    def _initialize_child_weight_directory(self):
        if self.initialize_child_weight_directory:
            print("initialize: {0}".format(self.child_weight_directory))
            if os.path.exists(self.child_weight_directory):
                shutil.rmtree(self.child_weight_directory)
        
    def get_child_index(self, y):
        return [i for i in range(len(y))]
    
    def define_controller_rnn(self, controller_network_name, model_file=None):
        return ControllerRNNController(controller_network_name=controller_network_name,
                                     num_nodes=self.num_nodes,
                                     num_opers=self.num_opers,
                                     input_x = self.controller_input_x,
                                     reward = self.reward,
                                     temperature = self.controller_temperature,
                                     tanh_constant = self.controller_tanh_constant,
                                     model_file = model_file,
                                     lstm_cell_units=self.controller_lstm_cell_units,
                                     baseline_decay=self.controller_baseline_decay,
                                     opt=self.controller_opt)
    
    def train_controller_rnn(self, normal_pred_dict, reduction_pred_dict):        
          self.NCRC.reward = self.reward
          self.RCRC.reward = self.reward
          print("{0} training {1} {0}".format(self._sep, self.NCRC.controller_network_name))
          self.NCRC.train_controller_rnn(targets=normal_pred_dict,
                                         batch_size = self.controller_batch_size,
                                         epochs = self.controller_epochs,
                                         callbacks=self.controller_callbacks)
          print("{0} training {1} {0}".format(self._sep, self.RCRC.controller_network_name))
          self.RCRC.train_controller_rnn(targets=reduction_pred_dict,
                                         batch_size = self.controller_batch_size,
                                         epochs = self.controller_epochs,
                                         callbacks=self.controller_callbacks)
        
    def define_network_operations(self):
        return NetworkOperationController(network_name=self.child_network_name,
                                         classes=self.child_classes,
                                         input_shape=self.child_input_shape,
                                         init_filters=self.child_init_filters,
                                         NetworkOperationInstance=NetworkOperation())
    
    def generate_child_cell(self, normal_cell, reduction_cell, NOC):
        return CellGenerator(num_nodes=self.num_nodes,                   
                           normal_cell=normal_cell,
                           reduction_cell=reduction_cell,
                           NetworkOperationControllerInstance=NOC)
        
    def define_chile_network(self, CG, opt):
        return ChildNetworkController(child_network_definition=self.child_network_definition,
                                   CellGeneratorInstance=CG,
                                   weight_dict=self.weight_dict, 
                                   weight_directory=self.child_weight_directory,
                                   opt_loss=self.child_opt_loss,
                                   opt=opt,
                                   opt_metrics=self.child_opt_metrics)
    
    def predict_architecture(self, CRC):
        controller_pred = CRC.softmax_predict()
        pred_dict = CRC.convert_pred_to_ydict(controller_pred)
        return controller_pred, pred_dict
    
    def get_sample_cells(self, normal_controller_pred, reduction_controller_pred):
        sample_cells = []
        for _ in range(self.sample_nums):
          sample_cell = {}
          random_normal_pred = self.NCRC.random_sample_softmax(normal_controller_pred)
          random_reduction_pred = self.RCRC.random_sample_softmax(reduction_controller_pred)

          sample_cell["normal_cell"] = self.NCRC.convert_pred_to_cell(random_normal_pred)
          sample_cell["reduction_cell"] = self.RCRC.convert_pred_to_cell(random_reduction_pred)
          sample_cells.append(sample_cell)
        return sample_cells
    
    
    def final_output(self, CNC, val_acc):
        if self.run_on_jupyter: 
            from IPython.display import clear_output
            clear_output(wait=True)
        print("{0} FINISHED NEURAL ARCHITECTURE SEARCH {0}".format(self._sep))
        print("training records:\n{0}".format(self.child_train_records))
        print("final child network:\n")
        print(CNC.model.summary())
        print("evaluation loss: {0}\nevaluation acc: {1}".format(val_acc[0],
                                                                 val_acc[1]))
    def get_batch(self, index, size, train=True):
        _batch = np.random.choice(index,
                                  size, 
                                  replace=False)
        if train:
             return self.x_train[_batch], self.y_train[_batch]
        else:
             return self.x_test[_batch], self.y_test[_batch]
        
    
    def search_neural_architecture(self):
        for e in range(self.search_epochs):
          print("SEARCH EPOCH: {0} / {1}".format(e, self.search_epochs))
          print("{0} sampling cells {0}".format(self._sep))
          normal_controller_pred, normal_pred_dict = self.predict_architecture(self.NCRC)
          reduction_controller_pred, reduction_pred_dict = self.predict_architecture(self.RCRC)
            
          sample_cells = self.get_sample_cells(normal_controller_pred, reduction_controller_pred)

          x_train_batch, y_train_batch = self.get_batch(self.child_train_index,
                                                        self.child_val_batch_size*10,
                                                        True)
          x_val_batch, y_val_batch = self.get_batch(self.child_val_index,
                                                    self.child_val_batch_size,
                                                    False)
        
          best_val_acc = 0
          best_cell_index = 0

          for i in range(len(sample_cells)):
            print("{0} evaluating sample: {1} {0}\ncell: ".format(self._sep, 
                                                                  i))
            for k,v in sample_cells[i].items():
              print("{0}: {1}".format(k,v))
            CG = self.generate_child_cell(sample_cells[i]["normal_cell"],
                                          sample_cells[i]["reduction_cell"],
                                          self.define_network_operations())
            CNC = self.define_chile_network(CG, self.child_sample_opt)
            CNC.set_weight_to_layer(set_from_dict=self.set_from_dict)
            CNC.train_child_network(x_train=x_train_batch, y_train=y_train_batch,
                                    batch_size = self.child_batch_size,
                                    epochs = 1,
                                    callbacks=None,
                                    data_gen=self.data_gen,
                                    data_flow_gen=self.data_flow_gen)
            val_acc = CNC.evaluate_child_network(x_val_batch, y_val_batch)
            print(val_acc)
            CNC.close_tf_session()
            
            del CNC.weight_dict
            del CNC.model
            del CNC.CG
            del CNC
            del CG.NOC
            del CG
            gc.collect()
            
            if best_val_acc < val_acc[1]:
              best_val_acc = val_acc[1]
              best_cell_index = i

          
          del x_train_batch
          del y_train_batch
          del x_val_batch
          del y_val_batch
          gc.collect()
            
            
          print("best val accuracy: {0}\nthe current best cell:".format(best_val_acc))
          for k,v in sample_cells[best_cell_index].items():
            print("{0}: {1}".format(k,v))
          print("{0} train child network with the current best cell {0}".format(self._sep))
        
          CG = self.generate_child_cell(sample_cells[best_cell_index]["normal_cell"],
                                        sample_cells[best_cell_index]["reduction_cell"],
                                        self.define_network_operations())
          CNC = self.define_chile_network(CG, self.child_opt)

          print("MODEL SUMMARY:\n")
          print(CNC.model.summary())
          CNC.set_weight_to_layer(set_from_dict=self.set_from_dict)
          CNC.train_child_network(x_train=self.x_train, y_train=self.y_train,
                                  validation_data=(self.x_test, self.y_test),
                                  batch_size = self.child_batch_size,
                                  epochs = self.child_epochs,
                                  callbacks=self.child_callbacks,
                                  data_gen=self.data_gen,
                                  data_flow_gen=self.data_flow_gen)
          CNC.fetch_layer_weight(save_to_disk=self.save_to_disk)
          for k,v in CNC.weight_dict.items():
              self.weight_dict[k] = v
          print("{0} training finished {0}".format(self._sep))

          val_acc = CNC.evaluate_child_network(self.x_test, self.y_test)

          self.reward = val_acc[1]
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
          self.child_train_records.append(child_train_record)

          if e == self.search_epochs - 1:
            self.final_output(CNC, val_acc)
            break
            
          CNC.close_tf_session()
          
                    
          del CNC.weight_dict
          del CNC.model
          del CNC.CG
          del CNC
          del CG.NOC
          del CG
          gc.collect()
            
                    
          print("{0} train controller rnn {0}".format(self._sep))
          self.train_controller_rnn(normal_pred_dict, reduction_pred_dict)
          self.NCRC.save_model()
          self.RCRC.save_model()
            
          print("{0} training finished {0}".format(self._sep))
          print("{0} FINISHED SEARCH EPOCH {1} / {2} {0}".format(self._sep,
                                                                 e, 
                                                                 self.search_epochs))
          if self.run_on_jupyter: 
                from IPython.display import clear_output
                clear_output(wait=True)