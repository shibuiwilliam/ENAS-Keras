import numpy as np
import os
import sys
import random
import string
from sklearn.externals import joblib

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras import Model
from keras.layers import Add, Concatenate, Reshape
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, ZeroPadding2D, Cropping2D
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras import losses, metrics

import tensorflow as tf

from src.keras_utils import get_weight_initializer
from src.keras_utils import get_weight_regularizer
from src.utils import get_random_str
from src.utils import get_size_str
from src.utils import get_int_list_in_str
from src.utils import generate_random_cell
from src.utils import make_dir


class NetworkOperation(object):
  def __init__(self):
    None  
  
  def input_layer(self, input_shape, name_prefix):
    return Input(shape=input_shape, 
                 name="{0}_input_".format(name_prefix))
  
  def relu_sepconv2d_bn(self, inputs, name_prefix, rep,
                        kernel_size, current_filters, strides=(1,1)):  
    x = Activation("relu", 
                   name="{0}_relu_{1}_".format(name_prefix,
                                              rep))(inputs)
    x = SeparableConv2D(filters=current_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="same", 
                        depthwise_initializer=get_weight_initializer(),
                        pointwise_initializer=get_weight_initializer(),
                        depthwise_regularizer=get_weight_regularizer(),
                        pointwise_regularizer=get_weight_regularizer(),
                        name="{0}_sepconv2d_{1}_".format(name_prefix,
                                                        rep))(x)
    x = BatchNormalization(name="{0}_bn_{1}_".format(name_prefix,
                                                     rep))(x)
    return x
  
  def separable_conv_3x3(self, inputs, name_prefix, current_filters,
                         strides=(1,1)):    
    x = self.relu_sepconv2d_bn(inputs, name_prefix, rep=0, kernel_size=(3,3),
                               current_filters=current_filters,
                               strides=strides)
    x = self.relu_sepconv2d_bn(x, name_prefix, rep=1, kernel_size=(3,3),
                               current_filters=current_filters, 
                               strides=(1,1))
    return x
  
  def separable_conv_5x5(self, inputs, name_prefix, current_filters,
                         strides=(1,1)):
    x = self.relu_sepconv2d_bn(inputs, name_prefix, rep=0, kernel_size=(5,5),
                               current_filters=current_filters, 
                               strides=strides)
    x = self.relu_sepconv2d_bn(x, name_prefix, rep=1, kernel_size=(5,5), 
                               current_filters=current_filters,
                               strides=(1,1))
    return x
  
  def max_pooling_3x3(self, inputs, name_prefix, strides=(1,1)):    
    x = MaxPooling2D(pool_size=(3,3), strides=strides, padding="same", 
                     name="{0}_maxpool2d_".format(name_prefix))(inputs)    
    return x
  
  def average_pooling_3x3(self, inputs, name_prefix, strides=(1,1)):    
    x = AveragePooling2D(pool_size=(3,3), strides=strides, padding="same", 
                         name="{0}_avepool2d_".format(name_prefix))(inputs)    
    return x
  
  def identity(self, inputs, name_prefix):
    x = Activation("linear", 
                   name="{0}_linear_".format(name_prefix))(inputs)    
    return x
 
  def adjust_output_depth(self, inputs, name_prefix, current_filters):
    x = Activation("relu", 
                   name="{0}_relu_".format(name_prefix))(inputs)
    x = Conv2D(filters=current_filters,
               kernel_size=(1,1),
               strides=(1,1), 
               padding="same", 
               kernel_initializer=get_weight_initializer(),
               kernel_regularizer=get_weight_regularizer(),
               name="{0}_conv2d_".format(name_prefix))(x)
    x = BatchNormalization(name="{0}_bn_".format(name_prefix))(x)
    return x
  
  def reduce_output_size(self, inputs, name_prefix, rep, 
                         half_current_filters):
    x_0 = AveragePooling2D(pool_size=(1,1),
                           strides=(2,2),
                           padding="valid", 
                           name="{0}_avepool2d_{1}a_".format(name_prefix, rep))(inputs)
    x_0 = Conv2D(filters=half_current_filters,
                 kernel_size=(1,1),
                 strides=(1,1), 
                 padding="valid", 
                 kernel_initializer=get_weight_initializer(),
                 kernel_regularizer=get_weight_regularizer(),
                 name="{0}_conv2d_{1}a_".format(name_prefix, rep))(x_0)
    
    x_1 = ZeroPadding2D(padding=((0,1),(0,1)),
                        name="{0}_zeropad2d_{1}b_".format(name_prefix, rep))(inputs)
    x_1 = Cropping2D(cropping=((1,0),(1,0)),
                     name="{0}_crop2d_{1}b_".format(name_prefix, rep))(x_1)
    x_1 = AveragePooling2D(pool_size=(1,1),
                           strides=(2,2), 
                           padding="valid", 
                           name="{0}_avepool2d_{1}b_".format(name_prefix, rep))(x_1)
    x_1 = Conv2D(filters=half_current_filters,
                 kernel_size=(1,1),
                 strides=(1,1), 
                 padding="valid", 
                 kernel_initializer=get_weight_initializer(),
                 kernel_regularizer=get_weight_regularizer(),
                 name="{0}_conv2d_{1}b_".format(name_prefix, rep))(x_1)

    x = Concatenate(name="{0}_concat_{1}_".format(name_prefix, rep))([x_0, x_1])
    x = BatchNormalization(name="{0}_bn_{1}_".format(name_prefix, rep))(x)
    return x
  
  def classification_layer(self, inputs, name_prefix, classes):
    x = GlobalAveragePooling2D(name="{0}_gap2d_".format(name_prefix))(inputs)
    x = Dense(classes,
              kernel_initializer=get_weight_initializer(),
              kernel_regularizer=get_weight_regularizer(),
              name="{0}_dense_".format(name_prefix))(x)
    x = Activation("softmax",
                   name="{0}_softmax_".format(name_prefix))(x)
    return x


class NetworkOperationController(object):
  def __init__(self, 
               network_name,
               classes,
               input_shape,
               init_filters,
               NetworkOperationInstance):
    self.network_name = network_name.replace("_","-")
    self.classes = classes
    self.input_shape = input_shape
    self.init_filters = init_filters
    self.NO = NetworkOperationInstance
    
    self.num_reductions = 0
    self.num_normals = 0
        
  def current_filters(self):
    return (2 ** self.num_reductions) * self.init_filters
  
  def generate_layer_name(self, cell_type, type_num,
                          node_num, inputs_node_num, LorR, operation):
    """
    layer_name = {network_name}_{cell_type}_{type_num}_{node_num}_{node_from}_{LorR}_{operation}_{opt}
    {cell_type} is either 'normal' or 'reduction'
    {type_num} is identity for the cell type starting from 0
    {node_num} is identity for node
    {node_from} is input node num
    {LorR} is 'L' for Left or 'R' for Right operation; 'C' for add, concat, input and classification layers
    {operation} is added when the layer generated
    {opt} is optional
    """
    return "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(self.network_name,
                                                cell_type, 
                                                type_num,
                                                node_num, 
                                                inputs_node_num,
                                                LorR, 
                                                operation)
  
  def generate_input_layer(self):
    name_prefix = self.generate_layer_name(cell_type="fixed", 
                                           type_num=999, 
                                           node_num=999,
                                           inputs_node_num=999,
                                           LorR="C",
                                           operation="input")
    return self.NO.input_layer(input_shape=self.input_shape, 
                               name_prefix=name_prefix)
  
  def generate_classification_layer(self, inputs):
    name_prefix = self.generate_layer_name(cell_type="fixed", 
                                           type_num=999, 
                                           node_num=999, 
                                           inputs_node_num=999,
                                           LorR="C",
                                           operation="classification")
    return self.NO.classification_layer(inputs=inputs,
                                        name_prefix=name_prefix,
                                        classes=self.classes)
  
  def get_node_operation_dicts(self):
    return {0:"sepconv3x3", 
            1:"sepconv5x5",
            2:"maxpool3x3",
            3:"avepool3x3",
            4:"identity"}

  def generate_node_operation(self,
                              oper_id, inputs_node_num, inputs, 
                              node_num, LorR, 
                              reduction=False):
    if reduction:
      strides= (1,1) if inputs_node_num > 1 else (2,2)
      cell_type="reduction"
      type_num = self.num_reductions
    else:
      strides=(1,1)
      cell_type="normal"
      type_num = self.num_normals
      
    name_prefix = self.generate_layer_name(cell_type=cell_type, 
                                           type_num=type_num, 
                                           node_num=node_num, 
                                           inputs_node_num=inputs_node_num,
                                           LorR=LorR,
                                           operation=self.get_node_operation_dicts()[oper_id])
    if oper_id == 0:
      return self.NO.separable_conv_3x3(inputs=inputs,
                                        name_prefix=name_prefix, 
                                        current_filters=self.current_filters(),
                                        strides=strides)
    elif oper_id == 1:
      return self.NO.separable_conv_5x5(inputs=inputs,
                                        name_prefix=name_prefix, 
                                        current_filters=self.current_filters(),
                                        strides=strides)
    elif oper_id == 2:
      x = self.NO.max_pooling_3x3(inputs=inputs,
                                  name_prefix=name_prefix,
                                  strides=strides)
      if x.get_shape().as_list()[-1] != self.current_filters():
        x =self.NO.adjust_output_depth(x, 
                                       name_prefix=name_prefix, 
                                       current_filters=self.current_filters())
      return x
    elif oper_id == 3:
      x = self.NO.average_pooling_3x3(inputs=inputs,
                                      name_prefix=name_prefix, 
                                      strides=strides)
      if x.get_shape().as_list()[-1] != self.current_filters():
        x = self.NO.adjust_output_depth(x, 
                                        name_prefix=name_prefix, 
                                        current_filters=self.current_filters())
      return x
    elif oper_id == 4:
      x = self.NO.identity(inputs=inputs,
                           name_prefix=name_prefix)
      if x.get_shape().as_list()[-1] != self.current_filters():
        x = self.NO.adjust_output_depth(x,
                                        name_prefix=name_prefix, 
                                        current_filters=self.current_filters())
      return x
  
  def adjust_layer_sizes(self, x_0, x_1, name_prefix, rep):    
    i = 0
    while True:
      x_0_size = x_0.get_shape().as_list()[1]
      x_1_size = x_1.get_shape().as_list()[1]
      if x_0_size == x_1_size:
        break
      elif x_0_size > x_1_size:
          x_0 = self.NO.reduce_output_size(x_0, 
                                           name_prefix,
                                           rep="{0}.{1}".format(str(rep), str(i)), 
                                           half_current_filters=self.current_filters() // 2)
      elif x_0_size < x_1_size:
          x_1 = self.NO.reduce_output_size(x_1, 
                                           name_prefix,
                                           rep="{0}.{1}".format(str(rep), str(i)), 
                                           half_current_filters=self.current_filters() // 2)
      i += 1
    return x_0, x_1
    
  def add_layers(self, x_0, x_1, node_num, reduction=False):
    """
    merge layers in addition
    """
    if reduction:
      cell_type="reduction"
      type_num = self.num_reductions
    else:
      cell_type="normal"
      type_num = self.num_normals
      
    name_prefix = self.generate_layer_name(cell_type=cell_type, 
                                           type_num=type_num, 
                                           node_num=node_num, 
                                           inputs_node_num=node_num,
                                           LorR="C",
                                           operation="add")
    x_0, x_1 = self.adjust_layer_sizes(x_0, x_1, name_prefix, rep=0)
    return Add(name="{0}_add_".format(name_prefix))([x_0, x_1])
  
  def concat_layers(self, x_list, node_num, reduction=False):
    """
    merge layers in concatenation
    """
    if reduction:
      cell_type="reduction"
      type_num = self.num_reductions
    else:
      cell_type="normal"
      type_num = self.num_normals 

    name_prefix = self.generate_layer_name(cell_type=cell_type, 
                                           type_num=type_num, 
                                           node_num=node_num, 
                                           inputs_node_num=node_num,
                                           LorR="C",
                                           operation="concat")    
    
    smallest_i = self.get_smallest_size_layer(x_list)
    resized_x_list = [x_list[smallest_i]]
    for i in range(len(x_list)):
      if i == smallest_i:
        continue
      else:
        resized_x, _ = self.adjust_layer_sizes(x_list[i], x_list[smallest_i], name_prefix, i)
        resized_x_list.append(resized_x)    
    return Concatenate(name="{0}_concat_".format(name_prefix))(resized_x_list)
  
  def get_smallest_size_layer(self, x_list):
    """
    get id of a layer with the smallest size in the x_list
    """
    smallest_i = None
    for i in range(len(x_list)):
      if smallest_i is None:
        smallest_i = i
      else:
        if x_list[smallest_i].get_shape().as_list()[1] > x_list[i].get_shape().as_list()[1]:
          smallest_i = i
    return smallest_i


class CellGenerator(object):
  def __init__(self,
               num_nodes,
               normal_cell,
               reduction_cell,
               NetworkOperationControllerInstance):
    self.num_nodes = num_nodes
    
    self.normal_cell = normal_cell
    self.reduction_cell = reduction_cell    
    """ cells
    node_num = operation node in int; starts from 2
    input_layer = input node num in int
    oper_id = operation id in int
    {node_num(int): {L: {input_layer:(int), oper_id:(int)},
                     R: {input_layer:(int), oper_id:(int)}},
     node_num(int): {L: {input_layer:(int), oper_id:(int)},
                     R: {input_layer:(int), oper_id:(int)}} ... }
    """
    
    self.NOC = NetworkOperationControllerInstance
    self.input_layer = self.NOC.generate_input_layer()
    
  def generate_classification_layer(self, inputs):
    return self.NOC.generate_classification_layer(inputs)
    
  def generate_cell_operation(self, node_0, node_1,
                              reduction=False):
    """
    node_0 = output from 2 layers before; Input layer for initial normal cell
    node_1 = output from 1 layer before; Input layer for initial normal cell
    """
    node_output = {}
    node_output[0] = node_0
    node_output[1] = node_1
    loose_ends = []
    
    if reduction:
      self.NOC.num_reductions += 1
      node_operation = self.reduction_cell
    else:
      self.NOC.num_normals += 1
      node_operation = self.normal_cell
    
    for n,lrop in node_operation.items():
      loose_ends.append(n)
      lr_outputs = {}
      for lr,op in lrop.items():
        lr_outputs[lr] = self.NOC.generate_node_operation(oper_id=op["oper_id"], 
                                                          inputs_node_num=op["input_layer"],
                                                          inputs=node_output[op["input_layer"]], 
                                                          node_num=n, 
                                                          LorR=lr,
                                                          reduction=reduction)
        if op["input_layer"] in loose_ends:
          loose_ends.remove(op["input_layer"])
      node_output[n] = self.NOC.add_layers(lr_outputs["L"], lr_outputs["R"], 
                                           node_num=n, 
                                           reduction=reduction)
    if len(loose_ends) == 1:
      return node_output[loose_ends[0]]
    else:
      concat_layers = []
      node_num = ""
      for e in loose_ends:
        concat_layers.append(node_output[e])
        node_num += str(e)
      return self.NOC.concat_layers(concat_layers, 
                                    node_num=node_num, 
                                    reduction=reduction)


class ChildNetworkController(object):
  def __init__(self, 
               child_network_definition,
               CellGeneratorInstance,
               weight_dict,
               weight_directory="./",
               opt_loss='categorical_crossentropy',
               opt=SGD(lr=0.001, decay=1e-6, nesterov=True),
               opt_metrics=['accuracy']):
    
    self.child_network_definition = child_network_definition
    """
    child_network_definition is like ["N","N","R","N","N","R"] 
    where N is for normal and R for reduction
    """
    
    self.CG = CellGeneratorInstance
    
    self.opt_loss = opt_loss
    self.opt = opt
    self.opt_metrics = opt_metrics
    
    self.model = self.generate_child_network()
    self.model_dict = self.generate_model_dict()
    
    self.weight_dict = weight_dict
    self.weight_directory = make_dir(weight_directory)
    
    self.graph = tf.get_default_graph()
    
  def generate_child_network(self):
    for i in range(len(self.child_network_definition)):
      if self.child_network_definition[i] == "R":
        reduction = True
      else:
        reduction = False

      if i == 0:
        node_0 = self.CG.input_layer
        node_1 = self.CG.input_layer

      node_0, node_1 = node_1, self.CG.generate_cell_operation(node_0=node_0, 
                                                               node_1=node_1,
                                                               reduction = reduction)
      
    classification_layer = self.CG.generate_classification_layer(node_1)

    return Model(inputs=self.CG.input_layer, 
                 outputs=classification_layer)
    
  def generate_model_dict(self):
    _model_dict = {}
    """
    _model_dict = {layer_num(int): {full_name(str): name,
                                    cell_type(str): normal, 
                                    type_num(str): (int),
                                    node_num(str): (int),
                                    node_from(str): (int),
                                    oper(str): sepconv3x3, 
                                    func(str): relu, 
                                    input_shape(str): [32,32,3](int tuple),
                                    output_shape(str): [32,32,3](int tuple),
                                    param(str): 283
                                    },
                   layer_num(next): {}, ...
                  }
    """
    for l in range(len(self.model.layers)):
      model_name = self.model.layers[l].name.split("_")
      _model_dict[l] = {"full_name": self.model.layers[l].name,
                        "cell_type": model_name[1],
                        "type_num": model_name[2],
                        "node_num": model_name[3],
                        "node_from": model_name[4],
                        "oper": model_name[6],
                        "func": model_name[7],
                        "input_shape": self.model.layers[l].input_shape,
                        "output_shape": self.model.layers[l].output_shape,
                        "param": self.model.layers[l].count_params()
                        }
    return _model_dict
    
  def fetch_layer_weight(self, save_to_disk=False):
    _weight_dict = {}
    for l,d in self.model_dict.items():
      if d["func"] == "bn":
        continue
      if d["param"] > 0:
        weight_name = self.generate_weight_name(d)
        print("keeping weight: {0}".format(weight_name))  
        if weight_name not in _weight_dict:
          _weight_dict[weight_name] = [self.model.layers[l].get_weights(), 1]
        else:
          _weight_dict[weight_name][0] = [_weight_dict[weight_name][0][i] + \
                                          self.model.layers[l].get_weights()[i] \
                                          for i in range(len(_weight_dict[weight_name][0]))]
          _weight_dict[weight_name][1] += 1
          
    for wn, wl in _weight_dict.items():
      w = [wl[0][i]/wl[1] for i in range(len(wl[0]))]
      self.weight_dict[wn] = w
      if save_to_disk:
        print("saving weight: {0}".format(wn))
        joblib.dump(w, 
                    os.path.join(self.weight_directory, "{0}.joblib".format(wn)))
      
  def generate_weight_name(self, d):
    """
    weight_name = 
    {operation}_{func}_{cell_type}_{type_num}_{node_num}_{node_from}_{input HxWxD}_{output HxWxD}_{param} if sepconv
    else {func}_{cell_type}_{type_num}_{node_num}_{node_from}_{input HxWxD}_{output HxWxD}_{param} 
    """
    if d["func"] == "sepconv2d":
      return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(d["oper"],
                                                  d["func"],
                                                  d["cell_type"],
                                                  d["type_num"],
                                                  d["node_num"],
                                                  d["node_from"],
                                                  get_int_list_in_str(d["input_shape"][1:]),
                                                  get_int_list_in_str(d["output_shape"][1:]),
                                                  d["param"])
    else:
      return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(d["func"],
                                              d["cell_type"],
                                              d["type_num"],
                                              d["node_num"],
                                              d["node_from"],
                                              get_int_list_in_str(d["input_shape"][1:]),
                                              get_int_list_in_str(d["output_shape"][1:]),
                                              d["param"])
    
  def load_weight_file(self, file_name):
    return joblib.load(os.path.join(self.weight_directory, file_name)) 

  def get_weight_file_list(self):
    return os.listdir(self.weight_directory)
    
  def set_weight_to_layer(self, set_from_dict=True):
    file_list = self.get_weight_file_list()
    weights = self.model.get_weights()
    i = 0
    changed = False
    for l,d in self.model_dict.items():
      if d["func"] != "bn" and d["param"] > 0:
        weight_name = self.generate_weight_name(d)
        if set_from_dict:
          if weight_name in self.weight_dict:
            changed = True
            print("loading weight: {0}".format(weight_name))
            for w in self.weight_dict[weight_name]:
              weights[i] = w
              i += 1
          elif "{0}.joblib".format(weight_name) in file_list:
            changed = True
            print("loading weight: {0}".format(weight_name))
            weights_from_file = self.load_weight_file("{0}.joblib".format(weight_name))
            for w in weights_from_file:
              weights[i] = w
              i += 1
          else:
            i += len(self.model.layers[l].weights)
        else:
          if weight_name in file_list:
            changed = True
            print("loading weight: {0}".format(weight_name))
            weights_from_file = self.load_weight_file("{0}.joblib".format(weight_name))
            for w in weights_from_file:
              weights[i] = w
              i += 1
          else:
            i += len(self.model.layers[l].weights)
      else:
        i += len(self.model.layers[l].weights)

    if changed:
      print("loading weights to model")
      self.model.set_weights(weights)
      print("loaded")

  def train_child_network(self,
                          x_train, y_train,
                          validation_data=None,
                          batch_size = 32,
                          epochs = 10,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')],
                          data_gen=None,
                          data_flow_gen=None):  
    with self.graph.as_default():
        self.model.compile(loss=self.opt_loss,
                           optimizer=self.opt,
                           metrics=self.opt_metrics)
        if data_flow_gen is not None:
          self.model.fit_generator(data_flow_gen,
                                   validation_data=validation_data,
                                   steps_per_epoch=x_train.shape[0] // batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=callbacks,
                                   max_queue_size=50,
                                   use_multiprocessing=True,
                                   workers=7)
        elif data_gen is not None:
          data_gen.fit(x_train)
          self.model.fit_generator(data_gen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                   validation_data=validation_data,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=callbacks,
                                   max_queue_size=50,
                                   use_multiprocessing=True,
                                   workers=7)
        else:
          self.model.fit(x_train, y_train,
                         validation_data=validation_data,
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         callbacks=callbacks)

  def evaluate_child_network(self, 
                             x_test, y_test):
    with self.graph.as_default():
        return self.model.evaluate(x_test, y_test)
  
  def close_tf_session(self):
    tf.Session().close()