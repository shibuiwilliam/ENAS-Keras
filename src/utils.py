import psutil
import humanize
import os
import GPUtil as GPU
import sys
import random
import string

def make_dir(dir_name):
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    return dir_name

def get_random_str(length=10, choices=None):
  choice_char = ""
  if choices is None:
    choice_char += string.ascii_letters
    choice_char += string.digits
  else:
    if "l" in choices:
      choice_char += string.ascii_lowercase
    if "c" in choices:
      choice_char += string.ascii_uppercase
    if "d" in choices:
      choice_char += string.digits
  if len(choice_char) == 0:
    choice_char += string.ascii_letters
    choice_char += string.digits    
  
  random_str = ''.join([random.choice(choice_char) for _ in range(length)])
  return random_str

def get_size_str(kernel_size, filters):
  if type(kernel_size) == tuple:
    size = list(kernel_size)
  size.append(filters)
  return "x".join([str(i) for i in size])


def get_int_list_in_str(int_list, separator="x"):
  if type(int_list) == tuple:
    int_list = list(int_list)
  return separator.join([str(i) for i in int_list])

def generate_random_cell(num_nodes=5, 
                         num_opers=5):
  """
  node_num = operation node in int; starts from 2
  inputs = input node num in int
  oper_id = operation id in int
  {node_num(int): {L: {input_layer:(int), oper_id:(int)},
                   R: {input_layer:(int), oper_id:(int)}},
   node_num(int): {L: {input_layer:(int), oper_id:(int)},
                   R: {input_layer:(int), oper_id:(int)}} ... }
  """
  
  cell = {}
  for i in range(2, num_nodes):
    cell[i] = {"L": {"input_layer":random.choices(list(range(i)), k=1)[0],
                     "oper_id":random.choices(list(range(num_opers)), k=1)[0]},
               "R": {"input_layer":random.choices(list(range(i)), k=1)[0], 
                     "oper_id":random.choices(list(range(num_opers)), k=1)[0]}}
  return cell

def print_gpu_ram(gpu_num):
  gpu = GPU.getGPUs()[gpu_num]
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: {0} | Proc size: {1}".format(humanize.naturalsize( psutil.virtual_memory().available ), 
                                                    humanize.naturalsize( process.memory_info().rss)))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, 
                                                                                              gpu.memoryUsed, 
                                                                                              gpu.memoryUtil*100,
                                                                                              gpu.memoryTotal))