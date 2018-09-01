import psutil
import humanize
import os
import GPUtil as GPU
import sys
import random
import string
import numpy as np


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


def generate_random_cell(num_nodes=5, num_opers=5):
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
        cell[i] = {
            "L": {
                "input_layer": random.choices(list(range(i)), k=1)[0],
                "oper_id": random.choices(list(range(num_opers)), k=1)[0]
            },
            "R": {
                "input_layer": random.choices(list(range(i)), k=1)[0],
                "oper_id": random.choices(list(range(num_opers)), k=1)[0]
            }
        }
    return cell


def sgdr_learning_rate(n_Max=0.05, n_min=0.001, ranges=4, init_cycle=10):
    d = [init_cycle * 2**i for i in range(ranges)]
    Tcur = np.hstack((np.array([list(range(i)) for i in d])))
    Ti = np.hstack((([np.full(i, i - 1) for i in d])))

    nt = n_min + (n_Max - n_min) * (1 + np.cos(np.pi * Tcur / Ti)) / 2
    return nt


def get_random_eraser(p=0.5,
                      s_l=0.02,
                      s_h=0.4,
                      r_1=0.3,
                      r_2=1 / 0.3,
                      v_l=0,
                      v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img

    return eraser


class MixupGenerator():
    def __init__(self,
                 X_train,
                 y_train,
                 batch_size=32,
                 alpha=0.2,
                 shuffle=True,
                 datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) *
                                    self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])
        if isinstance(self.y_train, list):
            y = []
            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


def print_gpu_ram(gpu_num):
    gpu = GPU.getGPUs()[gpu_num]
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: {0} | Proc size: {1}".format(
        humanize.naturalsize(psutil.virtual_memory().available),
        humanize.naturalsize(process.memory_info().rss)))
    print(
        "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".
        format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100,
               gpu.memoryTotal))
