import os
import glob
import json
import logging
import pickle
from tqdm import tqdm

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform
from PIL import Image


def init_logging(log_file):
    """Init for logging
    """
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s: %(message)s',
                        datefmt = '%m-%d %H:%M:%S',
                        filename = log_file,
                        filemode = 'w')
    # define a Handler which writes INFO message or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class LossManager(object):
    def __init__(self, print_step):
        self.print_step = print_step
        self.last_state = {'loss': -111.1}
        self.total_loss = []

    def update(self, loss, epoch, global_step):
        self.total_loss.append(loss)
        if (global_step % self.print_step) == 0:
            mean_loss = np.mean(self.total_loss)
            logging.info('Global step: {:d}, loss: {:.3f} -> {:.3f}'.\
                         format(global_step, self.last_state['loss'], mean_loss))
            self.last_state['loss'] = mean_loss
            self.total_loss = []

class ModelManager(object):
    def __init__(self, max_num_models=5):
        self.max_num_models = max_num_models
        self.best_epoch = 0
        self.best_err = 1000.
        # self.worst_err = 1000
        self.model_file_list = []

    def update(self, model_file, err, epoch):
        if len(self.model_file_list) >= self.max_num_models:
            worst_model_file = self.model_file_list.pop(-1)[0]
            if os.path.exists(worst_model_file):
                os.remove(worst_model_file)
        self.model_file_list.append((model_file, err))
        self.update_best_err(err, epoch)
        self.sort_model_list()
        logging.info('CURRENT BEST Performance (epoch: {:d}), TOP3-ERROR: {:.5f}'.format( \
            self.best_epoch, self.best_err))
        pass

    def update_best_err(self, err, epoch):
        if err < self.best_err:
            self.best_err = err
            self.best_epoch = epoch

    def sort_model_list(self):
        # batch_tupe.sort(key=lambda x: len(x[0]), reverse=True)
        self.model_file_list.sort(key=lambda x: x[1])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.deterministic = True

def save_dict(obj, name):
    with open(name, 'wb') as fid:
        pickle.dump(obj, fid, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as fid:
        return pickle.load(fid)

def strip_prefix_dict(state_dict, prefix='module.'):
    stripped_state_dict = {}
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, '')] = value
    return stripped_state_dict