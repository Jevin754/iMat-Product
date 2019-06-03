from tqdm import tqdm
import os
import logging
import time
import json
from glob import glob

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KDTree

from lib.util import save_dict, load_dict


class Ensembler():
    def __init__(self):
        self.num_class = 2019
        self.prior_y0_test = 1. / self.num_class
        self.prior_y1_test = 1. - self.prior_y0_test

    def load_json_file(self, json_file):
        with open(json_file, 'r') as fid:
            img_list = json.loads(fid.read())['images']
        return img_list

    def knn(self, result_dir):
        train_json = self.load_json_file('./data/train_fusesize.json')

        train_df = pd.DataFrame({'label_id': [x['class'] for x in train_json]})

        results_file_train = os.path.join(result_dir, 'train.logit.pkl')
        print('Loading training file...')
        dict_train = load_dict(results_file_train)
        train_data = []
        train_label = []
        for k,v in dict_train.items():
            train_data.append(v['logit'] / v['logit'].sum())
            train_label.append(v['label'])
        train_data = np.stack(train_data, axis=0)
        train_label = np.array(train_label)

        results_file_val = os.path.join(result_dir, 'val.logit.pkl')
        print('Loading validation file...')
        dict_val = load_dict(results_file_val)
        val_data = []
        val_label = []
        for k, v in dict_val.items():
            val_data.append(v['logit'] / v['logit'].sum())
            val_label.append(v['label'])
        val_data = np.stack(val_data, axis=0)
        val_label = np.array(val_label)

        print('Building KD tree...')
        kdt = KDTree(train_data, leaf_size=30, metric='euclidean')
        print('Predicting...')
        idx = kdt.query(val_data, k=500, return_distance=False)
        pred_tmp = train_label[idx]
        count = np.stack([np.bincount(x, minlength=2019) for x in pred_tmp], axis=0)
        pred = np.argsort(-count, axis=-1)[:, 0:3]
        err = 1. - np.sum(val_label[:, np.newaxis] == pred, axis=-1).mean()
        print('Error: {:f}'.format(err))
        pass


if __name__=='__main__':
    ens = Ensembler()
    ens.knn('./log/ensemble/tmp')