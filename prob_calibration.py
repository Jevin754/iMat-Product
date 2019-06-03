from tqdm import tqdm
import os
import logging
import time
import json
from glob import glob

import numpy as np
import pandas as pd

from lib.util import save_dict, load_dict

class ProbabilityCalibration():
    def __init__(self):
        self.num_class = 2019
        self.prior_y0_test = 1. / self.num_class
        self.prior_y1_test = 1. - self.prior_y0_test
        self.norm_weight = {'nasnetalarge': 10., 'resnet152': 20., 'resnet101': 20., 'inceptionresnetv2': 20., 'senet154': 20.}

    def load_json_file(self, json_file):
        with open(json_file, 'r') as fid:
            img_list = json.loads(fid.read())['images']
        return img_list

    def fuse(self, result_dir):
        train_json = self.load_json_file('./data/train_fusesize.json')

        train_df = pd.DataFrame({'label_id': [x['class'] for x in train_json]})
        results_file = glob(os.path.join(result_dir, '*.val.*.pkl'))
        results_file.sort()
        print('Total {} ensemble files.'.format(len(results_file)))
        new_dict = {}
        weight = [0.111111, 0.333333, 0.333333, 0.222222]
        weight = [0.2] * 5
        for i, res in tqdm(enumerate(results_file)):
            res_dict = load_dict(res)
            for k,v in res_dict.items():
                v = v['logit']
                if not k in new_dict.keys():
                    new_dict[k] = v / v.sum() * weight[i]
                else:
                    new_dict[k] += v / v.sum() * weight[i]
                    # tmp = np.sqrt(new_dict[k] * (v / v.sum()))
                    # new_dict[k] = tmp / tmp.sum()
        all_id_list = []
        all_predicted = []
        correct_count = 0
        prob_list = []
        label_list = []
        for k,v in tqdm(new_dict.items(), desc='Fusing result'):
            v /= v.sum()
            prob_list.append(v)
            top3_result = np.argsort(-v, axis=-1)[:3]
            all_predicted.append('{:d} {:d} {:d}'.format(top3_result[0], top3_result[1], top3_result[2]))
            all_id_list.append(k)
            label = res_dict[k]['label']
            label_list.append([label])
            a = -np.sort(-v)[:10] / 6
            correct_count += np.sum(top3_result == label)
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = os.path.join(result_dir, 'ensemble.csv')
        # datafame.to_csv(predict_csv, index=False, sep=',')
        print('Top3 error rate: {:.6f}'.format(1 - correct_count / float(len(all_id_list))))
        probs = np.stack(prob_list)
        label = np.array(label_list)

        # probability calibration
        # calibrated_prob = self.calibrate_all(probs, train_df)
        calibrated_prob = self.calibrate_probs(probs, train_df)
        topk_predict_calib = np.argsort(-calibrated_prob, axis=-1)[:, 0:3]
        err_calib = 1. - np.sum((topk_predict_calib == label), axis=-1).mean()
        print('Top3 error rate (calibration): {:.6f}'.format(err_calib))

        if False:
            calibrated_prob_sp = self.calibrate_superlabel(train_json, calibrated_prob)
            topk_predict_calib_sp = np.argsort(-calibrated_prob_sp, axis=-1)[:, 0:3]
            err_calib_sp = 1. - np.sum((topk_predict_calib_sp == label), axis=-1).mean()
            print('Top3 error rate (calibration hirechical label): {:.6f}'.format(err_calib_sp))

        all_predicted_calib = []
        for item in topk_predict_calib:
            all_predicted_calib.append('{:d} {:d} {:d}'.format(item[0], item[1], item[2]))
        datafame_calib = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted_calib})
        predict_csv_calib = os.path.join(result_dir, 'calib_ensemble_4.csv')
        # datafame_calib.to_csv(predict_csv_calib, index=False, sep=',')

    def calibrate_superlabel(self, train_json, probs):
        splabel = load_dict('./data/fglabel2splabel.pkl')
        sp_lab2idx_dict = {}
        dict_fg2sp = {}
        count = 0
        for k, v in splabel.items():
            tmp_str = ''.join([str(x) for x in v])
            splabel[k] = tmp_str
            if tmp_str not in sp_lab2idx_dict.keys():
                sp_lab2idx_dict[tmp_str] = count
                count += 1
        for k, v in splabel.items():
            tmp_str = ''.join([str(x) for x in v])
            dict_fg2sp[k] = sp_lab2idx_dict[tmp_str]
        topk_predict = np.argsort(-probs, axis=-1)[:, 0:3]
        fgidx = np.array([dict_fg2sp[x] for x in range(self.num_class)])
        new_probs = np.zeros_like(probs)
        for i in tqdm(range(fgidx[-1])):
            mask = fgidx == i
            mask_float = mask / np.sum(mask)
            mask_float = mask_float[np.newaxis, :]
            tmpp = np.matmul(probs, mask_float.transpose())
            new_probs += (probs + tmpp) * mask
        return new_probs

    def calibrate_all(self, prob, train_df):
        prior_train = [(train_df.label_id == class_).mean() for class_ in range(self.num_class)]
        prior_test = np.array([[1./self.num_class]*self.num_class])
        p = prior_test * (prob / prior_train)
        p /= np.sum(p, axis=-1, keepdims=True)
        return p

    def calibrate(self, prior_y0_train, prior_y0_test, prior_y1_train, prior_y1_test, predicted_prob_y0):
        predicted_prob_y1 = (1 - predicted_prob_y0)
        p_y0 = prior_y0_test * (predicted_prob_y0 / prior_y0_train)
        p_y1 = prior_y1_test * (predicted_prob_y1 / prior_y1_train)
        return p_y0 / (p_y0 + p_y1)  # normalization

    def calibrate_probs(self, prob, train_df):
        calibrated_prob = np.zeros_like(prob)
        nb_train = train_df.shape[0]
        for class_ in tqdm(range(self.num_class), desc='[Calibrating probability]'):  # enumerate all classes
            prior_y0_train = (train_df.label_id == class_).mean()
            prior_y1_train = 1 - prior_y0_train

            predicted_prob_y0 = prob[:, class_]
            calibrated_prob_y0 = self.calibrate(prior_y0_train, self.prior_y0_test, prior_y1_train,
                                                self.prior_y1_test, predicted_prob_y0)
            calibrated_prob[:, class_] = calibrated_prob_y0
        return calibrated_prob

if __name__=='__main__':
    pc = ProbabilityCalibration()
    pc.fuse('./log/ensemble/logit-0529')
    # pc.fuse('./log/ensemble/logit-0602')