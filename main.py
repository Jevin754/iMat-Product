import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from model import iMateriaList, EnsembleModel

import lib.util as util
from lib.util import setup_seed
from lib.util import init_logging, ModelManager
from lib.lib_data import ImageDataset, collate_fn_image

_DEBUG = False

def parse_args():
    p = argparse.ArgumentParser(description='SLR')
    p.add_argument('-t', '--task', type=str, default='train')
    p.add_argument('-g', '--gpu', type=str, default='1,2,3,4')

    # data
    p.add_argument('-dw', '--data_worker', type=int, default=16)

    # optimizer
    p.add_argument('-op', '--optimizer', type=str, default='adam')
    p.add_argument('-lr', '--learning_rate', type=float, default=4e-4)
    p.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
    p.add_argument('-mt', '--momentum', type=float, default=0.9)
    p.add_argument('-nepoch', '--num_epoch', type=int, default=1000)
    p.add_argument('-us', '--update_step', type=int, default=1)

    # cnn backbone
    p.add_argument('-cnn', '--cnn', type=str, default='inceptionresnetv2') # inceptionresnetv2

    # train
    p.add_argument('-db', '--DEBUG', type=bool, default=_DEBUG)
    # p.add_argument('-lg_d', '--log_dir', type=str, default='./log/nasnetalarge/fusesize_img_331_2/')
    # p.add_argument('-lg_d', '--log_dir', type=str, default='./log/resnet101_c/fusesize_img_224_1/')
    p.add_argument('-lg_d', '--log_dir', type=str, default='./log/inceptionresnetv2/fusesize_img_299_2/')
    # p.add_argument('-lg_d', '--log_dir', type=str, default='./log/resnet152/resize_img_224_1/')
    # p.add_argument('-lg_d', '--log_dir', type=str, default='./log/senet154/fusesize_img_224_1/')
    p.add_argument('-bs', '--batch_size', type=int, default=32)

    # test (decoding)
    p.add_argument('-bwd', '--beam_width', type=int, default=5)
    p.add_argument('-vbs', '--valid_batch_size', type=int, default=1)

    parameter = p.parse_args()
    return parameter

if __name__=='__main__':
    # setup_seed(8)
    torch.backends.cudnn.benchmark = True
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    if opts.task == 'train':
        if not os.path.exists(opts.log_dir):
            print('Log DIR ({:s}) not exist! Make new folder.'.format(opts.log_dir))
            os.makedirs(opts.log_dir)
        init_logging(os.path.join(opts.log_dir, 'log_{:s}.txt'.format(opts.task)))
        product_recog = iMateriaList(opts)
        # ckpt = ''
        ckpt = './log/nasnetalarge/fusesize_img_331_1/ep11.pkl'
        # ckpt = './log/inceptionresnetv2/fusesize_img_299_1/ep12.pkl'
        # ckpt = './log/resnet152/resize_img_224_1/ep5.pkl'
        # ckpt = './log/resnet152/resize_img/ep19.pkl'
        # ckpt = './log/senet154/resize_img_224_1/ep11.pkl'
        # ckpt = './log/resnet152/resize_img/ep19.pkl.new'
        product_recog.train(ckpt)
    elif opts.task == 'test':
        init_logging(os.path.join(opts.log_dir, 'log_{:s}.txt'.format(opts.task)))
        product_recog = iMateriaList(opts)
        # product_recog.test_ensemble(model_file='./log/nasnetalarge/fusesize_img_331_2/ep7.pkl')
        # product_recog.test_ensemble(model_file='./log/resnet101_c/fusesize_img_224_1/ep11.pkl')
        # product_recog.test_ensemble(model_file='./log/inceptionresnetv2/fusesize_img_299_2/ep1.pkl')
        # product_recog.test_ensemble(model_file='./log/inceptionresnetv2/fusesize_img_299_1/ep12.pkl')
        # product_recog.test_ensemble(model_file='./log/resnet152/fusesize_img_224_1/ep29.pkl')
        product_recog.test_ensemble(model_file='./log/senet154/fusesize_img_224_1/ep6.pkl')
        # product_recog.test_ensemble(model_file='./log/senet154/resize_img_224_1/ep11.pkl')
        # product_recog.test_ensemble(model_file='./log/resnet152/resize_img_224_1/ep5.pkl')
        # product_recog.test_ensemble(model_file='./log/ep19.pkl.new')
        # product_recog.test(model_file='./log/resnet152/resize_img_bin/ep7.pkl')
    elif opts.task == 'feature':
        init_logging(os.path.join(opts.log_dir, 'log_{:s}.txt'.format(opts.task)))
        product_recog = iMateriaList(opts)
        product_recog.extract_feature(model_file=os.path.join(opts.log_dir, 'ep1.pkl'))
        # product_recog.extract_feature(model_file='./log/nasnetalarge/fusesize_img_331_2/ep7.pkl')
        # product_recog.extract_feature(model_file='./log/senet154/fusesize_img_224_1/ep6.pkl')
        # product_recog.extract_feature(model_file='./log/resnet101_c/fusesize_img_224_1/ep11.pkl')
    elif opts.task == 'ensemble':
        ensemble = EnsembleModel()
        # ensemble.fuse_two('./log/ensemble')
        ensemble.fuse_probs('./log/ensemble/fuse')
        # ensemble.fuse('./log/ensemble/val')
        # ensemble.fuse_two_kinds('./log/ensemble/fuse')
    else:
        print('Run script with --task.')