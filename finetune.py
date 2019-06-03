from tqdm import tqdm
import os
import logging
import time
import json
from glob import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.util import save_dict, load_dict
from torch.utils.data import Dataset, DataLoader
from lib.util import init_logging
from lib.util import LossManager, ModelManager
from lib.lib_metric import get_correct_count

_DEBUG = False

def collate_fn_feature(batch):
    data, label, id = zip(*batch)
    data_list = []
    label_list = []
    id_list = []
    for i, tid in enumerate(id):
        data_list.append(data[i])
        label_list.append(label[i])
        id_list.append(i)
    data_tensor = torch.from_numpy(np.stack(data_list, axis=0))
    label_tensor = torch.from_numpy(np.array(label_list))
    return {'data': data_tensor, 'label': label_tensor, 'id': id_list}

class FeatureDataset(Dataset):
    def __init__(self, phase, feat_path, DEBUG=False):
        self.phase = phase
        self.feat_path = feat_path
        self.net_list = ['nasnetalarge', 'senet154', 'inceptionresnetv2', 'resnet152', 'resnet101']
        self.num_test_time = [10., 20., 20., 20., 20.]
        # self.net_list = ['nasnetalarge', 'resnet152', 'resnet101', 'inceptionresnetv2', 'senet154']
        self.num_nets = len(self.net_list)
        self.net_info = {'nasnetalarge': {'feat_dim': 4032, 'num_net': 10},
                         'resnet152': {'feat_dim': 2048, 'num_net': 20},
                         'resnet101': {'feat_dim': 2048, 'num_net': 20},
                         'inceptionresnetv2': {'feat_dim': 1536, 'num_net': 20},
                         'senet154': {'feat_dim': 2048, 'num_net': 20}}

        self.load_feat()

    def __len__(self):
        return len(self.dict_feat_list[0])

    def __getitem__(self, idx):
        data_list = []
        k = self.key_list[idx]
        for tmp_dict in self.dict_feat_list:
            logit = tmp_dict[k]['logit']
            # logit /= logit.sum()
            logit_norm = logit / logit.sum()
            label = tmp_dict[k]['label']
            data_list.append(logit_norm)
        data = np.stack(data_list, axis=0)
        data = data[np.newaxis, :]
        label = tmp_dict[k]['label']
        return data, label, k

    def __getitem1__(self, idx):
        data_list = []
        k = self.key_list[idx]
        for i, tmp_dict in enumerate(self.dict_feat_list):
            logit = tmp_dict[k]['logit']
            # logit /= logit.sum()
            logit_norm = logit / self.num_test_time[i]
            logit_norm /= np.sum(logit_norm**2)
            label = tmp_dict[k]['label']
            data_list.extend(list(logit_norm))
        # data = np.stack(data_list, axis=0)
        data = np.array(data_list)
        # data = data[np.newaxis, :]
        label = tmp_dict[k]['label']
        return data, label, k

    def load_feat(self):
        self.dict_feat_list = []
        for net in self.net_list:
            # filename = os.path.join(self.feat_path, '{:s}.*.{:s}.*.logit.pkl'.format(net, self.phase))
            filename = os.path.join(self.feat_path, '{:s}.*.{:s}.*.logit.pkl'.format(net, self.phase))
            fdl = self.load_feat_single(filename)
            self.dict_feat_list.extend(fdl)
        self.key_list = list(self.dict_feat_list[0].keys())

    def load_feat_single(self, filename):
        dict_feat_list = []
        filename_list = glob(filename)
        filename_list.sort()
        for fn in filename_list:
            print('Loading feature from {:s}'.format(fn))
            dict_feat = load_dict(fn)
            dict_feat_list.append(dict_feat)
        return dict_feat_list

from torch.autograd import Variable
class Network(nn.Module):
    def __init__(self, opts):
        super(Network, self).__init__()
        self.opts = opts
        ks = 1
        num_net = 5
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(num_net, ks), stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(11712, 4096)
        self.fc1 = nn.Linear(6080, 4096)
        self.fc2 = nn.Linear(2019-ks+1, 2019)

        # self.weight = Variable(torch.ones([2,1]), requires_grad=True).cuda()
        self.weight = torch.randn(2, 1, requires_grad=True).cuda()

    def forward(self, x):
        bs = x.shape[0]
        # x = x.reshape(x.shape[0], -1, x.shape[-1])
        # x = torch.matmul(x.permute(0, 2, 1), F.softmax(self.weight, dim=0))
        # x = x.reshape(bs, -1)
        x = self.conv1(x)
        # x = self.relu(x)
        x = x.reshape(bs, -1)
        # logit = self.relu(self.fc1(x))
        logit = self.fc2(x)
        # logit = x
        return logit

class Finetuner(object):
    def __init__(self, opts, feat_path):
        self.feat_path = feat_path
        self.opts = opts
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = Network(self.opts)
        self.criterion = nn.CrossEntropyLoss()
        self.params = self.network.parameters()
        self.optimizer = create_optimizer(self.opts.optimizer, self.params, lr=self.opts.learning_rate, momentum=self.opts.momentum,
                                          weight_decay=self.opts.weight_decay)

    def create_dataloader(self, phase, feat_path, batch_size, shuffle, num_workers=8, drop_last=False, DEBUG=False):
        dataset_product = FeatureDataset(phase, feat_path=feat_path, DEBUG=DEBUG)
        dataloader = DataLoader(dataset_product, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                num_workers=num_workers, collate_fn=collate_fn_feature, drop_last=drop_last)
        return dataloader

    def eval_batch(self, image, label, device=None):
        with torch.no_grad():
            bs = image.shape[0]
            image = image.to(device)
            label = label.to(device)

            self.network.eval()

            logit = self.network(image)
            correct_count_topk = get_correct_count(logit.data, label, topk=(1,3))
        return correct_count_topk, bs

    def train_batch(self, image, label, device=None, update_grad=True):
        '''
        :param image: [B, C, W, H]
        :param label:
        :param device:
        :param update_grad:
        :return:
        '''
        bs = image.shape[0]
        image = image.to(device)
        label = label.to(device)

        self.network.train()

        logit = self.network(image)
        loss_fg = self.criterion(logit, label)
        # loss_bin = self.criterion_bin(self.sigmoid(logit_bin), super_label) * 11.

        loss = loss_fg

        loss.backward()
        if update_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def train(self, ckpt=None):
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size * num_gpu
        dataloader_train = self.create_dataloader(phase='train', feat_path=self.feat_path, batch_size=bs, num_workers=self.opts.data_worker,
                                                  shuffle=True, drop_last=False, DEBUG=self.opts.DEBUG)
        dataloader_val = self.create_dataloader(phase='val', feat_path=self.feat_path, batch_size=bs, num_workers=self.opts.data_worker,
                                                  shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)

        num_epoch = self.opts.num_epoch
        global_step = 0
        loss_manage = LossManager(print_step=100)
        model_manager = ModelManager(max_num_models=5)

        if torch.cuda.device_count() > 1:
            logging.info('Total use {:d} GPUs! [ID: {:s}]'.format(torch.cuda.device_count(), self.opts.gpu))
            self.network = nn.DataParallel(self.network)
        else:
            logging.info('Only use 1 GPU! [ID: {:s}]'.format(self.opts.gpu))
        self.network.to(self.device)

        if ckpt != '' and ckpt is not None:
            logging.info('Restoring model from {:s}'.format(ckpt))
            self.network.load_state_dict(torch.load(ckpt), strict=True)

        self.optimizer.zero_grad()
        last_status = {'loss': -1.}
        for epoch in range(num_epoch):
            epoch_loss = []
            for i, item in tqdm(enumerate(dataloader_train), desc='[Training phase, epoch {:d}]'.format(epoch)):
                global_step += 1
                batch = item
                data = batch['data']
                label = batch['label']
                update_flag = True if (global_step % self.opts.update_step) == 0 else False
                loss = self.train_batch(data, label, device=self.device, update_grad=update_flag).item()
                loss_manage.update(loss, epoch, global_step)
                epoch_loss.append(loss)
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss)))
            last_status['loss'] = np.mean(epoch_loss)

            if epoch < 0 or epoch % 1 != 0:
                continue
            val_correct = 0
            val_count = 0
            correct_count_top1 = 0
            correct_count_top3 = 0
            for i, item in tqdm(enumerate(dataloader_val), desc='[Validation phase, epoch {:d}]'.format(epoch)):
                batch = item
                data = batch['data']
                label = batch['label']
                correct_count_topk, count = self.eval_batch(data, label, device=self.device)
                correct_count_top1 += correct_count_topk[0].data.item()
                correct_count_top3 += correct_count_topk[1].data.item()
                val_count += count
            logging.info('-' * 50)
            err_top1 = 1. - correct_count_top1 / val_count
            err_top3 = 1. - correct_count_top3 / val_count
            logging.info('VAL ERR (top1): {:.5f}, {:d}/{:d}'.format(err_top1, val_count - int(correct_count_top1), val_count))
            logging.info('VAL ERR (top3): {:.5f}, {:d}/{:d}'.format(err_top3, val_count - int(correct_count_top3), val_count))

    def fit(self):
        feat_dataset = FeatureDataset(phase='val', feat_path=self.feat_path, DEBUG=False)
        logit_all = []
        label_list = [feat_dataset.dict_feat_list[0][x]['label'] for x in feat_dataset.key_list]
        label = np.array(label_list)[:, np.newaxis]
        for k in tqdm(feat_dataset.key_list):
            tmp = np.stack([x[k]['logit'] for x in feat_dataset.dict_feat_list], axis=0)
            logit_all.append(tmp)
        logit_all = np.stack(logit_all, axis=0)
        logit_all = logit_all.transpose(0,2,1)

        for x1 in np.linspace(0,1,10):
            for x2 in np.linspace(0, 1, 10):
                for x3 in np.linspace(0, 1, 10):
                    if x1 + x2 + x3 > 1:
                        continue
                    x4 = 1. - x1 - x2 - x3
                    weight = np.array([x1, x2, x3, x4])
                    new_logit = np.matmul(logit_all, weight)

                    top3_result = np.argsort(-new_logit, axis=-1)[:, :3]
                    err_top3 = 1. - np.sum(top3_result == label, axis=-1).sum() / len(feat_dataset.key_list)
                    print('-'*50)
                    print(weight)
                    print('Error: {:f}'.format(err_top3))

        # weight = np.array([0.25]*4)
        # new_logit = np.matmul(logit_all, weight)
        #
        # top3_result = np.argsort(-new_logit, axis=-1)[:, :3]
        # err_top3 = 1. - np.sum(top3_result == label, axis=-1).sum() / len(feat_dataset.key_list)
        pass


def parse_args():
    p = argparse.ArgumentParser(description='Feature Finetune')
    p.add_argument('-t', '--task', type=str, default='train')
    p.add_argument('-g', '--gpu', type=str, default='1,2,3,4')

    # data
    p.add_argument('-dw', '--data_worker', type=int, default=16)
    p.add_argument('-fp', '--feat_path', type=str, default='./log/ensemble/logit-0529')

    # optimizer
    p.add_argument('-op', '--optimizer', type=str, default='adam')
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    p.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
    p.add_argument('-mt', '--momentum', type=float, default=0.9)
    p.add_argument('-nepoch', '--num_epoch', type=int, default=1000)
    p.add_argument('-us', '--update_step', type=int, default=1)

    # train
    p.add_argument('-db', '--DEBUG', type=bool, default=_DEBUG)
    # p.add_argument('-lg_d', '--log_dir', type=str, default='./log/nasnetalarge/fusesize_img_331_2/')
    p.add_argument('-lg_d', '--log_dir', type=str, default='./log/finetune/3/')
    p.add_argument('-bs', '--batch_size', type=int, default=512)

    parameter = p.parse_args()
    return parameter

def create_optimizer(optimizer, params, **kwargs):
    supported_optim = {
        'sgd': torch.optim.SGD, # momentum, weight_decay, lr
        'rmsprop': torch.optim.RMSprop, # momentum, weight_decay, lr
        'adam': torch.optim.Adam # weight_decay, lr
    }
    assert optimizer in supported_optim, 'Now only support {}'.format(supported_optim.keys())
    if optimizer == 'adam':
        del kwargs['momentum']
    optim = supported_optim[optimizer](params, **kwargs)
    logging.info('Create optimizer {}({})'.format(optimizer, kwargs))
    return optim

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
        finetuner = Finetuner(opts, opts.feat_path)
        ckpt = None
        # finetuner.fit()
        finetuner.train(ckpt)
    elif opts.task == 'test':
        init_logging(os.path.join(opts.log_dir, 'log_{:s}.txt'.format(opts.task)))
        finetuner.test_ensemble(model_file='./log/resnet101_c/fusesize_img_224_1/ep11.pkl')
    else:
        print('Run script with --task.')

#
# if __name__=='__main__':
#     fd = FeatureDataset('val', 'log/ensemble/0525')
#     dl = DataLoader(fd, batch_size=10, shuffle=False, pin_memory=True, num_workers=0, collate_fn=collate_fn_feature, drop_last=False)
#     for item in tqdm(dl):
#         pass
#     fd.__getitem__(0)
#     pass