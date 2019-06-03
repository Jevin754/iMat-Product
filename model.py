from tqdm import tqdm
import os
import logging
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
import torchvision
from backbone import resnet, inceptionresnetv2, senet, nasnet

from lib.lib_data import ImageDataset, collate_fn_image
from lib.util import LossManager, ModelManager
from lib.util import strip_prefix_dict, save_dict, load_dict
from lib.lib_metric import get_correct_count

class Network(nn.Module):
    def __init__(self, opts, backbone='resnet152', pretrained=False, dropout=0.0):
        super(Network, self).__init__()
        self.backbone_name = backbone
        self.num_classes = 2019

        backbone_list = {'resnet18': {'model': resnet.resnet18(num_classes=self.num_classes), 'weights': './pretrained_weight/resnet18.pth'},
                         'resnet34': {'model': resnet.resnet34(num_classes=self.num_classes), 'weights': './pretrained_weight/resnet34.pth'},
                         'resnet101_c': {'model': resnet.resnet101(num_classes=self.num_classes), 'weights': './pretrained_weight/resnet101_cdiscount.pth'},
                         'resnet152': {'model': resnet.resnet152(num_classes=self.num_classes), 'weights': './pretrained_weight/resnet152.pth'},
                         'senet154': {'model': senet.senet154(num_classes=self.num_classes), 'weights': './pretrained_weight/senet154.pth'},
                         'nasnetalarge': {'model': nasnet.nasnetalarge(num_classes=self.num_classes, pretrained=None), 'weights': './pretrained_weight/nasnetalarge-a1897284.pth'},
                         'inceptionresnetv2': {'model': inceptionresnetv2.InceptionResNetV2(num_classes=self.num_classes), 'weights': './pretrained_weight/inceptionresnetv2.pth'}}

        # self.backbone = resnet.resnet152(pretrained=True, num_classes=self.num_classes)
        self.backbone = backbone_list[backbone]['model']
        weight_pth = backbone_list[backbone]['weights']

        if pretrained:
            self.load_pretrained_model(weight_pth)

    def forward(self, img):
        logits, logits_bin = self.backbone(img)
        return logits, logits_bin

    def load_pretrained_model(self, model_file):
        logging.info('Restoring CNN backbone model parameters from {:s}'.format(model_file))
        pretrained_weight_dict = torch.load(model_file)
        if self.backbone_name == 'resnet101_c':
            pretrained_weight_dict = self.replace_prefix(pretrained_weight_dict)
        self.backbone.load_state_dict(pretrained_weight_dict, strict=False)

    def replace_prefix(self, pretrained_dict):
        new_pretrained_dict = {}
        for key,v in pretrained_dict.items():
            new_key = key
            if 'layer0.0.conv.' in key: new_key = key.replace('layer0.0.conv.', 'conv1.')
            if 'layer0.0.bn.' in key: new_key = key.replace('layer0.0.bn.', 'bn1.')
            if '.conv_bn1.conv.' in key: new_key = key.replace('.conv_bn1.conv.', '.conv1.')
            if '.conv_bn1.bn.' in key: new_key = key.replace('.conv_bn1.bn.', '.bn1.')
            if '.conv_bn2.conv.' in key: new_key = key.replace('.conv_bn2.conv.', '.conv2.')
            if '.conv_bn2.bn.' in key: new_key = key.replace('.conv_bn2.bn.', '.bn2.')
            if '.conv_bn3.conv.' in key: new_key = key.replace('.conv_bn3.conv.', '.conv3.')
            if '.conv_bn3.bn.' in key: new_key = key.replace('.conv_bn3.bn.', '.bn3.')
            if '.downsample.conv.' in key: new_key = key.replace('.downsample.conv.', '.downsample.0.')
            if '.downsample.bn.' in key: new_key = key.replace('.downsample.bn.', '.downsample.1.')
            new_pretrained_dict[new_key] = v
        return new_pretrained_dict



class iMateriaList(object):
    def __init__(self, opts):
        self.opts = opts
        self.network = Network(self.opts, backbone=self.opts.cnn, pretrained=True, dropout=0.0)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cuda = True if torch.cuda.is_available() else False

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_bin = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        if self.cuda:
            self.criterion.cuda()

        self.params = self.network.parameters()
        self.optimizer = create_optimizer(self.opts.optimizer, self.params, lr=self.opts.learning_rate, momentum=self.opts.momentum,
                                          weight_decay=self.opts.weight_decay)

    def create_dataloader(self, phase, batch_size, shuffle, num_workers=8, drop_last=False, DEBUG=False):
        dataset_product = ImageDataset(phase, is_transform=True, DEBUG=DEBUG)
        dataloader = DataLoader(dataset_product, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                num_workers=num_workers, collate_fn=collate_fn_image, drop_last=drop_last)
        return dataloader

    def eval_batch(self, image, label, device=None):
        with torch.no_grad():
            bs = image.shape[0]
            image = image.to(device)
            label = label.to(device)

            self.network.eval()

            logit, _ = self.network(image)
            correct_count_topk = get_correct_count(logit.data, label, topk=(1,3))
        return correct_count_topk, bs

    def train_batch(self, image, label, super_label, device=None, update_grad=True):
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
        super_label = super_label.to(device)

        self.network.train()

        logit, logit_bin = self.network(image)
        loss_fg = self.criterion(logit, label)
        # loss_bin = self.criterion_bin(self.sigmoid(logit_bin), super_label) * 11.

        loss = loss_fg

        loss.backward()
        if update_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def train(self, ckpt):
        logging.info('CNN backnobe: {:s}'.format(self.opts.cnn))
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size * num_gpu
        dataloader_train = self.create_dataloader(phase='train', batch_size=bs, num_workers=self.opts.data_worker,
                                                  shuffle=True, drop_last=False, DEBUG=self.opts.DEBUG)
        dataloader_val = self.create_dataloader(phase='val', batch_size=bs, num_workers=self.opts.data_worker,
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

        if ckpt != '' or ckpt is not None:
            logging.info('Restoring model from {:s}'.format(ckpt))
            self.network.load_state_dict(torch.load(ckpt), strict=True)

        self.optimizer.zero_grad()
        last_status = {'loss': -1.}
        for epoch in range(num_epoch):
            epoch_loss = []
            for i, item in tqdm(enumerate(dataloader_train), desc='[Training phase, epoch {:d}]'.format(epoch)):
                global_step += 1
                batch = item
                image = batch['image']
                label = batch['label']
                super_label = batch['super_label']
                update_flag = True if (global_step % self.opts.update_step) == 0 else False
                loss = self.train_batch(image, label, super_label, device=self.device, update_grad=update_flag).item()
                loss_manage.update(loss, epoch, global_step)
                epoch_loss.append(loss)
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss)))
            last_status['loss'] = np.mean(epoch_loss)

            if epoch < 0:
                continue
            val_correct = 0
            val_count = 0
            correct_count_top1 = 0
            correct_count_top3 = 0
            for i, item in tqdm(enumerate(dataloader_val), desc='[Validation phase, epoch {:d}]'.format(epoch)):
                batch = item
                image = batch['image']
                label = batch['label']
                correct_count_topk, count = self.eval_batch(image, label, device=self.device)
                correct_count_top1 += correct_count_topk[0].data.item()
                correct_count_top3 += correct_count_topk[1].data.item()
                val_count += count
            logging.info('-' * 50)
            err_top1 = 1. - correct_count_top1 / val_count
            err_top3 = 1. - correct_count_top3 / val_count
            logging.info('VAL ERR (top1): {:.5f}, {:d}/{:d}'.format(err_top1, val_count - int(correct_count_top1), val_count))
            logging.info('VAL ERR (top3): {:.5f}, {:d}/{:d}'.format(err_top3, val_count - int(correct_count_top3), val_count))

            model_name = os.path.join(self.opts.log_dir, 'ep{:d}.pkl'.format(epoch))
            torch.save(self.network.state_dict(), model_name)
            model_manager.update(model_name, err_top3, epoch)

    def inference_batch(self, image, device=None):
        with torch.no_grad():
            bs = image.shape[0]
            image = image.to(device)
            self.network.eval()
            logit, _ = self.network(image)
            logit = torch.nn.functional.softmax(logit, dim=-1)
            top3_result = torch.argsort(logit, dim=-1, descending=True)[:, :3]
        return top3_result, logit

    def test(self, model_file):
        self.network.training = False
        logging.info('Restoring full model parameters from {:s}'.format(model_file))
        trained_model_paras = torch.load(model_file)
        self.network.load_state_dict(strip_prefix_dict(trained_model_paras))
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size * num_gpu
        dataloader_test = self.create_dataloader(phase='test', batch_size=bs, num_workers=self.opts.data_worker,
                                                shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)
        if torch.cuda.device_count() > 1:
            logging.info('Total use {:d} GPUs! [ID: {:s}]'.format(torch.cuda.device_count(), self.opts.gpu))
            self.network = nn.DataParallel(self.network)
        else:
            logging.info('Only use 1 GPU! [ID: {:s}]'.format(self.opts.gpu))
        self.network.to(self.device)

        all_id_list = []
        all_predicted = []
        for i, item in tqdm(enumerate(dataloader_test), desc='[Testing phase]'):
            batch = item
            image = batch['image']
            label = batch['label']
            img_id = batch['id']
            top3_result, _ = self.inference_batch(image, device=self.device)
            top3_result = top3_result.cpu().data.numpy()
            for j, tmpid in enumerate(img_id):
                all_id_list.append(tmpid)
                all_predicted.append('{:d} {:d} {:d}'.format(top3_result[j][0], top3_result[j][1], top3_result[j][2]))
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = model_file + '.csv'
        datafame.to_csv(predict_csv, index=False, sep=',')

    def test_ensemble(self, model_file):
        phase = 'test'
        self.network.training = False
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size * num_gpu
        dataloader_test = self.create_dataloader(phase=phase, batch_size=bs, num_workers=self.opts.data_worker,
                                                 shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)
        if torch.cuda.device_count() > 1:
            logging.info('Total use {:d} GPUs! [ID: {:s}]'.format(torch.cuda.device_count(), self.opts.gpu))
            self.network = nn.DataParallel(self.network)
        else:
            logging.info('Only use 1 GPU! [ID: {:s}]'.format(self.opts.gpu))
        self.network.to(self.device)

        logging.info('Restoring full model parameters from {:s}'.format(model_file))
        model_paras = torch.load(model_file)
        self.network.load_state_dict(torch.load(model_file))

        all_id_list = []
        all_predicted = []
        all_gt_list = []
        N = 20
        logit_list = []
        logit_dict = {}
        for n in range(N):
            for i, item in tqdm(enumerate(dataloader_test), desc='[Testing phase (Ensemble {:d})]'.format(n)):
                batch = item
                image = batch['image']
                label = batch['label']
                img_id = batch['id']
                _, logit = self.inference_batch(image, device=self.device)
                logit = logit.cpu()
                if n == 0:
                    logit_list.append(logit)
                    all_id_list.extend(img_id)
                    all_gt_list.extend(label.data.cpu().numpy().tolist())
                else:
                    logit_list[i] += logit

        label_dict = {}
        for item in dataloader_test.dataset.img_list:
            label_dict[item['id']] = item['class'] if phase == 'val' else -1

        count = 0
        correct_count = 0
        for item_logit in logit_list:
            top3_result = torch.argsort(item_logit, dim=-1, descending=True)[:, :3]
            for j in range(len(item_logit)):
                all_predicted.append('{:d} {:d} {:d}'.format(top3_result[j][0], top3_result[j][1], top3_result[j][2]))
                logit_dict[all_id_list[count]] = {'logit': item_logit[j].data.numpy(), 'label': all_gt_list[count]}
                correct_count += int(np.sum(top3_result[j].cpu().data.numpy() == label_dict[all_id_list[count]]))
                count += 1
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        # predict_csv = model_file + '.csv'
        predict_csv = model_file.replace('pkl', '{:s}.ens{:d}.csv'.format(phase, N))
        datafame.to_csv(predict_csv, index=False, sep=',')
        save_dict(logit_dict, model_file.replace('pkl', '{:s}.ens{:d}.logit.pkl'.format(phase, N)))
        logging.info('Top-3 error rate: {:.5f}'.format(1. - correct_count*1.0/count))

    def extract_feature_batch(self, image, device=None):
        with torch.no_grad():
            bs = image.shape[0]
            image = image.to(device)
            self.network.eval()
            logit, feat = self.network(image)
            logit = torch.nn.functional.softmax(logit, dim=-1)
            top3_result = torch.argsort(logit, dim=-1, descending=True)[:, :3]
        return top3_result, logit, feat

    def extract_feature(self, model_file):
        phase = 'train'
        self.network.training = False
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size * num_gpu
        dataloader_test = self.create_dataloader(phase=phase, batch_size=bs, num_workers=self.opts.data_worker,
                                                 shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)
        if torch.cuda.device_count() > 1:
            logging.info('Total use {:d} GPUs! [ID: {:s}]'.format(torch.cuda.device_count(), self.opts.gpu))
            self.network = nn.DataParallel(self.network)
        else:
            logging.info('Only use 1 GPU! [ID: {:s}]'.format(self.opts.gpu))
        self.network.to(self.device)

        logging.info('Restoring full model parameters from {:s}'.format(model_file))
        model_paras = torch.load(model_file)
        self.network.load_state_dict(torch.load(model_file))

        all_id_list = []
        all_predicted = []
        all_gt_list = []
        N = 20
        logit_list = []
        logit_dict = {}
        for n in range(N):
            for i, item in tqdm(enumerate(dataloader_test), desc='[Testing phase (Ensemble {:d})]'.format(n)):
                batch = item
                image = batch['image']
                label = batch['label']
                img_id = batch['id']
                _, logit, feat = self.extract_feature_batch(image, device=self.device)
                # logit = logit.cpu()
                logit = feat.cpu()
                if n == 0:
                    logit_list.append(logit)
                    all_id_list.extend(img_id)
                    all_gt_list.extend(label.data.cpu().numpy().tolist())
                else:
                    logit_list[i] += logit

        label_dict = {}
        for item in dataloader_test.dataset.img_list:
            label_dict[item['id']] = item['class'] if phase == 'val' else -1

        count = 0
        correct_count = 0
        for item_logit in logit_list:
            top3_result = torch.argsort(item_logit, dim=-1, descending=True)[:, :3]
            for j in range(len(item_logit)):
                all_predicted.append('{:d} {:d} {:d}'.format(top3_result[j][0], top3_result[j][1], top3_result[j][2]))
                logit_dict[all_id_list[count]] = {'logit': item_logit[j].data.numpy(), 'label': all_gt_list[count]}
                correct_count += int(np.sum(top3_result[j].cpu().data.numpy() == label_dict[all_id_list[count]]))
                count += 1
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        # predict_csv = model_file + '.csv'
        predict_csv = model_file.replace('pkl', '{:s}.ens{:d}.feat.csv'.format(phase, N))
        datafame.to_csv(predict_csv, index=False, sep=',')
        save_dict(logit_dict, model_file.replace('pkl', '{:s}.ens{:d}.feat.pkl'.format(phase, N)))
        logging.info('Top-3 error rate: {:.5f}'.format(1. - correct_count*1.0/count))


class EnsembleModel(object):
    def __init__(self):
        pass

    def fuse(self, result_dir):
        results_file = glob(os.path.join(result_dir, '*.pkl'))
        print('Total {} ensemble files.'.format(len(results_file)))
        new_dict = {}
        for res in tqdm(results_file):
            res_dict = load_dict(res)
            for k,v in res_dict.items():
                v = v['logit']
                if not k in new_dict.keys():
                    new_dict[k] = v / v.sum()
                else:
                    new_dict[k] += v / v.sum()
        all_id_list = []
        all_predicted = []
        correct_count = 0
        for k,v in tqdm(new_dict.items(), desc='Fusing result'):
            top3_result = np.argsort(-v, axis=-1)[:10]
            all_predicted.append('{:d} {:d} {:d}'.format(top3_result[0], top3_result[1], top3_result[2]))
            all_id_list.append(k)
            label = res_dict[k]['label']
            a = -np.sort(-v)[:10] / 6
            correct_count += np.sum(top3_result == label)
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = os.path.join(result_dir, 'ensemble.csv')
        datafame.to_csv(predict_csv, index=False, sep=',')
        print('Top3 error rate: {:.6f}'.format(1 - correct_count / float(len(all_id_list))))

    def fuse_two(self, result_dir):
        results_file = glob(os.path.join(result_dir, '*.pkl'))
        num_models = len(results_file)
        if num_models != 2:
            exit('No. models large than 2.')
        new_dict_list = []
        for res in tqdm(results_file):
            res_dict = load_dict(res)
            new_dict_list.append(res_dict)
        all_id_list = []
        all_predicted = []
        for k, v in tqdm(new_dict_list[0].items(), desc='Fusing result'):
            p0 = v
            p1 = new_dict_list[1][k]
            p0 /= p0.sum()
            p1 /= p1.sum()
            p = p0*p1 / (p0*p1 + (1-p0) * (1-p1))
            top3_result = np.argsort(-p, axis=-1)[:3]
            all_predicted.append('{:d} {:d} {:d}'.format(top3_result[0], top3_result[1], top3_result[2]))
            all_id_list.append(k)
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = os.path.join(result_dir, 'ensemble_two.csv')
        datafame.to_csv(predict_csv, index=False, sep=',')

    def fuse_two_kinds(self, result_dir):
        net1 = 'resnet152'
        net2 = 'senet154'
        results_file1 = glob(os.path.join(result_dir, net1 + '*.pkl'))
        results_file2 = glob(os.path.join(result_dir, net2 + '*.pkl'))

        new_dict1 = {}
        for res in tqdm(results_file1):
            res_dict = load_dict(res)
            for k, v in res_dict.items():
                v = v['logit']
                if not k in new_dict1.keys():
                    new_dict1[k] = v / v.sum()
                else:
                    new_dict1[k] += v / v.sum()
        new_dict2 = {}
        for res in tqdm(results_file2):
            res_dict = load_dict(res)
            for k, v in res_dict.items():
                v = v['logit']
                if not k in new_dict2.keys():
                    new_dict2[k] = v / v.sum()
                else:
                    new_dict2[k] += v / v.sum()

        all_id_list = []
        all_predicted = []
        for k, v in tqdm(new_dict1.items(), desc='Fusing result'):
            p0 = v
            p1 = new_dict2[k]
            p0 /= p0.sum()
            p1 /= p1.sum()
            # p = p0*p1 / (p0*p1 + (1-p0) * (1-p1))
            p = p0 + p1
            top3_result = np.argsort(-p, axis=-1)[:3]
            all_predicted.append('{:d} {:d} {:d}'.format(top3_result[0], top3_result[1], top3_result[2]))
            all_id_list.append(k)
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = os.path.join(result_dir, 'ensemble_two_kinds.csv')
        datafame.to_csv(predict_csv, index=False, sep=',')

    def fuse_probs(self, result_dir):
        results_file = glob(os.path.join(result_dir, '*.pkl'))
        num_models = len(results_file)
        print('Total {} ensemble files.'.format(len(results_file)))
        new_dict_list = []
        for res in tqdm(results_file):
            res_dict = load_dict(res)
            new_dict_list.append(res_dict)
        all_id_list = []
        all_predicted = []
        pi_product = 1.
        tmp_product = 1.
        for k, v in tqdm(new_dict_list[0].items(), desc='Fusing result'):
            pi = [tv[k]['logit'] for tv in new_dict_list]
            for it in pi:
                it /= it.sum()
                pi_product *= it
                tmp_product *= (1 - it)
            p = pi_product / (pi_product + tmp_product)
            # p0 = v
            # p1 = new_dict_list[1][k]
            # p0 /= p0.sum()
            # p1 /= p1.sum()
            # p = p0*p1 / (p0*p1 + (1-p0) * (1-p1))
            top3_result = np.argsort(-p, axis=-1)[:3]
            all_predicted.append('{:d} {:d} {:d}'.format(top3_result[0], top3_result[1], top3_result[2]))
            all_id_list.append(k)
        datafame = pd.DataFrame({'id': all_id_list, 'predicted': all_predicted})
        predict_csv = os.path.join(result_dir, 'ensemble_prob.csv')
        datafame.to_csv(predict_csv, index=False, sep=',')


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

