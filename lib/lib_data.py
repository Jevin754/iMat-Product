import os
import glob
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform
from PIL import Image

from lib.util import save_dict, load_dict

def collate_fn_image(batch):
    image_list = []
    label_list = []
    super_label_list = []
    id_list = []
    for i, bat in enumerate(batch):
        image = bat['image']
        label = bat['label']
        super_label = bat['super_label']
        id = bat['id']
        image_list.append(image)
        label_list.append(label)
        super_label_list.append(super_label)
        id_list.append(id)
    image_tensor = torch.stack(image_list, dim=0)
    label_tensor = torch.LongTensor(label_list)
    super_label_tensor = torch.FloatTensor(super_label_list)
    return {'image': image_tensor, 'label': label_tensor, 'id': id_list, 'super_label': super_label_tensor}


class DataProcessor():
    def __init__(self):
        self.img_root = '/data1/zhaoj/data/challenge/iMaterialist/image_fullsize'
        self.img_list_root = '/data1/zhaoj/data/challenge/iMaterialist/img_list'
        self.new_json_root = './data'

    def fuse_bothsize_image(self, phase):
        img_root_fullsize = '/data1/zhaoj/data/challenge/iMaterialist/image_fullsize'
        img_root_resize = '/data1/zhaoj/data/challenge/iMaterialist/image_resize'
        img_root_fusesize = '/data1/zhaoj/data/challenge/iMaterialist/image_fusesize'
        img_list_root = '/data1/zhaoj/data/challenge/iMaterialist/img_list'
        json_file = os.path.join(img_list_root, phase + '.json')
        non_exist_id = []
        exist_item = []
        with open(json_file, 'r') as fid:
            img_list = json.loads(fid.read())['images']
        for item in tqdm(img_list):
            img_file_fullsize = os.path.join(img_root_fullsize, phase, item['id'])
            img_file_resize = os.path.join(img_root_resize, phase, item['id'])
            target = os.path.join(img_root_fusesize, phase, item['id'])
            if os.path.exists(img_file_fullsize):
                exist_item.append(item)
                os.system('cp {:s} {:s}'.format(img_file_fullsize, target))
            elif os.path.exists(img_file_resize):
                exist_item.append(item)
                os.system('cp {:s} {:s}'.format(img_file_resize, target))
            else:
                non_exist_id.append(item['id'])
        print('Number of images not exist in {:s}: {:d}'.format(phase + '.json', len(non_exist_id)))
        new_dict = {'images': exist_item}
        new_json_file = os.path.join(self.new_json_root, phase + '_fusesize.json')
        with open(new_json_file, 'w') as fid:
            json.dump(new_dict, fid)
        print('Data written to {:s}'.format(new_json_file))

    def check_exist(self, img_list_file):
        phase = img_list_file.split('.')[0]
        img_list_file_full = os.path.join(self.img_list_root, img_list_file)
        non_exist_id = []
        exist_item = []
        with open(img_list_file_full, 'r') as fid:
            img_list = json.loads(fid.read())['images']
        for item in tqdm(img_list):
            img_file = os.path.join(self.img_root, phase, item['id'])
            if not os.path.exists(img_file):
                non_exist_id.append(img_file)
            else:
                exist_item.append(item)
        print('Number of images not exist in {:s}: {:d}'.format(img_list_file, len(non_exist_id)))
        new_dict = {'images': exist_item}
        new_json_file = os.path.join(self.new_json_root, phase + '_fullsize.json')
        with open(new_json_file, 'w') as fid:
            json.dump(new_dict, fid)
        print('Data written to {:s}'.format(new_json_file))

    def get_superlabel_binary(self):
        self.label_tree_file = './data/product_tree.json'
        with open(self.label_tree_file, 'r') as fid:
            product_tree = json.loads(fid.read())
        N_l1_item = len(product_tree)
        label2superlabel_bin_dict = {}
        for l1 in range(1, N_l1_item+1):
            l1_key = 'level1_node{:d}'.format(l1)
            l1_value = product_tree[l1_key]
            l1_bin_code = bin(l1-1)[2:].zfill(3)
            N_l2_item = len(l1_value)
            for l2 in range(1, N_l2_item+1):
                l2_key = 'level2_node{:d}'.format(l2)
                l2_value = l1_value[l2_key]
                l2_bin_code = bin(l2-1)[2:].zfill(4)
                N_l3_item = len(l2_value)
                for l3 in range(1, N_l3_item+1):
                    l3_key = 'level3_node{:d}'.format(l3)
                    l3_value = l2_value[l3_key]
                    l3_bin_code = bin(l3-1)[2:].zfill(4)
                    N_l4_item = len(l3_value)
                    superlabel_bin = l1_bin_code + l2_bin_code + l3_bin_code
                    for item in l3_value:
                        label2superlabel_bin_dict[int(item)] = [int(i) for i in superlabel_bin]
                    # pass
        label2superlabel_bin_dict[-1] = [1] * 11
        save_dict(label2superlabel_bin_dict, './data/fglabel2splabel.pkl')
        pass

class ImageDataset(Dataset):
    def __init__(self, phase, is_transform=True, DEBUG=False):
        self.img_size = 'fusesize'
        self.img_root = '/data1/zhaoj/data/challenge/iMaterialist/image_' + self.img_size
        self.img_list_root = './data'
        self.phase = phase
        self.fglabel2splabel_dict = load_dict('./data/fglabel2splabel.pkl')

        reimgsize = 256
        self.transform_dict = {'resnet':
                                   {'train': transforms.Compose([transforms.Resize(reimgsize), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                                    'test': transforms.Compose([transforms.Resize(reimgsize), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])},
                                    # 'test': transforms.Compose([transforms.Resize(reimgsize), transforms.CenterCrop(224), transforms.ToTensor(),
                                    #                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])},
                               'senet':
                                   {'train': transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                                    'test': transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])},
                               'inceptionresnet':
                                   {'train': transforms.Compose([transforms.Resize(342), transforms.RandomResizedCrop(299), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                                    'test': transforms.Compose([transforms.Resize(342), transforms.RandomResizedCrop(299), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])},
                               'nasnet':
                                   {'train': transforms.Compose([transforms.Resize(378), transforms.RandomResizedCrop(331), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                                    'test': transforms.Compose([transforms.Resize(378), transforms.RandomResizedCrop(331), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
        }

        if phase == 'train' and is_transform:
            self.transform = self.transform_dict['inceptionresnet']['train']
            # self.transform = transforms.Compose([transforms.Resize(512), transforms.RandomResizedCrop(448), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = self.transform_dict['inceptionresnet']['test']
            # self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # self.transform = transforms.Compose([transforms.Resize(512), transforms.RandomCrop(448), transforms.ToTensor(),
            #                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.img_list = self.load_json_file()
        if DEBUG:
            self.img_list = self.img_list[:301]

    def load_json_file(self):
        json_file = os.path.join(self.img_list_root, self.phase + '_{:s}.json'.format(self.img_size))
        with open(json_file, 'r') as fid:
            img_list = json.loads(fid.read())['images']
        return img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cur_img_info = self.img_list[idx]
        id = cur_img_info['id']
        if self.phase in ['train', 'val']:
            label = int(cur_img_info['class'])
        else:
            label = -1
        img_tensor = self.load_image(os.path.join(self.img_root, self.phase, id))
        super_label = self.fglabel2splabel_dict[label]
        sample = {'id': id, 'image': img_tensor, 'label': label, 'super_label': super_label}
        return  sample

    def load_image(self, img_name):
        image = Image.open(img_name)
        image = image.convert("RGB")
        image = self.transform(image)
        return image
