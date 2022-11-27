import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import argparse

import torch.autograd as autograd
import math
import torch.nn.functional as F

import pathlib
from pdb import set_trace as st
import copy
import os

from torch.utils.data import Dataset
from PIL import Image

class Dataset_npy_batch(Dataset):
    def __init__(self, npy_dir, transform, f='batch.npy'):
        self.data = np.load(os.path.join(npy_dir, f), allow_pickle=True)
        self.transform = transform
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][0])
        img = self.transform(img)
        label = self.data[index][1]
        label = torch.LongTensor([label])[0]
        return img.squeeze(0), label
    def __len__(self):
        return len(self.data)
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('folder exists')

def write_result_to_csv(args, **kwargs):
    results = pathlib.Path(args.save_file)

    if not results.exists():
        results.write_text(
            "epoch, "
            "depth, "
            "act, "
            "model_type, "
            "bn_type, "
            "hdim, "
            "clone_type, "
            "MP_RP_ratio, "
            "MP_RP_mode, "
            "save_file, "
            "exp_id, "
            "best_acc\n "
        )

    with open(results, "a+") as f:
        f.write(
            (
                "{epoch}, "
                "{depth}, "
                "{act}, "
                "{model_type}, "
                "{bn_type}, "
                "{hdim}, "
                "{clone_type}, "
                "{MP_RP_ratio}, "
                "{MP_RP_mode}, "
                "{save_file}, "
                "{exp_id}, "
                "{best_acc}\n "
            ).format(**kwargs)
        )

def one_layer(model):
    print("==> Clone layer-wise weights")

    proto_size = []
    proto_weight=[]
    proto_name = []

    for name, module in model.named_modules():
        if type(module).__name__ == 'SupermaskConv':
            if module.weight.data.size() not in proto_size:
                proto_size.append(module.weight.data.size())
                proto_weight.append(module.weight.data)
                proto_name.append(name + '-' + type(module).__name__)
                print("==> save proto layer: ", name + '-' + type(module).__name__)
            else:
                cloned_index = proto_size.index(module.weight.data.size())
                cloned_data = proto_weight[cloned_index]
                module.weight.data = copy.deepcopy(cloned_data)
                print("==> clone proto from ", proto_name[cloned_index], 'to', name + '-' + type(module).__name__)

def MP_RP(args, model):
    print("Start augment")
    vector = []
    largest_num = 0

    for name, module in model.named_modules():
        if type(module).__name__ == 'SupermaskConv' or type(module).__name__ == 'SupermaskLinear':
            if module.weight.numel() > largest_num:
                largest_num = module.weight.numel()

                if len(vector) == 1:
                    vector[0] = copy.deepcopy(module.weight.data)
                else:
                    vector.append(copy.deepcopy(module.weight.data))
    
    one_vec = vector[0]
    one_vec = one_vec.view(-1, one_vec.numel()).squeeze()
    one_vec_numel = one_vec.numel()
    print('Random vec length', len(one_vec))

    if args.MP_RP_ratio < 1.0:
        print('Basic ratio: ', args.MP_RP_ratio)
        dict_length = int(one_vec_numel * args.MP_RP_ratio) + 1
        basic_dict = one_vec[:dict_length]
        num_basic_dict = int(one_vec_numel / dict_length) + 1

        if args.MP_RP_mode == 'rand':
            augmented_dict = basic_dict.repeat(num_basic_dict)
            cut_dict = augmented_dict[:one_vec_numel]
            
            random_idx = torch.randperm(one_vec_numel)
            one_vec = cut_dict[random_idx]

        elif args.MP_RP_mode == 'copy':
            augmented_dict = basic_dict.repeat(num_basic_dict)
            one_vec = augmented_dict[:one_vec_numel]

        else:
            raise NotImplementedError

    for name, module in model.named_modules():
        if type(module).__name__ == 'SupermaskConv' or type(module).__name__ == 'SupermaskLinear':
            cur_length = module.weight.data.numel()
            cur_size = module.weight.data.size()

            cur_vec = copy.deepcopy(one_vec[:cur_length])
            cur_param = cur_vec.view(cur_size)
            module.weight.data = cur_param
    
    print('Augmented')
