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

from options import args

'''
Popup from What's Hidden in a Randomly Weighted Neural Network? CVPR'21
'''

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1-k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity='relu')

        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


'''
set model type
'''

if args.model_type == 'mask':
    linear_block = SupermaskLinear
    conv_block = SupermaskConv
elif args.model_type == 'regular':
    linear_block = nn.Linear
    conv_block = nn.Conv2d
else:
    raise NotImplementedError

if args.bn_type == 'learn':
    bn_block = nn.BatchNorm2d
elif args.bn_type == 'not-learn':
    bn_block = NonAffineBatchNorm
else:
    raise NotImplementedError

if args.act == 'relu':
    act_block = nn.ReLU
elif args.act == 'gelu':
    act_block = nn.GELU
else:
    raise NotImplementedError


'''
convmixer model from Patches Are All You Need?, arXiv'22
'''

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        conv_block(3, dim, kernel_size=patch_size, stride=patch_size, bias=False),
        act_block(),
        bn_block(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    conv_block(dim, dim, kernel_size, groups=dim, padding="same", bias=False),
                    act_block(),
                    bn_block(dim)
                )),
                conv_block(dim, dim, kernel_size=1, bias=False),
                act_block(),
                bn_block(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        linear_block(dim, n_classes, bias=False)
    )