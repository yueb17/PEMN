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

from models import ConvMixer
from utils import write_result_to_csv, mkdir
from utils import MP_RP, one_layer
from utils import Dataset_npy_batch

from options import args
import os

if args.dataset == 'cf10':
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=args.reprob)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
elif args.dataset == 'cf100':
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
        transforms.RandomErasing(p=args.reprob)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
elif args.dataset == 'tiny':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    trainset = Dataset_npy_batch('data/tiny_imagenet/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    
    testset = Dataset_npy_batch('data/tiny_imagenet/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    
else:
    raise NotImplementedError

model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=args.num_class)

if args.parallel:
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda()

if args.pipe == 'test':
    print('Test model acc')
    model.load_state_dict(torch.load(args.saved_model_path + 'best.pt'))
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

        print('Test acc: ', test_acc/m)

    model.load_state_dict(torch.load(args.saved_model_path + 'start.pt'))
    model.eval()

elif args.pipe == 'train':
    # change model #
    if args.clone_type == 'clone':
        one_layer(model)
    elif args.clone_type == 'augment':
        MP_RP(args, model)
    elif args.clone_type == 'no':
        print("KEEP MODEL AS IT IS")
    else:
        raise NotImplementedError

    # save model path
    saved_path = 'saved_models/' + \
                     str(args.epochs) + '_' + \
                     str(args.depth) + '_' + \
                     str(args.act) + '_' + \
                     str(args.model_type) + '_' + \
                     str(args.bn_type) + '_' + \
                     str(args.hdim) + '_' + \
                     str(args.clone_type) + '_' + \
                     str(args.MP_RP_ratio) + '_' + \
                     str(args.MP_RP_mode) + '_' + \
                     str(args.save_file) + '_' + \
                     str(args.exp_id) + '_' + \
                     '/'
    mkdir(saved_path)

    if args.save_start:
        model_path = saved_path + 'start.pt'
        start_state = model.state_dict()
        torch.save(start_state, model_path)

    # train
    print('Train model')
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_max, weight_decay=args.wd)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
                                  [0, args.lr_max, args.lr_max/20.0, 0])[0]
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, n = 0, 0, 0
        for i, (X, y) in enumerate(trainloader):
            model.train()
            X, y = X.cuda(), y.cuda()

            lr = lr_schedule(epoch + (i + 1)/len(trainloader))
            opt.param_groups[0].update(lr=lr)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)

            scaler.update()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        model.eval()
        test_acc, m = 0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(testloader):
                X, y = X.cuda(), y.cuda()
                with torch.cuda.amp.autocast():
                    output = model(X)
                test_acc += (output.max(1)[1] == y).sum().item()
                m += y.size(0)

        if test_acc >= best_acc:
            best_acc = test_acc

            if args.save:
                best_state = model.state_dict()
                model_path = saved_path + 'best.pt'
                torch.save(best_state, model_path)

        print(f'[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

    if args.save:
        print('saving')
        write_result_to_csv(args,
            epoch=args.epochs,
            depth=args.depth,
            act=args.act,
            model_type=args.model_type,
            bn_type=args.bn_type,
            hdim=args.hdim,
            clone_type=args.clone_type,
            MP_RP_ratio=args.MP_RP_ratio,
            MP_RP_mode=args.MP_RP_mode,
            save_file=args.save_file,
            exp_id=args.exp_id,
            best_acc=best_acc/m
            )

