from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import os
from torchvision import transforms
import torchvision.datasets as datasets
from adv_defence.datasets import MNIST as MyMNIST

DATASETS = ['mnist', 'cifar10', 'tinyimagenet', 'mnist_rot', 'celeba']


def get_loader(dataset_name, root, batch_size, split='train', num_workers=2, shuffle=True):
    if dataset_name not in DATASETS:
        raise Exception("[!] No data loader found for the dataset: {}.".format(dataset_name))

    # transform chain
    transform_list = []
    if split == 'train':
        if dataset_name == 'celeba':
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomCrop(256, 16))
        if dataset_name == 'mnist_rot':
            transform_list.append(transforms.RandomCrop(32, 4))
        if dataset_name == 'cifar10':
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomCrop(32, 4))
        if dataset_name == 'mnist':
            transform_list.append(transforms.RandomCrop(28, 4))
        if dataset_name == 'tinyimagenet':
            transform_list.append(transforms.RandomRotation(20))
            transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())
    transform_chain = transforms.Compose(transform_list)

    if dataset_name == 'celeba':
        item = datasets.CelebA(root=root, split=split, target_type='attr', transform=transform_chain,
                               download=True)
    if dataset_name == 'mnist_rot':
        item = MyMNIST(root=root, train=split=='train', transform=transform_chain, download=True)

    if dataset_name == 'mnist':
        item = datasets.MNIST(root=root, train=split=='train', transform=transform_chain, download=True)
    elif dataset_name == 'cifar10':
        item = datasets.CIFAR10(root=root, train=split=='train', transform=transform_chain, download=True)
    elif dataset_name == 'tinyimagenet':
        item = datasets.ImageFolder(os.path.join(root, ('train' if split=='train' else 'val')), transform=transform_chain)
    print(dataset_name, split, item.__len__(), batch_size)

    data_loader = torch.utils.data.DataLoader(dataset=item,
                                              batch_size=batch_size,
                                              shuffle=split == 'train',
                                              num_workers=num_workers)
    return data_loader
