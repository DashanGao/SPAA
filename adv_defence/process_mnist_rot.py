import sys

import numpy as np # linear algebra
from PIL import Image
import random
import torch
from torch.nn import ZeroPad2d
from numpy.random import choice
from torchvision import transforms
from torchvision.utils import save_image
import os, codecs


def process_data(data, target, aug_fold=5, min_rot=-45, max_rot=45):
    """
    rotate image with given angle range. Each image rotate for aug_fold times.
    :param data_path: path to load original processed .pt data.
    :param aug_fold: Path to
    :param min_rot:
    :param max_rot:
    :return:
    """
    if len(target.shape) == 2 and target.shape[1] == 2:
        raise RuntimeError("Data has been rotated, target shape is (*, 2)")

    rot_data = []
    rot_target = []
    rot_label = []

    trans2PIL = transforms.ToPILImage(mode='L')
    trans2tensor = transforms.ToTensor()

    assert max_rot > min_rot, "max_rot must > min_rot"
    pad_img = ZeroPad2d((2, 2, 2, 2))

    for idx in range(data.shape[0]):
        # rot_list = choice(np.arange(min_rot, max_rot), aug_fold, replace=False)
        rot_list = choice((min_rot, 0, max_rot), aug_fold, replace=False)

        for rot_angle in rot_list:
            img = trans2PIL(data[idx]).rotate(rot_angle)
            rot_img = trans2tensor(img)
            rot_img = pad_img(rot_img)
            rot_data.append(rot_img)
            norm_rot_angle = 1.0 * (rot_angle - min_rot) / (max_rot - min_rot)
            rot_target.append(torch.tensor([norm_rot_angle, target[idx]]))

    # List of tensors to tensor
    rot_data = torch.cat(rot_data, dim=0)
    rot_target = torch.cat(rot_target, dim=0).reshape(-1, 2)

    return rot_data, rot_target


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


if __name__ == '__main__':
    project_root = '/home/cseadmin/urt_unified_robust_training/l2l-da/'
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    data_path = os.path.join(project_root, 'datasets/mnist_rot/MNIST/processed/')
    raw_folder = os.path.join(project_root, 'datasets/mnist_rot/MNIST/raw/')

    # Processing training data
    print("processing training data ...")
    train_data = (
        read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte'))
    )
    # train_data, train_target = torch.load(os.path.join(data_path, 'training.pt'))
    train_data, train_target = process_data(train_data[0], train_data[1], aug_fold=3, min_rot=-40, max_rot=40)
    print(train_data[0].shape)

    print(train_target[:3])
    with open(os.path.join(data_path, 'training.pt'), 'wb') as f:
        torch.save((train_data, train_target), f)

    # Processing test data
    print("processing test data ...")
    test_data = (
        read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(raw_folder, 't10k-labels-idx1-ubyte'))
    )
    # test_data, test_target = torch.load(os.path.join(data_path, 'test.pt'))
    test_data, test_target = process_data(test_data[0], test_data[1], aug_fold=3, min_rot=-40, max_rot=40)
    with open(os.path.join(data_path, 'test.pt'), 'wb') as f:
        torch.save((test_data, test_target), f)
