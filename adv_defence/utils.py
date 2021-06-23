from __future__ import print_function

import os
import json
import logging
import numpy as np
from datetime import datetime
import torch
from torch.nn import functional as F


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def model_name_generator(config):
    name_str = []
    name_str.append('dataset={}'.format(config.dataset))
    name_str.append('lambda={}'.format(config.dsgan_lambda))
    name_str.append('g_method={}'.format(config.g_method))
    name_str.append('mini_step={}'.format(config.g_ministep_size))
    name_str.append('g_z_dim={}'.format(config.g_z_dim))
    name_str.append('cond_batch={}'.format(config.num_classes > 1))
    name_str.append('f_update={}'.format(config.f_update_style))
    name_str.append('g_mini_update={}'.format(config.g_mini_update_style))
    name_str.append('f={}'.format(config.f_classifier_name))
    name_str.append('f_pre={}'.format(config.f_pretrain))
    name_str.append('g_ce={}'.format(config.use_cross_entropy_for_g))
    name_str.append('g_grad={}'.format(config.g_normalize_grad))
    name_str.append('g_step={}'.format(config.train_g_iter))
    name_str.append('f_lr={}'.format(config.f_lr))
    name_str.append('g_lr={}'.format(config.g_lr))
    name_str.append('g_use_grad={}'.format(config.g_use_grad))
    name_str.append('gamma={}'.format(config.lr_gamma))
    name_str.append('b={}'.format(config.single_batch_size))
    name_str.append(get_time())
    return '-'.join(name_str)


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        config.model_dir = config.load_path
        config.model_name = os.path.basename(config.load_path)
    else:
        config.model_name = model_name_generator(config)

    if (not hasattr(config, 'model_dir')) or (len(config.model_dir) == 0):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if ('CIFAR10' in config.f_classifier_name) or ('MNIST' in config.f_classifier_name) or ('TinyImagenet' in config.f_classifier_name):
        config.model_dir = os.path.join(config.log_dir, config.f_classifier_name)
        config.model_name = config.f_classifier_name + '_adapt'

    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] Model Name: %s" % config.model_name)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)




##############
#  MSP attribute_disentanglement
##############

def _loss_reconstruction(predict, orig):
    """
    Mean square error (MSE) loss for image reconstruction. Even under attribute-perturbation.
    :param predict:
    :param orig:
    :return:
    """
    batch_size = predict.shape[0]
    a = predict.view(batch_size, -1)
    b = orig.view(batch_size, -1)
    L = F.mse_loss(a, b, reduction='sum')
    return L

def _loss_vae(mu, logvar):
    # https://arxiv.org/abs/1312.6114
    # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def _loss_msp(M, attr_label, latent_embedding):
    """
    MSP loss for attribute disentanglement.
    :param attr_label:  the attributes of the image
    :param latent_embedding:  the extracted embedding
    :return:  MSP loss
    """
    latent_embedding = torch.flatten(latent_embedding, 1, -1)
    attr_label = attr_label.view(-1, 1)
    L1 = F.mse_loss((latent_embedding @ M.t()).view(-1), attr_label.view(-1), reduction="none").sum()
    L2 = F.mse_loss((attr_label @ M).view(-1), latent_embedding.view(-1), reduction="none").sum()
    return L1 + L2, L1, L2


def loss(pred, real_img, attr_label, attr_pred, mu, logvar, M):
    L_rec = _loss_reconstruction(pred, real_img)
    L_vae = _loss_vae(mu, logvar)
    L_msp, L_msp_1, L_msp_2 = _loss_msp(M, attr_label, attr_pred)
    _msp_weight = real_img.numel()/(attr_label.numel()+attr_pred.numel())
    Loss = 0.01 * L_rec + 0.0001 * L_vae + 0.0001 * L_msp * _msp_weight
    return Loss, L_rec, L_vae, L_msp * _msp_weight, L_msp_1 * _msp_weight, L_msp_2 * _msp_weight


def acc(M, attr_pred, attr_label):
    """
    compute accuracy of predicted attributes
    :param attr_pred:
    :param attr_label:
    :return:
    """
    zl = attr_pred @ M.t()
    a = zl.clamp(-1, 1) * attr_label * 0.5 + 0.5
    return a.round().mean().item()


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

