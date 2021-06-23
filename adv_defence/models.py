""" models.py """
# Author: Dashan
# Email: dgaoaa@connect.ust.hk
# Date: Jun 4th, 2021
# Description: This code is based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/bicyclegan/models.py


from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Function, Variable
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from collections import OrderedDict
from adv_defence.sync_batchnorm import SynchronizedBatchNorm2d
import cifar10.cifar_resnets as cifar_resnets
import os
import re

from adv_defence import helper


def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif (classname.find('BatchNorm') != -1):
        if (m.weight is not None) and (m.weight.data is not None):
            if (m.num_category is not None):
                # class-conditional batch norm
                nn.init.normal_(m.weight.data[:-1], 0.0, 0.02)
                nn.init.normal_(m.weight.data[-1], 1.0, 0.02)
            else:
                # general batch norm
                nn.init.normal_(m.weight.data, 1.0, 0.02)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


def reparameterize(mu, logvar):
    """
    For forward() method in attribute disentanglement.
    :param mu:
    :param logvar:
    :return:
    """
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


##############################
#           Generator
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, num_class, normalize=True, kernel=4, stride=2, dropout=0.0):
        super(UNetDown, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride, padding=1, bias=False)
        self.num_class = num_class
        if normalize:
            if num_class <= 1:
                self.norm = SynchronizedBatchNorm2d(out_size, eps=1e-10)
            else:
                self.norm = SynchronizedBatchNorm2d(out_size, num_category=num_class, eps=1e-10)
        else:
            self.norm = None
        self.fn = nn.LeakyReLU(0.2)
        if dropout:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, x, y, z=None):
        if z is not None:
            width = x.shape[2]
            spatial_tile_z = torch.unsqueeze(torch.unsqueeze(z, -1).expand(-1, -1, width), -1).expand(-1, -1, -1, width)
            out = self.conv(torch.cat((x, spatial_tile_z), 1))
        else:
            out = self.conv(x)

        if self.norm is not None:
            if self.num_class <= 1:
                out = self.norm(out)
            else:
                out = self.norm(out, y)
        out = self.fn(out)
        if self.drop is not None:
            out = self.drop(out)
        return out


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, num_class, dropout=0.0):
        super(UNetUp, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.num_class = num_class
        if num_class <= 1:
            self.norm = SynchronizedBatchNorm2d(out_size, eps=1e-10)
        else:
            self.norm = SynchronizedBatchNorm2d(out_size, num_category=num_class, eps=1e-10)
        self.fn = nn.ReLU(inplace=True)
        if dropout:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, x, y, skip_input=None):
        out = self.upconv(x)
        if self.num_class <= 1:
            out = self.norm(out)
        else:
            out = self.norm(out, y)
        out = self.fn(out)
        if self.drop is not None:
            out = self.drop(out)

        if skip_input is not None:
            out = torch.cat((out, skip_input), 1)
        return out


class NoiseGenerator(nn.Module):
    def __init__(self, base_channel_dim, input_img_channel, z_channel, deeper_layer, num_class, last_dim,
                 skip_connect=False):
        super(NoiseGenerator, self).__init__()
        self.deeper_layer = deeper_layer
        self.skip_connect = skip_connect
        self.down0 = UNetDown(input_img_channel + z_channel, base_channel_dim, num_class, kernel=3, stride=1,
                              normalize=False)
        self.down1 = UNetDown(base_channel_dim + z_channel, base_channel_dim * 1, num_class)
        self.down2 = UNetDown(base_channel_dim * 1 + z_channel, base_channel_dim * 2, num_class, normalize=deeper_layer)
        if deeper_layer:
            self.down3 = UNetDown(base_channel_dim * 2 + z_channel, base_channel_dim * 4, num_class, normalize=False)
            self.up3 = UNetUp(base_channel_dim * 4, base_channel_dim * 2, num_class)
            self.up2 = UNetUp(base_channel_dim * 2 * (2 if skip_connect else 1), base_channel_dim * 1, num_class)
        else:
            self.up2 = UNetUp(base_channel_dim * 2, base_channel_dim * 1, num_class)
        self.up1 = UNetUp(base_channel_dim * 1 * (2 if skip_connect else 1), base_channel_dim, num_class)

        final = [nn.Conv2d(base_channel_dim * (2 if skip_connect else 1), last_dim,
                           kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()]
        self.final = nn.Sequential(*final)

        if z_channel > 0:
            self.z_encoder = nn.Sequential(
                nn.Linear(z_channel, z_channel),
                nn.ReLU(),
                nn.Linear(z_channel, z_channel),
                nn.ReLU(),
            )
        else:
            self.z_encoder = None

        if deeper_layer:
            self.hidden_size = base_channel_dim * 4 * 4 * 4
        else:
            self.hidden_size = base_channel_dim * 2 * 8 * 8

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.M = nn.Parameter(torch.empty(1, self.hidden_size))
        nn.init.xavier_normal_(self.M)

        if deeper_layer:
            self.nonupdate_params = [self.down0, self.down1, self.down2, self.down3, self.M]
            self.uper_input_dim = base_channel_dim * 4
        else:
            self.nonupdate_params = [self.down0, self.down1, self.down2, self.M]
            self.uper_input_dim = base_channel_dim * 2

    # !!!!!! Notice: these methods are moved to utils.py !!!!!!!
    #
    # def _loss_reconstruction(self, predict, orig):
    #     """
    #     Mean square error (MSE) loss for image reconstruction. Even under attribute-perturbation.
    #     :param predict:
    #     :param orig:
    #     :return:
    #     """
    #     batch_size = predict.shape[0]
    #     a = predict.view(batch_size, -1)
    #     b = orig.view(batch_size, -1)
    #     L = F.mse_loss(a, b, reduction='sum')
    #     return L
    #
    # def _loss_vae(self, mu, logvar):
    #     # https://arxiv.org/abs/1312.6114
    #     # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return KLD
    #
    # def _loss_msp(self, attr_label, latent_embedding):
    #     """
    #     MSP loss for attribute disentanglement.
    #     :param attr_label:  the attributes of the image
    #     :param latent_embedding:  the extracted embedding
    #     :return:  MSP loss
    #     """
    #     latent_embedding = torch.flatten(latent_embedding, 1, -1)
    #     attr_label = attr_label.view(-1, 1)
    #     L1 = F.mse_loss((latent_embedding @ self.M.t()).view(-1), attr_label.view(-1), reduction="none").sum()
    #     L2 = F.mse_loss((attr_label @ self.M).view(-1), latent_embedding.view(-1), reduction="none").sum()
    #     return L1 + L2
    #
    # def loss(self, pred, real_img, attr_label, latent_embedding, mu, logvar):
    #     L_rec = self._loss_reconstruction(pred, real_img)
    #     L_vae = self._loss_vae(mu, logvar)
    #     L_msp = self._loss_msp(attr_label, latent_embedding)
    #     _msp_weight = real_img.numel()/(attr_label.numel()+latent_embedding.numel())
    #     Loss = L_rec + L_vae + L_msp * _msp_weight
    #     return Loss, L_rec.item(), L_vae.item(), L_msp.item()

    def predict(self, x, y, z, new_ls=None, weight=1.0):
        """
        Generate adversarial samples with given attribute vector
        :param x: raw image
        :param y: label
        :param z: adversarial perturbation
        :param new_ls: attribute perturbation
        :param weight: weight of attribute perturbation
        :return: generated adv-sample
        """
        # TODO: add skip_connection choise
        if self.z_encoder is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None

        d0 = self.down0(x, y, z_encoded)
        d1 = self.down1(d0, y, z_encoded)
        d2 = self.down2(d1, y, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, y, z_encoded)
            # Do attribute disentanglement on d3
            z_, _ = self._encode(d3)
        else:
            # Do attribute disentanglement on d2
            z_, _ = self._encode(d2)

        if new_ls is not None:
            zl = z_ @ self.M.t()
            d = torch.zeros_like(zl)
            for i, v in new_ls:
                d[:, i] = v * weight - zl[:, i]
            z_ += d @ self.M

        if self.deeper_layer:
            u3 = self.up3(z_, y, d2)
        else:
            u3 = z_
        u2 = self.up2(u3, y, d1)
        u1 = self.up1(u2, y, d0)
        result = self.final(u1)
        return result

    def forward(self, x, y, z):
        """
        Forward with attribute disentanglement.
        :param x: original images
        :param y: labels
        :param z: used to generate diverse adv examples
        :return:
            result: generated adv_examples
            z_: encoding
            mu: attribute vector
            logvar: I don't know what's this shit
        """
        # TODO: add skip_connection choise
        if self.z_encoder is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None
        d0 = self.down0(x, y, z_encoded)
        d1 = self.down1(d0, y, z_encoded)
        d2 = self.down2(d1, y, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, y, z_encoded)
            # Do attribute disentanglement on d3
            mu, logvar = self._encode(d3)
            z_ = reparameterize(mu, logvar)
            z_ = z_.view(z_.shape[0], self.uper_input_dim, d2.shape[-2], d2.shape[-1])
            u3 = self.up3(z_, y, d2)
        else:
            # Do attribute disentanglement on d2
            mu, logvar = self._encode(d2)
            z_ = reparameterize(mu, logvar)
            z_ = z_.view(z_.shape[0], self.uper_input_dim, d2.shape[-2], d2.shape[-1])
            u3 = z_

        u2 = self.up2(u3, y, d1)
        u1 = self.up1(u2, y, d0)
        result = self.final(u1)

        return result, z_, mu, logvar

    def forward_without_attr_disentangle(self, x, y, z):
        """
        Deprecated. Original method with NO attribute perturbation
        :param x: image
        :param y: label
        :param z: adv-noise for generator
        :return: adv-sample
        """
        if self.z_encoder is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None
        d0 = self.down0(x, y, z_encoded)
        d1 = self.down1(d0, y, z_encoded)
        d2 = self.down2(d1, y, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, y, z_encoded)
            u3 = self.up3(d3, y, d2)
        else:
            u3 = d2

        u2 = self.up2(u3, y, d1)
        u1 = self.up1(u2, y, d0)
        result = self.final(u1)

        return result

    def _encode(self, h):
        h = torch.flatten(h, 1, -1)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def acc(self, attr_pred, attr_label):
        """
        compute accuracy of predicted attributes
        :param attr_pred:
        :param attr_label:
        :return:
        """
        zl = attr_pred @ self.M.t()
        a = zl.clamp(-1, 1) * attr_label * 0.5 + 0.5
        return a.round().mean().item()


class AttrAdvNoiseGenerator(NoiseGenerator):
    """
    Modified based on UNet with one encoder and two decoders: [dec_1, dec_2].
    The decoder dec_1 generates adversarial example with both 1) attribute perturbation, via MSP-loss, and
    2) adversarial perturbation, via adversarial loss.

    Two decoders shares the same latent subspace embedding. MSP loss is added to the embedding.
    """
    def __init__(self, base_channel_dim, input_img_channel, z_channel, deeper_layer, num_class, last_dim,
                 skip_connect=False):
        super().__init__(base_channel_dim, input_img_channel, z_channel, deeper_layer, num_class, last_dim,
                         skip_connect)

        if deeper_layer:
            self.dec2_up3 = UNetUp(base_channel_dim * 4, base_channel_dim * 2, num_class)
            self.dec2_up2 = UNetUp(base_channel_dim * 2 * (2 if skip_connect else 1), base_channel_dim * 1, num_class)
        else:
            self.dec2_up2 = UNetUp(base_channel_dim * 2, base_channel_dim * 1, num_class)
        self.dec2_up1 = UNetUp(base_channel_dim * 1 * (2 if skip_connect else 1), base_channel_dim, num_class)

        dec2_final = [nn.Conv2d(base_channel_dim * (2 if skip_connect else 1), last_dim,
                                kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()]
        self.dec2_final = nn.Sequential(*dec2_final)

        if deeper_layer:
            self.nonupdate_params += [self.dec2_up1, self.dec2_up2, self.dec2_up3, self.dec2_final]
        else:
            self.nonupdate_params += [self.dec2_up1, self.dec2_up2, self.dec2_final]

    def predict(self, x, y, z, new_ls=None, weight=1.0):
        """
        Generate adversarial samples with given attribute vector
        :param x: raw image
        :param y: label
        :param z: adversarial perturbation
        :param new_ls: attribute perturbation
        :param weight: weight of attribute perturbation
        :return: generated adv-sample
        """
        if self.z_encoder is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None

        d0 = self.down0(x, y, z_encoded)
        d1 = self.down1(d0, y, z_encoded)
        d2 = self.down2(d1, y, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, y, z_encoded)
            # Do attribute disentanglement on d3
            mu, _ = self._encode(d3)
        else:
            # Do attribute disentanglement on d2
            mu, _ = self._encode(d2)

        if new_ls is not None:
            zl = mu @ self.M.t()
            d = torch.zeros_like(zl)
            for i, v in new_ls:  # index, value
                d[:, i] = v * weight - zl[:, i]
            mu += d @ self.M
            if self.deeper_layer:
                mu = mu.view(mu.shape[0], self.uper_input_dim, d3.shape[-2], d3.shape[-1])
            else:
                mu = mu.view(mu.shape[0], self.uper_input_dim, d2.shape[-2], d2.shape[-1])

        if self.deeper_layer:
            if self.skip_connect:
                u3 = self.up3(mu, y, d2)
                dec2_u3 = self.dec2_up3(mu, y, d2)
            else:
                u3 = self.up3(mu, y)
                dec2_u3 = self.dec2_up3(mu, y)
        else:
            u3 = mu
            dec2_u3 = mu

        # TODO: Consider to add soft-parameter transfer between low layers: [up3, dec2_up3], [up2, dec2_up2]
        if self.skip_connect:
            # attribute + adversarial perturbation sample.
            u2 = self.up2(u3, y, d1)
            u1 = self.up1(u2, y, d0)

            # attribute perturbation sample.
            dec2_u2 = self.dec2_up2(dec2_u3, y, d1)
            dec2_u1 = self.dec2_up1(dec2_u2, y, d0)
        else:
            # attribute + adversarial perturbation sample.
            u2 = self.up2(u3, y)
            u1 = self.up1(u2, y)

            # attribute perturbation sample.
            dec2_u2 = self.dec2_up2(dec2_u3, y)
            dec2_u1 = self.dec2_up1(dec2_u2, y)

        adv_attr_result = self.final(u1)
        attr_result = self.dec2_final(dec2_u1)

        return adv_attr_result, attr_result

    def forward(self, x, y, z):
        """
        Attribute perturbation + adversarial perturbation
        Two decoders:
        :param x: original images
        :param y: labels
        :param z: used to generate diverse adv examples
        :return:
            result: generated adv_examples
            z_: encoding
            mu: attribute vector
            logvar: I don't know what's this shit
        """
        if self.z_encoder is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None
        d0 = self.down0(x, y, z_encoded)
        d1 = self.down1(d0, y, z_encoded)
        d2 = self.down2(d1, y, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, y, z_encoded)
            # Do attribute disentanglement on d3
            mu, logvar = self._encode(d3)
            print(d3.shape)
            exit()
            z_ = reparameterize(mu, logvar)
            z_ = z_.view(z_.shape[0], self.uper_input_dim, d3.shape[-2], d3.shape[-1])
            if self.skip_connect:
                u3 = self.up3(z_, y, d2)
                dec2_u3 = self.dec2_up3(z_, y, d2)
            else:
                u3 = self.up3(z_, y)
                dec2_u3 = self.dec2_up3(z_, y)
        else:
            # Do attribute disentanglement on d2
            mu, logvar = self._encode(d2)
            z_ = reparameterize(mu, logvar)
            z_ = z_.view(z_.shape[0], self.uper_input_dim, d2.shape[-2], d2.shape[-1])
            u3 = z_
            dec2_u3 = z_

        if self.skip_connect:
            # attribute + adversarial perturbation sample.
            u2 = self.up2(u3, y, d1)
            u1 = self.up1(u2, y, d0)

            # attribute perturbation sample.
            dec2_u2 = self.dec2_up2(dec2_u3, y, d1)
            dec2_u1 = self.dec2_up1(dec2_u2, y, d0)
        else:
            # attribute + adversarial perturbation sample.
            u2 = self.up2(u3, y)
            u1 = self.up1(u2, y)

            # attribute perturbation sample.
            dec2_u2 = self.dec2_up2(dec2_u3, y)
            dec2_u1 = self.dec2_up1(dec2_u2, y)

        adv_attr_result = self.final(u1)
        attr_result = self.dec2_final(dec2_u1)

        return adv_attr_result, attr_result, z_, mu, logvar


##############################
#        Classifier
##############################

class Classifier(nn.Module):
    def __init__(self, num_classes, classifier_name="resnet50", dataset="mnist",
                 pretrained=False, pretrained_dir="./pretrained_models"):
        # TODO: add dataset CelebA, and its [mean, std]
        #  Download dataset CelebA

        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.dataset = dataset
        self.classifier_name = classifier_name

        if classifier_name == 'lenet':
            self.mean = None
            self.std = None
        elif dataset == 'cifar10':
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif dataset == 'tinyimagenet':
            self.mean = [0.4802, 0.4481, 0.3975]
            self.std = [0.2302, 0.2265, 0.2262]
        elif dataset == 'celeba':  # TODO
            self.mean = [0, 0, 0]
            self.std = [0, 0, 0]
        else:
            self.mean = None
            self.std = None

        # map_location = None
        map_location = (lambda s, _: s)

        # https://pytorch.org/docs/stable/torchvision/models.html#id3
        if classifier_name == "resnet50":
            if dataset == "cifar10":
                self.feature_extractor = cifar_resnets.resnet56()
                if pretrained:
                    weight_path = os.path.join(pretrained_dir, 'cifar10_resnet56.th')
                    bad_state_dict = torch.load(weight_path, map_location=map_location)
                    correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                                          bad_state_dict['state_dict'].items()}
                    self.feature_extractor.load_state_dict(correct_state_dict)
                self.fc = None
            else:
                model = resnet50(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-3])
                self.fc = nn.Linear(4096, num_classes)
                nn.init.xavier_uniform_(self.fc.weight)
                nn.init.zeros_(self.fc.bias)
        elif classifier_name == "resnet20":
            self.feature_extractor = cifar_resnets.resnet20()
            if pretrained:
                weight_path = os.path.join(pretrained_dir, 'cifar10_resnet20.th')
                bad_state_dict = torch.load(weight_path, map_location=map_location)
                correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                                      bad_state_dict['state_dict'].items()}
                self.feature_extractor.load_state_dict(correct_state_dict)
            self.fc = None
        elif classifier_name == 'resnet18':
            if dataset == 'tinyimagenet':
                # code from https://github.com/tjmoon0104/Tiny-ImageNet-Classifier
                model = resnet18()
                # Finetune Final few layers to adjust for tiny imagenet input
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Sequential()
                model.avgpool = nn.AdaptiveAvgPool2d(1)
                model.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
                self.feature_extractor = model
                if pretrained:
                    weight_path = os.path.join(pretrained_dir, 'tinyimagenet_resnet18.th')
                    pretrained_dict = torch.load(weight_path, map_location=map_location)
                    model_ft_dict = {re.sub(r'^1\.', '', k): v for k, v in pretrained_dict.items()}
                    correct_state_dict = {re.sub(r'^.*feature_extractor\.', '', k): v for k, v in
                                          model_ft_dict.items()}
                    self.feature_extractor.load_state_dict(correct_state_dict)
                self.fc = None
            else:
                raise Exception("not supported yet for the other datasets")
        elif classifier_name == "lenet":
            if (dataset != 'mnist') or pretrained:
                raise Exception("No you can't use lenet for other data because of dimension")
            # this is LeNet-5 by the way
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)),
                ('relu1', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2))),
                ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
                ('relu3', nn.ReLU()),
                ('s4', nn.MaxPool2d(kernel_size=(2, 2))),
                ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
                ('relu5', nn.ReLU())
            ]))
            self.fc = nn.Sequential(OrderedDict([
                ('f6', nn.Linear(120, 84)),
                ('relu6', nn.ReLU()),
                ('f7', nn.Linear(84, 10)),
                ('sig7', nn.LogSoftmax(dim=-1))
            ]))
        # note: these are uppercase letters. not trained from scratch cases
        elif ('CIFAR10' in classifier_name) or ('MNIST' in classifier_name) or ('TinyImagenet' in classifier_name):
            self.feature_extractor = helper.load_classifier(classifier_name)
            self.fc = None
            self.mean = None
            self.std = None
        else:
            raise Exception("Undefined!")

    def forward(self, img):
        if (self.dataset != "mnist") and (img.shape[1] == 1):
            img = img.repeat(1, 3, 1, 1)

        if (self.mean is not None) and (self.std is not None):
            mean_var = Variable(img.data.new(self.mean).view(1, 3, 1, 1))
            std_var = Variable(img.data.new(self.std).view(1, 3, 1, 1))
            img = (img - mean_var) / std_var

        out = self.feature_extractor(img)
        if self.fc is not None:
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out
