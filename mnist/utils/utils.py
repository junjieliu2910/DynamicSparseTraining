import os
import json
import logging

import numpy as np

import torch
import torch.nn as nn
from model import MaskedMLP, MaskedConv2d


def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    assert len(ids.shape) == 1, 'the ids should be 1-D'

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def get_tensor_sparsity(inputs):
    flatten = inputs.view(-1)
    absolute = torch.abs(flatten)
    total = absolute.size()[0]
    mask = absolute > 0.
    non_zero = torch.sum(mask)
    return 1. - int(non_zero)/total



def print_layer_keep_ratio(model, logger):
    total = 0. 
    keep = 0.
    for layer in model.modules():
        if isinstance(layer, MaskedMLP):
            abs_weight = torch.abs(layer.weight)
            threshold = layer.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = layer.step(abs_weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
        if isinstance(layer, MaskedConv2d):
            weight_shape = layer.weight.shape 
            threshold = layer.threshold.view(weight_shape[0], -1)
            weight = torch.abs(layer.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = layer.step(weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            #print("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
    logger.info("Model keep ratio {:.4f}".format(keep/total))
    return keep / total


