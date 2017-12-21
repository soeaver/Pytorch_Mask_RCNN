import datetime
import glob
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from config import Config
from network.mask_rcnn import MaskRCNN


def get_state_dict(weight_file):

    weight_dict = OrderedDict()

    f = h5py.File(weight_file, mode='r')
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    fc_layers = ['mrcnn_class_logits', 'mrcnn_bbox_fc']
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        if len(weight_names) == 0:
            continue

        if 'kernel' in weight_names[0]:
            name = weight_names[0].split('/')[0]

            if 'rpn' in name:
                name1 = weight_names[0].split('/')[0]
                data = np.transpose(
                    weight_values[0], (3, 2, 0, 1)).astype(np.float32)
                weight_dict[name1 + '.weight'] = data
                weight_dict[name1 +
                            '.bias'] = np.array(weight_values[1]).astype(np.float32)

                name2 = weight_names[2].split('/')[0]
                data = np.transpose(
                    weight_values[2], (3, 2, 0, 1)).astype(np.float32)
                weight_dict[name2 + '.weight'] = data
                weight_dict[name2 +
                            '.bias'] = np.array(weight_values[3]).astype(np.float32)

                name3 = weight_names[4].split('/')[0]
                data = np.transpose(
                    weight_values[4], (3, 2, 0, 1)).astype(np.float32)
                weight_dict[name3 + '.weight'] = data

                weight_dict[name3 +
                            '.bias'] = np.array(weight_values[5]).astype(np.float32)

            elif name not in fc_layers:
                weight_dict[name + '.weight'] = np.transpose(
                    weight_values[0], (3, 2, 0, 1)).astype(np.float32)
                weight_dict[name +
                            '.bias'] = np.array(weight_values[1]).astype(np.float32)

            else:

                weight_dict[name + '.weight'] = np.transpose(
                    weight_values[0], (1, 0)).astype(np.float32)
                weight_dict[name +
                            '.bias'] = np.array(weight_values[1]).astype(np.float32)
        else:
            name = weight_names[0].split('/')[0]

            weight_key = name + '.weight'
            bias_key = name + '.bias'
            running_mean_key = name + '.running_mean'
            running_var_key = name + '.running_var'
            rescale_factor = float(
                weight_values[0].shape[0]) / (float(weight_values[0].shape[0]) - 1.)
            rescale_factor = np.sqrt(rescale_factor)

            gamma = np.array(weight_values[0]).astype(np.float32)
            beta = np.array(weight_values[1]).astype(np.float32)
            mean = np.array(weight_values[2]).astype(np.float32)
            var = np.array(weight_values[3]).astype(np.float32)

            weight_dict[weight_key] = gamma
            weight_dict[bias_key] = beta
            weight_dict[running_mean_key] = mean
            weight_dict[running_var_key] = var

    for name, param in weight_dict.items():
        weight_dict[name] = torch.from_numpy(param).float()

    return weight_dict

def make_name_right(state_dict, model):
    pytorch_name1 = ['resnet_graph.layer1.0.bn3.weight',
     'resnet_graph.layer1.0.bn3.bias',
     'resnet_graph.layer1.0.bn3.running_mean',
     'resnet_graph.layer1.0.bn3.running_var',
     'resnet_graph.layer1.0.downsample.0.weight',
     'resnet_graph.layer1.0.downsample.0.bias',
     'resnet_graph.layer2.0.bn3.weight',
     'resnet_graph.layer2.0.bn3.bias',
     'resnet_graph.layer2.0.bn3.running_mean',
     'resnet_graph.layer2.0.bn3.running_var',
     'resnet_graph.layer2.0.downsample.0.weight',
     'resnet_graph.layer2.0.downsample.0.bias',
     'resnet_graph.layer3.0.bn3.weight',
     'resnet_graph.layer3.0.bn3.bias',
     'resnet_graph.layer3.0.bn3.running_mean',
     'resnet_graph.layer3.0.bn3.running_var',
     'resnet_graph.layer3.0.downsample.0.weight',
     'resnet_graph.layer3.0.downsample.0.bias',
     'resnet_graph.layer4.0.bn3.weight',
     'resnet_graph.layer4.0.bn3.bias',
     'resnet_graph.layer4.0.bn3.running_mean',
     'resnet_graph.layer4.0.bn3.running_var',
     'resnet_graph.layer4.0.downsample.0.weight',
     'resnet_graph.layer4.0.downsample.0.bias']

    keras_name1 = [
        u'bn2a_branch2c.weight',
        u'bn2a_branch2c.bias',
        u'bn2a_branch2c.running_mean',
        u'bn2a_branch2c.running_var',
        u'res2a_branch1.weight',
        u'res2a_branch1.bias',
        u'bn3a_branch2c.weight',
        u'bn3a_branch2c.bias',
        u'bn3a_branch2c.running_mean',
        u'bn3a_branch2c.running_var',
        u'res3a_branch1.weight',
        u'res3a_branch1.bias',
        u'bn4a_branch2c.weight',
        u'bn4a_branch2c.bias',
        u'bn4a_branch2c.running_mean',
        u'bn4a_branch2c.running_var',
        u'res4a_branch1.weight',
        u'res4a_branch1.bias',
        u'bn5a_branch2c.weight',
        u'bn5a_branch2c.bias',
        u'bn5a_branch2c.running_mean',
        u'bn5a_branch2c.running_var',
        u'res5a_branch1.weight',
        u'res5a_branch1.bias'
    ]

    match_dict1 = dict(zip(pytorch_name1, keras_name1))

    keras_keys = list(state_dict.keys())

    pytorch_keys = list(model.state_dict().keys())

    match_dict = dict()

    for i in range(624):
        match_dict[pytorch_keys[i]] = keras_keys[i]

    match_dict.update(match_dict1)

    for i in range(624, 640):
        match_dict[pytorch_keys[i]] = pytorch_keys[i]

    for i in range(640, len(pytorch_keys)):
        match_dict[pytorch_keys[i]] = pytorch_keys[i].split('.', 1)[1]

    new_state_dict = model.state_dict()

    for j in range(len(model.state_dict().keys())):
        key_in_pytorch = list(model.state_dict().keys())[j]

        key_in_keras = match_dict[key_in_pytorch]
        val_in_keras = state_dict[key_in_keras]

        new_state_dict[key_in_pytorch] = val_in_keras

    return new_state_dict

if __name__ == "__main__":


    torch.backends.cudnn.enabled = False

    class InferenceConfig(Config):

        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Number of classes (including background)
        NUM_CLASSES = 1 + 80  # COCO has 80 classes


    config = InferenceConfig()
    config.display()

    model = MaskRCNN(config=config)
    model.cuda()
    model.eval()
        
    tf_weights= '/home/tensorboy/Downloads/mask_rcnn_coco.h5'
    state_dict = get_state_dict(tf_weights)
        

    new_state_dict = make_name_right(state_dict, model)
    torch.save(new_state_dict, '../mrcnn.pth')
