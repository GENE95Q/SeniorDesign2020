"""
create_modules() - take parsed modules and return DNN model with all layers
parse_config() - parse config file and return a list of layer blocks
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np


def parse_model_cfg(cfg_path):
    """
    parse cofig file
    :param cfg_path:
    :return:
    """

    with open(cfg_path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if not x.startswith('#')]  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    model_defs = []  # model definitions

    for line in lines:
        if line.startswith('['):  # new block
            model_defs.append({})
            model_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            model_defs[-1][key.rstrip()] = value.rstrip()

    return model_defs


def create_modules(mddefs):
    """
    Constructs module list from layer blocks
    """
    net_info = mddefs.pop(0)  # [net] block
    module_list = nn.Modulelist()
    output_filters = []
    prev_filters = 3
    yolo_index = -1
    routs = []

    for i, m in enumerate(mddefs):
        modules = nn.Sequential()

        # Convolutional layer
        if m['type'] == 'convolutional':
            bn = int(m['batch_normalize'])
            filters = int(m['filters'])
            k = int(m['size'])
            stride = int(m['stride'])

            modules.add_module("Conv2d_{0}".format(i), nn.Conv2d(in_channels=prev_filters,
                                                                 out_channels=filters,
                                                                 kernel_size=k,
                                                                 stride=stride,
                                                                 padding=k // 2 if m['pad'] else 0,
                                                                 bias=not bn))
            # Batch normalization
            if bn:
                modules.add_module("BatchNorm2d_{0}".format(i), nn.BatchNorm2d(filters))
            else:
                routs.append(i)

            # Activation
            if m['activation'] == 'leaky':
                modules.add_module('Leaky', nn.LeakyReLU(0.1, inplace=True))

        # Upsample
        if m['type'] == 'upsample':
            stride = int(m['stride'])
            modules.add_module("Upsample_{0}".format(i), nn.Upsample(scale_factor=2, mode="nearest"))

        # Route
        if m['type'] == 'route':
            m['layers'] = m['layers'].split(',')
            # Start  of a route
            start = int(m["layers"][0])
            # end, if there exists one.
            try:
                end = int(m["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i
            modules.add_module("Route_{0}".format(i), EmptyLayer())
            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i + start]

        # Shortcut
        if m['type'] == 'shortcut':
            modules.add_module("Shortcut_{0}".format(i), EmptyLayer())

        # Yolo Detection
        if m['type'] == 'yolo':
            yolo_index += 1
            mask = m["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = m["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            modules.add_module("Yolo_{0}".format(i), YoloLayer(anchors))

        # Append current module block
        module_list.append(modules)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class EmptyLayer(nn.Module):
    """
    Does nothing
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YoloLayer(nn.Module):
    """
    YOLO Detection layer
    strides used in YOLOv3 : [32, 16, 8]
    """
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x

        batch_size = prediction.size(0)
        stride = inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(self.anchors)

        prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

        # Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

        # Add the center offsets
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        # if CUDA:
        #     x_offset = x_offset.cuda()
        #     y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        # log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        # if CUDA:
        #     anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

        prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

        prediction[:, :, :4] *= stride

        # prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)

        return prediction
