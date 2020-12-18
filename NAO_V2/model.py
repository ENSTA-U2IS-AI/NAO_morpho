import os
import sys
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPERATIONS_small, OPERATIONS_middle, OPERATIONS_large, OPERATIONS_small_Sep, ReLUConvBN, MaybeCalibrateSize, FactorizedReduce, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path, FinalCombine,OPERATIONS_small_without_mor
from torch import Tensor
import utils
    

class Node(nn.Module):
    def __init__(self, search_space, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride=1, drop_path_keep_prob=None,
                 layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.x_id = x_id
        self.x_op_id = x_op
        self.y_id = y_id
        self.y_op_id = y_op
        x_shape = list(x_shape)
        y_shape = list(y_shape)
        
        if search_space == 'small':
            OPERATIONS = OPERATIONS_small
        elif search_space == 'small_with_Sep':
            OPERATIONS = OPERATIONS_small_Sep
        elif search_space == 'small_without_mor':
            OPERATIONS = OPERATIONS_small_without_mor
        elif search_space == 'middle':
            OPERATIONS = OPERATIONS_middle
        elif search_space == 'large':
            OPERATIONS = OPERATIONS_large
        else:
            OPERATIONS = OPERATIONS_small

        x_stride = stride if x_id in [0, 1] else 1
        self.x_op = OPERATIONS[x_op](channels, channels, x_stride, x_shape, True)
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]

        y_stride = stride if y_id in [0, 1] else 1
        self.y_op = OPERATIONS[y_op](channels, channels, y_stride, y_shape, True)
        y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, channels]
        
        assert x_shape[0] == y_shape[0] and x_shape[1] == y_shape[1]
        self.out_shape = list(x_shape)
        
    def forward(self, x, y, step):
        x = self.x_op(x)
        y = self.y_op(y)
        X_DROP = False
        Y_DROP = False
        if self.search_space == 'small':
            if self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        elif self.search_space == 'middle':
            if self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        elif self.search_space == 'large':
            if self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True        

        if X_DROP:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        if Y_DROP:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        out = x + y
        return out
    

class Cell(nn.Module):
    def __init__(self, search_space, arch, prev_layers, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        #print(prev_layers)
        assert len(prev_layers) == 2
        self.search_space = search_space
        self.arch = arch
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = len(arch) // 4
        self.used = [0] * (self.nodes + 2)
        
        # maybe calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape

        stride = 2 if self.reduction else 1
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            x_shape, y_shape = prev_layers[x_id], prev_layers[y_id]
            node = Node(self.search_space, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride, drop_path_keep_prob, layer_id, layers, steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            prev_layers.append(node.out_shape)
        
        self.concat = [i for i in range(self.nodes+2) if self.used[i] == 0]
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers) if i in self.concat])
        self.final_combine = FinalCombine(prev_layers, out_hw, channels, self.concat)
        self.out_shape = [out_hw, out_hw, channels * len(self.concat)]
    
    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        states = [s0, s1]
        for i in range(self.nodes):
            x_id = self.arch[4*i]
            y_id = self.arch[4*i+2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y, step)
            states.append(out)
        return self.final_combine(states)
        

class NASNetworkCIFAR(nn.Module):
    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkCIFAR, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels],[32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, self.conv_arch, outs, channels, False, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, self.reduc_arch, outs, channels, True, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class NASNetworkImageNet(nn.Module):
    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkImageNet, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        arch = list(map(int, arch.strip().split()))
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]
        
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers + 2):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, self.conv_arch, outs, channels, False, i, self.layers + 2, self.steps,
                            self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, self.reduc_arch, outs, channels, True, i, self.layers + 2, self.steps,
                            self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
