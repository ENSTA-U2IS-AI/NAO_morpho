import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.operations import OPERATIONS_search_with_mor,OPERATIONS_search_without_mor, MaybeCalibrateSize, WSReLUConvBN, FactorizedReduce, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path,ConvNet, Aux_dropout
from utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# customise the cell for segmentation
class Node(nn.Module):
    def __init__(self, search_space, prev_layers, channels, stride, drop_path_keep_prob=None, node_id=0, layer_id=0,
                 layers=0, steps=0):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.node_id = node_id
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()

        num_possible_inputs = node_id + 2

        if search_space == 'small_with_mor':
            OPERATIONS = OPERATIONS_search_with_mor
        elif search_space == 'small_without_mor':
            OPERATIONS = OPERATIONS_search_without_mor
        # elif search_space == 'middle':
        #     OPERATIONS = OPERATIONS_search_middle
        # else:
        #     OPERATIONS = OPERATIONS_search_small

        for k, v in OPERATIONS.items():
            self.x_op.append(v(num_possible_inputs, channels, channels, stride, True))
            self.y_op.append(v(num_possible_inputs, channels, channels, stride, True))

        self.out_shape = [prev_layers[0][0] // stride, prev_layers[0][1] // stride, channels]

    def forward(self, x, x_id, x_op, y, y_id, y_op, step, bn_train=False):
        stride = self.stride if x_id in [0, 1] else 1
        x = self.x_op[x_op](x, x_id, stride, bn_train)
        stride = self.stride if y_id in [0, 1] else 1
        y = self.y_op[y_op](y, y_id, stride, bn_train)

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
        if X_DROP:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        if Y_DROP:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)

        return x + y
      

# def consistent_dim(states):
#     """the aim of this fonction is to make sure that the dimensions of state are consistent """
#     h_max, w_max = 0, 0
#     for ss in states:
#         if h_max < ss.size()[2]:
#             h_max = ss.size()[2]
#         if w_max < ss.size()[3]:
#             w_max = ss.size()[3]
#     return [interpolate(ss, (h_max, w_max)) for ss in states]

# customise the  dowm cell for segmentation
class Cell_down(nn.Module):
    def __init__(self, search_space, prev_layers, nodes, channels, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell_down, self).__init__()
        self.search_space = search_space
        assert len(prev_layers) == 2
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes

        #if calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape

        # the prev_layers represents chs_prev_2, chs_prev and the channels represents chs
        stride = 2
        for i in range(self.nodes):
            node = Node(search_space, prev_layers, channels, stride, drop_path_keep_prob, i, layer_id, layers, steps)
            self.ops.append(node)
            prev_layers.append(node.out_shape)

        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])

        self.fac_1 = FactorizedReduce(prev_layers[0][-1], channels, prev_layers[0])
        self.fac_2 = FactorizedReduce(prev_layers[1][-1], channels, prev_layers[1])
        self.final_combine_conv = WSReLUConvBN(self.nodes + 2, channels, channels, 1)

        self.out_shape = [out_hw, out_hw, channels]

    def forward(self, s0, s1, arch, step, bn_train=False):
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]
        used = [0] * (self.nodes + 2)
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, step, bn_train=bn_train)
            states.append(out)
        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)

        # Notice that in reduction cell, 0, 1 might be concated and they might have to be factorized
        if 0 in concat:
            states[0] = self.fac_1(states[0])
        if 1 in concat:
            states[1] = self.fac_2(states[1])
        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat, bn_train=bn_train)
        print(out.size())
        return out


# customise the  dowm cell for segmentation
class Cell_up(nn.Module):
    def __init__(self, search_space, prev_layers, nodes, channels, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell_up, self).__init__()
        self.search_space = search_space
        assert len(prev_layers) == 2
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes

        # calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape


        # the prev_layers represents chs_prev_2, chs_prev and the channels represents chs
        stride = 1
        for i in range(self.nodes):
            node = Node(search_space, prev_layers, channels, stride, drop_path_keep_prob, i, layer_id, layers, steps)
            self.ops.append(node)
            prev_layers.append(node.out_shape)

        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])

        self.final_combine_conv = WSReLUConvBN(self.nodes + 2, channels, channels, 1)

        self.out_shape = [out_hw*2, out_hw*2, channels]

    def forward(self, s0, s1, arch, step, bn_train=False):
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]
        used = [0] * (self.nodes + 2)
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, step, bn_train=bn_train)
            states.append(out)
        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)

        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat, bn_train=bn_train)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        print(out.size())
        return out


class NASUNetSegmentationWS(nn.Module):
    def __init__(self, args, drop_path_keep_prob, steps, depth=4, classes=2, nodes=5, input_chs=3, channels=16, keep_prob=0.9, use_softmax_head=False,use_aux_head=False):
        super(NASUNetSegmentationWS, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.classes = classes
        self.nodes = nodes
        self.keep_prob = keep_prob
        self.use_softmax_head=use_softmax_head
        self.multiplier = nodes
        self.use_aux_head = use_aux_head
        self.drop_path_keep_prob = drop_path_keep_prob
        self.steps = steps
        self.channels = channels
        self.total_layers = self.depth * 2

        self.down_layer = [i for i in range(self.depth)]
        self.up_layer = [i for i in range(self.depth, self.depth * 2)]
        channels = self.nodes*channels

        self.stem0 = ConvNet(input_chs, channels, kernel_size=1, op_type='pre_ops')
        # self.stem1 = ConvNet(input_chs, channels*2, kernel_size=3, stride=2, op_type='pre_ops')

        # the size of img
        self.cells = nn.ModuleList()
        outs = [[416, 416, channels], [416, 416, channels]]
        channels = self.channels
        # this is the left part of U-Net (encoder) down sampling -- learn the down cell
        for _,i in enumerate(self.down_layer):
            channels *=2
            cell = Cell_down(self.search_space, outs, self.nodes, channels, i, self.total_layers, self.steps,
                             self.drop_path_keep_prob)
            self.cells.append(cell)
            outs =[outs[-1],cell.out_shape]
        # this is the right part of U-Net (decoder) up sampling -- learn the down cell
        for _,i in enumerate(self.up_layer):
            channels = channels//2
            cell = Cell_up(self.search_space, outs, self.nodes, channels, i, self.total_layers, self.steps,
                           self.drop_path_keep_prob)
            self.cells.append(cell)
            outs =[outs[-1],cell.out_shape]


        if use_aux_head:
          self.ConvSegmentation = Aux_dropout(outs[-1][-1], self.classes, nn.BatchNorm2d)
        else:
          self.ConvSegmentation = ConvNet(outs[-1][-1], self.classes, kernel_size=1, dropout_rate=0.1, op_type='SC')

        if use_softmax_head:
            self.softmax = nn.Softmax(dim=1)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, arch, step=None, bn_train=False):

        _,_,h,w = input.size()

        s0= s1 = self.stem0(input)
        print(s0.size())
        cells_recorder = []

        DownCell_arch,UpCell_arch=arch

        for i, cell in enumerate(self.cells):
            if i in self.down_layer:
                s0,s1=s1,cell(s0,s1,DownCell_arch,step,bn_train=bn_train)
            elif i in self.up_layer:
                s0,s1=s1,cell(s0,s1,UpCell_arch,step,bn_train=bn_train)

        # exit()
        if self.use_aux_head:
            x = self.ConvSegmentation(s1)
            x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        else:
            x = self.ConvSegmentation(s1)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_softmax_head:
            x = self.softmax(x)

        logits=x
        return logits