import torch
import torch.nn as nn
from ops.operations import WSReLUConvBN, OPERATIONS_search_with_mor, OPERATIONS_search_without_mor
import model.resnet as ResNet
import torch.nn.functional
import numpy as np


class Node(nn.Module):
    def __init__(self, search_space, channels, stride, node_id, drop_path_keep_prob=None):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.node_id = node_id
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()

        num_possible_inputs = node_id + 2

        if search_space == 'with_mor_ops':
            OPERATIONS = OPERATIONS_search_with_mor
        elif search_space == 'without_mor_ops':
            OPERATIONS = OPERATIONS_search_without_mor

        for k, v in OPERATIONS.items():
            self.x_op.append(v(num_possible_inputs, channels, channels, stride, affine=True))
            self.y_op.append(v(num_possible_inputs, channels, channels, stride, affine=True))

    def forward(self, x, x_id, x_op, y, y_id, y_op, bn_train=False):

        stride = 1
        x = self.x_op[x_op](x, x_id, stride, bn_train)
        y = self.y_op[y_op](y, y_id, stride, bn_train)

        return x + y


class Cell(nn.Module):
    def __init__(self, search_space, nodes, channels,
                 drop_path_keep_prob=None):
        super(Cell, self).__init__()
        self.search_space = search_space
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes

        stride = 1
        for i in range(self.nodes):
            node = Node(search_space, channels, stride, i, drop_path_keep_prob)
            self.ops.append(node)

        self.final_combine_conv = WSReLUConvBN(self.nodes + 2, channels, channels, 1)

    def forward(self, s0, s1, arch, bn_train=False):

        states = [s0, s1]
        used = [0] * (self.nodes + 2)

        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, bn_train=bn_train)
            states.append(out)

        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)

        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat, bn_train=bn_train)
        return out


class NAODecoderSearch(nn.Module):
    def __init__(self, args, c_in, c_out, nodes):
        super(NAODecoderSearch, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.c_in = c_in
        self.c_out = c_out
        self.nodes = nodes

        # NAONet search process
        self.cells = nn.ModuleList()
        self.s0 = self._stem0(self.c_in, self.c_out)
        self.s1 = self._stem1(self.c_out)

        cell = Cell(self.search_space, self.nodes, self.c_out)
        self.cells.append(cell)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def _stem0(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=False),
        )

    def _stem1(self, c_out):
        return nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, size, arch, bn_train=False):

        C_S0 = self.s0(x)
        C_s1 = self.s1(C_S0)
        decorder_out = self.cells[0](C_S0, C_s1, arch, bn_train=bn_train)

        return decorder_out


class NAOMSCBC_search(nn.Module):
    def __init__(self, args, classes, nodes, channels, pretrained=False, res='101'):
        super(NAOMSCBC_search, self).__init__()
        self.classes = classes
        self.decoder = nn.ModuleList()

        # resnet
        if res == '101':
            self.C_down_channel = [64, 256, 512, 1024, 2048]
            self.resnet_ = ResNet.resnet101(pretrained=pretrained)
        elif res == '18':
            self.C_down_channel = [64, 64, 128, 256, 512]
            self.resnet_ = ResNet.resnet18(pretrained=pretrained)
        elif res == '34':
            self.C_down_channel = [64, 64, 128, 256, 512]
            self.resnet_ = ResNet.resnet34(pretrained=pretrained)
        elif res == '50':
            self.C_down_channel = [64, 256, 512, 1024, 2048]
            self.resnet_ = ResNet.resnet50(pretrained=pretrained)
        else:
            self.C_down_channel = [64, 256, 512, 1024, 2048]
            self.resnet_ = ResNet.resnet152(pretrained=pretrained)

        # decoder
        for i, cin in enumerate(self.C_down_channel):
            self.decoder.append(NAODecoderSearch(args=args, c_in=cin, c_out=channels, nodes=nodes))

        self.score_dsn1 = nn.Conv2d(channels, 1, 1)
        self.score_dsn2 = nn.Conv2d(channels, 1, 1)
        self.score_dsn3 = nn.Conv2d(channels, 1, 1)
        self.score_dsn4 = nn.Conv2d(channels, 1, 1)
        self.score_dsn5 = nn.Conv2d(channels, 1, 1)
        self.score_final = nn.Conv2d(5, self.classes, 1)
        self.relu = nn.ReLU()

    def forward(self, x, size, arch, bn_train=False):

        c_outs = self.resnet_(x)
        R1 = self.decoder[0](self.relu(c_outs[0]), size, arch[0], bn_train=bn_train)
        R2 = self.decoder[1](self.relu(c_outs[1]), size, arch[0], bn_train=bn_train)
        R3 = self.decoder[2](self.relu(c_outs[2]), size, arch[0], bn_train=bn_train)
        R4 = self.decoder[3](self.relu(c_outs[3]), size, arch[0], bn_train=bn_train)
        R5 = self.decoder[4](self.relu(c_outs[4]), size, arch[0], bn_train=bn_train)

        so1_out = self.score_dsn1(R1)
        so2_out = self.score_dsn2(R2)
        so3_out = self.score_dsn3(R3)
        so4_out = self.score_dsn4(R4)
        so5_out = self.score_dsn4(R5)

        # upsample = nn.UpsamplingBilinear2d(size)
        upsample = nn.Upsample(size)

        out1 = upsample(so1_out)
        out2 = upsample(so2_out)
        out3 = upsample(so3_out)
        out4 = upsample(so4_out)
        out5 = upsample(so5_out)

        # weight_deconv1 = make_bilinear_weights(4, 1).cuda()
        # weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        # weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        # weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        # weight_deconv5 = make_bilinear_weights(32, 1).cuda()
        #
        # # print(so1_out.size())
        # upsample1 = torch.nn.functional.conv_transpose2d(so1_out, weight_deconv1, stride=4)
        # upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=4)
        # upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=8)
        # upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=16)
        # upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=32)
        #
        # out1 = crop(upsample1, size[0], size[1])
        # out2 = crop(upsample2, size[0], size[1])
        # out3 = crop(upsample3, size[0], size[1])
        # out4 = crop(upsample4, size[0], size[1])
        # out5 = crop(upsample5, size[0], size[1])
        # # print(out1.size(),out2.size(),out3.size(),out4.size(),out5.size())

        fuse = torch.cat([out1, out2, out3, out4, out5], dim=1)
        final_out = self.score_final(fuse)
        # print(final_out.size())

        results = [out1, out2, out3, out4, out5, final_out]
        results = [torch.sigmoid(r) for r in results]
        return results


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]