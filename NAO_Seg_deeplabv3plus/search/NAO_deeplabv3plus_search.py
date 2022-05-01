import torch
import torch.nn as nn
from ops.operations import WSReLUConvBN, OPERATIONS_search_with_mor,OPERATIONS_search_without_mor
import model.resnet as ResNet
import model.network as network
import torch.nn.functional as F

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

        #NAONet search process
        self.cells = nn.ModuleList()
        self.s0 = self._stem0(self.c_in,self.c_out)
        self.s1 = self._stem1(self.c_out)

        cell = Cell(self.search_space, self.nodes, self.c_out)
        self.cells.append(cell)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def _stem0(self, c_in,c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_out // 2, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def _stem1(self,c_out):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x, size, arch, bn_train=False):

        C_S0 = self.s0(x)
        C_s1 = self.s1(C_S0)
        decorder_out = self.cells[0](C_S0, C_s1, arch, bn_train=bn_train)

        return decorder_out

class NAO_deeplabv3plus_search(nn.Module):
    def __init__(self, args, classes, nodes,pretrained=False,res='deeplabv3plus_resnet50'):
        super(NAO_deeplabv3plus_search, self).__init__()
        self.classes = classes
        self.decoder = nn.ModuleList()

        # resnet deeplab
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }
        self.NAO_deeplabv3plus = model_map[res](num_classes=self.classes,
                                                output_stride=8)

        # decoder for deeplabv3plus
        self.decoder.append(NAODecoderSearch(args=args, c_in=256, c_out=256, nodes=nodes))

        # score
        self.score = nn.Conv2d(256, self.classes, 1)

    def forward(self, x, size, arch, bn_train=False):

        c_outs = self.NAO_deeplabv3plus(x)
        results = self.decoder[0](c_outs,size,arch[0],bn_train=bn_train)
        results = self.score(results)
        results = F.interpolate(results, size=input_shape, mode='bilinear', align_corners=False)
        return results

