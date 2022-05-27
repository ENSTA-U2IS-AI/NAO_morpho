import torch
import torch.nn as nn
from ops.operations import OPERATIONS_with_mor,OPERATIONS_without_mor
import model.resnet as ResNet
import model.network as network
import torch.nn.functional as F

class Node(nn.Module):
    def __init__(self,search_space,x_id,x_op,y_id,y_op,channels,stride):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.x_id = x_id
        self.x_op_id = x_op
        self.y_id = y_id
        self.y_op_id = y_op

        if search_space == 'with_mor_ops':
            OPERATIONS = OPERATIONS_with_mor
        elif search_space == 'without_mor_ops':
            OPERATIONS = OPERATIONS_without_mor

        x_stride = stride
        self.x_op = OPERATIONS[x_op](channels, channels, x_stride, affine=True)

        y_stride = stride
        self.y_op = OPERATIONS[y_op](channels, channels, y_stride, affine=True)

    def forward(self, x, y):
        x = self.x_op(x)
        y = self.y_op(y)
        return x+y

class Cell(nn.Module):
    def __init__(self, search_space, arch, channels,
                 drop_path_keep_prob=None):
        super(Cell, self).__init__()
        self.search_space = search_space
        self.arch = arch
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = len(arch) // 4
        self.used = [0] * (self.nodes + 2)

        stride = 1
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            node = Node(self.search_space, x_id, x_op, y_id, y_op, channels, stride)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1

        self.concat = [i for i in range(self.nodes + 2) if self.used[i] == 0]

        self.final_combine = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(channels*len(self.concat), channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
        )

    def forward(self, s0, s1):

        states = [s0, s1]
        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y)
            states.append(out)

        out = torch.cat([states[i] for _,i in enumerate(self.concat)], dim=1)
        return self.final_combine(out)

class NAODecoder(nn.Module):
    def __init__(self, args, c_in, c_out, arch):
        super(NAODecoder, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.c_in = c_in
        self.c_out = c_out

        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list):
            arch = arch[0]
        self.cell_arch = arch

        self.cells = nn.ModuleList()
        self.s0 = self._stem0(self.c_in,self.c_out)
        self.s1 = self._stem1(self.c_out)

        cell = Cell(self.search_space, self.cell_arch, self.c_out)
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

    def forward(self, x, size):

        C_S0 = self.s0(x)
        C_s1 = self.s1(C_S0)
        decorder_out = self.cells[0](C_S0, C_s1)

        return decorder_out

class NAO_deeplabv3plus_size(nn.Module):
    def __init__(self, args, classes, arch,res='deeplabv3plus_resnet50'):
        super(NAO_deeplabv3plus_size, self).__init__()
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
        self.decoder.append(NAODecoder(args, c_in=256, c_out=256, arch=arch))
        self.score = nn.Conv2d(256, self.classes, 1)

class NAO_deeplabv3plus(nn.Module):
    def __init__(self, args, classes, arch, pretrained=True, res='deeplabv3plus_resnet50'):
        super(NAO_deeplabv3plus, self).__init__()
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
        self.decoder.append(NAODecoder(args,c_in=256,c_out=256,arch=arch))

        # score
        self.score = nn.Conv2d(256, self.classes, 1)

    def forward(self, x, size):

        # architecture for NAO + deeplabv3plus
        c_outs = self.NAO_deeplabv3plus(x)
        results = self.decoder[0](c_outs,size)
        results = self.score(results)
        results = F.interpolate(results, size=x.size()[2:4], mode='bilinear', align_corners=False)
        return results

