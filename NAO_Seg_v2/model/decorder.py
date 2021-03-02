import torch
import torch.nn as nn
from ops.operations import OPERATIONS_with_mor,OPERATIONS_without_mor
import model.resnet as ResNet

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

class NAOMSCBC_decoder_size(nn.Module):
    def __init__(self, args, classes, arch, channels,res='101'):
        super(NAOMSCBC_decoder_size, self).__init__()
        self.classes = classes
        self.decoder = nn.ModuleList()
        if res == '101':
            self.C_down_channel = [64, 256, 512, 1024, 2048]
        elif res == '18':
            self.C_down_channel = [64, 64, 128, 256, 512]
        elif res == '34':
            self.C_down_channel = [64, 64, 128, 256, 512]
        elif res == '50':
            self.C_down_channel = [64, 256, 512, 1024, 2048]
        else:
            self.C_down_channel = [64, 256, 512, 1024, 2048]
        # decoder
        for i, cin in enumerate(self.C_down_channel):
            self.decoder.append(NAODecoder(args=args, c_in=cin, c_out=channels, arch=arch))


class NAOMSCBC(nn.Module):
    def __init__(self, args, classes, arch, channels,pretrained=True,res='101'):
        super(NAOMSCBC, self).__init__()
        self.classes = classes
        self.decoder = nn.ModuleList()

        #resnet basic network
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
            self.decoder.append(NAODecoder(args=args, c_in=cin, c_out=channels, arch=arch))

        self.score_dsn1 = nn.Conv2d(channels, 1, self.classes)
        self.score_dsn2 = nn.Conv2d(channels, 1, self.classes)
        self.score_dsn3 = nn.Conv2d(channels, 1, self.classes)
        self.score_dsn4 = nn.Conv2d(channels, 1, self.classes)
        self.score_dsn5 = nn.Conv2d(channels, 1, self.classes)
        self.score_final = nn.Conv2d(5*self.classes, self.classes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, size):

        c_outs = self.resnet_(x)
        R1 = self.relu(self.decoder[0](c_outs[0],size))
        R2 = self.relu(self.decoder[1](c_outs[1],size))
        R3 = self.relu(self.decoder[2](c_outs[2],size))
        R4 = self.relu(self.decoder[3](c_outs[3],size))
        R5 = self.relu(self.decoder[4](c_outs[4],size))

        so1_out = self.score_dsn1(R1)
        so2_out = self.score_dsn2(R2)
        so3_out = self.score_dsn3(R3)
        so4_out = self.score_dsn4(R4)
        so5_out = self.score_dsn4(R5)

        upsample = nn.UpsamplingBilinear2d(size)

        out1 = upsample(so1_out)
        out2 = upsample(so2_out)
        out3 = upsample(so3_out)
        out4 = upsample(so4_out)
        out5 = upsample(so5_out)

        fuse = torch.cat([out1, out2, out3, out4, out5], dim=1)
        final_out = self.score_final(fuse)

        results = [out1, out2, out3, out4, out5, final_out]
        results = [torch.sigmoid(r) for r in results]
        return results

