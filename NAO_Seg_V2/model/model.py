import torch.nn as nn
from ops.operations import OPERATIONS_with_mor,OPERATIONS_without_mor, ConvNet, MaybeCalibrateSize, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path, FinalCombine, Aux_dropout, OPERATIONS_without_mor_ops
import torch
import torch.nn.functional as F

# customise the cell for segmentation
class Node(nn.Module):
    def __init__(self, search_space, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride=1,
                 drop_path_keep_prob=None,
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

        if search_space == 'small_with_mor':
            OPERATIONS = OPERATIONS_with_mor
        elif search_space == 'small_without_mor':
            OPERATIONS = OPERATIONS_without_mor
        # elif search_space == 'middle':
        #     OPERATIONS = OPERATIONS_middle
        # elif search_space == 'large':
        #     OPERATIONS = OPERATIONS_large
        # else:
        #     OPERATIONS = OPERATIONS_small

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

# customise the  dowm cell for segmentation
class Cell_down(nn.Module):
    def __init__(self, search_space, arch, prev_layers, channels, layer_id, layers, steps,
                 drop_path_keep_prob=None):
        super(Cell_down, self).__init__()
        # print(prev_layers)
        assert len(prev_layers) == 2
        self.search_space = search_space
        self.arch = arch
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

        stride = 2
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            x_shape, y_shape = prev_layers[x_id], prev_layers[y_id]
            node = Node(self.search_space, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride,
                        drop_path_keep_prob, layer_id, layers, steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            prev_layers.append(node.out_shape)

        self.concat = [i for i in range(self.nodes + 2) if self.used[i] == 0]
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers) if i in self.concat])
        self.final_combine = FinalCombine(prev_layers, out_hw, channels, self.concat)
        self.out_shape = [out_hw, out_hw, channels * len(self.concat)]

    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        states = [s0, s1]
        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y, step)
            states.append(out)
        return self.final_combine(states)

# customise the  up cell for segmentation
class Cell_up(nn.Module):
    def __init__(self, search_space, arch, prev_layers, channels, layer_id, layers, steps,
                 drop_path_keep_prob=None):
        super(Cell_up, self).__init__()
        # print(prev_layers)
        assert len(prev_layers) == 2
        self.search_space = search_space
        self.arch = arch
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

        stride = 1
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            x_shape, y_shape = prev_layers[x_id], prev_layers[y_id]
            node = Node(self.search_space, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride,
                        drop_path_keep_prob, layer_id, layers, steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            prev_layers.append(node.out_shape)

        self.concat = [i for i in range(self.nodes + 2) if self.used[i] == 0]
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers) if i in self.concat])
        self.final_combine = FinalCombine(prev_layers, out_hw, channels, self.concat)
        self.out_shape = [out_hw*2, out_hw*2, channels * len(self.concat)]

    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        states = [s0, s1]
        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y, step)
            states.append(out)
        out = self.final_combine(states)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class NASUNetBSD(nn.Module):
    """construct the Ulike-net according to these searched cells"""

    def __init__(self, args, keep_prob, drop_path_keep_prob,  steps,
                 nclass=1, in_channels=3, aux=False,
                 channels=8, depth=4, dropout_rate=0., nodes=5, arch=None,
                 use_aux_head=False,use_softmax_head=False):
        super(NASUNetBSD, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.nodes = nodes
        self.multiplier = nodes
        self.use_aux_head = use_aux_head
        self.use_softmax_head = use_softmax_head
        self.classes = nclass
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.channels = channels

        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]

        self.Cell_down_arch = arch[:4 * self.nodes]
        self.Cell_up_arch = arch[4 * self.nodes:]
        channels = self.nodes * channels

        self.stem0 = ConvNet(in_channels, channels, kernel_size=1, op_type='pre_ops')

        self.cells = nn.ModuleList()
        outs = [[416, 416, channels], [416, 416, channels]]
        channels = self.channels

        # this is the left part of U-Net (encoder) down sampling -- learn the down cell
        for _, i in enumerate(self.down_layer):
            channels *= 2
            cell = Cell_down(self.search_space, self.Cell_down_arch, outs, channels, i, self.layers+2, self.steps,
                             self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
        # this is the right part of U-Net (decoder) up sampling -- learn the down cell
        for _, i in enumerate(self.up_layer):
            channels /= 2
            cell = Cell_up(self.search_space, self.Cell_up_arch, outs, channels, i, self.layers+2, self.steps,
                           self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

        # self.ConvSegmentation = ConvNet(ch_prev, nclass, kernel_size=1, dropout_rate=0.1, op_type='SC')

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

    def forward(self, input, step=None):
        """bchw for tensor"""
        aux_logits = None
        s0=s1 = self.stem0(input)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)

        if self.use_aux_head:
            x = self.ConvSegmentation(s1)
            # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        else:
            x = self.ConvSegmentation(s1)
            # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_softmax_head:
            x = self.softmax(x)

        logits = x
        return logits


if __name__ == '__main__':
    batch_size = 8
    img_height = 416
    img_width = 416

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = NASUNetBSD().to(device)
    output = model(input)
    print(output.size())
    # print(f"output shapes: {[t.shape for t in output]}")

    # for i in range(20000):
    #     print(i)
    #     output = model(input)
    #     loss = nn.MSELoss()(output[-1], target)
    #     loss.backward()