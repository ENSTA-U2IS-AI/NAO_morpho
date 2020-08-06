import torch.nn as nn
from ops.operations import OPERATIONS_small, OPERATIONS_middle, ConvNet, MaybeCalibrateSize, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path, FinalCombine
import torch

# customise the cell for segmentation
class NodeSegmentation(nn.Module):
    def __init__(self,search_space,x_id,x_op,y_id,y_op,channels,stride,drop_path_keep_prob=None,transpose=False):
        super(NodeSegmentation, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.x_id = x_id
        self.x_op_id = x_op
        self.y_id = y_id
        self.y_op_id = y_op
        self.transpose = transpose

        if search_space == 'small':
            OPERATIONS = OPERATIONS_small
        elif search_space == 'middle':
            OPERATIONS = OPERATIONS_middle
        else:
            OPERATIONS = OPERATIONS_small
        if self.transpose==False:
          x_stride = stride if x_id in [0, 1] else 1
          self.x_op = OPERATIONS[x_op](channels, channels, x_stride, True)

          y_stride = stride if y_id in [0, 1] else 1
          self.y_op = OPERATIONS[y_op](channels, channels, y_stride, True)
        else:
          x_stride = stride if x_id==1 else 1
          self.x_op = OPERATIONS[x_op](channels, channels, x_stride, True)

          y_stride = stride if y_id==1 else 1
          self.y_op = OPERATIONS[y_op](channels, channels, y_stride, True)

    def forward(self, x, y):
        # this mean that only the inputs to the intermediate nodes exists the down sampling ops
        input_to_intermediate_node = []
        x = self.x_op(x)
        y = self.y_op(y)
        input_to_intermediate_node+=[x]
        input_to_intermediate_node+=[y]
        out = sum(consistent_dim(input_to_intermediate_node))
        return out
        
from torch.nn.functional import interpolate
def consistent_dim(states):
    # handle the un-consistent dimension
    # Todo: zbabby
    # concatenate all meta-node to output along channels dimension
    h_max, w_max = 0, 0
    for ss in states:
        if h_max < ss.size()[2]:
            h_max = ss.size()[2]
        if w_max < ss.size()[3]:
            w_max = ss.size()[3]
    return [interpolate(ss, (h_max, w_max)) for ss in states]

# customise the cell for segmentation
class CellSegmentation(nn.Module):
    def __init__(self, search_space, arch, ch_prev_2, ch_prev, channels, dropout_prob=0, type='down'):
        super(CellSegmentation, self).__init__()
        self.search_space = search_space
        self.dropout_prob = dropout_prob
        self.ops = nn.ModuleList()
        self.nodes = len(arch)//4
        self.type = type
        self.concatenate_nodes = self.nodes
        self.arch = arch
        self._multiplier = self.nodes

        if self.type == 'down':
            self.preprocess0 = nn.Sequential(
            nn.Conv2d(ch_prev_2, channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(channels)
        )
        else:
            self.preprocess0 = nn.Sequential(
            nn.Conv2d(ch_prev_2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(ch_prev, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        stride = 2
        for i in range(self.nodes):
          x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
          if self.type =='down':
            node = NodeSegmentation(search_space,  x_id, x_op, y_id, y_op, channels, stride,
                                  dropout_prob )
          else:
            node = NodeSegmentation(search_space,  x_id, x_op, y_id, y_op, channels, stride,
                                              dropout_prob,transpose=True )
          self.ops.append(node)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        # this mean that every intermediate node if and only if two inputs
        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x,y)
            states.append(out)

        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        return out



class NASUNetBSD(nn.Module):
    """construct the Ulike-net according to these searched cells"""

    def __init__(self, args, nclass, in_channels=3, backbone=None, aux=False,
                 c=48, depth=5, keep_prob=0.9, nodes=5, arch=None,
                 double_down_channel=False, use_aux_head=False,use_softmax_head=False):
        super(NASUNetBSD, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.nodes = nodes 
        self.double_down_channel=double_down_channel
        self.multiplier = nodes
        self.use_aux_head = use_aux_head
        self.use_softmax_head = use_softmax_head

        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]
        self.DownCell_arch = arch[:4 * self.nodes] #every cell contain 4 nodes
        # print(self.DownCell_arch)
        self.UpCell_arch = arch[4 * self.nodes:] #every cell contain 4 nodes
        # print(self.UpCell_arch)

        ch_prev_2, ch_prev, ch_curr = self.multiplier * c, self.multiplier * c, c

        # s0 = 1×1 convolution
        self._stem0 = nn.Sequential(
            nn.Conv2d(in_channels, ch_prev_2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_prev_2)
        )
        # s1 = 3×3 convolution
        self._stem1 = nn.Sequential(
            nn.Conv2d(in_channels, ch_prev, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_prev)
        )

        self.cells_down = nn.ModuleList()
        self.cells_up = nn.ModuleList()

        path_recorder = []
        path_recorder += [ch_prev]
        path_recorder += [ch_prev_2]

        # this is the left part of U-Net (encoder) down sampling
        for i in range(depth):
            ch_curr = 2*ch_curr if self.double_down_channel else ch_curr
            cell_down = CellSegmentation(self.search_space,self.DownCell_arch,ch_prev_2,ch_prev,ch_curr,1-keep_prob,type='down')
            self.cells_down +=[cell_down]
            ch_prev_2,ch_prev = ch_prev,cell_down._multiplier*ch_curr
            path_recorder +=[ch_prev]

        # this is the right part of U-Net (decoder) up sampling
        for i in range(depth+1):
            ch_prev_2 = path_recorder[-(i+2)]
            cell_up = CellSegmentation(self.search_space,self.UpCell_arch,ch_prev_2,ch_prev,ch_curr,1-keep_prob,type='up')
            self.cells_up += [cell_up]
            ch_prev = cell_up._multiplier*ch_curr
            ch_curr = ch_curr//2 if self.double_down_channel else ch_curr

        self.ConvSegmentation = ConvNet(ch_prev, nclass, kernel_size=1, dropout_rate=0.1)

        # if use_aux_head:
        #     self.aux_output_layer = FCNHead(ch_prev, nclass, nn.BatchNorm2d)
        if self.use_softmax_head:
          self.softmax = nn.Softmax(dim=1)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input):
        """bchw for tensor"""
        _,_,h,w = input.size()
        # s0: remain the original image size
        # s1: halve image size
        s0, s1 = self._stem0(input), self._stem1(input)
        cells_recorder = []

        cells_recorder.append(s0)
        cells_recorder.append(s1)


        #the left part of U-Net
        for i, cell in enumerate(self.cells_down):
            s0,s1 = s1,cell(s0,s1)
            cells_recorder.append(s1)

        #the right part of U-Net
        for i, cell in enumerate(self.cells_up):
            s0 = cells_recorder[-(i+2)] # get the chs_prev_prev
            s1 = cell(s0,s1)

        x = self.ConvSegmentation(s1)
        if self.use_softmax_head:
           x = self.softmax(x)
        return x
