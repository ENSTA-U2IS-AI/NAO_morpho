import torch.nn as nn
from ops.operations import OPERATIONS_with_mor, ConvNet,Aux_dropout, OPERATIONS_without_mor
import torch
import torch.nn.functional as F

# customise the cell for segmentation
class Node(nn.Module):
    def __init__(self,search_space,x_id,x_op,y_id,y_op,channels,stride,transpose=False):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.x_id = x_id
        self.x_op_id = x_op
        self.y_id = y_id
        self.y_op_id = y_op
        self.transpose = transpose

        if search_space == 'with_mor_ops':
            OPERATIONS = OPERATIONS_with_mor
        elif search_space == 'without_mor_ops':
            OPERATIONS = OPERATIONS_without_mor

        if self.transpose==False:
          x_stride = stride if x_id in [0, 1] else 1
          self.x_op = OPERATIONS[x_op](channels, channels, x_stride, affine=True)

          y_stride = stride if y_id in [0, 1] else 1
          self.y_op = OPERATIONS[y_op](channels, channels, y_stride, affine=True)
        else:
          x_stride = stride
          self.x_op = OPERATIONS[x_op](channels, channels, x_stride, affine=True)

          y_stride = stride
          self.y_op = OPERATIONS[y_op](channels, channels, y_stride, affine=True)

    def forward(self, x, y):

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

# customise down cell
class cellDown(nn.Module):
    def __init__(self, search_space, arch, ch_prev_2, ch_prev, channels):
        super(cellDown, self).__init__()
        self.search_space = search_space
        self.ops = nn.ModuleList()
        self.nodes = len(arch)//4
        self.concatenate_nodes = self.nodes
        self.arch = arch
        self._multiplier = self.nodes

        self.preprocess0 = ConvNet(ch_prev_2, channels, kernel_size=1, stride=2, affine=True, op_type='pre_ops_cell')
        self.preprocess1 = ConvNet(ch_prev, channels, kernel_size=1, stride=1, affine=True, op_type='pre_ops_cell')
        
        stride = 2
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            node = Node(search_space,  x_id, x_op, y_id, y_op, channels, stride)
            self.ops.append(node)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x,y)
            states.append(out)

        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        # print(out.size())
        return out


# customise the cell for segmentation
class cellUp(nn.Module):
    def __init__(self, search_space, arch, ch_prev_2, ch_prev, channels):
        super(cellUp, self).__init__()
        self.search_space = search_space
        self.ops = nn.ModuleList()
        self.nodes = len(arch) // 4
        self.concatenate_nodes = self.nodes
        self.arch = arch
        self._multiplier = self.nodes

        self.preprocess0 = ConvNet(ch_prev_2, channels, kernel_size=1, stride=1, affine=True,
                                   op_type='pre_ops_cell')
        # self.preprocess1 = ConvNet(ch_prev, channels, stride=2, transpose=True, affine=True),  # 'up_conv_3×3'
        self.preprocess1 = ConvNet(ch_prev, channels, kernel_size=3, stride=2, affine=True, transpose=True,op_type='ops')


        stride = 1
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            node = Node(search_space, x_id, x_op, y_id, y_op, channels, stride,transpose=True)
            self.ops.append(node)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(self.nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y)
            states.append(out)
        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        # print(out.size())
        return out

class NASUNetBSD(nn.Module):
    """construct the Ulike-net according to these searched cells"""

    def __init__(self, args, nclass=1, in_channels=3, backbone=None, aux=False,
                 c=8, depth=4, keep_prob=1, nodes=5, arch=None,
                 double_down_channel=False, use_aux_head=False):
        super(NASUNetBSD, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.nodes = nodes 
        self.double_down_channel=double_down_channel
        self.multiplier = nodes
        self.use_aux_head = use_aux_head
        self.keep_prob=keep_prob

        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]

        self.DownCell_arch = arch[:4 * self.nodes]
        self.UpCell_arch = arch[4 * self.nodes:]

        ch_prev_2, ch_prev, ch_curr = self.multiplier * c, self.multiplier * c, c

        self._stem0 = ConvNet(in_channels, ch_prev_2, kernel_size=1, op_type='pre_ops')
        self._stem1 = ConvNet(in_channels, ch_prev, kernel_size=3, stride=2, op_type='pre_ops')
        self.cells_down = nn.ModuleList()
        self.cells_up = nn.ModuleList()

        path_recorder = []
        path_recorder += [ch_prev]
        path_recorder += [ch_prev_2]

        # this is the left part of U-Net (encoder) down sampling
        for i in range(depth):
            ch_curr = 2*ch_curr if self.double_down_channel else ch_curr
            cell_down = cellDown(self.search_space,self.DownCell_arch,ch_prev_2,ch_prev,ch_curr)
            self.cells_down +=[cell_down]
            ch_prev_2,ch_prev = ch_prev,cell_down._multiplier*ch_curr
            path_recorder +=[ch_prev]

        # this is the right part of U-Net (decoder) up sampling
        for i in range(depth+1):
            ch_prev_2 = path_recorder[-(i+2)]
            cell_up = cellUp(self.search_space,self.UpCell_arch,ch_prev_2,ch_prev,ch_curr)
            self.cells_up += [cell_up]
            ch_prev = cell_up._multiplier*ch_curr
            ch_curr = ch_curr//2 if self.double_down_channel else ch_curr

        if self.use_aux_head:
          self.ConvSegmentation = Aux_dropout(ch_prev, nclass, nn.BatchNorm2d,dropout_rate=1-self.keep_prob,)
        else:
          self.ConvSegmentation = ConvNet(ch_prev, nclass, kernel_size=1, dropout_rate=1-self.keep_prob, op_type='SC')

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input):
        """bchw for tensor"""
        _,_,h,w = input.size()

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

        x= F.interpolate(x, size=input.size()[2:4], mode='bilinear', align_corners=True)
        x=torch.sigmoid(x)
        return x
