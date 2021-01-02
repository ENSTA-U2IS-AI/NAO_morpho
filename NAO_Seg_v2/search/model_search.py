import torch
import torch.nn as nn
from ops.operations import OPERATIONS_search_with_mor, ConvNet, Aux_dropout,OPERATIONS_search_without_mor
import torch.nn.functional as F

# customise the cell for segmentation
class Node(nn.Module):
    def __init__(self,search_space,channels,node_id,stride,transpose=False):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()
        self.transpose=transpose
        possible_connection_nums = node_id+2

        if search_space == 'with_mor_ops':
            OPERATIONS = OPERATIONS_search_with_mor
        elif search_space == 'without_mor_ops':
            OPERATIONS = OPERATIONS_search_without_mor

        for k, v in OPERATIONS.items():
            self.x_op.append(v(possible_connection_nums, channels, channels, stride, affine=True))
            self.y_op.append(v(possible_connection_nums, channels, channels, stride, affine=True))

    def forward(self, x, x_id, x_op, y, y_id, y_op,bn_train=False):
        # this mean that only the inputs to the intermediate nodes exists the down sampling ops
        input_to_intermediate_node = []
        if self.transpose==True:
          stride =  1
          x = self.x_op[x_op](x, x_id, stride,bn_train=bn_train)
          y = self.y_op[y_op](y, y_id, stride,bn_train=bn_train)
        else:
          stride = self.stride if x_id in [0, 1] else 1
          x = self.x_op[x_op](x, x_id, stride,bn_train=bn_train)
          stride = self.stride if y_id in [0, 1] else 1
          y = self.y_op[y_op](y, y_id, stride,bn_train=bn_train)

        input_to_intermediate_node+=[x]
        input_to_intermediate_node+=[y]
        out = sum(consistent_dim(input_to_intermediate_node))
        return out
      
from torch.nn.functional import interpolate
def consistent_dim(states):
    """the aim of this fonction is to make sure that the dimensions of state are consistent """
    h_max, w_max = 0, 0
    for ss in states:
        if h_max < ss.size()[2]:
            h_max = ss.size()[2]
        if w_max < ss.size()[3]:
            w_max = ss.size()[3]
    return [interpolate(ss, (h_max, w_max)) for ss in states]

# customise the cell for segmentation
class cellDown(nn.Module):
    def __init__(self, search_space, ch_prev_2,ch_prev, nodes, channels):
        super(cellDown, self).__init__()
        self.search_space = search_space
        self.ops = nn.ModuleList()
        self.nodes = nodes
        self.nums_inputs_to_intermediate_nodes = 2
        self.concatenate_nodes = nodes

        self.preprocess0 = ConvNet(ch_prev_2, channels, kernel_size=1, stride=2, affine=False, op_type='pre_ops_cell')
        self.preprocess1 = ConvNet(ch_prev, channels, kernel_size=1, stride=1, affine=False, op_type='pre_ops_cell')

        # the prev_layers represents chs_prev_2, chs_prev and the channels represents chs
        stride=2
        for i in range(self.nodes):
            node = Node(search_space, channels, i, stride)
            self.ops.append(node)

    def forward(self, s0, s1, arch,bn_train=False):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        # this mean that every intermediate node if and only if two inputs
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op,bn_train=bn_train)
            states.append(out)
            # print('\n')

        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        # print(out.size())
        return out


# customise the cell for segmentation
class cellUp(nn.Module):
    def __init__(self, search_space, ch_prev_2, ch_prev, nodes, channels):
        super(cellUp, self).__init__()
        self.search_space = search_space
        self.ops = nn.ModuleList()
        self.nodes = nodes
        self.nums_inputs_to_intermediate_nodes = 2
        self.concatenate_nodes = nodes

        self.preprocess0 = ConvNet(ch_prev_2, channels, kernel_size=1, stride=1, affine=False,
                                       op_type='pre_ops_cell')

        self.preprocess1 = ConvNet(ch_prev, channels*4, kernel_size=1, stride=1, affine=False, op_type='pre_ops_cell')
        # self.PS = nn.PixelShuffle(2)
        self.preprocess1_ = nn.ConvTranspose2d(channels * 4, channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1, bias=False)
        self._ops = nn.ModuleList()

        # the prev_layers represents chs_prev_2, chs_prev and the channels represents chs
        stride = 1
        for i in range(self.nodes):
            node = Node(search_space, channels, i, stride, transpose=True)
            self.ops.append(node)

    def forward(self, s0, s1, arch, bn_train=False):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        # s1 = self.PS(s1)
        s1 = self.preprocess1_(s1)

        states = [s0, s1]
        # this mean that every intermediate node if and only if two inputs
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, bn_train=bn_train)
            states.append(out)

        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        # print(out.size())
        return out


class NASUNetSegmentationWS(nn.Module):
    #args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps
    def __init__(self, args, depth=4, classes=2, nodes=5, input_chs=3, chs=16, keep_prob=1, double_down_channel=False, use_aux_head=False):
        super(NASUNetSegmentationWS, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.classes = classes
        self.nodes = nodes
        self.keep_prob = keep_prob
        self.double_down_channel=double_down_channel
        self.multiplier = nodes
        self.use_aux_head = use_aux_head

        ch_prev_2, ch_prev, ch_curr = self.nodes * chs, self.nodes * chs, chs #chs = channels

        self._stem0 = ConvNet(input_chs, ch_prev_2, kernel_size=1, op_type='pre_ops')
        self._stem1 = ConvNet(input_chs,ch_prev, kernel_size=3, stride=2, op_type='pre_ops')
        self.cells_down = nn.ModuleList()
        self.cells_up = nn.ModuleList()
        self.score_outs = nn.ModuleList()

        path_recorder = []
        path_recorder += [ch_prev]
        path_recorder += [ch_prev_2]

        # this is the left part of U-Net (encoder) down sampling
        for i in range(depth):
            ch_curr = 2*ch_curr if self.double_down_channel else ch_curr
            cell_down = cellDown(self.search_space,ch_prev_2,ch_prev,self.nodes,ch_curr)
            self.cells_down +=[cell_down]
            ch_prev_2,ch_prev = ch_prev,self.multiplier*ch_curr
            path_recorder +=[ch_prev]

        # this is the right part of U-Net (decoder) up sampling
        for i in range(depth):
            ch_prev_2 = path_recorder[-(i+2)]
            cell_up = cellUp(self.search_space,ch_prev_2,ch_prev,self.nodes,ch_curr)
            self.cells_up += [cell_up]
            ch_prev = self.multiplier*ch_curr
            self.score_outs.append(nn.Conv2d(ch_prev,1,1))
            ch_curr = ch_curr//2 if self.double_down_channel else ch_curr

        self.score_final = nn.Conv2d(depth, self.classes, 1)
        if use_aux_head:
          self.ConvSegmentation = Aux_dropout(ch_prev, self.classes, nn.BatchNorm2d,dropout_rate=1-self.keep_prob)
        else:
          self.ConvSegmentation = ConvNet(ch_prev, self.classes, kernel_size=1, dropout_rate=1-self.keep_prob, op_type='SC')

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, arch, size,bn_train=False):

        s0, s1 = self._stem0(input), self._stem1(input)
        cells_recorder = []
        
        cells_recorder.append(s0)
        cells_recorder.append(s1)
      
        DownCell_arch,UpCell_arch=arch
       
        #the left part of U-Net
        for i, cell in enumerate(self.cells_down):
            s0,s1 = s1,cell(s0,s1,DownCell_arch,bn_train=bn_train)
            cells_recorder.append(s1)
            
        outs=[]
        upsample = nn.UpsamplingBilinear2d(size)
        #the right part of U-Net
        for i,cell in enumerate(self.cells_up):
            s0 = cells_recorder[-(i+2)] # get the chs_prev_prev
            s1 = cell(s0,s1,UpCell_arch,bn_train=bn_train)
            s1_out=self.score_outs[i](s1)
            outs.append(upsample(s1_out))

        fuse = torch.cat(outs[:], dim=1)
        fuse_out = self.score_final(fuse)
        outs.append(fuse_out)
        results = [torch.sigmoid(out) for out in outs]
        # exit()
        # if self.use_aux_head:
        #   x = self.ConvSegmentation(s1)
        #   x = interpolate(x, (h,w))
        # else:
        #   x = self.ConvSegmentation(s1)

        return results
