import torch
import torch.nn as nn
from ops.operations import OPERATIONS_search_small, OPERATIONS_search_middle, WSReLUConvBN, FactorizedReduce, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path,ConvNet
from ops.genotype import DownOps,NormalOps,UpOps

# customise the cell for segmentation
class NodeSegmentation(nn.Module):
    def __init__(self, search_space, channels, node_id, stride, drop_path_keep_prob=None,transpose=False):
        super(NodeSegmentation, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()
        self.transpose=transpose
        possible_connection_nums = node_id

        if search_space == 'small':
            OPERATIONS = OPERATIONS_search_small
        elif search_space == 'middle':
            OPERATIONS = OPERATIONS_search_middle
        else:
            OPERATIONS = OPERATIONS_search_small

        if self.stride>=2:
            pris=UpOps if transpose else DownOps
        else:
            pris=NormalOps

        for pri in pris:
            self.x_op.append(OPERATIONS[pri](possible_connection_nums, channels, channels, stride, True))
            self.y_op.append(OPERATIONS[pri](possible_connection_nums, channels, channels, stride, True))

    def forward(self, x, x_id, x_op, y, y_id, y_op,bn_train=False):
        # this mean that only the inputs to the intermediate nodes exists the down sampling ops
        input_to_intermediate_node = []
        stride = self.stride if x_id in [0, 1] else 1
        x = self.x_op[x_op](x, x_id, stride,bn_train=bn_train)
        stride = self.stride if y_id in [0, 1] else 1
        y = self.y_op[y_op](y, y_id, stride,bn_train=bn_train)
        input_to_intermediate_node+=x
        input_to_intermediate_node+=y
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
class CellSegmentation(nn.Module):
    def __init__(self, search_space, ch_prev_2,ch_prev, nodes, channels, drop_path_keep_prob=None, type='down'):
        super(CellSegmentation, self).__init__()
        self.search_space = search_space
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes
        self.type = type
        self.nums_inputs_to_intermediate_nodes = 2
        self.concatenate_nodes = nodes

        if self.type == 'down':
            self.preprocess0 = nn.Sequential(
            nn.Conv2d(ch_prev_2, channels, kernel_size=1, stride=2, affine=False,),
            nn.BatchNorm2d(ch_prev_2)
        )
        else:
            self.preprocess0 = nn.Sequential(
            nn.Conv2d(ch_prev_2, channels, kernel_size=1, affine=False,),
            nn.BatchNorm2d(ch_prev_2)
        )
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(ch_prev, channels, kernel_size=1, affine=False,),
            nn.BatchNorm2d(ch_prev_2)
        )

        self._ops = nn.ModuleList()

        # the prev_layers represents chs_prev_2, chs_prev and the channels represents chs
        # initial_id_for_up_or_down=0 if self.type=='down' else 1
        stride = 2
        for i in range(self.nodes):
          # for j in range(self.self.nums_inputs_to_intermediate_nodes+i):
          #     stride=2 if j>=initial_id_for_up_or_down and j<2 else 1
          if self.type=='up':
              node = NodeSegmentation(search_space, channels, i, stride, drop_path_keep_prob, transpose=True)
          else:
              node = NodeSegmentation(search_space, channels, i, stride, drop_path_keep_prob, )
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

        out = torch.cat(states[-self.concatenate_nodes:], dim=1)
        return out



class NASUNetSegmentationWS(nn.Module):
    #args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps
    def __init__(self, args, depth=4, classes=1, nodes=4, input_chs=3, chs=16, keep_prob=0.9, double_down_channel=True, use_softmax_head = False):
        super(NASUNetSegmentationWS, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.depth = depth
        self.classes = classes
        self.nodes = nodes
        self.keep_prob = keep_prob
        self.double_down_channel=double_down_channel
        self.use_softmax_head=use_softmax_head
        self.multiplier = nodes

        ch_prev_2, ch_prev, ch_curr = self._nodes_num * chs, self._nodes_num * chs, chs #chs = channels

        # s0 = 1×1 convolution
        self._stem0 = nn.Sequential(
            nn.Conv2d(input_chs, ch_prev_2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_prev_2)
        )
        # s1 = 3×3 convolution
        self._stem1 = nn.Sequential(
            nn.Conv2d(input_chs, ch_prev, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_prev)
        )
        self.cells_down = nn.ModuleList
        self.cells_up = nn.ModuleList

        path_recorder = []
        path_recorder += [ch_prev]
        path_recorder += [ch_prev_2]

        # this is the left part of U-Net (encoder) down sampling
        for i in range(depth):
            ch_curr = 2*ch_curr if self._double_down_channel else ch_curr
            cell_down = CellSegmentation(self._search_space,ch_prev_2,ch_prev,self.nodes,ch_curr,self.keep_prob,type='down')
            self.cells_down +=[cell_down]
            ch_prev_2,ch_prev = ch_prev,self._multiplier*ch_curr
            path_recorder +=[ch_prev]

        # this is the right part of U-Net (decoder) up sampling
        for i in range(depth+1):
            ch_prev_2 = path_recorder[-(i+2)]
            cell_up = CellSegmentation(self._search_space,ch_prev_2,ch_prev,self.nodes,ch_curr,self.keep_prob,type='up')
            self.cells_up += [cell_up]
            ch_prev = self._multiplier*ch_prev
            ch_curr = ch_curr//2 if self._double_down_channel else ch_curr

        self.ConvSegmentation = ConvNet(ch_prev, self._classes, kernel_size=1, dropout_rate=0.1)

        if use_softmax_head:
            self.softmax = nn.Softmax(dim=1)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, arch, bn_train=False):
        # s0: remain the original image size
        # s1: halve image size
        s0, s1 = self._stem0(input), self._stem1(input)
        path_recorder = []

        path_recorder.append(s0)
        path_recorder.append(s1)

        DownCell_arch = arch[:self.nodes*4] 
        UpCell_arch = arch[self.nodes*4:]

        #the left part of U-Net
        for i, cell in enumerate(self.cells_down):
            s0,s1 = s1,cell(s0,s1,DownCell_arch,bn_train=bn_train)
            path_recorder.append(s1)

        #the right part of U-Net
        for i,cell in enumerate(self.cells_up):
            s0 = path_recorder[-(i+2)] # get the chs_prev_prev
            s1 = cell(s0,s1,UpCell_arch,bn_train=bn_train)

        x = self.ConvSegmentation(s1)

        if self._use_softmax_head:
            x = self.softmax(x)
        return x