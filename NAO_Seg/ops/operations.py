import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INPLACE=False
BIAS=False


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
        
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        x = self.ops(x)
        return x


class WSReLUConvBN(nn.Module):
    def __init__(self, num_possible_inputs, C_out, C_in, kernel_size, stride=1, padding=0):
        super(WSReLUConvBN, self).__init__()
        self.stride = stride
        self.padding = padding
        self.relu = nn.ReLU(inplace=INPLACE)
        self.w = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, kernel_size, kernel_size)) for _ in range(num_possible_inputs)])
        self.bn = nn.BatchNorm2d(C_out, affine=True)
    
    def forward(self, x, x_id, bn_train=False):
        x = self.relu(x)
        w = torch.cat([self.w[i] for i in x_id], dim=1)
        x = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class WSBN(nn.Module):

    def __init__(self, num_possible_inputs, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(WSBN, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            for i in range(self.num_possible_inputs):
                self.weight[i].data.fill_(1)
                self.bias[i].data.zero_()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight[x_id], self.bias[x_id],
            training, self.momentum, self.eps)
    
    
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
       
    def forward(self, x):
        return self.op(x)


class WSSepConv(nn.Module):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, padding, affine=True):
        super(WSSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        
        self.relu1 = nn.ReLU(inplace=INPLACE)
        self.W1_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W1_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn1 = WSBN(num_possible_inputs, C_in, affine=affine)

        self.relu2 = nn.ReLU(inplace=INPLACE)
        self.W2_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W2_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn2 = WSBN(num_possible_inputs, C_in, affine=affine)
    
    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu1(x)
        x = F.conv2d(x, self.W1_depthwise[x_id], stride=stride, padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W1_pointwise[x_id], padding=0)
        x = self.bn1(x, x_id, bn_train=bn_train)

        x = self.relu2(x)
        x = F.conv2d(x, self.W2_depthwise[x_id], padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W2_pointwise[x_id], padding=0)
        x = self.bn2(x, x_id, bn_train=bn_train)
        return x


class DilSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
      )
        
    def forward(self, x):
        return self.op(x)


class WSDilSepConv(nn.Module):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, padding, dilation=2, affine=True):
        super(WSDilSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        self.dilation = dilation
        
        self.relu = nn.ReLU(inplace=INPLACE)
        self.W_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn = WSBN(num_possible_inputs, C_in, affine=affine)
    
    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu(x)
        x = F.conv2d(x, self.W_depthwise[x_id], stride=stride, padding=self.padding, dilation=self.dilation, groups=self.C_in)
        x = F.conv2d(x, self.W_pointwise[x_id], padding=0)
        x = self.bn(x, x_id, bn_train=bn_train)

        return x


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, count_include_pad=False):
        super(AvgPool, self).__init__()
        self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
    
    def forward(self, x):
        return self.op(x)


class WSAvgPool2d(nn.Module):
    def __init__(self, kernel_size, padding):
        super(WSAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
    
    def forward(self, x, x_id, stride=1, bn_train=False):
        return F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool, self).__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.op(x)


class WSMaxPool2d(nn.Module):
    def __init__(self, kernel_size, padding):
        super(WSMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
    
    def forward(self, x, x_id, stride=1, bn_train=False):
        return F.max_pool2d(x, self.kernel_size, stride, self.padding)


class Identity(nn.Module):
    def __init__(self,in_channels,out_channels,norm_type='gn',affine=True):
        super(Identity, self).__init__()
        # self.bn = nn.BatchNorm2d(out_channels, affine=False)
        group = 1 if out_channels % 8 != 0 else out_channels // 8
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)
        else:
            self.norm = nn.BatchNorm2d(out_channels, affine=affine)
            
        #activate function
        self.activate = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.norm(x)
        x = self.activate(x)
        return x


class WSIdentity(nn.Module):
    def __init__(self, c_in, c_out, stride, affine=True,norm_type='gn'):
        super(WSIdentity, self).__init__()
        if stride == 2:
            self.reduce = nn.ModuleList()
            self.reduce.append(FactorizedReduce(c_in, c_out, [0, 0, 0], affine=affine))
            self.reduce.append(FactorizedReduce(c_in, c_out, [0, 0, 0], affine=affine))
        # self.bn = nn.BatchNorm2d(out_channels, affine=False)
        group = 1 if c_out % 8 != 0 else c_out // 8
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(group, c_out, affine=affine)
        else:
            self.norm = nn.BatchNorm2d(c_out, affine=affine)
            
        #activate function
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x, x_id, stride=1, bn_train=False):
        if stride == 2:
            return self.reduce[x_id](x, bn_train=bn_train)
        x = self.norm(x)
        x = self.activate(x)
        return x



class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class WSZero(nn.Module):
    def __init__(self):
        super(WSZero, self).__init__()

    def forward(self, x, x_id, stride=1, bn_train=False):
        if stride == 1:
            return x.mul(0.)
        return x[:,:,::stride,::stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, [hw[0], hw[0], c[0]], affine)
            x_out_shape = [hw[1], hw[1], channels]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, [hw[0], hw[0]], affine)
            x_out_shape = [hw[0], hw[0], channels]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, [hw[1], hw[1]], affine)
            y_out_shape = [hw[1], hw[1], channels]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1, bn_train=False):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1, bn_train=bn_train)
        return [s0, s1]


class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0,1]
                self.concat_fac_op_dict[i] = len(self.ops)
                op = FactorizedReduce(layers[i][-1], channels, [hw, hw], affine)
                self.ops.append(op)
        
    def forward(self, states, bn_train=False):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], bn_train)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out

class Depthwise_separable_conv(nn.Module):
    def __init__(self, cin, cout, kernel_size):
        super(Depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(cin, cin, kernel_size=kernel_size, padding=1, groups=cin,bias=False)
        self.pointwise = nn.Conv2d(cin, cout, kernel_size=1,bias=False)
        # self.bn = nn.BatchNorm2d(cout, affine=True)
        # self.relu= nn.ReLU(inplace=INPLACE)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        # out = self.bn(out)
        # out = self.relu(out)
        return out

class Pseudo_Shuff_dilation(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, degree=3, stride=1, type=None, affine=True):
      super(Pseudo_Shuff_dilation, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.pading = kernel_size// 2
      self.degree = degree
      self.stride = stride 
      self.convmorph = Depthwise_separable_conv(in_channels, out_channels * kernel_size * kernel_size, kernel_size)
      self.pixel_shuffle = nn.PixelShuffle(kernel_size)
      self.pool_ = nn.MaxPool2d(kernel_size, stride=kernel_size)
      self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
      # gp = 1 if out_channels%8 !=0 else out_channels//8
      # self.norm = nn.GroupNorm(gp, out_channels, affine=affine)
      self.bn = nn.BatchNorm2d(out_channels, affine=True)
      #activate function
      self.activate = nn.ReLU(inplace=True)
      
  def forward(self, x):
      '''
      x: tensor of shape (B,C,H,W)
      '''
      x = self.bn(x)
      y = self.convmorph(x)# / self.degree
      y = self.pixel_shuffle(y)
      y = self.pool_(y)
      if self.stride==2:
        y = self.pool(y)
      return y

class WSPseudo_Shuff_dilation(nn.Module):
  def __init__(self, num_possible_inputs, in_channels, out_channels, kernel_size=3, degree=3, stride=1, type=None, affine=True):
      super(WSPseudo_Shuff_dilation, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.padding = kernel_size// 2
      self.degree = degree
      self.stride = stride 
      self.convmorph = Depthwise_separable_conv(in_channels, out_channels * kernel_size * kernel_size, kernel_size)
      self.pixel_shuffle = nn.PixelShuffle(kernel_size)
      self.pool_ = nn.MaxPool2d(kernel_size, stride=kernel_size)
      self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
      # gp = 1 if out_channels%8 !=0 else out_channels//8
      # self.norm = nn.GroupNorm(gp, out_channels, affine=affine)
      self.bn = nn.BatchNorm2d(out_channels, affine=False)
       #activate function
      self.activate = nn.ReLU(inplace=True)
      
  def forward(self, x, x_id, stride, bn_train=False):
      '''
      x: tensor of shape (B,C,H,W)
      '''
      x = self.bn(x)
      y = self.convmorph(x)# / self.degree
      y = self.pixel_shuffle(y)
      y = self.pool_(y)
      if stride==2:
        y = self.pool(y)
      return y


class Pseudo_Shuff_gradient(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, degree=3, stride=1, type=None, affine=True):
        super(Pseudo_Shuff_gradient, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.degree = degree
        self.stride = stride
        # self.convmorph = Depthwise_separable_conv(in_channels, out_channels * kernel_size * kernel_size, kernel_size)
        self.convmorph = nn.Conv2d(in_channels,out_channels*kernel_size*kernel_size,kernel_size,stride=1,padding=1)
        self.pixel_shuffle = nn.PixelShuffle(kernel_size)
        self.pool_ = nn.MaxPool2d(kernel_size, stride=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        # activate function
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        x = self.bn(x)
        y = self.convmorph(x)  # / self.degree
        y = self.pixel_shuffle(y)
        y = self.pool_(y)
        gradient = y-x
        if self.stride == 2:
            gradient = self.pool(gradient)
        return gradient


class WSPseudo_Shuff_gradient(nn.Module):
    def __init__(self, num_possible_inputs, in_channels, out_channels, kernel_size=3, degree=3, stride=1, type=None,
                 affine=True):
        super(WSPseudo_Shuff_gradient, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.degree = degree
        self.stride = stride
        # self.convmorph = Depthwise_separable_conv(in_channels, out_channels * kernel_size * kernel_size, kernel_size)
        self.convmorph = nn.Conv2d(in_channels,out_channels*kernel_size*kernel_size,kernel_size,stride=1,padding=1)
        self.pixel_shuffle = nn.PixelShuffle(kernel_size)
        self.pool_ = nn.MaxPool2d(kernel_size, stride=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        # activate function
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x, x_id, stride, bn_train=False):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        x = self.bn(x)
        y = self.convmorph(x)  # / self.degree
        y = self.pixel_shuffle(y)
        y = self.pool_(y)
        gradient = y-x
        if stride == 2:
            gradient = self.pool(gradient)
        return gradient

#Operation 1
class CWeightNet(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=None,
                 bias=False, transpose=False, out_padding=1,use_norm=False, affine=True, dropout_rate=0):
        super(CWeightNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose
        self.out_padding = out_padding

        padding = self.kernel_size//2

        if isinstance(padding,int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        self.globalPool = nn.AdaptiveAvgPool2d(1)
        #Squeeze-and-Excitation Networks
        self.SEnet = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, out_channels),
            nn.Sigmoid()
        )
        if (self.stride >=2):
            if self.transpose:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding, output_padding=self.out_padding,
                                                bias=False)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=False)

        gp = 1 if out_channels%8 !=0 else out_channels//8
        self.norm = nn.GroupNorm(gp, out_channels, affine=affine)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.globalPool(x).view(b, c)
        y = self.SEnet(y).view(b, c, 1, 1)
        SENet = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
        return SENet

class WSCWeightNet(nn.Module):
    def __init__(self,num_possible_inputs,in_channels, out_channels, kernel_size=3, stride=1,dilation=1, groups=None,
                 bias=False, transpose=False,out_padding=1, use_norm=False, affine=True, dropout_rate=0):
        super(WSCWeightNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose
        self.out_padding = out_padding

        padding = self.kernel_size//2

        if isinstance(padding,int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        self.globalPool = nn.AdaptiveAvgPool2d(1)
        #Squeeze-and-Excitation Networks
        self.SEnet = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),#16
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, out_channels),
            nn.Sigmoid()
        )
        if (self.stride >=2):
            if self.transpose:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding, output_padding=self.out_padding,
                                                bias=False)
                # self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels*2*2,kernel_size,stride=1,padding=1),
                #           nn.PixelShuffle(2))
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=False)

        gp = 1 if out_channels%8 !=0 else out_channels//8    #16
        self.norm = nn.GroupNorm(gp, out_channels, affine=affine)

    def forward(self, x, x_id, stride, bn_train=False):
        b, c, _, _ = x.size()
        y = self.globalPool(x).view(b, c)
        y = self.SEnet(y).view(b, c, 1, 1)
        SENet = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
        return SENet

#Operation 2
class ConvNet(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1,dilation=1,groups=1,
                 bias=False, transpose=False, out_padding=1,use_norm=True, affine=True, dropout_rate=0,
                 norm_type='gn', op_type='ops' ):
        super(ConvNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose =transpose
        self.out_padding = out_padding
        self.op_type = op_type
        self.dropout_rate = dropout_rate

        padding = self.kernel_size // 2

        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        if self.transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                           stride=self.stride, padding=padding,
                                           output_padding=self.out_padding, bias=self.bias)

        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=padding,
                                  dilation=self.dilation, bias=True)
        # self.bn = nn.BatchNorm2d(out_channels, affine=False)
        group = 1 if out_channels % 8 != 0 else out_channels // 8
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)
        else:
            self.norm = nn.BatchNorm2d(out_channels, affine=affine)

        #activate function
        self.activate = nn.ReLU(inplace=True)
        self.activate2 = nn.ReLU(inplace=False)

        #dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

    def forward(self, x):
        if self.op_type=='ops':
          x = self.conv(x)
          x = self.norm(x)
          x = self.activate(x)
        elif self.op_type=='SC':
          if self.dropout_rate>0:
            x = self.dropout(x)
          x = self.conv(x)
        elif self.op_type=='pre_ops':
          x = self.conv(x)
          x = self.norm(x)
        elif self.op_type=='pre_ops_cell':
          x = self.activate2(x)
          x = self.conv(x)
          x = self.norm(x)  
        return x

class WSConvNet(nn.Module):
    def __init__(self, num_possible_inputs, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=False, transpose=False, out_padding=1, use_norm=True, affine=True, dropout_rate=0,
                 norm_type='gn', op_type='ops'):
        super(WSConvNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose
        self.out_padding = out_padding
        self.op_type = op_type
        self.dropout_rate = dropout_rate

        padding = self.kernel_size // 2

        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        if self.transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                           stride=self.stride, padding=padding,
                                           output_padding=self.out_padding, bias=self.bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=padding,
                                  dilation=self.dilation, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels, affine=False)
        group = 1 if out_channels % 8 != 0 else out_channels // 8
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)
        else:
            self.norm = nn.BatchNorm2d(out_channels, affine=affine)
            
        #activate function
        self.activate = nn.ReLU(inplace=True)
        # self.activate2 = nn.ReLU(inplace=False)

        #dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

    def forward(self, x, x_id, stride, bn_train=False):
        if self.op_type=='ops':
          x = self.conv(x)
          x = self.norm(x)# add batchnorm to the input
          x = self.activate(x)

        return x

class Aux_dropout(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer,dropout_rate):
        super(Aux_dropout, self).__init__()
        inter_channels = in_channels // 4
        if dropout_rate>0:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(),
                                        nn.Dropout2d(dropout_rate, False),
                                        nn.Conv2d(inter_channels, out_channels, 1))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                      norm_layer(inter_channels),
                                      nn.ReLU(),
                                      # nn.Dropout2d(dropout_rate, False),
                                      nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x):
        return self.conv(x)

"""
operation set for cell of U-net segmentation network
DownOps = [
            'max_pool',
            'down_cweight_3×3',
            'down_conv_3×3',
            'pix_shuf_gradient'
]

NormalOps = [
            'identity',
            'cweight_3×3',
            'conv_3×3',
            'pix_shuf_gradient',
]

UpOps = [
            'up_conv_3×3'
]
"""

OPERATIONS_with_mor = {
    0:lambda c_in, c_out, stride, affine: MaxPool(3, stride, 1),#'max_pool'
    1:lambda c_in, c_out, stride, affine: CWeightNet(c_in,c_out,stride=2,affine=affine),#'down_cweight_3×3'
    2:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,stride=2,affine=affine),#'down_conv_3×3'
    3:lambda c_in, c_out, stride, affine: Pseudo_Shuff_gradient(c_in, c_out, 3, 3, stride=2,affine=affine),#'pix_shuf_pool'
    4:lambda c_in, c_out, stride, affine: Identity(c_in, c_out, affine),#'identity'
    5:lambda c_in, c_out, stride, affine: CWeightNet(c_in,c_out,affine=affine),#'cweight_3×3'
    6:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,affine=affine),#'conv_3×3'
    7:lambda c_in, c_out, stride, affine: Pseudo_Shuff_gradient(c_in, c_out, 3, 3,affine=affine),#'pix_shuf_3×3'
    8:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,stride=2,transpose=True,affine=affine),#'up_conv_3×3'
}

OPERATIONS_search_with_mor = {
    0:lambda n, c_in, c_out, stride, affine: WSMaxPool2d(3, padding=1),#'max_pool'
    1:lambda n, c_in, c_out, stride, affine: WSCWeightNet(n,c_in,c_out,stride=2,affine=affine),#'down_cweight_3×3'
    2:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,stride=2,affine=affine),#'down_conv_3×3'
    3:lambda n, c_in, c_out, stride, affine: WSPseudo_Shuff_gradient(n, c_in, c_out, 3, 3, stride=2,affine=affine),#'pix_shuf_pool'
    4:lambda n, c_in, c_out, stride, affine: WSIdentity(c_in, c_out, stride, affine=affine),#'identity'
    5:lambda n, c_in, c_out, stride, affine: WSCWeightNet(n,c_in,c_out,affine=affine),#'cweight_3×3'
    6:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,affine=affine),#'conv_3×3'
    7:lambda n, c_in, c_out, stride, affine: WSPseudo_Shuff_gradient(n, c_in, c_out, 3, 3,affine=affine),#'pix_shuf_3×3'
    8:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,stride=2,transpose=True,affine=affine),#'up_conv_3×3'
}

"""
0-3 down ops
4-6 normal ops
7-8 up ops
"""
OPERATIONS_without_mor_ops = {
    0:lambda c_in, c_out, stride, affine: AvgPool(3, stride, 1),#'avg_pool'
    1:lambda c_in, c_out, stride, affine: MaxPool(3, stride, 1),#'max_pool'
    2:lambda c_in, c_out, stride, affine: CWeightNet(c_in,c_out,stride=2,affine=affine),#'down_cweight_3×3'
    3:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,stride=2,affine=affine),#'down_conv_3×3'
    4:lambda c_in, c_out, stride, affine: Identity(c_in, c_out, affine),#'identity'
    5:lambda c_in, c_out, stride, affine: CWeightNet(c_in,c_out,affine=affine),#'cweight_3×3'
    6:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,affine=affine),#'conv_3×3'
    7:lambda c_in, c_out, stride, affine: CWeightNet(c_in,c_out,stride=2,transpose=True,affine=affine),#'up_cweight_3×3'
    8:lambda c_in, c_out, stride, affine: ConvNet(c_in,c_out,stride=2,transpose=True,affine=affine),#'up_conv_3×3'
}
OPERATIONS_search_without_mor_ops = {
    0:lambda n, c_in, c_out, stride, affine: WSAvgPool2d(3, padding=1),#'avg_pool'
    1:lambda n, c_in, c_out, stride, affine: WSMaxPool2d(3, padding=1),#'max_pool'
    2:lambda n, c_in, c_out, stride, affine: WSCWeightNet(n,c_in,c_out,stride=2,affine=affine),#'down_cweight_3×3'
    3:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,stride=2,affine=affine),#'down_conv_3×3'
    4:lambda n, c_in, c_out, stride, affine: WSIdentity(c_in, c_out, stride, affine=affine),#'identity'
    5:lambda n, c_in, c_out, stride, affine: WSCWeightNet(n,c_in,c_out,affine=affine),#'cweight_3×3'
    6:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,affine=affine),#'conv_3×3'
    7:lambda n, c_in, c_out, stride, affine: WSCWeightNet(n,c_in,c_out,stride=2,transpose=True,affine=affine),#'up_cweight_3×3'
    8:lambda n, c_in, c_out, stride, affine: WSConvNet(n,c_in,c_out,stride=2,transpose=True,affine=affine),#'up_conv_3×3'
}
