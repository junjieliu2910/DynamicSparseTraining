import math
import torch 
import torch.nn as nn

"""
Function for activation binarization
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.step = BinaryStep.apply
        #self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0.)
    
    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight-threshold
        mask = self.step(abs_weight)
        ratio = torch.sum(mask) / mask.numel()
        #print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0.)
            abs_weight = torch.abs(self.weight)
            threshold = self.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = self.step(abs_weight)
        masked_weight = self.weight * mask 
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

    


class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c 
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups

        ## define weight 
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_c))
        self.step = BinaryStep.apply
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        weight_shape = self.weight.shape 
        threshold = self.threshold.view(weight_shape[0], -1)
        weight = torch.abs(self.weight)
        weight = weight.view(weight_shape[0], -1)
        weight = weight - threshold
        mask = self.step(weight)
        mask = mask.view(weight_shape)
        ratio = torch.sum(mask) / mask.numel()
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            threshold = self.threshold.view(weight_shape[0], -1)
            weight = torch.abs(self.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = self.step(weight)
            mask = mask.view(weight_shape)
        masked_weight = self.weight * mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out