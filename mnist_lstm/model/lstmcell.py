import math
import torch 
import torch.nn as nn
from .binarization import BinaryStep

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.weight_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # forget gate
        self.weight_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # update 
        self.weight_xu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # ouput gate
        self.weight_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.bias_i = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_u = nn.Parameter(torch.Tensor(hidden_size))
    
        self.activation = nn.Sigmoid()
       
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, prev_h, prev_c):
        input_gate = torch.matmul(input, self.weight_xi) + torch.matmul(prev_h, self.weight_hi) + self.bias_i 
        input_gate = self.activation(input_gate)

        forget_gate = torch.matmul(input, self.weight_xf) + torch.matmul(prev_h, self.weight_hf) + self.bias_f
        forget_gate = self.activation(forget_gate)

        output_gate = torch.matmul(input, self.weight_xo) + torch.matmul(prev_h, self.weight_ho) + self.bias_o
        output_gate = self.activation(output_gate) 

        update = torch.matmul(input, self.weight_xu) + torch.matmul(prev_h, self.weight_hu) + self.bias_u
        update = self.tanh(update)

        current_c = forget_gate * prev_c +  input_gate * update
        current_h = output_gate * self.tanh(current_c)     

        return current_h, current_c



class MaskedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.weight_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.threshold_xi = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.threshold_hi = nn.Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.weight_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.threshold_xf = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.threshold_hf = nn.Parameter(torch.Tensor(hidden_size))
        # update 
        self.weight_xu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.threshold_xu = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.threshold_hu = nn.Parameter(torch.Tensor(hidden_size))
        # ouput gate
        self.weight_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.threshold_xo = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.threshold_ho = nn.Parameter(torch.Tensor(hidden_size))
        
        self.bias_i = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_u = nn.Parameter(torch.Tensor(hidden_size))
    
        self.activation = nn.Sigmoid()
        self.unit_step = BinaryStep.apply
        self.tanh = nn.Tanh()

        self.keep_ratio = 1.
        self.keep_number = -1

        self.reset_parameters()

        self.total_number = self.weight_xi.numel() + self.weight_hi.numel() + self.weight_xf.numel() + \
            self.weight_hf.numel() + self.weight_xu.numel() + self.weight_hu.numel() + \
            self.weight_xo.numel() + self.weight_ho.numel()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        
        self.weight_xi.data.uniform_(-stdv, stdv)
        self.weight_hi.data.uniform_(-stdv, stdv)
        self.weight_xf.data.uniform_(-stdv, stdv)
        self.weight_hf.data.uniform_(-stdv, stdv)
        self.weight_xu.data.uniform_(-stdv, stdv)
        self.weight_hu.data.uniform_(-stdv, stdv)
        self.weight_xo.data.uniform_(-stdv, stdv)
        self.weight_ho.data.uniform_(-stdv, stdv)

        self.threshold_xi.data.fill_(0.)
        self.threshold_hi.data.fill_(0.)
        self.threshold_xf.data.fill_(0.)
        self.threshold_hf.data.fill_(0.)
        self.threshold_xu.data.fill_(0.)
        self.threshold_hu.data.fill_(0.)
        self.threshold_xo.data.fill_(0.)
        self.threshold_ho.data.fill_(0.)
        
        self.bias_i.data.fill_(0.)
        self.bias_f.data.fill_(0.)
        self.bias_o.data.fill_(0.)
        self.bias_u.data.fill_(0.)



    def forward(self, input, prev_h, prev_c):
        hidden_size = self.hidden_size

        abs_weight_xi = torch.abs(self.weight_xi.transpose(1, 0))
        mask_weight_xi = self.unit_step(abs_weight_xi - self.threshold_xi.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_xi = mask_weight_xi * self.weight_xi

        abs_weight_hi = torch.abs(self.weight_hi.transpose(1, 0))
        mask_weight_hi = self.unit_step(abs_weight_hi - self.threshold_hi.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_hi = mask_weight_hi * self.weight_hi

        abs_weight_xf = torch.abs(self.weight_xf.transpose(1, 0))
        mask_weight_xf = self.unit_step(abs_weight_xf - self.threshold_xf.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_xf = mask_weight_xf * self.weight_xf

        abs_weight_hf = torch.abs(self.weight_hf.transpose(1, 0))
        mask_weight_hf = self.unit_step(abs_weight_hf - self.threshold_hf.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_hf = mask_weight_hf * self.weight_hf

        abs_weight_xu = torch.abs(self.weight_xu.transpose(1, 0))
        mask_weight_xu = self.unit_step(abs_weight_xu - self.threshold_xu.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_xu = mask_weight_xu * self.weight_xu

        abs_weight_hu = torch.abs(self.weight_hu.transpose(1, 0))
        mask_weight_hu = self.unit_step(abs_weight_hu - self.threshold_hu.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_hu = mask_weight_hu * self.weight_hu

        abs_weight_xo = torch.abs(self.weight_xo.transpose(1, 0))
        mask_weight_xo = self.unit_step(abs_weight_xo - self.threshold_xo.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_xo = mask_weight_xo * self.weight_xo

        abs_weight_ho = torch.abs(self.weight_ho.transpose(1, 0))
        mask_weight_ho = self.unit_step(abs_weight_ho - self.threshold_ho.view(hidden_size, -1)).transpose(1, 0)
        masked_weight_ho = mask_weight_ho * self.weight_ho

        self.keep_number = torch.sum(mask_weight_xi) + torch.sum(mask_weight_hi) + torch.sum(mask_weight_xf) + \
            torch.sum(mask_weight_hf) + torch.sum(mask_weight_xu) + torch.sum(mask_weight_hu) + \
            torch.sum(masked_weight_xo) + torch.sum(masked_weight_ho)

        self.keep_ratio = float(self.keep_number) / float(self.total_number)

        input_gate = torch.matmul(input, masked_weight_xi) + torch.matmul(prev_h, masked_weight_hi) + self.bias_i
        input_gate = self.activation(input_gate)

        forget_gate = torch.matmul(input, masked_weight_xf) + torch.matmul(prev_h, masked_weight_hf) + self.bias_f
        forget_gate = self.activation(forget_gate)

        output_gate = torch.matmul(input, masked_weight_xo) + torch.matmul(prev_h, masked_weight_ho) + self.bias_o
        output_gate = self.activation(output_gate) 

        update = torch.matmul(input, masked_weight_xu) + torch.matmul(prev_h, masked_weight_hu) + self.bias_u
        update = self.tanh(update)

        current_c = forget_gate * prev_c +  input_gate * update
        current_h = output_gate * self.tanh(current_c)  
   

        return current_h, current_c


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=False, mask=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_major = not batch_first
        if mask:
            self.cell = MaskedLSTMCell(input_size, hidden_size)
        else:
            self.cell = LSTMCell(input_size, hidden_size)
        

    def forward(self, inputs, initial_h, initial_c, h_sparsity=0., h_threshold=0., block_size=-1):
        # Input need to have size [batch_size, time_step, input_size]
        if self.time_major:
            time_steps = inputs.size(0)
        else:
            time_steps = inputs.size(1)

        
        h_f = initial_h
        c_f = initial_c 
        
        outputs = []
        
        for t in range(time_steps):
            if self.time_major:
                h_f, c_f = self.cell(inputs[t, :, :], h_f, c_f)    
            else:
                h_f, c_f = self.cell(inputs[:, t, :], h_f, c_f)
                  
            outputs.append(h_f)
           
        if self.time_major:
            outputs = torch.stack(outputs)       
        else:
            outputs = torch.stack(outputs, 1)
        
            
        out_h = h_f
        out_c = c_f 

        return outputs, (out_h, out_c)