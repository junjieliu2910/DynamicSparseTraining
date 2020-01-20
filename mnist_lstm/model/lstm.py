import torch 
import torch.nn as nn 
from .lstmcell import LSTM


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first=True, mask=False):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm1 = LSTM(input_size, hidden_size, batch_first, mask=mask)
        self.lstm2 = LSTM(hidden_size, hidden_size, batch_first, mask=mask)
        #self.lstm3 = LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.keep_ratio=1.
        self.mask = mask
        self.init_weights()

    def forward(self, x, states):
        
        final_state = []
        layer1, state = self.lstm1(x, states[0][0], states[1][0])
        final_state.append(state)
        layer2, state = self.lstm2(layer1, states[0][1], states[1][1])
        final_state.append(state)
        if self.batch_first:
            out = self.fc(layer2[:, -1, :])
        else:
            out = self.fc(layer2[-1, :, :])

        if self.mask:
            total_number = self.lstm1.cell.total_number + self.lstm2.cell.total_number
            keep_number = self.lstm1.cell.keep_number + self.lstm2.cell.keep_number
            self.keep_ratio = keep_number / total_number
        
        return out, tuple(final_state)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
       
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))
       
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)