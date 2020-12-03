import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LSTM(nn.Module):
    '''
    Written by KKL in 2020-12-1

    parameters: INPUT_SIZE, HIDDEN_SIZE, num_layers, batch_first
    input:  mag: mixture feature spectra, shape (batch_size, time_step, input_size)
    output: the estimated magnitude spectra of source1 and source2

            estimate src: shape (batch_size, num_speaker, time_step, output_size)
    '''

    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=41,
            hidden_size=20,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True,
            # dropout=0.8,
        )
        self.hidden = nn.Linear(20, 10)

        self.y_hidden1 = nn.Linear(20, 10)
        self.y_hidden2 = nn.Linear(20, 10)
        self.y_hidden3 = nn.Linear(10, 1)
        self.y_hidden3 = nn.Linear(20, 1)


        self._initialize_weights()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # mag shape (batch, time_step, input_size)

        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)

        r_out, (h_t, c_t) = self.rnn(x, None)
        y_hat = []
        for time in range(r_out.size(1)):
            out = self.hidden(r_out[:, time, :])
            # out = F.gelu(out)
            out = torch.tanh(out)
            out = self.y_hidden1(out)
            out = F.gelu(out)
            out = self.y_hidden2(out)
            out = torch.tanh(out)
            out = self.y_hidden3(out)
            out = torch.sigmoid(out)

            y_hat.append(out)
        return torch.stack(y_hat, dim=1)

    def _initialize_weights(self):
        # Xavier Init
        for name, param in self.named_parameters():
            if len(param.size())>1: #权重初始化
                # init.xavier_normal_(param, gain=1)
                init.xavier_uniform_(param, gain=1)
