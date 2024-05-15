
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers=3, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout_p, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(2*hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class AttnDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=3, dropout_p=0.5):
        super(AttnDecoder, self).__init__()
        
        
        self.dropout = nn.Dropout(dropout_p)
        
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(output_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, hidden, cell, encoder_outputs):
        a = self.attention(hidden[-1], encoder_outputs)
        weighted = torch.bmm(a.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((x, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.fc(output)

        return output, hidden, cell


class LSTMAttn(nn.Module):
    def __init__(self, input_size, 
                        output_dim=300, 
                        hidden_dim=128, 
                        num_layers = 3,
                        fc_hidden_dims=[60, 400], 
                        number_classes=4,
                        dropout_p=0.1):
        super(LSTMAttn, self).__init__()

        self.encoder = EncoderLSTM(input_size,hidden_dim,num_layers,dropout_p)
        self.decoder = AttnDecoder(output_dim,hidden_dim,num_layers,dropout_p)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.decoder.output_dim, fc_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dims[1])  # Batch normalization before final classification
        )
        self.final_fc = nn.Linear(fc_hidden_dims[1], number_classes) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial hidden and cell states for decoder
        encoder_outputs, hidden, cell = self.encoder(x)
        
        current_batch_size = x.size(0) 
        initial_input = torch.zeros(current_batch_size, 1, self.decoder.output_dim)
        decoder_output, hidden, cell = self.decoder(initial_input, hidden, cell, encoder_outputs)
        
        output = self.fc_layers(decoder_output.squeeze(1))
        output = self.softmax(self.final_fc(output)) 
        return output
