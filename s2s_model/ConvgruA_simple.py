import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        timesteps = encoder_outputs.size(1)
        h = hidden.repeat(timesteps, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        attn_weights = torch.softmax(energy.squeeze(1), dim=1)
        return attn_weights

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        out, hidden = self.gru(x)
        return out, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, attention):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, hidden, encoder_outputs):
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)

        # print(f"x size: {x.size()}")
        # print(f"context size: {context.size()}")
        if x.size(2) != self.hidden_size:
            x = nn.functional.pad(x, (0, self.hidden_size - x.size(2)))
        gru_input = torch.cat((x, context), 2)
        # print(f"gru_input size: {gru_input.size()}")
        
        out, hidden = self.gru(gru_input, hidden)
        out = out.squeeze(1)
        context = context.squeeze(1)
        out = self.fc(torch.cat((out, context), 1))
        return out, hidden

class CONVGRUA_SIMPLE(nn.Module):
    def __init__(self, encoder, decoder, conv_output_dim, dense_output_dim,device,num_classes=4):
        super(CONVGRUA_SIMPLE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
       
        decoder_output_size = decoder.fc.out_features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=decoder_output_size, out_channels=conv_output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_output_dim, out_channels=conv_output_dim, kernel_size=3, padding=1),
            nn.ReLU()
)
        self.dense_layers = nn.Sequential(
            nn.Linear(conv_output_dim, dense_output_dim),
            nn.LayerNorm(dense_output_dim),
            nn.ReLU(),
            nn.Linear(dense_output_dim, decoder_output_size)
        )
        
        self.fc2 = nn.Linear(decoder_output_size, num_classes)  
        
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, src, seq_len=10):
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, seq_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        # Initial decoder input (start token or zeros)
        decoder_input = torch.zeros(batch_size, 1, self.decoder.hidden_size).to(self.device)
        
        for t in range(seq_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            # Use the current output as the next input
            decoder_input = output.unsqueeze(1)

        outputs = torch.mean(outputs, dim=1)
        # outputs = outputs.transpose(1, 2).contiguous()

        # print(f"outputs size before conv: {outputs.size()}")
        outputs = self.conv_layers(outputs.unsqueeze(2)).squeeze(2)
        # outputs = self.conv_layers(outputs).transpose(1, 2)
        # print(f"outputs size after conv: {outputs.size()}")

        outputs = self.dense_layers(outputs)
        outputs = self.fc2(outputs) 
        # Apply softmax to get probability distribution
        outputs = self.softmax(outputs)
        return outputs
