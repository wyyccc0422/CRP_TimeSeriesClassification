import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, inputs):
        weights = torch.matmul(inputs, self.attention_weights).squeeze(2)
        weights = F.softmax(weights, dim=1).unsqueeze(2)
        output = torch.sum(inputs * weights, dim=1)
        return output

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.self_attention = SelfAttention(hidden_size * 2)  # *2 for bidirectional

    def forward(self, x):
        outputs, _ = self.bigru(x)
        outputs = self.self_attention(outputs)
        return outputs
        

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.adjust_dim = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, encoder_outputs):
        adjusted_encoder_outputs = self.adjust_dim(encoder_outputs)
        output, hidden = self.gru(x)
        output = self.attention(output + adjusted_encoder_outputs)  
        output = self.fc(output)
        return output

class BiGRUAttn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=4):
        super(BiGRUAttn, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.norm = nn.LayerNorm(hidden_size) 
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(x, encoder_outputs)
        
  
        residual = decoder_outputs
        decoder_outputs = F.relu(self.fc1(decoder_outputs))
        decoder_outputs = self.norm(decoder_outputs + residual)  
        decoder_outputs = self.fc2(decoder_outputs) 
        decoder_outputs = self.softmax(decoder_outputs)
        return decoder_outputs


