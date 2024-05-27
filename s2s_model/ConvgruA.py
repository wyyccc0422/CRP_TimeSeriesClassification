import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Ensure the number of channels after the division is at least 1
        reduced_channels = max(in_channels // ratio, 1)
        
        self.fc1 = nn.Conv1d(in_channels, reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv1d(reduced_channels, in_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        fc_out = self.relu(self.fc1(avg_out))
        fc_out = self.fc2(fc_out)
        y = self.sigmoid(fc_out)
        return x * y


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
    
    def forward(self, x):
        out1, hidden1 = self.gru1(x)
        out2, hidden2 = self.gru2(out1)
        out3, hidden3 = self.gru3(out2)
        return out1, out2, out3, hidden3


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.gru1 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size*2, hidden_size, 1, batch_first=True)
        self.gru3 = nn.GRU(hidden_size*2, hidden_size, 1, batch_first=True)
        self.self_attention = SelfAttention(input_dim=hidden_size*2)
        self.hidden_size = hidden_size

    def forward(self, context, encoder_outputs1, encoder_outputs2, hidden, seq_len):
        outputs = []
        input_step = context
        
        for t in range(seq_len):
            gru_out1, hidden = self.gru1(input_step, hidden)
            
            # SKIP CONNECTION1: Concatenate encoder's second output with the output from the first GRU in decoder
            concat_input = torch.cat((gru_out1, encoder_outputs2[:, t:t+1]), dim=-1)
            
            # Apply attention to the concatenated input
            attention_output1 = self.self_attention(concat_input)
            
            # Second GRU in the decoder
            gru_out2, hidden = self.gru2(attention_output1, hidden)
            
            # SKIP CONNECTION2: Concatenate encoder's first output with the output from the second GRU in decoder
            concat_input2 = torch.cat((gru_out2, encoder_outputs1[:, t:t+1]), dim=-1)
            
            # Apply attention to the concatenated input
            attention_output2 = self.self_attention(concat_input2)

            # Third GRU in the decoder
            gru_out3, hidden = self.gru3(attention_output2, hidden)
            outputs.append(gru_out3)
            input_step = gru_out3  # Use the current output as the next input
        
        
        outputs = torch.cat(outputs, dim=1)  # Concatenate along the time dimension
        return outputs, hidden


class DAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size)
        self.self_attention = SelfAttention(input_dim=hidden_size)


    def forward(self, x):
        encoder_outputs1, encoder_outputs2, encoder_final_output, hidden = self.encoder(x)
        context = self.self_attention(encoder_final_output)
        
        seq_len = x.size(1)
        decoder_output, hidden = self.decoder(context, encoder_outputs1, encoder_outputs2, hidden, seq_len)
        return decoder_output, hidden


class CONVGRUA(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3,num_class=4):
        super(CONVGRUA, self).__init__()
        self.dae = DAE(input_size, hidden_size)
        self.self_attention = SelfAttention(input_dim=hidden_size)
        self.channel_attention = ChannelAttention(in_channels=1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),  # Changed to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size)
        )
        self.fc = nn.Linear(hidden_size, num_class)  
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size = x.shape[0] 
        dae_output, hidden = self.dae(x)
       
        # Apply Channel Attention to the decoder output
        attention_output = self.self_attention(dae_output)
        
        # 1D Convolutional layer
        conv_output = self.conv1d(attention_output).squeeze(1)
        dense_output = self.dense_layers(conv_output)

        outputs = self.fc(dense_output)
        outputs = self.softmax(outputs)

        return outputs
