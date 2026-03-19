import torch
import torch.nn as nn

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dense_dim, output_dim, dropout_rate=0.0):
        super(LSTMStockPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.fc1 = nn.Linear(hidden_dim2, dense_dim)
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # We only want the output of the last time step
        # out shape: (batch_size, seq_length, hidden_dim2)
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.fc2(out)
        return out