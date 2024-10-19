import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()

        self.layer_nums = len(hidden_layer_size)
        self.hidden_layer_size = hidden_layer_size

        print(self.layer_nums)

        if self.layer_nums <= 0:
            raise RuntimeError("provided hidden_layer_size can not be zero")

        self.layers = [nn.LSTM(input_size, hidden_layer_size[0], batch_first=True)]

        for i in range(1, self.layer_nums, 1):
            self.layers.append(
                nn.LSTM(hidden_layer_size[i - 1], hidden_layer_size[i], batch_first=True)
            )

        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Linear(hidden_layer_size[-1], output_size)

    def forward(self, x):
        for i, lstm_layer in enumerate(self.layers):
            h0 = torch.zeros(1, x.size(0), self.hidden_layer_size[i]).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_layer_size[i]).to(x.device)
            x, _ = lstm_layer(x, (h0, c0))

        out = self.fc(x[:, -1, :])
        return out


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x).view(x.size(0), -1)
        # Excite
        y = self.fc(y).view(x.size(0), x.size(1), 1)
        return x * y.expand_as(x)


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MLSTMFCN(nn.Module):
    def __init__(self, input_size, conv_filters, kernel_sizes, lstm_hidden_sizes, output_size):
        super(MLSTMFCN, self).__init__()

        # Validate input sizes
        if len(conv_filters) < 3:
            raise ValueError("conv_filters must contain at least three elements for the three convolutional layers.")
        if len(lstm_hidden_sizes) == 0:
            raise ValueError("lstm_hidden_sizes must contain at least one element.")

        # Convolutional branch - dynamically create convolutional layers
        self.conv_branch = nn.Sequential()
        input_channels = input_size
        for i, (out_channels, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_branch.add_module(f'conv1d_{i + 1}', Conv1DBlock(input_channels, out_channels, kernel_size))
            if i < len(conv_filters) - 1:  # Add Squeeze-and-Excite blocks except for the last convolutional layer
                self.conv_branch.add_module(f'se_{i + 1}', SqueezeExciteBlock(out_channels))
            input_channels = out_channels

        # LSTM branch - dynamically create LSTM layers
        self.lstm_branch = nn.ModuleList()
        for i, hidden_size in enumerate(lstm_hidden_sizes):
            lstm_input_size = input_size if i == 0 else lstm_hidden_sizes[i - 1]
            self.lstm_branch.append(nn.LSTM(lstm_input_size, hidden_size, batch_first=True))
        self.dropout = nn.Dropout(0.5)

        # Fully connected output layer
        final_lstm_output_size = lstm_hidden_sizes[-1]
        self.fc = nn.Linear(conv_filters[-1] + final_lstm_output_size, output_size)

    def forward(self, x):
        # Convolutional branch
        conv_x = x.transpose(1, 2)  # Conv1d expects (batch, channels, sequence)
        conv_x = self.conv_branch(conv_x)
        conv_x = torch.mean(conv_x, dim=2)  # Global Average Pooling (along the sequence dimension)

        # LSTM branch
        lstm_x = x
        for lstm_layer in self.lstm_branch:
            h0 = torch.zeros(1, lstm_x.size(0), lstm_layer.hidden_size).to(lstm_x.device)
            c0 = torch.zeros(1, lstm_x.size(0), lstm_layer.hidden_size).to(lstm_x.device)
            lstm_x, _ = lstm_layer(lstm_x, (h0, c0))
        lstm_x = self.dropout(lstm_x[:, -1, :])  # We take only the last output

        # Concatenation of both branches
        combined_x = torch.cat((conv_x, lstm_x), dim=1)

        # Fully connected layer
        out = self.fc(combined_x)
        return out