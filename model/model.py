from torch import nn
import torch

# Stage 1 Teacher Model
class Net1D(torch.nn.Module):
    '''
    shallow cnn network
    Args:
        in_channels : 1 (lead-ii)
        num_classes : 이진 분류
    Returns:
        fc(x)
    '''
    def __init__(self, in_channels: int, num_classes: int) -> None:
        # TODO: you can modify this model if you want
        super(Net1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.mean(x, dim=-1)
        return self.fc(x)

# Stage 2 : Student Model
class ResNet_1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.maxpool = nn.MaxPool1d(2)
        self.downsampling = downsampling

    def forward(self, x):
        identity = self.downsampling(x)
        out = self.relu(self.bn1(x))
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out += identity
        return out

class ECGNet(nn.Module):
    '''
    https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs 참고
    기본 kernel = [3,5,7,9]
    Args:
        in_channels : 1 (lead-ii)
        num_classes : 이진 분류
    Returns:
        fc(x)
    '''
    def __init__(self, kernels, in_channels=1, fixed_kernel_size=17, num_classes=1):
        super().__init__()
        self.planes = 24
        self.parallel_conv = nn.ModuleList([
            nn.Conv1d(in_channels, self.planes, k, 1, 0, bias=False) for k in kernels
        ])
        self.bn1 = nn.BatchNorm1d(self.planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(self.planes, self.planes, fixed_kernel_size, 2, 2, bias=False)
        self.block = self._make_resnet_layer(fixed_kernel_size, 1, 9, fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(self.planes)
        self.avgpool = nn.AvgPool1d(6, 6, 2)
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(328, num_classes)

    def _make_resnet_layer(self, kernel_size, stride, blocks, padding):
        layers = []
        for _ in range(blocks):
            downsampling = nn.Sequential(nn.MaxPool1d(2))
            layers.append(ResNet_1D_Block(self.planes, self.planes, kernel_size, stride, padding, downsampling))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.parallel_conv], dim=2)
        out = self.relu(self.bn1(out))
        out = self.conv1(out)
        out = self.block(out)
        out = self.relu(self.bn2(out))
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]
        new_out = torch.cat([out, new_rnn_h], dim=1)

        return self.fc(new_out)