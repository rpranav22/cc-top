import pdb
import torch
import torch.nn as nn

ACTIVATION = nn.ReLU

class ResNet(nn.Module):
    """
    """
    def __init__(self,
                out_dim,
                latent_dim,
                in_channel,
                **kwargs):
        super(ResNet, self).__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.in_channel = in_channel
        self.bottleneck = True

        hidden_channels = [16, 32, 64, 128]
        kernel_sizes = [5, 3, 3, 3]
        strides = [2, 2, 2, 2]
        padding = [2, 1, 1, 1]
        conv_block = conv2d_bn_block

        layers = []
        layers.append(conv_block(in_channels=self.in_channel,
                                 out_channels=hidden_channels[0],
                                 kernel=kernel_sizes[0],
                                 stride=strides[0],
                                 padding=padding[0]))

        layers.append(ResidualBlock(hidden_channels[0]))
        layers.append(conv_block(in_channels=hidden_channels[0],
                                 out_channels=hidden_channels[1],
                                 kernel=kernel_sizes[0],
                                 stride=strides[0],
                                 padding=padding[1]))

        layers.append(ResidualBlock(hidden_channels[1]))
        layers.append(conv_block(in_channels=hidden_channels[1],
                       out_channels=hidden_channels[2],
                       kernel=kernel_sizes[2],
                       stride=strides[2],
                       padding=padding[2]))

        if self.bottleneck:
            layers.append(ResidualBottleneckBlock(hidden_channels[2]))
        else:
            layers.append(ResidualBlock(hidden_channels[2]))
        layers.append(conv_block(in_channels=hidden_channels[2],
                                 out_channels=hidden_channels[3],
                                 kernel=kernel_sizes[3],
                                 stride=strides[3],
                                 padding=padding[3]))

        if self.bottleneck:
            layers.append(ResidualBottleneckBlock(hidden_channels[3]))
        else:
            layers.append(ResidualBlock(hidden_channels[3]))
        layers.append(conv_block(in_channels=hidden_channels[3],
                                 out_channels=4,
                                 kernel=1,
                                 stride=1,
                                 padding=0))

        self.main = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.latent_dim, out_features=self.out_dim)
        # self.linear = nn.Linear(528, out_features=self.out_dim)

    def forward(self, x):
        out = self.main(x)
        out = self.flatten(out)
        out = self.linear(out)

        return out

def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block linear-bn-activation
    '''
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )

def conv2d_bn_block(in_channels, out_channels, kernel=3, stride=1, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        activation(),
        nn.BatchNorm2d(out_channels, momentum=momentum),
    )


class ResidualBlock(nn.Module):
    """Residual block architecture."""

    def __init__(self, in_channels: int):
        """Initialize module."""
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block architecture."""

    def __init__(self, in_channels: int, bottleneck_filters: int = 64):
        """Initialize module."""
        super(ResidualBottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_filters,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_filters)
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_filters,
            out_channels=bottleneck_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_filters)
        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_filters,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += x
        out = self.relu(out)

        return out