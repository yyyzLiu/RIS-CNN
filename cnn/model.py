"""RIS CNN architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNetAttention(nn.Module):
    def __init__(self, config):
        super(UNetAttention, self).__init__()

        in_channels = config.get('in_channels')
        out_channels = config.get('out_channels')
        features = config.get('features')
        dropout_p = config.get('dropout_rate')

        self.downs = nn.ModuleList()
        self.dropouts_down = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            layer = nn.Sequential(
                nn.Conv2d(prev_channels, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SEBlock(feature),
                # nn.Dropout2d(p=dropout_p),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(p=dropout_p),
                SEBlock(feature),
                nn.Dropout2d(p=dropout_p)
            )
            self.downs.append(layer)
            prev_channels = feature

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(features[-1] * 2),
            # nn.Dropout2d(p=dropout_p),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=dropout_p),
            SEBlock(features[-1] * 2),
            nn.Dropout2d(p=dropout_p)
        )

        self.ups = nn.ModuleList()
        self.conv_ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    SEBlock(feature),  # Adding SE block
                )
            )
            self.conv_ups.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SEBlock(feature),
                # nn.Dropout2d(p=dropout_p),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SEBlock(feature),
                nn.Dropout2d(p=dropout_p)
            ))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        skip_connections = reversed(skip_connections)
        for up, conv_up, skip in zip(self.ups, self.conv_ups, skip_connections):
            x = up(x)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat((x, skip), dim=1)
            x = conv_up(x)

        return self.final_conv(x)


