import torch
import math
from torch import nn


class UpConvBatch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        output_padding,
        output_layer=False,
    ):
        super(UpConvBatch, self).__init__()
        if output_layer:
            # activation = nn.Sigmoid()
            activation = nn.Identity()
            self.up = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                    output_padding=output_padding,
                ),
                nn.BatchNorm3d(out_channels),
                activation,
            )

        else:
            activation = nn.LeakyReLU(0.2, inplace=False)

            self.up = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                    output_padding=output_padding,
                ),
                activation,
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        return self.up(x)


class ConvBatch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    ):
        super(ConvBatch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=False),
            nn.BatchNorm3d(out_channels),
        )

    def weights_init_normal(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.conv(x)


class ResNetBatch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    ):
        super(ResNetBatch, self).__init__()
        conv_only1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        batch1 = nn.BatchNorm3d(out_channels)
        relu1 = nn.ReLU(inplace=False)

        conv_only2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        batch2 = nn.BatchNorm3d(out_channels)
        nn.LeakyReLU(0.2, inplace=False)

        layers = [conv_only1, batch1, relu1, conv_only2, batch2, relu2]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x) + x
