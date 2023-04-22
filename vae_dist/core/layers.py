import torch
import math
import torch


class UpConvBatch(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        padding_mode: str,
        output_padding: int,
        output_layer: bool=False,
    ):
        super(UpConvBatch, self).__init__()
        if output_layer:
            # activation = nn.Sigmoid()
            activation = torch.nn.Identity()
            self.up = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(
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
                torch.nn.BatchNorm3d(out_channels),
                activation,
            )

        else:
            activation = torch.nn.LeakyReLU(0.2, inplace=False)

            self.up = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(
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
                torch.nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        return self.up(x)


class ConvBatch(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        padding_mode: str,
    ):
        super(ConvBatch, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(
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
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.BatchNorm3d(out_channels),
        )

    def weights_init_normal(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.conv(x)


class ResNetBatch(torch.nn.Module):
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
        conv_only1 = torch.nn.Conv3d(
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
        batch1 = torch.nn.BatchNorm3d(out_channels)
        relu1 = torch.nn.ReLU(inplace=False)

        conv_only2 = torch.nn.Conv3d(
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
        batch2 = torch.nn.BatchNorm3d(out_channels)
        relu2 = torch.nn.LeakyReLU(0.2, inplace=False)

        layers = [conv_only1, batch1, relu1, conv_only2, batch2, relu2]
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x) + x
