import torch
import math
import torch
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *
from torch import nn
from typing import Tuple, Union


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
        output_layer: bool = False,
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


class ResBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        channels: int,
        out_type: FieldType = None,
        stride: int = 1,
        features: str = "2_96",
    ):
        super(ResBlock, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if features == "ico":
            L = 2
            grid = {"type": "ico"}
        elif features == "2_96":
            L = 2
            grid = {"type": "thomson_cube", "N": 4}
        elif features == "2_72":
            L = 2
            grid = {"type": "thomson_cube", "N": 3}
        elif features == "3_144":
            L = 3
            grid = {"type": "thomson_cube", "N": 6}
        elif features == "3_192":
            L = 3
            grid = {"type": "thomson_cube", "N": 8}
        elif features == "3_160":
            L = 3
            grid = {"type": "thomson", "N": 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))

        # We try to keep the width of the model approximately constant
        _channels = channels / S
        _channels = int(round(_channels))

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = FourierELU(
            self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid
        )
        res_type = ftelu.in_type

        print(
            f"ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}"
        )

        self.res_block = SequentialModule(
            R3Conv(
                in_type,
                res_type,
                kernel_size=3,
                padding=1,
                bias=False,
                initialize=False,
            ),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(
                res_type,
                self.out_type,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
                initialize=False,
            ),
        )

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased3D(in_type, 0.33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R3Conv(
                self.in_type, self.out_type, kernel_size=1, padding=0, bias=False
            )
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):
        assert input.type == self.in_type
        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        else:
            return input_shape
