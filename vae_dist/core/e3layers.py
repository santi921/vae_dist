from torch import nn
import torch
from functools import partial
from e3nn import o3
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.nn.models.v2103.voxel_convolution import LowPassFilter
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.math import soft_unit_step, soft_one_hot_linspace
from e3nn import o3
import math


class Convolution(torch.nn.Module):
    r"""convolution on voxels
    Parameters
    ----------
    irreps_in : `Irreps`
        input irreps
    irreps_out : `Irreps`
        output irreps
    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``
    diameter : float
        diameter of the filter in physical units
    num_radial_basis : int
        number of radial basis functions
    steps : tuple of float
        size of the pixel in physical units
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter,
        num_radial_basis,
        steps=(1.0, 1.0, 1.0),
        cutoff=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        self.register_buffer("lattice", lattice)

        if "padding" not in kwargs:
            kwargs["padding"] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=cutoff,
        )
        self.register_buffer("emb", emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh, x=lattice, normalize=True, normalization="component"
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer("sh", sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(
            torch.randn(self.num_radial_basis, self.tp.weight_numel)
        )

    def kernel(self):
        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(
            self.sh, weight
        )  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum("xyzio->oixyz", kernel)
        return kernel

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        return sc + torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return x


class ConvolutionVoxel(torch.nn.Module):
    r"""convolution on voxels
    Parameters
    ----------
    irreps_in : `Irreps`
    irreps_out : `Irreps`
    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``
    size : int
    steps : tuple of int
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        size,
        steps=(1, 1, 1),
        cutoff=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.size = size
        self.num_rbfs = self.size

        if "padding" not in kwargs:
            kwargs["padding"] = self.size // 2
        self.kwargs = kwargs

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)

        # connection with neighbors
        r = torch.linspace(-1, 1, self.size)
        x = r * steps[0] / min(steps)
        x = x[x.abs() <= 1]
        y = r * steps[1] / min(steps)
        y = y[y.abs() <= 1]
        z = r * steps[2] / min(steps)
        z = z[z.abs() <= 1]
        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=1.0,
            number=self.num_rbfs,
            basis="smooth_finite",
            cutoff=cutoff,
        )
        self.register_buffer("emb", emb)

        sh = o3.spherical_harmonics(
            self.irreps_sh, lattice, True, "component"
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer("sh", sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False
        )

        self.weight = torch.nn.Parameter(
            torch.randn(self.num_rbfs, self.tp.weight_numel)
        )

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        weight = self.emb @ self.weight
        weight = weight / (self.size ** (3 / 2))
        kernel = self.tp.right(
            self.sh, weight
        )  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum("xyzio->oixyz", kernel)
        return sc + 0.1 * torch.nn.functional.conv3d(x, kernel, **self.kwargs)


class LowPassFilter(torch.nn.Module):
    def __init__(self, scale, stride=1, transposed=False, steps=(1, 1, 1)):
        super().__init__()

        sigma = 0.5 * (scale**2 - 1) ** 0.5

        size = int(1 + 2 * 2.5 * sigma)
        if size % 2 == 0:
            size += 1

        r = torch.linspace(-1, 1, size)
        x = r * steps[0] / min(steps)
        x = x[x.abs() <= 1]
        y = r * steps[1] / min(steps)
        y = y[y.abs() <= 1]
        z = r * steps[2] / min(steps)
        z = z[z.abs() <= 1]
        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        lattice = (size // 2) * lattice

        kernel = torch.exp(-lattice.norm(dim=-1).pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        if transposed:
            kernel = kernel * stride**3
        kernel = kernel[None, None]
        self.register_buffer("kernel", kernel)

        self.scale = scale
        self.stride = stride
        self.size = size
        self.transposed = transposed

    def forward(self, image):
        """
        Parameters
        ----------
        image : `torch.Tensor`
            tensor of shape ``(..., x, y, z)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., x, y, z)``
        """
        if self.scale <= 1:
            assert self.stride == 1
            return image

        out = image
        out = out.reshape(-1, 1, *out.shape[-3:])
        if self.transposed:
            out = torch.nn.functional.conv_transpose3d(
                out, self.kernel, padding=self.size // 2, stride=self.stride
            )
        else:
            out = torch.nn.functional.conv3d(
                out, self.kernel, padding=self.size // 2, stride=self.stride
            )
        out = out.reshape(*image.shape[:-3], *out.shape[-3:])
        return out


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        input,
        irreps_hidden,
        activation,
        irreps_sh,
        normalization,
        kernel_size,
        dropout_prob,
        cutoff,
        num_radial_basis,
    ):
        super().__init__()

        if normalization == "None":
            BN = Identity
        elif normalization == "batch":
            BN = BatchNorm
        elif normalization == "instance":
            BN = partial(BatchNorm, instance=True)

        irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])
        fe = sum(mul for mul, ir in irreps_gated if ir.p == 1)
        fo = sum(mul for mul, ir in irreps_gated if ir.p == -1)
        irreps_gates = Irreps(f"{fe}x0e+{fo}x0o").simplify()

        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid, torch.tanh][: len(activation)]

        self.gate1 = Gate(
            irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated
        )
        self.conv1 = Convolution(
            input,
            self.gate1.irreps_in,
            irreps_sh,
            kernel_size,
            cutoff=cutoff,
            num_radial_basis=num_radial_basis,
        )
        self.batchnorm1 = BN(self.gate1.irreps_in)
        self.dropout1 = Dropout(self.gate1.irreps_out, dropout_prob)

        self.gate2 = Gate(
            irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated
        )
        self.conv2 = Convolution(
            self.gate1.irreps_out,
            self.gate2.irreps_in,
            irreps_sh,
            kernel_size,
            cutoff=cutoff,
            num_radial_basis=num_radial_basis,
        )
        self.batchnorm2 = BN(self.gate2.irreps_in)
        self.dropout2 = Dropout(self.gate2.irreps_out, dropout_prob)

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)
        return x


class Down(nn.Module):
    def __init__(
        self,
        n_blocks_down,
        activation,
        irreps_sh,
        ne,
        no,
        BN,
        input,
        kernel_size,
        down_op,
        scale,
        stride,
        dropout_prob,
        cutoff,
        num_radial_basis,
    ):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_blocks_down + 1):
            irreps_hidden = Irreps(
                f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o"
            ).simplify()
            block = ConvolutionBlock(
                input,
                irreps_hidden,
                activation,
                irreps_sh,
                BN,
                kernel_size,
                dropout_prob,
                cutoff,
                num_radial_basis,
            )
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            input = block.irreps_out
            ne *= 2
            no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        # change to pooling
        if down_op == "lowpass":
            self.pool = LowPassFilter(scale, stride=stride)
        elif down_op == "maxpool3d":
            self.pool = nn.MaxPool3d(scale, stride=stride)
        elif down_op == "average":
            self.pool = nn.AvgPool3d(scale, stride=stride)

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            ftrs.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.pool(x)
        return ftrs


class Up(nn.Module):
    def __init__(
        self,
        n_blocks_up,
        activation,
        irreps_sh,
        ne,
        no,
        BN,
        downblock_irreps,
        kernel_size,
        up_op,
        scale,
        stride,
        dropout_prob,
        scalar_upsampling,
        cutoff,
        num_radial_basis,
    ):
        super().__init__()

        self.n_blocks_up = n_blocks_up
        if up_op == "lowpass":
            self.upsamp = LowPassFilter(scale, stride=stride, transposed=True)
        else:
            self.upsamp = nn.Upsample(
                scale_factor=scale, mode="trilinear", align_corners=True
            )

        input = downblock_irreps[-1]
        blocks = []

        for n in range(n_blocks_up):
            if scalar_upsampling:
                irreps_hidden = Irreps(f"{8*ne}x0e+{8*no}x0o").simplify()
            else:
                irreps_hidden = Irreps(
                    f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o"
                ).simplify()

            block = ConvolutionBlock(
                input + downblock_irreps[::-1][n + 1],
                irreps_hidden,
                activation,
                irreps_sh,
                BN,
                kernel_size,
                dropout_prob,
                cutoff,
                num_radial_basis,
            )
            blocks.append(block)
            input = block.irreps_out
            ne //= 2
            no //= 2

        self.up_blocks = nn.ModuleList(blocks)

    def forward(self, x, down_features):
        for i in range(self.n_blocks_up):
            x = self.upsamp(x)
            x = torch.cat([x, down_features[::-1][i + 1]], dim=1)
            x = self.up_blocks[i](x)
        return x
