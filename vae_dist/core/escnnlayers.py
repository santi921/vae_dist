import torch
import math

import torch
import numpy as np
from torch.nn.functional import interpolate

from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules.equivariant_module import EquivariantModule

# from escnn.gspaces import GSpace

from escnn.nn import GeometricTensor
from escnn.nn import FieldType
from typing import Tuple, Optional, Union


# __all__ = ["R2Upsampling"]


class R3Upsampling(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        scale_factor: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        r"""

        Wrapper for :func:`torch.nn.functional.interpolate`. Check its documentation for further details.

        Only ``"bilinear"`` and ``"nearest"`` methods are supported.
        However, ``"nearest"`` is not equivariant; using this method may result in broken equivariance.
        For this reason, we suggest to use ``"bilinear"`` (default value).

        .. warning ::
            The module supports a ``size`` parameter as an alternative to ``scale_factor``.
            However, the use of ``scale_factor`` should be *preferred*, since it guarantees both axes are scaled
            uniformly, which preserves rotation equivariance.
            A misuse of the parameter ``size`` can break the overall equivariance, since it might scale the two axes by
            two different factors.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.


        Args:
            in_type (FieldType): the input field type
            size (optional, int or tuple): output spatial size.
            scale_factor (optional, int): multiplier for spatial size
            mode (str): algorithm used for upsampling: ``nearest`` | ``bilinear``. Default: ``bilinear``
            align_corners (bool): if ``True``, the corner pixels of the input and output tensors are aligned, and thus
                    preserving the values at those pixels. This only has effect when mode is ``bilinear``.
                    Default: ``False``

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 3

        super(R3Upsampling, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        assert (
            size is None or scale_factor is None
        ), f'Only one of "size" and "scale_factor" can be set, but found scale_factor={scale_factor} and size={size}'

        self._size = (size, size) if isinstance(size, int) else size
        assert self._size is None or (
            isinstance(self._size, tuple) and len(self._size) == 2
        ), self._size
        self._scale_factor = scale_factor
        self._mode = mode
        self._align_corners = align_corners if mode != "nearest" else None

        if mode not in ["nearest", "bilinear"]:
            raise ValueError(
                f"Error Upsampling mode {mode} not recognized! Mode should be `nearest` or `bilinear`."
            )

    def forward(self, input: GeometricTensor):
        r"""

        Args:
            input (torch.Tensor): input feature map

        Returns:
             the result of the convolution

        """

        assert input.type == self.in_type
        # assert len(input.shape) == 4

        if self._align_corners is None:
            output = interpolate(
                input.tensor,
                size=self._size,
                scale_factor=self._scale_factor,
                mode=self._mode,
            )
        else:
            output = interpolate(
                input.tensor,
                size=self._size,
                scale_factor=self._scale_factor,
                mode=self._mode,
                align_corners=self._align_corners,
            )

        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return super().evaluate_output_shape(input_shape)


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, s, dtype=dtype)
    c = (s - 1) / 2
    t = (c - margin / 100.0 * c) ** 2
    sig = 2.0
    for x in range(s):
        for y in range(s):
            for z in range(s):
                r = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                if r > t:
                    mask[..., x, y, z] = math.exp((t - r) / sig**2)
                else:
                    mask[..., x, y, z] = 1.0
    return mask


class MaskModule3D(EquivariantModule):
    def __init__(self, in_type: FieldType, S: int, margin: float = 0.0):
        r"""

        Performs an element-wise multiplication of the input with a *mask* of shape ``S x S x S``.

        The mask has value :math:`1` in all pixels with distance smaller than ``(S-1)/2 * (1 - margin)/100`` from the
        center of the mask and :math:`0` elsewhere. Values change smoothly between the two regions.

        This operation is useful to remove from an input image or feature map all the part of the signal defined on the
        pixels which lay outside the circle inscribed in the grid.
        Because a rotation would move these pixels outside the grid, this information would anyways be
        discarded when rotating an image. However, allowing a model to use this information might break the guaranteed
        equivariance as rotated and non-rotated inputs have different information content.


        .. note::

            In order to perform the masking, the module expects an input with the same spatial dimensions as the mask.
            Then, input tensors must have shape ``B x C x S x S x S``.


        Args:
            in_type (FieldType): input field type
            S (int): the shape of the mask and the expected inputs
            margin (float, optional): margin around the mask in percentage with respect to the radius of the mask
                                      (default ``0.``)

        """
        super(MaskModule3D, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(
            build_mask(S, margin=margin), requires_grad=False
        )

        self.in_type = self.out_type = in_type

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        assert input.tensor.shape[2:] == self.mask.shape[2:]
        assert input.tensor.shape[3:] == self.mask.shape[3:]
        assert input.tensor.shape[4:] == self.mask.shape[4:]

        out = input.tensor * self.mask
        return GeometricTensor(out, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape
