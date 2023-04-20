from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor


from escnn.nn.modules.equivariant_module import EquivariantModule
from typing import Tuple, Optional, Union

import torch
import numpy as np

import math

from torch.nn.functional import interpolate

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
