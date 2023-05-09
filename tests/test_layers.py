import torch
from vae_dist.core.layers import UpConvBatch, ConvBatch
from vae_dist.core.escnnlayers import R3Upsampling, MaskModule3D
from vae_dist.core.parameters import build_representation, pull_escnn_params


def test_mask_module_3d():
    # generate random input

    params = {
        "escnn_group": "so3",
        "flips_r3": True,
        "max_freq": 4,
        "scalar": False,
        "l_max": 3,
    }
    channels_in = [3]
    channels_out = []
    group, gspace, rep_list_in = pull_escnn_params(params, channels_in, channels_out)
    print(rep_list_in)
    # build mask
    mask_layer = MaskModule3D(in_type=rep_list_in[0], S=21, margin=0.0)
    input = torch.rand(1, 3, 21, 21, 21)
    input_escnn = rep_list_in[0](input)
    mask_out = mask_layer(input_escnn)
    print("masked!")
    assert mask_out.tensor.shape == input.shape
    print("shapes match!")


def main():
    # throw non implemented error
    test_mask_module_3d()


main()
