import torch
from escnn import gspaces, group, nn
from escnn.group import directsum


def build_representation(groupspace, o3: bool, K: int):
    assert K >= 0

    if K == 0:
        return groupspace.trivial_repr

    group = groupspace.fibergroup
    if o3:
        polynomials = [groupspace.trivial_repr, group.irrep(l=1, j=1)]
    else:
        polynomials = [groupspace.trivial_repr, group.irrep(l=1)]

    for k in range(2, K + 1):
        if o3:
            polynomials.append(polynomials[-1].tensor(group.irrep(l=1, j=1)))
        else:
            polynomials = [groupspace.trivial_repr, group.irrep(l=1)]
    final_rep = directsum(polynomials, name=f"polynomial_{K}")
    return final_rep


def pull_escnn_params(params: dict, channels_in: list, channels_out: list = []):
    rep_list_out, rep_list_in = [], []
    params_keys = params.keys()
    print("keys read from escnn: " + str(params_keys))
    if "escnn_group" not in params_keys:
        params["escnn_group"] = "so3"
    if "flips_r3" not in params_keys:
        params["flips_r3"] = True
    if "max_freq" not in params_keys:
        params["max_freq"] = 4
    if "scalar" not in params_keys:
        params["scalar"] = False

    # if params["escnn_group"] == "o3":
    #    g = group.o3_group(params["max_freq"])
    # else:
    #    g = group.so3_group(params["max_freq"])

    if params["flips_r3"]:
        print("flips enabled")
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=params["max_freq"])
    else:
        print("flips disabled")
        gspace = gspaces.rot3dOnR3(maximum_frequency=params["max_freq"])

    if params["scalar"]:
        input_out_reps = [build_representation(gspace, o3=params["flips_r3"], K=0)]
    else:
        input_out_reps = 3 * [build_representation(gspace, o3=params["flips_r3"], K=0)]

    rep_list_in.append(nn.FieldType(gspace, input_out_reps))
    # feat_type_in = nn.FieldType(gspace, input_out_reps)

    for i in range(len(channels_in)):
        # print(i)
        rep_list_in.append(
            nn.FieldType(
                gspace,
                channels_in[i]
                * [
                    build_representation(
                        gspace, o3=params["flips_r3"], K=params["l_max"]
                    )
                ],
            )
        )

    if channels_out == []:
        # print(rep_list_in)
        return group, gspace, rep_list_in

    else:
        for i in range(len(channels_out)):
            rep_list_out.append(
                nn.FieldType(
                    gspace,
                    channels_out[i]
                    * [
                        build_representation(
                            gspace, o3=params["flips_r3"], K=params["l_max"]
                        )
                    ],
                )
            )
        # print(rep_list_out)
        return group, gspace, rep_list_in, rep_list_out


def set_enviroment():
    from torch.multiprocessing import set_start_method

    torch.set_float32_matmul_precision("high")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass


def hyperparameter_dicts(image_size: int = 21):
    assert image_size == 21 or image_size == 51, "image size must be 21 or 51"

    dict_ret = {}

    dict_escnn = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[1300, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "reconstruction_loss": {
            "values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]
        },
        "padding": {"values": [0]},
    }

    dict_esvae = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "beta": {"values": [0.001, 0.01, 1, 10, 100]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "reconstruction_loss": {
            "values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]
        },
        "padding": {"values": [0]},
    }

    dict_vae = {
        "initializer": {"values": ["equi_var", "kaiming"]},
        "beta": {"values": [0.001, 0.01, 1, 10, 100]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
    }

    dict_auto = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
    }

    dict_cnn_supervised = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "irreps": {"values": [None]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.001,
            "max": 0.05,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
        "padding_mode": {"values": ["zeros"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [3]},
        "activation": {"values": ["relu"]},
        "log_wandb": {"values": [True]},
        "gradient_clip_val": {"values": [0.5, 1.0, 5.0, 10.0]},
        "accumulate_grad_batches": {"values": [1, 2, 4, 8]},
        "standardize": {"values": [False, True]},
        "lower_filter": {"values": [False, True]},
        "log_scale": {"values": [False, True]},
        "min_max_scale": {"values": [False, True]},
        "wrangle_outliers": {"values": [False, True]},
        "scalar": {"values": [False]},
    }

    dict_escnn_supervised = {
        "initializer": {"values": ["xavier", "kaiming"]},
        "irreps": {"values": [None]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.001,
            "max": 0.05,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
        "padding_mode": {"values": ["zeros"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [3]},
        "activation": {"values": ["relu"]},
        "log_wandb": {"values": [True]},
        "gradient_clip_val": {"values": [0.5, 1.0, 5.0, 10.0]},
        "accumulate_grad_batches": {"values": [1, 2, 4, 8]},
        "standardize": {"values": [False, True]},
        "lower_filter": {"values": [False, True]},
        "log_scale": {"values": [False, True]},
        "min_max_scale": {"values": [False, True]},
        "wrangle_outliers": {"values": [False, True]},
        "lr_patience": {"values": [10, 20, 50]},
        "lr_decay_factor": {"values": [0.1, 0.5, 0.8]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "scalar": {"values": [False]},
    }

    if image_size == 21:
        dict_cnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [3, 256, 128, 128],
                    "kernel_size_in": [9, 9, 3],
                    "stride_in": [1, 1, 3],
                    "max_pool": False,
                    "max_pool_kernel_size_in": [0],
                    "max_pool_loc_in": [0],
                    "padding": [0],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 32, 64, 128],
                    "kernel_size_in": [4, 5, 4],
                    "stride_in": [1, 1, 1],
                    "max_pool": True,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [1, 3],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 32, 32, 64, 64, 256],
                    "kernel_size_in": [5, 5, 5, 5, 5],
                    "stride_in": [1, 1, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_escnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [256, 128, 128],
                    "kernel_size_in": [9, 9, 3],
                    "stride_in": [1, 1, 3],
                    "max_pool": False,
                    "max_pool_kernel_size_in": [0],
                    "max_pool_loc_in": [0],
                    "padding": [0],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [32, 64, 128],
                    "kernel_size_in": [4, 5, 4],
                    "stride_in": [1, 1, 1],
                    "max_pool": True,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [1, 3],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [32, 64, 128, 256, 512],
                    "kernel_size_in": [5, 5, 5, 5, 5],
                    "stride_in": [1, 1, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_auto["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_vae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_esvae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

    else:
        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_auto["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_vae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_escnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [256, 512, 1024],
                    "kernel_size_in": [7, 7, 3],
                    "stride_in": [3, 3, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [64, 128, 128, 256],
                    "kernel_size_in": [7, 7, 5, 5],
                    "stride_in": [3, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [64, 128, 128, 256],
                    "kernel_size_in": [12, 9, 5],
                    "stride_in": [2, 2, 2],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_cnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [3, 256, 512, 1024],
                    "kernel_size_in": [7, 7, 3],
                    "stride_in": [3, 3, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 64, 128, 128, 256],
                    "kernel_size_in": [7, 7, 5, 5],
                    "stride_in": [3, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 64, 128, 128, 256],
                    "kernel_size_in": [12, 9, 5],
                    "stride_in": [2, 2, 2],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

    dict_ret["escnn"] = dict_escnn
    dict_ret["esvae"] = dict_esvae
    dict_ret["auto"] = dict_auto
    dict_ret["vae"] = dict_vae
    dict_ret["cnn_supervised"] = dict_cnn_supervised
    dict_ret["escnn_supervised"] = dict_escnn_supervised

    return dict_ret
