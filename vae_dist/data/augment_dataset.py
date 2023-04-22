import os
import numpy as np
from vae_dist.dataset.fields import mat_pull
from vae_dist.data.augmentation import Augment


def write_mat(
        mat:np.ndarray, 
        filename:str, 
        start_line:str="", 
        spacing:float=0.3):
    # write seven empty lines

    with open(filename, "w") as f:
        for i in range(6):
            if start_line != "" and i == 0:
                f.write(start_line)
            f.write("\n")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(mat.shape[2]):
                    # round to 2 decimal places
                    i_ = np.around(i - np.floor(mat.shape[0] / 2), decimals=4)
                    j_ = np.around(j - np.floor(mat.shape[1] / 2), decimals=4)
                    k_ = np.around(k - np.floor(mat.shape[2] / 2), decimals=4)

                    f.write(
                        "{:.3f} {:.3f} {:.3f} {:.6f} {:.6f} {:.6f}\n".format(
                            i_ * spacing,
                            j_ * spacing,
                            k_ * spacing,
                            mat[i][j][k][0],
                            mat[i][j][k][1],
                            mat[i][j][k][2],
                        )
                    )


def main():
    # input_dir
    # output_dir
    aug_obj = Augment(xy=True, z=False, rot=-1)
    input_dir = "../../data/cpet/"
    output_dir = "../../data/cpet_augment/"

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate through files in input directory ending in .dat
    for ind, file in enumerate(os.listdir(input_dir)):
        if file.endswith(".dat"):
            meta_data = mat_pull(input_dir + file, meta_data=True, offset=0)
            mat = mat_pull(input_dir + file, offset=0)
            # print(meta_data)
            print("original shape: ", mat.shape)
            mat_aug = aug_obj(
                mat.reshape(
                    [
                        1,
                        meta_data["steps_x"],
                        meta_data["steps_y"],
                        meta_data["steps_z"],
                        3,
                    ]
                )
            )
            # save augmented data to output directory as .dat file
            # print("aug len: ", len(mat_aug))
            for i in range(len(mat_aug)):
                # print(meta_data)
                mat_aug_reshape = mat_aug[i].reshape(
                    [
                        meta_data["steps_x"],
                        meta_data["steps_y"],
                        meta_data["steps_z"],
                        3,
                    ]
                )
                # print(mat_aug_reshape.shape)
                write_mat(
                    mat_aug_reshape,
                    output_dir
                    + file.split(".")[0]
                    + "_"
                    + str(i)
                    + "."
                    + file.split(".")[1],
                    start_line=meta_data["first_line"],
                    spacing=meta_data["step_size_x"],
                )


main()
