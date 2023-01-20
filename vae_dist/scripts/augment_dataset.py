import os 
import numpy as np 
from vae_dist.dataset.fields import mat_pull
from vae_dist.data.augmentation import Augment


def write_mat(mat, filename, start_line='', spacing = 0.3):
    # write seven empty lines 

    with open(filename, 'w') as f:
        for i in range(7):
            if (start_line != '' and i == 0):
                f.write(start_line)
            f.write("\n")
        #print(mat.shape)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(mat.shape[2]):
                    # round to 2 decimal places
                    i_ = np.around(i - np.floor(mat.shape[1]/2), decimals=2)
                    j_ = np.around(j - np.floor(mat.shape[2]/2), decimals=2)
                    k_ = np.around(k - np.floor(mat.shape[3]/2), decimals=2)

                    f.write("{:.3f} {:.3f} {:.3f} {:.6f} {:.6f} {:.6f}\n".format(i_*spacing, j_*spacing, k_*spacing, mat[0][i][j][k], mat[1][i][j][k], mat[2][i][j][k]))            
                    #f.write(str(mat[i][j][k]) + " ")
            
def main():     
    # input_dir 
    # output_dir
    aug_obj = Augment(xy = True, z = True)
    input_dir = "../../data/cpet/"
    output_dir = "../../data/cpet_augmented_flips/"
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_line = "#Sample Density: 10 10 10; Volume: Box: 3.000000 3.000000 3.000000"
    # iterate through files in input directory ending in .dat
    for file in os.listdir(input_dir):
        if file.endswith(".dat"):
            mat, shape = mat_pull(input_dir + file)
            mat_aug = aug_obj(mat.reshape([1, 3, shape[0], shape[1], shape[2]]))
            # save augmented data to output directory as .dat file
            for i in range(len(mat_aug)):
                write_mat(
                    mat_aug[i], 
                    output_dir + file.split(".")[0] + "_" + str(i) + "." + file.split(".")[1],
                    start_line = start_line)
            
    

main()