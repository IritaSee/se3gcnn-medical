import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

count = 0

def get_cube_rotation_inds(img_size):
    return np.random.randint(60, size=img_size)

def save_rotation_inds(path):
    dirs = os.listdir(path)
    save_path = path.replace('classical_grid_size_7', 'ico_rotation_inds')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    for d in dirs:
        if d.endswith('.npz'):
            img_size = len(np.load(f"{path}/{d}")['labels'])
            inds = get_cube_rotation_inds(img_size)
            save_name = f"{save_path}/{d.replace('.npz', '.npy')}"
            np.save(save_name, inds)
            global count
            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/renfei/Documents/HCP')
    args = parser.parse_args()
    path = f"{args.path}/data_aligned/classical_grid_size_7"
    save_rotation_inds(path)

    global count
    print(count)
if __name__ == '__main__':
    main()