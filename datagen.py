import argparse
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm

import mesh_util
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_spatial_blocks(name, image_raw, all_labels, bvecs, bvals, brain_mask, num_shells, manifold_coords, args):
    """
    Function to process and save the data and labels needed for training.
    """

    # labels interpreted from free surfer
    group1 = [4, 5, 14, 15, 24, 43, 44, 72]
    group2 = [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60]
    group3 = [251, 252, 253, 254, 255]

    # create label image
    label_mask = np.zeros(all_labels.shape)
    label_mask[np.isin(all_labels, group1)] = 1
    label_mask[np.isin(all_labels, group2)] = 2
    label_mask[np.isin(all_labels, group3)] = 3
    label_mask[np.logical_and(all_labels >= 1000, all_labels < 3000)] = 4
    label_mask[all_labels >= 3000] = 3
    
    # resample label image to the resolution of the data
    i_vals, j_vals, k_vals = np.meshgrid(range(brain_mask.shape[0]), range(brain_mask.shape[1]), range(brain_mask.shape[2]), indexing='ij')
    downSamplingFactors = np.array(label_mask.shape) / np.array(brain_mask.shape)
    i_vals = i_vals / (1 / downSamplingFactors[0])
    j_vals = j_vals / (1 / downSamplingFactors[1])
    k_vals = k_vals / (1 / downSamplingFactors[2])
    down_mask = map_coordinates(label_mask, np.array([i_vals, j_vals, k_vals]), order=0)
    down_mask = torch.from_numpy(down_mask * brain_mask)
    
    # create bounding boxes for each voxel for data voxel grid generation
    vox_coords = down_mask.nonzero()
    bb_window = (args.grid_size - 1) // 2
    vox_bb_lu = vox_coords - bb_window
    vox_bb_rd = vox_coords + bb_window
    
    # get all labels in integer numbers
    all_labels = (down_mask[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]] - 1).numpy().astype(int)
    
    # get all grids for all voxels of interest
    all_coords = []
    for lu, rd in tqdm(zip(vox_bb_lu, vox_bb_rd)):
        i = torch.arange(lu[0], (rd[0] + 1))
        j = torch.arange(lu[1], (rd[1] + 1))
        k = torch.arange(lu[2], (rd[2] + 1))
        try:
            # New versions of torch assumes xy indexing.
            gi, gj, gk = torch.meshgrid(i, j, k, indexing='ij')
        except TypeError as e:
            # Old versions of torch does not have the indexing argument and assumes ij indexing.
            gi, gj, gk = torch.meshgrid(i, j, k)
        coords = torch.stack([gi, gj, gk], dim=-1)
        all_coords.append(coords)
    all_coords = torch.stack(all_coords)
    n, x, y, z, _ = all_coords.shape

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    # get rid of repeating ones to save storage space
    all_coords = all_coords.contiguous().reshape(-1, 3).to(device)    # it runs way faster on gpu...

    # get inverse indices of the repeating ones for later recovery of the grid structure
    unique_coords, inv_inds = torch.unique(all_coords, return_inverse=True, dim=0)
    unique_coords = unique_coords.cpu()
    inv_inds = inv_inds.cpu()
    inv_inds = inv_inds.contiguous().reshape(n, x, y, z).numpy()

    # normalize images with b0
    b0_mask = bvals==5
    b0_img = np.mean(image_raw[:, :, :, b0_mask], axis=-1, keepdims=True)
    val = image_raw[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], :]
    b0_val = b0_img[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], :]
    val = val / b0_val
    val[np.isnan(val)] = 1
    val[np.isinf(val)] = 1
    
    # interpolate with watson kernel
    watson_inter = []
    for i in range(num_shells):
        shell = np.abs(bvals - 1000 * (i + 1)) < 30
        bvecs_val = bvecs[shell]
        v = val[..., shell]
        if not args.interpolate:
            watson_inter.append(v)
        else:
            w = watson_interpolation(torch.from_numpy(v), torch.from_numpy(bvecs_val), manifold_coords.reshape(-1, 3), args.watson_param).reshape(len(v), *manifold_coords.shape[:-1]).numpy() # N x 12 x 11
            watson_inter.append(w)
    watson_inter = np.stack(watson_inter)   # 3 x N x 12 x 11

    # save data
    data_path = f"{args.path}/data_aligned"
    if not args.interpolate:
        folder_name = f"classical_grid_size_{args.grid_size}"
    else:
        folder_name = f"num_rays_{args.num_rays}_samples_per_ray_{args.samples_per_ray}_ray_len_{args.ray_len}_watson_{args.watson_param}_grid_size_{args.grid_size}"
    data_path = f"{data_path}/{folder_name}"
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
    data_path = f"{data_path}/{name}"
    print(data_path)
    np.savez_compressed(data_path, data=watson_inter, labels=all_labels, inds=inv_inds, vox_coords=vox_coords.numpy())

    return

def generate_data_blocks(dirs, manifold_coords, args, num_shells=3):
    """
    Function to iterate and process each scan, and save the processed data from each scan.
    """
    for i in tqdm(range(len(dirs))):
        p = dirs[i]
        print(i, p)
        label_path = f"{args.path}/{p}/T1w"
        img_path = f"{args.path}/{p}/T1w/Diffusion"
        img = nib.load(f"{img_path}/data.nii.gz").get_fdata()
        brain_mask = nib.load(f"{img_path}/nodif_brain_mask.nii.gz").get_fdata()
        wm = nib.load(f"{label_path}/wmparc.nii.gz").get_fdata()
        bvals = np.loadtxt(f"{img_path}/bvals")
        bvecs = np.loadtxt(f"{img_path}/bvecs").T

        get_spatial_blocks(p, img, wm, bvecs, bvals, brain_mask, num_shells, manifold_coords, args)
    return

def watson_interpolation(vals, vn, v, k):
    """
    Function to interpolate spherical signals using a watson kernel.
    
    # params:
        vals: N x 90
        vn: 90 x 3
        v: M x 3
    """

    mat = torch.exp(k * torch.mm(vn, v.T) ** 2)    # 90 x M
    sums = torch.sum(mat, dim=0)                   # 1 x M
    ds_norm = mat / sums                           # 90 x M
    ds = torch.einsum("ij,ki->kij", (ds_norm, vals)) # N x 90 x M
    res = torch.sum(ds, dim=1)    # N x M
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/renfei/Documents/HCP')
    parser.add_argument("--num_rays", type=int, default=5)
    parser.add_argument("--ray_len", default=None)
    parser.add_argument("--samples_per_ray", type=int, default=2)
    parser.add_argument("--watson_param", type=int, default=10)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--interpolate", default=False, action='store_true')
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    # args.ray_len is the radius of the spherical kernel. If it is not given, it is assumed that the radius is the arc length
    # of an icosahedral edge, and args.ray_len is set to None.
    if not type(args.ray_len) == float:
        args.ray_len = None

    path = args.path
    watson_param = args.watson_param
    num_rays = args.num_rays
    ray_len = args.ray_len
    samples_per_ray = args.samples_per_ray

    dirs = os.listdir(path)
    dirs = [d for d in dirs if not 'data' in d]

    # To generate the processed data for all scans, comment the line bellow and uncomment the two lines above.
    #dirs = ['100206', '100408']
    vertices, edge_rings, faces = mesh_util.get_icosahedron_aligned()

    manifold_coords = mesh_util.get_sphere_points(vertices, edge_rings, samples_per_ray, ray_len)  # 12 x 11 x 3

    generate_data_blocks(dirs, manifold_coords, args)
    
    return

if __name__ == '__main__':
    main()