import argparse
import copy
import os
import random
from random import shuffle

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import wandb

from focal_loss import focal_loss
import mesh_util, model_util

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_' and not 'training' in i and not 'args' in i]

def run_training(args, cuda, train_dl, test_dl, network):

    device = "cuda:{}".format(cuda)
    model, wandb_name = eval(network)
    model_props = props(model)
    run_path = args.run_path
    run_path += f"/{wandb_name}_b0_{args.b0}"

    for p in model_props:
        run_path += f"_{p}_{getattr(model, p)}"

    img_path = run_path + '/images/'
    if not os.path.isdir(img_path):
        os.makedirs(img_path, exist_ok=True)

    model = model.to(device)

    wandb.init(project=args.exp_name, name=wandb_name)

    wandb.config.update({'cuda': cuda, 'b0': args.b0})
    for p in model_props:
        wandb.config.update({p: getattr(model, p)})

    print("#params", sum(x.numel() for x in model.parameters()))

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.config.update({'alpha': args.alpha, 'gamma': args.gamma})
    criterion = focal_loss(alpha=args.alpha, gamma=args.gamma, device=device)

    prediction = model_util.train_model(args.iter, model,
                                        criterion, optimiser,
                                        train_dl, test_dl,
                                        run_path, args.clip,
                                        device)
    wandb.finish()
    return

def run_voxelcnn(args):
    wandb_name = "voxel_cnn"

    model = model_util.ModelVoxel(args)

    return model, wandb_name

def run_classicalcnn(args):
    wandb_name = "classical_cnn"

    model = model_util.ClassicalCNN(args)

    return model, wandb_name

def run_augment_classicalcnn(args, device, full_group=True):
    wandb_name = "augment_classical"
    if full_group:
        wandb_name += "_full"
    
    model = model_util.ClassicalAugmentCNN(args, device=device, full_group=full_group)
    return model, wandb_name

def run_steerable_gcnn(args, device, full_group=True):
    wandb_name = "steerable_spatial"
    if full_group:
        wandb_name += "_full"
    model =  model_util.ModelSpatialSteer(args, device=device, full_group=full_group)

    return model, wandb_name

def run_steerable_param_gcnn(args, device, full_group=True):
    wandb_name = "steerable_spatial_param"
    if full_group:
        wandb_name += "_full"
    model =  model_util.ModelSpatialSteerParamCompare(args, device=device, full_group=full_group)

    return model, wandb_name

def run_spatial_gcnn(args):
    wandb_name = "unsteerable_spatial"
    model =  model_util.ModelSpatial(args)

    return model, wandb_name

def read_scans(path, scans):
    data = []
    lbs = []
    inds = []
    all_coords = []
    for s in tqdm(scans):
        d_path = f"{path}/{s}"
        df = np.load(d_path)
        val = df['data']
        labels = df['labels']
        inv_inds = df['inds']
        coords = df['vox_coords']
        data.append(torch.Tensor(val))
        lbs.append(torch.Tensor((labels)))
        inds.append(torch.Tensor(inv_inds))
        all_coords.append(coords)

    return data, lbs, inds, all_coords


def main():

    # TODO: maybe we don't need so many arguments. Keep the necessary ones and delete the others.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/renfei/Documents/HCP')
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--model_capacity", type=str, default='small')
    parser.add_argument("--b_size", type=int, default=100)
    parser.add_argument("--num_rays", type=int, default=5)
    parser.add_argument("--ray_len", default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--samples_per_ray", type=int, default=2)
    parser.add_argument("--watson_param", type=int, default=10)
    parser.add_argument("--bias", default=True, action='store_false')
    parser.add_argument("--lin_bias", default=True, action='store_false')
    parser.add_argument("--spatial_bias", default=True, action='store_false')
    parser.add_argument("--lin_bn", default=True, action='store_false')
    parser.add_argument("--alpha", nargs='+', type=float, default=[0.35, 0.35, 0.15, 0.15])
    parser.add_argument("--gamma", type=float, default=2.)
    parser.add_argument("--num_shells", type=int, default=1)
    parser.add_argument("--b0", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default='testnow')     # the experiment name in wandb.
    parser.add_argument("--run_path", type=str, default='testnow')		# the name of the folder that is going to be generated later.
    parser.add_argument("--spatial_kernel_size", nargs='+', type=int, default=[3, 3, 3])
    parser.add_argument("--interpolate", default=False, action='store_true')    # if true, the program runs GCNN, if false, the program runs classical CNN.
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--data_aug", default=False, action='store_true')	# if true, data augmentation is applied to the training set.
    parser.add_argument("--aug_grid", type=int, default=7)
    parser.add_argument("--network", type=str, default='ours_full')    # if the network chosen is a classical CNN of any kind, --interpolate should be set to False.

    args = parser.parse_args()

    # args.ray_len is the radius of the spherical kernel. If it is not given, it is assumed that the radius is the arc length
    # of an icosahedral edge, and args.ray_len is set to None.
    if not type(args.ray_len)==float:
        args.ray_len = None

    rotations = args.num_rays

    data_path = f"{args.path}/data_aligned/"
    data_folder = os.listdir(data_path)

    if not args.interpolate:
        folder_name = f"classical_grid_size_{args.aug_grid}"
    else:
        folder_name = f"num_rays_{args.num_rays}_samples_per_ray_{args.samples_per_ray}_ray_len_{args.ray_len}_watson_{args.watson_param}_grid_size_{args.grid_size}"
    print(folder_name in data_folder)
    
    data_path_train = data_path + folder_name
    if args.data_aug:
        data_path_train = f"{data_path}classical_grid_size_{args.aug_grid}"
    data_path += folder_name
    print(data_path_train, data_path)

    scans = os.listdir(data_path)
    train = ['100206']
    test = ['100408']

    train_scans = [s for s in scans if train[0] in s]
    test_scans = [s for s in scans if test[0] in s]
 
    data, lbs, inds, vox_coords = read_scans(data_path_train, train_scans)
    train_len = [d.shape[1] for d in data]
    data = torch.cat(data, dim=1)
    lbs = torch.cat(lbs)
    inds = torch.cat(inds)

    test_data, test_labels_sep, test_inds, test_vox_coords = read_scans(data_path, test_scans)
    test_len = [d.shape[1] for d in test_data]
    test_data = torch.cat(test_data, dim=1)
    test_inds = torch.cat(test_inds)
    test_labels = torch.cat(test_labels_sep)

    if args.num_shells == 1:
        b0_idx = (args.b0 // 1000) - 1

        if not args.interpolate:
            b_data = data[b0_idx, ...].permute(1, 0)
            test_b_data = test_data[b0_idx, ...].permute(1, 0)
        else:
            if not args.data_aug:
                b_data = data[b0_idx, ...].permute(1, 2, 0).unsqueeze(-2)
            else:
                b_data = data[b0_idx, ...].permute(1, 0)
            test_b_data = test_data[b0_idx, ...].permute(1, 2, 0).unsqueeze(-2)

    else:
        if not args.interpolate:
            b_data = data[:args.num_shells, ...].permute(0, 2, 1)
            n = b_data.shape[-1]
            b_data = b_data.reshape(-1, n)

            test_b_data = test_data[:args.num_shells, ...].permute(0, 2, 1)
            n = test_b_data.shape[-1]
            test_b_data = test_b_data.reshape(-1, n)

        else:
            if not args.data_aug:
                b_data = data[:args.num_shells, ...].permute(2, 3, 0, 1)
            else:
                b_data = data[:args.num_shells, ...].permute(0, 2, 1)
                n = b_data.shape[-1]
                b_data = b_data.reshape(-1, n)
            test_b_data = test_data[:args.num_shells, ...].permute(2, 3, 0, 1)

    if args.data_aug:
        vertices, edge_rings, faces = mesh_util.get_icosahedron_aligned()
        manifold_coords = mesh_util.get_sphere_points(vertices, edge_rings, args.samples_per_ray, args.ray_len).float()  # 12 x 11 x 3
        
        bvecs = []
        for n in train:
            bvec_path = data_path_train.replace(f"data_aligned/classical_grid_size_{args.aug_grid}", f"{n}/T1w/Diffusion/bvecs")
            bvec = np.loadtxt(bvec_path)

            bvals_path = bvec_path.replace('/bvecs', '/bvals')
            bvals = np.loadtxt(bvals_path)

            shell = np.abs(bvals - args.b0) < 30
            bvec = bvec[:, shell]
            bvecs.append(bvec)
        bvecs = torch.from_numpy(np.stack(bvecs))
        binds = torch.cat([torch.ones(train_len[i]) * i for i in range(len(train_len))]).long()
        train_set = model_util.DataCubeRotation(b_data, lbs, inds, args, manifold_coords, binds,
                                                bvecs, device=f"cuda:{args.cuda}")
    else:
        train_set = model_util.DataSpatial(b_data, lbs, inds)
    test_ds = model_util.DataSpatial(test_b_data, test_labels, test_inds)

    train_dl = DataLoader(train_set, batch_size=args.b_size, num_workers=0, shuffle=True)

    test_dl = DataLoader(test_ds, batch_size=200, num_workers=4)

    if args.network == 'ours_full':
    	assert args.interpolate
    	run_training(args, args.cuda, train_dl, test_dl, "run_steerable_gcnn(args, device, True)")
    elif args.network == 'ours_part':
    	assert args.interpolate
    	run_training(args, args.cuda, train_dl, test_dl, "run_steerable_gcnn(args, device, False)")
    elif args.network == 'classical':
    	assert args.interpolate == False
    	run_training(args, args.cuda, train_dl, test_dl, "run_classicalcnn(args)")
    elif args.network == 'ours_decoupled':
    	assert args.interpolate
    	run_training(args, args.cuda, train_dl, test_dl, "run_spatial_gcnn(args)")
    elif args.network == 'baseline':
    	assert args.interpolate
    	run_training(args, args.cuda, train_dl, test_dl, "run_voxelcnn(args)")
    elif args.network == 'classical_augment_full':
    	assert args.interpolate == False
    	run_training(args, args.cuda, train_dl, test_dl, "run_augment_classicalcnn(args, device, True)")
    elif args.network == 'classical_augment_part':
    	assert args.interpolate == False
    	run_training(args, args.cuda, train_dl, test_dl, "run_augment_classicalcnn(args, device, False)")
    elif args.network == 'ours_compare':
    	assert args.interpolate
    	run_training(args, args.cuda, train_dl, test_dl, "run_steerable_param_gcnn(args, device, True)")
    else:
    	raise ValueError('Invalid network!')

if __name__ == '__main__':
    main()