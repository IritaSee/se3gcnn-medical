import copy
import random
from random import shuffle

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from torch.nn.modules import Module
import torch.nn.functional as F
import wandb

from mesh_util import compute_rotations
from equiv_test import watson_interpolation

class Lifting(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, perm, use_bias=True):
        """
        Lifting layer for GCNN

        :param nfeature_in: number of shells
        :param nfeature_out: number of output features
        :param kernel_size: size of each kernel
        :param perm: permutations of the rotational kernel
        """
        super(Lifting, self).__init__()
        self.kernel_size = kernel_size
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.perm = perm
        self.rotations = len(perm)
        self.kernel = torch.nn.Parameter(torch.empty(kernel_size, nfeature_in, nfeature_out))
        self.use_bias = use_bias
        torch.nn.init.xavier_normal_(self.kernel)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.ones(1, 1, 1, nfeature_out, 1, 1, 1)*0.01)
        else:
            self.register_parameter(name="bias", param=None)

    def forward(self, data):  # pylint: disable=W
        """
        data: batch_size x 12 x 11 x shell x 7 x 7 x 7
        """

        # rotate the kernels by permutation
        kernel_size, f_in, f_out = self.kernel.shape     # 11, f_in, f_out
        kernel_center = self.kernel[:1, ...]    # 1 x f_in x f_out
        kernel_rot = self.kernel[1:, ...]       # 10 x f_in x f_out
        kernel_rot = kernel_rot.reshape(self.rotations, (self.kernel_size-1)//self.rotations, f_in, f_out)       # 5 x 2 x f_in x f_out
        kernel_rot = kernel_rot[np.concatenate(self.perm), ...].view(self.rotations, -1, f_in, f_out)     # 5 x 10 x f_in x f_out
        kernel_center = kernel_center.repeat(self.rotations, 1, 1).view(self.rotations, -1, f_in, f_out)  # 5 x 1 x f_in x f_out
        kernel_all = torch.cat([kernel_center, kernel_rot], dim=1)                                      # 5 x 11 x f_in x f_out

        rot_response = torch.tensordot(data, kernel_all, dims=([2, 3], [1, 2]))                    # batch_size x 12 x 7 x 7 x 7 x 5 x f_out
        inds = list(np.arange(len(rot_response.shape)))
        new_inds = [*inds[:2], *inds[-2:], *inds[2:-2]]
        rot_response = rot_response.permute(*new_inds)                                                  # batch_size x 12 x 5 x f_out x 7 x 7 x 7
        if self.use_bias:
            rot_response += self.bias
        return rot_response

class SteerableSpatialConv(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, rotation_matrices, use_bias=True, device="cuda:0"):
        super(SteerableSpatialConv, self).__init__()
        """
        Steerable spatial convolution layer

        :param nfeature_in:        input channels
        :param nfeature_out:       output channels
        :param kernel_size:        kernel size for 3d spatial convolution
        :param rotation_matrices:  rotation matrices aligned with parallel transport
        """

        # initialize params
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.kernel = torch.nn.Parameter(torch.empty(nfeature_out*nfeature_in, *kernel_size))
        torch.nn.init.xavier_normal_(self.kernel)
        self.kernel_size = kernel_size

        # make rotation matrices affine-like
        rotation_matrices = torch.cat([rotation_matrices, torch.zeros([*list(rotation_matrices.shape[:-1]), 1])], dim=-1).float()     # 12 x 5 x 3 x 4
        m, n = rotation_matrices.shape[-2:]
        rotation_matrices = rotation_matrices.reshape(-1, m, n)                                 # 60 x 3 x 4

        out_size = torch.Size((rotation_matrices.shape[0], *list(self.kernel.shape)))           # 60, f_out*f_in, 3, 3, 3

        # for later interpolation
        self.inter_grid = F.affine_grid(rotation_matrices, out_size).to(device)
        self.num_rot = self.inter_grid.shape[0]
        if use_bias:
            self.bias = torch.nn.Parameter(torch.ones(self.nfeature_out)*0.01)
        else:
            self.bias = None

    def forward(self, data):
        """
        data: batch_size x 12 x 5 x f_in x grid x grid x grid
        """

        batch_size, n_ver, kernel_dim, f_in, x, y, z = data.shape                    # batch, 12, 5, f_in, grid, grid, grid
        kn = self.kernel.repeat(self.num_rot, 1, 1, 1, 1)   # steer x f_out*f_in x 3 x 3 x 3

        # rotate and interpolate the spatial kernel grid  
        rotated_kernel = F.grid_sample(kn, self.inter_grid).reshape(self.num_rot, self.nfeature_out, self.nfeature_in, *self.kernel_size)  # 60, f_out, f_in, 3, 3, 3

        """
        ###################################################################################################
        The following code should be equivalent to:                                                      
        rotated_kernel = rotated_kernel.reshape(n_ver, kernel_dim, self.nfeature_out, self.nfeature_in, *self.kernel_size)                                                                                    
        res = []
        for i in range(nver):
            dd = data[:, i, ...]
            kk = rotated_kernel[i, ...]
            rr = []
            for j in range(kernel_dim):
                d = dd[:, j, ...]
                k = kk[j, ...]
                r = F.conv3d(d, k, bias=self.bias)
                rr.append(r)
            rr = torch.stack(rr, dim=1)
            res.append(rr)
        data = torch.stack(res, dim=1)
        return data
        ###################################################################################################
        """

        rotated_kernel = rotated_kernel.reshape(-1, self.nfeature_in, *self.kernel_size)

        if self.num_rot == 60:
            data = data.reshape(batch_size, n_ver*kernel_dim*f_in, x, y, z) # batch x 12*5*f_in x grid x grid x grid
        elif self.num_rot == 12:
            data = data.transpose(1, 2).reshape(batch_size*kernel_dim, n_ver*f_in, x, y, z)

        data = F.conv3d(data, rotated_kernel, groups=self.num_rot, bias=self.bias.repeat(self.num_rot))
        _, _, out_x, out_y, out_z = data.shape

        if self.num_rot == 60:
            data = data.reshape(batch_size, n_ver, kernel_dim, self.nfeature_out, out_x, out_y, out_z) # batch x 12 x 5 x f_out x grid x grid x grid
        elif self.num_rot == 12:
            data = data.reshape(batch_size, kernel_dim, n_ver, self.nfeature_out, out_x, out_y, out_z).transpose(1, 2)
        return data

class GroupConv(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, perm, use_bias=True):
        """
        Group (fibre) convolution layer

        :param nfeature_in: number of shells
        :param nfeature_out: number of output features
        :param kernel_size: size of each kernel
        :param perm: permutations for rotational kernels
        """
        super(GroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.perm = perm
        self.rotations = len(perm)
        self.kernel = torch.nn.Parameter(torch.empty(kernel_size, nfeature_in, nfeature_out))
        self.use_bias = use_bias
        torch.nn.init.xavier_normal_(self.kernel)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.ones(1, 1, 1, nfeature_out, 1, 1, 1)*0.01)
        else:
            self.register_parameter(name="bias", param=None)

    def forward(self, data):  # pylint: disable=W
        """
        data: batch_size x 12 x 5 x f_in x grid x grid x grid
        """

        kernel_size, f_in, f_out = self.kernel.shape
        kernel_rot = self.kernel[np.concatenate(self.perm), ...].view(self.rotations, -1, f_in, f_out)       # 5 x kernel_size x f_in x f_out
        data = torch.tensordot(data, kernel_rot, dims=([2, 3], [1, 2]))             # batch_size x 12 x grid x grid x grid x 5 x f_out
        inds = list(np.arange(len(data.shape)))
        new_inds = [*inds[:2], *inds[-2:], *inds[2:-2]]

        data = data.permute(*new_inds)                                          # batch_size x 12 x 5 x f_out x grid x grid x grid
        if self.use_bias:
            data += self.bias
        return data

class SpatialConv(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, use_bias=True):
        """
        Classical CNN layer
        """
        super(SpatialConv, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.kernel_size = kernel_size
        self.conv3d = torch.nn.Conv3d(in_channels=nfeature_in, out_channels=nfeature_out, kernel_size=tuple(kernel_size), bias=use_bias)
    
    def forward(self, data):
        batch_size, n_ver, kernel_dim, f_in, x, y, z = data.shape
        data = self.conv3d(data.reshape(-1, f_in, x, y, z))
        out_b, out_f, out_x, out_y, out_z = data.shape
        data = data.reshape(batch_size, n_ver, kernel_dim, out_f, out_x, out_y, out_z)
        return data

class DataSpatial(Dataset):
    def __init__(self, vals, labels, inds, device="cpu"):
        """
        Dataset to load the data.
        """
        self.vals = vals.to(device)
        self.labels = labels.long().to(device)
        self.inds = inds.long()
        v_shape = self.vals.shape
        self.v_shape = list(v_shape[:-1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ind = self.inds[idx]
        x, y, z = ind.shape
        val = self.vals[..., ind.view(-1)].view(*self.v_shape, x, y, z)

        return val, self.labels[idx]

class DataCubeRotation(Dataset):
    def __init__(self, vals, labels, inds, args, manifold_coords=None, binds=None, bvecs=None, device="cpu"):
        """
        Dataset to load the data and randomly rotate them using rotations from the octahedral group.
        """
        self.vals = vals.float().to(device)
        self.labels = labels.long().to(device)
        self.inds = inds.long()
        self.rotations = torch.from_numpy(R.create_group('O').as_matrix())
        self.s2conv = args.interpolate
        self.bvecs = bvecs.float().to(device)
        self.binds = binds
        self.manifold_coords = manifold_coords.to(device)
        self.watson_param = args.watson_param
        self.model_voxel = args.voxel_cnn

        inv_rots = []
        for rot in self.rotations:
            inv_r = np.linalg.inv(rot.numpy())
            inv_rots.append(torch.from_numpy(inv_r))
        self.inv_rotations = torch.stack(inv_rots).float()

        self.rotations = self.rotations.float().to(device)
        rotation_matrices = torch.cat([self.inv_rotations, torch.zeros([*list(self.inv_rotations.shape[:-1]), 1]).float()], dim=-1).float()
        out_size = torch.Size((rotation_matrices.shape[0], self.vals.shape[0], *list(self.inds.shape[1:])))           # 24, 90, 7, 7, 7

        # for later interpolation
        self.inter_grid = F.affine_grid(rotation_matrices, out_size).to(device)  # 24, 7, 7, 7, 3
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ind = self.inds[idx]
        x, y, z = ind.shape

        val_origin = self.vals[:, ind.view(-1)].view(self.vals.shape[0], x, y, z)  # 90, 7, 7, 7
        grid_ind = torch.randint(self.inter_grid.shape[0], (1,))

        if not self.model_voxel:
            grid = self.inter_grid[grid_ind]
            val_rot = F.grid_sample(val_origin.unsqueeze(0), grid).squeeze(0)  # 90, 7, 7, 7
        else:
            val_rot = val_origin[:, 3:4, 3:4, 3:4]    # 90, 1, 1, 1

        bind = self.binds[idx]
        bvec = self.bvecs[bind]
        rot_mat = self.rotations[grid_ind[0]]
        bvec_rot = torch.mm(rot_mat, bvec).T

        if self.s2conv:
            val_rot = watson_interpolation(val_rot, bvec_rot, self.manifold_coords, self.watson_param).unsqueeze(2)   # 12, 11, 1, 7, 7, 7
        else:
            val_rot = watson_interpolation(val_rot, bvec_rot, bvec.T, self.watson_param)   # 90, 7, 7, 7
        return val_rot, self.labels[idx]
        
class Projection(Module):
    def __init__(self, dim=1):
        super(Projection, self).__init__()
        """
        Projection layer
        """
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim, keepdim=True).values

class MeanPooling(Module):
    def __init__(self, dim=1):
        super(MeanPooling, self).__init__()
        """
        Projection layer
        """
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)

class ClassicalLifting(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, rotation_matrices, use_bias=True, device="cuda:0"):
        super(ClassicalLifting, self).__init__()
        """
        Steerable convolution layer for classical CNN setup

        :param nfeature_in:        input channels
        :param nfeature_out:       output channels
        :param kernel_size:        kernel size for 3d spatial convolution
        :param rotation_matrices:  rotation matrices to discretize SO(3)
        """

        # initialize params
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.kernel = torch.nn.Parameter(torch.empty(nfeature_out*nfeature_in, *kernel_size))
        torch.nn.init.xavier_normal_(self.kernel)
        self.kernel_size = kernel_size

        # make rotation matrices affine-like
        rotation_matrices = torch.cat([rotation_matrices, torch.zeros([*list(rotation_matrices.shape[:-1]), 1])], dim=-1).float()     # 12 x 5 x 3 x 4
        m, n = rotation_matrices.shape[-2:]
        rotation_matrices = rotation_matrices.reshape(-1, m, n)                                 # 60 x 3 x 4

        out_size = torch.Size((rotation_matrices.shape[0], *list(self.kernel.shape)))           # 60, f_out*f_in, 3, 3, 3

        # for later interpolation
        self.inter_grid = F.affine_grid(rotation_matrices, out_size).to(device)
        self.num_rot = self.inter_grid.shape[0]
        if use_bias:
            self.bias = torch.nn.Parameter(torch.ones(self.nfeature_out)*0.01)
        else:
            self.bias = None

    def forward(self, data):
        """
        data: batch_size x f_in x grid x grid x grid
        """

        batch_size, f_in, x, y, z = data.shape                    # batch, f_in, grid, grid, grid
        kn = self.kernel.repeat(self.num_rot, 1, 1, 1, 1)   # 60 x f_out*f_in x 3 x 3 x 3

        # rotate and interpolate the spatial kernel grid  
        rotated_kernel = F.grid_sample(kn, self.inter_grid).reshape(self.num_rot, self.nfeature_out, self.nfeature_in, *self.kernel_size)  # 60, f_out, f_in, 3, 3, 3

        """
        ###################################################################################################
        The following code should be equivalent to:                                                                                                                                       
        res = []
        for i in range(self.num_rot):
            k = rotated_kernel[i]
            r = F.conv3d(data, k, bias=self.bias)
            res.append(r)
        data = torch.stack(res, dim=1)
        return data
        ###################################################################################################
        """

        rotated_kernel = rotated_kernel.reshape(-1, self.nfeature_in, *self.kernel_size) # 60*f_out, f_in, 3, 3, 3
        data = F.conv3d(data, rotated_kernel, bias=self.bias.repeat(self.num_rot)) # batch x 60*f_out x grid x grid x grid
        _, _, out_x, out_y, out_z = data.shape
        data = data.reshape(batch_size, self.num_rot, self.nfeature_out, out_x, out_y, out_z) # batch x 60 x f_out x grid x grid x grid

        return data

class ClassicalAugmentconv(Module):
    def __init__(self, nfeature_in, nfeature_out, kernel_size, rotation_matrices, use_bias=True, device="cuda:0"):
        super(ClassicalAugmentconv, self).__init__()
        """
        Steerable convolution layer for classical CNN setup

        :param nfeature_in:        input channels
        :param nfeature_out:       output channels
        :param kernel_size:        kernel size for 3d spatial convolution
        :param rotation_matrices:  rotation matrices to discretize SO(3)
        """

        # initialize params
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.kernel = torch.nn.Parameter(torch.empty(nfeature_out*nfeature_in, *kernel_size))
        torch.nn.init.xavier_normal_(self.kernel)
        self.kernel_size = kernel_size

        # make rotation matrices affine-like
        rotation_matrices = torch.cat([rotation_matrices, torch.zeros([*list(rotation_matrices.shape[:-1]), 1])], dim=-1).float()     # 12 x 5 x 3 x 4
        m, n = rotation_matrices.shape[-2:]
        rotation_matrices = rotation_matrices.reshape(-1, m, n)                                 # 60 x 3 x 4

        out_size = torch.Size((rotation_matrices.shape[0], *list(self.kernel.shape)))           # 60, f_out*f_in, 3, 3, 3

        # for later interpolation
        self.inter_grid = F.affine_grid(rotation_matrices, out_size).to(device)
        self.num_rot = self.inter_grid.shape[0]
        if use_bias:
            self.bias = torch.nn.Parameter(torch.ones(self.nfeature_out)*0.01)
        else:
            self.bias = None

    def forward(self, data):
        """
        data: batch_size x 60 x f_in x grid x grid x grid
        """

        batch_size, num_rot, f_in, x, y, z = data.shape                    # batch, 60, f_in, grid, grid, grid
        kn = self.kernel.repeat(self.num_rot, 1, 1, 1, 1)   # steer x f_out*f_in x 3 x 3 x 3

        # rotate and interpolate the spatial kernel grid  
        rotated_kernel = F.grid_sample(kn, self.inter_grid).reshape(self.num_rot, self.nfeature_out, self.nfeature_in, *self.kernel_size)  # 60, f_out, f_in, 3, 3, 3

        """
        ###################################################################################################
        The following code should be equivalent to:                                                                                                                                       
        res = []
        for i in range(self.num_rot):
            d = data[:, i, ...]
            k = rotated_kernel[i]
            r = F.conv3d(d, k, bias=self.bias)
            res.append(r)
        data = torch.stack(res, dim=1)
        return data
        ###################################################################################################
        """

        rotated_kernel = rotated_kernel.reshape(-1, self.nfeature_in, *self.kernel_size) # 60*f_out, f_in, 3, 3, 3
        data = data.reshape(batch_size, num_rot*f_in, x, y, z) # batch x 60*f_in x grid x grid x grid
        data = F.conv3d(data, rotated_kernel, groups=self.num_rot, bias=self.bias.repeat(self.num_rot)) # batch x 60*f_out x grid x grid x grid
        _, _, out_x, out_y, out_z = data.shape
        data = data.reshape(batch_size, self.num_rot, self.nfeature_out, out_x, out_y, out_z) # batch x 60 x f_out x grid x grid x grid

        return data

def construct_linearlayers(dims, lin_bias, lin_bn):
    linear_layers = []
    for i in range(len(dims) - 1):
        D_in = dims[i]
        D_out = dims[i + 1]
        linear_layers.append(torch.nn.Linear(D_in, D_out, bias=lin_bias))

        if lin_bn and i < (len(dims) - 2):
            linear_layers.append(torch.nn.BatchNorm1d(D_out))

        if i < (len(dims) - 2):
            linear_layers.append(torch.nn.ReLU())
    return linear_layers

class ModelVoxel(torch.nn.Module):
    def __init__(self, args):
        super(ModelVoxel, self).__init__()
        """
        Model without spatial convolution
        """
        perm = [list(np.roll(np.arange(args.num_rays), i)) for i in range(args.num_rays)]
        input_kernel_size = args.num_rays * args.samples_per_ray + 1
        # f_lift = 20
        # f_conv = 40
        if args.model_capacity == 'small':
            f_lift = 1
            f_conv = 5
        else:
            f_lift = 10
            f_conv = 20

        self.num_shells = args.num_shells
        self.ray_len = args.ray_len
        self.b_size = args.b_size
        self.watson_param = args.watson_param
        self.num_epoch = args.iter
        self.lr = args.lr
        self.aug = args.data_aug

        self.f_lift = f_lift
        self.f_conv = f_conv

        self.conv_layers = torch.nn.Sequential(

            Lifting(
                nfeature_in=args.num_shells,
                nfeature_out=f_lift,
                kernel_size=input_kernel_size,
                perm=perm,
                use_bias=args.bias
                ),
                
            torch.nn.ReLU(),

            GroupConv(
                nfeature_in=f_lift,
                nfeature_out=f_conv,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),

            torch.nn.ReLU(),

            Projection(dim=2)

            )
        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f_conv*12, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x

class ModelSpatialSteer(torch.nn.Module):
    def __init__(self, args, device='cuda:0', full_group=True):
        super(ModelSpatialSteer, self).__init__()
        """
        Model with steerable spatial kernels.
        """
        perm = [list(np.roll(np.arange(args.num_rays), i)) for i in range(args.num_rays)]
        _, inv_rotation_matrices = compute_rotations(SO2=full_group)
        inv_rotation_matrices = inv_rotation_matrices.float()

        input_kernel_size = args.num_rays * args.samples_per_ray + 1
        if args.model_capacity == 'small':
            f_lift = 5
            f_conv1, f_conv2, f_conv3 = 5,5,5
            f_spatial1, f_spatial2, f_spatial3 = 5,5,5
        else:
            f_lift = 10
            f_conv1, f_conv2, f_conv3 = 20,40,10
            f_spatial1, f_spatial2, f_spatial3 = 20,40,20

        self.num_shells = args.num_shells
        self.ray_len = args.ray_len
        self.b_size = args.b_size
        self.watson_param = args.watson_param
        self.num_epoch = args.iter
        self.lr = args.lr
        self.pooling = args.pooling
        self.aug = args.data_aug

        self.f_lift = f_lift

        self.f_conv1 = f_conv1
        self.f_conv2 = f_conv2
        self.f_conv3 = f_conv3

        self.f_spatial1 = f_spatial1
        self.f_spatial2 = f_spatial2
        self.f_spatial3 = f_spatial3

        self.lifting_spatial_groupconv = torch.nn.Sequential(

            Lifting(
                nfeature_in=args.num_shells,
                nfeature_out=f_lift,
                kernel_size=input_kernel_size,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_lift,
                nfeature_out=f_spatial1,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU()

            )

        self.spatial_groupconv1 = torch.nn.Sequential(

            GroupConv(
                nfeature_in=f_spatial1,
                nfeature_out=f_conv1,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_conv1,
                nfeature_out=f_spatial2,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU()

            )

        self.spatial_groupconv2 = torch.nn.Sequential(

            GroupConv(
                nfeature_in=f_spatial2,
                nfeature_out=f_conv2,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_conv2,
                nfeature_out=f_spatial3,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            GroupConv(
                nfeature_in=f_spatial3,
                nfeature_out=f_conv3,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU()
            )

        if self.pooling == 'mean':
            print("mean pooling")
            self.projection = [MeanPooling(dim=2), MeanPooling(dim=1)]
        else:
            self.projection = [Projection(dim=2), Projection(dim=1)]

        self.projection = torch.nn.Sequential(*self.projection)


        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f_conv3, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.lifting_spatial_groupconv(x)
        x = self.spatial_groupconv1(x)
        x = self.spatial_groupconv2(x)
        x = self.projection(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x

class ModelSpatial(torch.nn.Module):
    def __init__(self, args):
        super(ModelSpatial, self).__init__()
        """
        Model with unsteerable spatial kernels.
        """
        perm = [list(np.roll(np.arange(args.num_rays), i)) for i in range(args.num_rays)]
        
        input_kernel_size = args.num_rays * args.samples_per_ray + 1

        if args.model_capacity == 'small':
            f_lift = 5
            f_conv1, f_conv2, f_conv3 = 5,5,5
            f_spatial1, f_spatial2, f_spatial3 = 5,5,5
        else:
            f_lift = 10
            f_conv1, f_conv2, f_conv3 = 20,40,10
            f_spatial1, f_spatial2, f_spatial3 = 20,40,20

        self.num_shells = args.num_shells
        self.ray_len = args.ray_len
        self.b_size = args.b_size
        self.watson_param = args.watson_param
        self.num_epoch = args.iter
        self.lr = args.lr
        self.aug = args.data_aug

        self.f_lift = f_lift

        self.f_conv1 = f_conv1
        self.f_conv2 = f_conv2
        self.f_conv3 = f_conv3

        self.f_spatial1 = f_spatial1
        self.f_spatial2 = f_spatial2
        self.f_spatial3 = f_spatial3

        self.lifting_spatial_groupconv = torch.nn.Sequential(

            Lifting(
                nfeature_in=args.num_shells,
                nfeature_out=f_lift,
                kernel_size=input_kernel_size,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SpatialConv(
                nfeature_in=f_lift,
                nfeature_out=f_spatial1,
                kernel_size=args.spatial_kernel_size,
                use_bias=args.spatial_bias
                ),
            torch.nn.ReLU(),

            GroupConv(
                nfeature_in=f_spatial1,
                nfeature_out=f_conv1,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU()

            )

        self.spatial_groupconv1 = torch.nn.Sequential(

            SpatialConv(
                nfeature_in=f_conv1,
                nfeature_out=f_spatial2,
                kernel_size=args.spatial_kernel_size,
                use_bias=args.spatial_bias
                ),
            torch.nn.ReLU(),

            GroupConv(
                nfeature_in=f_spatial2,
                nfeature_out=f_conv2,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU()
            )

        self.spatial_groupconv2 = torch.nn.Sequential(

            SpatialConv(
                nfeature_in=f_conv2,
                nfeature_out=f_spatial3,
                kernel_size=args.spatial_kernel_size,
                use_bias=args.spatial_bias
                ),
            torch.nn.ReLU(),

            GroupConv(
                nfeature_in=f_spatial3,
                nfeature_out=f_conv3,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU()
            )


        self.projection = [Projection(dim=2), Projection(dim=1)]
        self.projection = torch.nn.Sequential(*self.projection)


        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f_conv3, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.lifting_spatial_groupconv(x)
        x = self.spatial_groupconv1(x)
        x = self.spatial_groupconv2(x)
        x = self.projection(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x

class ClassicalCNN(torch.nn.Module):
    def __init__(self, args):
        super(ClassicalCNN, self).__init__()
        """
        Classical CNN model with no geometry involved.
        """
        if args.model_capacity == 'small':
            f1, f2, f3, f4 = 90, 5, 5, 5
        else:
            f1, f2, f3, f4 = 90, 120, 120, 90

        self.num_shells = args.num_shells
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.b_size = args.b_size
        self.num_epoch = args.iter
        self.lr = args.lr
        self.aug = args.data_aug

        self.conv_layers = torch.nn.Sequential(

            torch.nn.Conv3d(
                in_channels=f1,
                out_channels=f2,
                kernel_size=tuple(args.spatial_kernel_size),
                bias=args.spatial_bias),
            torch.nn.ReLU(),

            torch.nn.Conv3d(
                in_channels=f2,
                out_channels=f3,
                kernel_size=tuple(args.spatial_kernel_size),
                bias=args.spatial_bias),
            torch.nn.ReLU(),

            torch.nn.Conv3d(
                in_channels=f3,
                out_channels=f4,
                kernel_size=tuple(args.spatial_kernel_size),
                bias=args.spatial_bias),
            torch.nn.ReLU()

            )

        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f4, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x

class ClassicalAugmentCNN(torch.nn.Module):
    def __init__(self, args, device='cuda:0', full_group=True):
        super(ClassicalAugmentCNN, self).__init__()
        """
        Test augmentation classical CNN
        """

        _, inv_rotation_matrices = compute_rotations(SO2=full_group)
        inv_rotation_matrices = inv_rotation_matrices.float()

        # These are for logging
        #f1, f2, f3, f4 = 90, 120, 120, 90
        f1, f2, f3, f4 = 90, 5, 5, 5
        self.num_shells = args.num_shells
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.b_size = args.b_size
        self.num_epoch = args.iter
        self.lr = args.lr
        self.aug = args.data_aug

        self.conv_layers = torch.nn.Sequential(

            ClassicalLifting(
                nfeature_in=f1,
                nfeature_out=f2,
                kernel_size=tuple(args.spatial_kernel_size),
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            
            ClassicalAugmentconv(
                nfeature_in=f2,
                nfeature_out=f3,
                kernel_size=tuple(args.spatial_kernel_size),
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            ClassicalAugmentconv(
                nfeature_in=f3,
                nfeature_out=f4,
                kernel_size=tuple(args.spatial_kernel_size),
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            MeanPooling(dim=1)
            # Projection(dim=1)
            )

        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f4, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x

def train_model(num_epoch, model, criterion, optimiser, train_loader, test_loader, run_path, clip, device):
    its = 0
    #prediction = evaluate(test_loader, model, criterion, run_path, its, 0, device)
    model.train()
    for i in tqdm(range(num_epoch)):
        if (np.abs(((i+1) / num_epoch) - 0.4) < 0.001) or (np.abs(((i+1) / num_epoch) - 0.5) < 0.001) or (np.abs(((i+1) / num_epoch) - 0.8) < 0.001) or (np.abs(((i+1) / num_epoch) - 0.9) < 0.001):
            for g in optimiser.param_groups:
                g['lr'] *= 0.5

        for m, lb in tqdm(train_loader):
            m = m.to(device)
            out = model(m)
            loss = criterion(out, lb.to(device))
            wandb.log({"Train loss": loss.item()}, step=its)
            optimiser.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip, norm_type='inf')
            optimiser.step()
            its += 1

        prediction = evaluate(test_loader, model, criterion, run_path, its, i+1, device)
        model.train()
    return prediction

def evaluate(test_loader, model, criterion, run_path, iteration, epoch, device):
    model.eval()
    l = []
    prediction = []
    test_labels = []
    
    with torch.no_grad():
        for m, lb in tqdm(test_loader):
            m = m.to(device)
            out = model(m)
            loss = criterion(out, lb.to(device)).cpu().item()
            loss_cp = copy.deepcopy(loss)
            l.append(loss_cp)

            res = torch.nn.functional.softmax(out, dim=1)
            res = torch.argmax(res, dim=1)
            res_cp = copy.deepcopy(res.cpu())
            lb_cp = copy.deepcopy(lb.cpu())
            del res, lb, loss
            prediction.append(res_cp)
            test_labels.append(lb_cp)

    print("Eval finished")
    l = np.mean(l)
    prediction = torch.cat(prediction)
    test_labels = torch.cat(test_labels)
    print("info captured")
    n_labels = len(torch.unique(test_labels))
    cond = prediction == test_labels
    pred = prediction[cond]
    accs = []
    for i in range(n_labels):
        class_ac = torch.sum(pred==i).item()/torch.sum(test_labels==i).item()
        class_ac_cp = copy.deepcopy(class_ac)
        del class_ac
        accs.append(class_ac_cp)
        wandb.log({f"Val class {i+1}": class_ac_cp}, step=iteration)
        print("logged {}th class accuracy".format(i+1))

    acc = torch.sum(cond).item() / len(test_labels)
    wandb.log({"Val loss": l, "Val accuracy": acc}, step=iteration)
    print("logged overall accuracy")
    accs.append(acc)
    print("before writing into files")

    with open(f"{run_path}/epoch_{str(epoch)}_accuracy.txt", 'w') as f:
        for item in accs:
            f.write("%s\n" % item)

    print("Saving model...")
    torch.save(model.state_dict(), f"{run_path}/model_epoch_{str(epoch)}.ckpt")
    print("Model saved.")
    return prediction

class ModelSpatialSteerParamCompare(torch.nn.Module):
    def __init__(self, args, device='cuda:0', full_group=True):
        super(ModelSpatialSteerParamCompare, self).__init__()
        """
        Model with steerable spatial kernels.
        """
        perm = [list(np.roll(np.arange(args.num_rays), i)) for i in range(args.num_rays)]
        _, inv_rotation_matrices = compute_rotations(SO2=full_group)
        inv_rotation_matrices = inv_rotation_matrices.float()

        input_kernel_size = args.num_rays * args.samples_per_ray + 1
        f_lift = 10
        f_conv1, f_conv2 = 20,20
        f_spatial1, f_spatial2, f_spatial3 = 10,40,10

        self.num_shells = args.num_shells
        self.ray_len = args.ray_len
        self.b_size = args.b_size
        self.watson_param = args.watson_param
        self.num_epoch = args.iter
        self.lr = args.lr
        self.pooling = args.pooling
        self.aug = args.data_aug

        self.f_lift = f_lift

        self.f_conv1 = f_conv1
        self.f_conv2 = f_conv2
#         self.f_conv3 = f_conv3

        self.f_spatial1 = f_spatial1
        self.f_spatial2 = f_spatial2
        self.f_spatial3 = f_spatial3

        self.lifting_spatial_groupconv = torch.nn.Sequential(

            Lifting(
                nfeature_in=args.num_shells,
                nfeature_out=f_lift,
                kernel_size=input_kernel_size,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_lift,
                nfeature_out=f_spatial1,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            )

        self.spatial_groupconv1 = torch.nn.Sequential(
            
            GroupConv(
                nfeature_in=f_spatial1,
                nfeature_out=f_conv1,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_conv1,
                nfeature_out=f_spatial2,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            )

        self.spatial_groupconv2 = torch.nn.Sequential(
            GroupConv(
                nfeature_in=f_spatial2,
                nfeature_out=f_conv2,
                kernel_size=args.num_rays,
                perm=perm,
                use_bias=args.bias
                ),
            torch.nn.ReLU(),

            SteerableSpatialConv(
                nfeature_in=f_conv2,
                nfeature_out=f_spatial3,
                kernel_size=args.spatial_kernel_size,
                rotation_matrices=inv_rotation_matrices,
                use_bias=args.spatial_bias,
                device=device
                ),
            torch.nn.ReLU(),

            )

        if self.pooling == 'mean':
            print("mean pooling")
            self.projection = [MeanPooling(dim=2), MeanPooling(dim=1)]
        else:
            self.projection = [Projection(dim=2), Projection(dim=1)]
        self.projection = torch.nn.Sequential(*self.projection)


        self.linear_layers = torch.nn.Sequential(*construct_linearlayers([f_spatial3, 4], args.lin_bias, args.lin_bn))

    def forward(self, x):
        x = self.lifting_spatial_groupconv(x)
        x = self.spatial_groupconv1(x)
        x = self.spatial_groupconv2(x)
        x = self.projection(x)
        x = self.linear_layers(x.reshape(len(x), -1))
        return x