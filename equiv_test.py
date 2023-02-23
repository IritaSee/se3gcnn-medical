import argparse
import copy
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method

import mesh_util, model_util
from train_drivedata import run_voxelcnn, run_steerable_gcnn, run_steerable_param_gcnn, run_spatial_gcnn, run_augment_classicalcnn, run_classicalcnn
from train_drivedata import props

def predict(model, args, epoch, img_shape, test_dl, img_path, test_coords, test_name, device):

    def dice_(seg, gt, k):
        d = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))
        return d

    b_size = args.b_size
    save_folder = img_path.replace(args.run_path, 'pred').replace('images', '')
    print(save_folder)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    test_labels = []
    prediction = []
    prediction_rot = []
    with torch.no_grad():    
        for data, data_rot, lb in tqdm(test_dl):

            out = torch.nn.functional.softmax(model(torch.cat([data, data_rot], dim=0)), dim=1)
            out = torch.argmax(out, dim=1).cpu()

            res, res_rot = torch.split(out, len(out)//2, dim=0)

            prediction.append(copy.deepcopy(res))

            prediction_rot.append(copy.deepcopy(res_rot))

            test_labels.append(copy.deepcopy(lb))
            del res, res_rot, lb

    prediction = torch.cat(prediction)
    prediction_rot = torch.cat(prediction_rot)

    test_labels = torch.cat(test_labels).cpu()
    print(prediction.shape, prediction_rot.shape, test_labels.shape)
    assert len(prediction) == len(test_labels)

    cond = prediction == test_labels
    pred = prediction[cond]

    cond_rot = prediction_rot == test_labels
    pred_rot = prediction_rot[cond_rot]

    n_labels = len(torch.unique(test_labels))

    accs = []
    dices = []
    correct = []

    accs_rot = []
    dices_rot = []
    correct_rot = []

    all_samples = []
    for i in range(n_labels):
        corr = torch.sum(pred==i).item()
        corr_copy = copy.deepcopy(corr)
        correct.append(corr_copy)

        corr_rot = torch.sum(pred_rot==i).item()
        corr_rot_copy = copy.deepcopy(corr_rot)
        correct_rot.append(corr_rot_copy)

        gt = torch.sum(test_labels==i).item()
        gt_copy = copy.deepcopy(gt)
        all_samples.append(gt_copy)

        class_ac = torch.sum(pred==i).item()/torch.sum(test_labels==i).item()
        class_ac_cp = copy.deepcopy(class_ac)
        accs.append(class_ac_cp)

        class_ac_rot = torch.sum(pred_rot==i).item()/torch.sum(test_labels==i).item()
        class_ac_rot_cp = copy.deepcopy(class_ac_rot)
        accs_rot.append(class_ac_rot_cp)

        del corr, corr_rot, gt, class_ac, class_ac_rot
        bool_gt = np.zeros(len(test_labels))
        bool_gt[test_labels==i] = 1

        bool_pred = np.zeros(len(prediction))
        bool_pred[prediction==i] = 1

        d = dice_(bool_pred, bool_gt, 1)
        d_copy = copy.deepcopy(d)
        dices.append(d_copy)

        bool_pred_rot = np.zeros(len(prediction_rot))
        bool_pred_rot[prediction_rot==i] = 1

        d_rot = dice_(bool_pred_rot, bool_gt, 1)
        d_rot_copy = copy.deepcopy(d_rot)
        dices_rot.append(d_rot_copy)

        del d, d_rot
    total_corr = len(pred)
    total_corr_rot = len(pred_rot)

    total_samples = len(test_labels)
    
    total_acc = torch.sum(cond).item()/len(test_labels)
    total_acc_rot = torch.sum(cond_rot).item()/len(test_labels)

    total_corr_copy = copy.deepcopy(total_corr)
    total_corr_rot_copy = copy.deepcopy(total_corr_rot)

    total_samples_copy = copy.deepcopy(total_samples)
    
    total_acc_copy = copy.deepcopy(total_acc)
    total_acc_rot_copy = copy.deepcopy(total_acc_rot)

    correct.append(total_corr_copy)
    correct_rot.append(total_corr_rot_copy)

    all_samples.append(total_samples_copy)

    accs.append(total_acc_copy)
    accs_rot.append(total_acc_rot_copy)

    del total_corr, total_corr_rot, total_samples, total_acc, total_acc_rot
    with open(f"{img_path}/{test_name}_epoch_{str(epoch)}_accuracy.txt", 'w') as f:
        for item in accs:
            f.write("%s\n" % item)
    with open(f"{img_path}/{test_name}_epoch_{str(epoch)}_dice.txt", 'w') as f:
        for item in dices:
            f.write("%s\n" % item)

    with open(f"{img_path}/{test_name}_equiv_epoch_{str(epoch)}_accuracy.txt", 'w') as f:
        for item in accs_rot:
            f.write("%s\n" % item)
    with open(f"{img_path}/{test_name}_equiv_epoch_{str(epoch)}_dice.txt", 'w') as f:
        for item in dices_rot:
            f.write("%s\n" % item)

    prediction = prediction.numpy()
    save_name = f"{save_folder}/{test_name}_epoch_{str(epoch)}_pred.npy"
    np.save(save_name, prediction)

    prediction_rot = prediction_rot.numpy()
    save_name = f"{save_folder}/{test_name}_epoch_{str(epoch)}_pred_rot.npy"
    np.save(save_name, prediction_rot)

    return correct, correct_rot, all_samples

def run_testing(args, epoch, cuda, all_paths, names, network):

    device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    model, wandb_name = eval(network)

    model_props = props(model)
    run_path = args.run_path
    run_path += f"/{wandb_name}_b0_{args.b0}"

    for p in model_props:
        run_path += f"_{p}_{getattr(model, p)}"
    print(run_path)
    img_path = f"{run_path}/images"
    if not os.path.isdir(img_path):
        os.makedirs(img_path, exist_ok=True)

    model_path = f"{run_path}/model_epoch_{epoch}.ckpt"
    print(f"Loading model from {epoch}th epoch, path {model_path}...")
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path, map_location={'cuda:1': 'cuda:0'}))
    model.eval()
    print("Model loaded.")

    model = model.to(device)

    #img_shape = get_img_shape(args.path, names)
    pred_correct, pred_correct_rot, all_prediction = [], [], []
    vertices, edge_rings, faces = mesh_util.get_icosahedron_aligned()

    manifold_coords = mesh_util.get_sphere_points(vertices, edge_rings, args.samples_per_ray, args.ray_len).float()  # 12 x 11 x 3

    for p, n in tqdm(zip(all_paths, names)):
        """
        If the prediction of some scans are already generated, skip them.
        """
        acc_path = f"{img_path}/{n}_epoch_{str(epoch)}_accuracy.txt"
        dice_path = f"{img_path}/{n}_epoch_{str(epoch)}_dice.txt"
        acc_rot_path = f"{img_path}/{n}_equiv_epoch_{str(epoch)}_accuracy.txt"
        dice_rot_path = f"{img_path}/{n}_equiv_epoch_{str(epoch)}_dice.txt"
        exist_count = 0
        if os.path.isfile(acc_path) and os.path.isfile(dice_path) and os.path.isfile(acc_rot_path) and os.path.isfile(dice_rot_path):
            exist_count += 1
            continue
        print("skipping {} existing results".format(exist_count))


        data, labels, inv_inds, vox_coords = read_scan(p)

        b0_idx = (args.b0//1000)-1
        b_data = data[b0_idx, ...].permute(1, 0)

        cube_path = p.replace('classical_grid_size_7', 'cube_rotation_inds').replace('.npz', '.npy')
        cube_inds = torch.Tensor(np.load(cube_path))

        bvec_path = p.replace(f"data_aligned/classical_grid_size_7/{n}.npz", f"{n}/T1w/Diffusion/bvecs")
        bvecs = np.loadtxt(bvec_path)

        bvals_path = bvec_path.replace('/bvecs', '/bvals')
        bvals = np.loadtxt(bvals_path)

        shell = np.abs(bvals - args.b0) < 30
        bvecs = torch.Tensor(bvecs[:, shell])

        test_ds = DataRotation(b_data, labels, inv_inds, args, cube_inds, bvecs, manifold_coords, device=device)

        dl = DataLoader(test_ds, batch_size=args.b_size, num_workers=0)

        correct, correct_rot, all_samples = predict( model, args,
                                                    epoch, None, dl,
                                                    img_path, vox_coords,
                                                    n, device )
        correct = np.array(correct)
        correct_rot = np.array(correct_rot)
        all_samples = np.array(all_samples)

        correct_cp = copy.deepcopy(correct)
        correct_rot_cp = copy.deepcopy(correct_rot)
        all_samples_cp = copy.deepcopy(all_samples)

        pred_correct.append(correct_cp)
        pred_correct_rot.append(correct_rot_cp)
        all_prediction.append(all_samples_cp)
        del correct, correct_rot, all_samples
    #
    #
    # 
    # pred_correct = np.stack(pred_correct)
    # pred_correct_rot = np.stack(pred_correct_rot)

    # all_prediction = np.stack(all_prediction)

    # accs = np.sum(pred_correct, axis=0)/np.sum(all_prediction, axis=0)
    # with open(f"{img_path}/all_accuracy_epoch_{str(epoch)}.txt", 'w') as f:
    #     for item in accs:
    #         f.write("%s\n" % item)

    # accs_rot = np.sum(pred_correct_rot, axis=0)/np.sum(all_prediction, axis=0)
    # with open(f"{img_path}/all_accuracy_equiv_epoch_{str(epoch)}.txt", 'w') as f:
    #     for item in accs_rot:
    #         f.write("%s\n" % item)
    return

def watson_interpolation(vals, vn, v, k):
    """
    Function to interpolate spherical signals using a watson kernel.

    This is a batched version.
    
    # params:
        vals: 90 x 7 x 7 x 7
        vn: 90 x 3
        v: 12 x 11 x 3 or N x 3
    """

    mat = torch.exp(k * torch.tensordot(vn, v, dims=[[-1], [-1]]) ** 2)    # 90 x 12 x 11 or 90 x N
    sums = torch.sum(mat, dim=0, keepdim=True)                   # 1 x 12 x 11 or 1 x N
    ds_norm = mat / sums                           # 90 x 12 x 11 or 90 x N
    ds = torch.tensordot(ds_norm, vals, dims=([0], [0]))    # 12 x 11 x 7 x 7 x 7 or N x 7 x 7 x 7
    return ds

def read_scan(data_path):
    df = np.load(data_path)
    val = df['data']
    labels = df['labels']
    inv_inds = df['inds']
    coords = df['vox_coords']
    return torch.Tensor(val), torch.Tensor(labels), torch.Tensor(inv_inds).long(), torch.Tensor(coords).long()

def get_all_scan_paths(base_path, all_scans):
    all_paths = []
    for scan in all_scans:
        p = '{}/{}'.format(base_path, scan)
        all_paths.append(p)
    return all_paths

class DataRotation(Dataset):
    def __init__(self, vals, labels, inds, args, cube_inds, bvecs, manifold_coords, device="cpu"):
        """
        Dataset to load the data.
        """
        self.vals = vals.to(device)
        self.labels = labels.long().to(device)
        self.inds = inds.long()
        self.cube_inds = cube_inds.long()
        self.bvecs = bvecs.to(device)    # 3 x 90
        self.s2conv = args.interpolate
        self.manifold_coords = manifold_coords.to(device) # 12 x 11 x 3
        self.watson_param = args.watson_param
        self.rotations = torch.from_numpy(R.create_group('O').as_matrix())
        self.network = args.network

        inv_rots = []
        for rot in self.rotations:
            inv_r = np.linalg.inv(rot.numpy())
            inv_rots.append(torch.from_numpy(inv_r))
        self.inv_rotations = torch.stack(inv_rots).float().to(device)

        self.rotations = self.rotations.float().to(device)
        rotation_matrices = torch.cat([self.inv_rotations, torch.zeros([*list(self.inv_rotations.shape[:-1]), 1]).float().to(device)], dim=-1).float()
        out_size = torch.Size((rotation_matrices.shape[0], self.vals.shape[0], *list(self.inds.shape[1:])))           # 24, 90, 7, 7, 7

        # for later interpolation
        self.inter_grid = F.affine_grid(rotation_matrices, out_size).to(device)  # 24, 7, 7, 7, 3
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ind = self.inds[idx]
        cube_ind = self.cube_inds[idx]
        x, y, z = ind.shape

        if not self.network == 'baseline':
            val_origin = self.vals[:, ind.view(-1)].view(self.vals.shape[0], x, y, z)  # 90, 7, 7, 7
            grid = self.inter_grid[cube_ind:(cube_ind+1)]
            val_rot = F.grid_sample(val_origin.unsqueeze(0), grid).squeeze(0)  # 90, 7, 7, 7
        else:
            val_origin = self.vals[:, ind.view(-1)].view(self.vals.shape[0], x, y, z)[:, 3:4, 3:4, 3:4]  # 90, 1, 1, 1
            val_rot = val_origin

        rot_mat = self.rotations[cube_ind]
        bvecs_rot = torch.mm(rot_mat, self.bvecs).T  # 90, 3
        if self.s2conv:
            val_origin = watson_interpolation(val_origin, self.bvecs.T, self.manifold_coords, self.watson_param).unsqueeze(2)#[:,:,:,3:4,3:4,3:4]
            val_rot = watson_interpolation(val_rot, bvecs_rot, self.manifold_coords, self.watson_param).unsqueeze(2)#[:,:,:,3:4,3:4,3:4] # 12 x 11 x 1 x 7 x 7 x 7
        else:
            val_rot = watson_interpolation(val_rot, bvecs_rot, self.bvecs.T, self.watson_param)   # 90, 7, 7, 7

        return val_origin, val_rot, self.labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/renfei/Documents/HCP')
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--model_capacity", type=str, default='small')
    parser.add_argument("--b_size", type=int, default=100)
    parser.add_argument("--num_rays", type=int, default=5)
    parser.add_argument("--ray_len", default=None)
    parser.add_argument("--samples_per_ray", type=int, default=2)
    parser.add_argument("--watson_param", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bias", default=True, action='store_false')
    parser.add_argument("--lin_bias", default=True, action='store_false')
    parser.add_argument("--spatial_bias", default=True, action='store_false')
    parser.add_argument("--lin_bn", default=True, action='store_false')
    parser.add_argument("--num_shells", type=int, default=1)
    parser.add_argument("--b0", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--run_path", type=str, default='data_aug_exp')
    parser.add_argument("--spatial_kernel_size", nargs='+', type=int, default=[3, 3, 3])
    parser.add_argument("--interpolate", default=False, action='store_true')    # if true, the program runs GCNN, if false, the program runs classical CNN.
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--data_aug", default=False, action='store_true')
    parser.add_argument("--network", type=str, default='ours_full')    # if the network chosen is a classical CNN of any kind, --interpolate should be set to False.
    args = parser.parse_args()

    if not type(args.ray_len)==float:
        args.ray_len = None

    rotations = args.num_rays

    data_path = f"{args.path}/data_aligned/classical_grid_size_7"

    scans = os.listdir(data_path)
    exclude_sub = ['100206', '100408']
    all_scans = [x for x in scans if x.replace('.npz', '') not in exclude_sub]

    all_paths = get_all_scan_paths(data_path, all_scans)
    names = [n.replace('.npz', '') for n in all_scans]

    pred_path =  'pred/'
    if not os.path.isdir(pred_path):
        os.makedirs(pred_path, exist_ok=True)

    if args.network == 'ours_full':
        assert args.interpolate
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_steerable_gcnn(args, device, True)")
    elif args.network == 'ours_part':
        assert args.interpolate
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_steerable_gcnn(args, device, False)")
    elif args.network == 'classical':
        assert args.interpolate == False
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_classicalcnn(args)")
    elif args.network == 'ours_decoupled':
        assert args.interpolate
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_spatial_gcnn(args)")
    elif args.network == 'baseline':
        assert args.interpolate
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_voxelcnn(args)")
    elif args.network == 'classical_augment_full':
        assert args.interpolate == False
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_augment_classicalcnn(args, device, True)")
    elif args.network == 'classical_augment_part':
        assert args.interpolate == False
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_augment_classicalcnn(args, device, False)")
    elif args.network == 'ours_compare':
        assert args.interpolate
        run_testing(args, args.epoch, args.cuda, all_paths, names, "run_steerable_param_gcnn(args, device, True)")
    else:
        raise ValueError('Invalid network!')

if __name__ == '__main__':
    main()