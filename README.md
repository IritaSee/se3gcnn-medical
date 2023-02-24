# SE(3) group convolutional neural networks and a study on group convolutions and equivariance for DWI segmentation
![Visualization of the network](visualization/demo.gif)

Conda virtual environment is used. Packages needed:

python 3.6

pytorch 1.7.1

wandb

scipy 1.5.2

tqdm 4.57.0

nibabel 3.2.1

numpy

## Data

In this work, Human-connectome Project (HCP) data are used. The data can be downloaded from [here](https://www.humanconnectome.org/study/hcp-young-adult/). Registration is needed.

After registering and downloading the data (you can download one scan at a time), each scan is a folder named by an ID (e.g. 100408) containing all the data needed. All scan folders are contained in a root folder. The structure of the folders should be:
```
Folder that contains the root folder
├──	HCP(root folder)
	├──	100206 (scan folder)
	├──	100307 (scan folder)
	...	
	...
```

The scans we downloaded have the following IDs:
103414, 105620, 101309, 106521, 107725, 103818, 110007, 104012, 113316, 105115, 108828, 111312, 112112, 111211, 100408, 108222, 110411, 108525, 110613, 111009, 104820, 111716, 103111, 102614, 101915, 103515, 101006, 106016, 112920, 111514, 103212, 105014, 106319, 101107, 103010, 111413, 102816, 113215, 101410, 102109, 109830, 109123, 112314, 112516, 108121, 108020, 104416, 102715, 105216, 100206, 107422, 105923.

## Steps to run the networks

The data need to be pre-processed before being fed into the networks.

### Data pre-processing and generation

The script to process the data is datagen.py.

There are 3 types of pre-processed data that need to be generated. The brain mask is used to extract meaningful voxels that are in the brain, and for all the networks, either a grid of voxels centered at a meaningful voxel or just the voxel itself is used, based what network it is. Details are explained below.

* Data grids of size 7x7x7 in which each voxel contains a spherical function, interpolated using the directional signals in the original scan. This type of data is used to train the SE(3) group CNN. In the ablation study, this type of data is also used to train the T<sup>3</sup> x SO(3) group CNN. In the paper, the networks that use this type of data are called Ours (SE(3) group CNNs including OursFull and OursPart) and OursDecoupled (T<sup>3</sup> x SO(3) group CNN). To generate this type of data, run ```python datagen.py --path [your path to the root folder] --interpolate --grid_size 7```

* Data grids of size 7x7x7 in which each voxel contains flattened signals with no directional information. This type of data is used to train classical CNNs. To generate this type of data, run ```python datagen.py --path [your path to the data folder] --grid_size 7```

* Single voxels. Each voxel is a spherical function interpolated using the directional signals in the original scan. To generate this typd of data, run ```python datagen.py --path [your path to the root folder] --interpolate --grid_size 1```

The generated data will be stored in a created folder in the HCP root folder called data_aligned.
### Train the networks

The script to train the networks is train_drivedata.py. In the script, we use the scan with ID 100206 to train the network, and the scan with ID 100408 for validation during training. Feel free to change it to other scans.

Weights and Biases (WandB) package is used to log the training. To run our script, you simply have to provide in the command line an experiment name (that you think of yourself) for WandB for the logging, and follow the instructions shown in the console from WandB.

It is suggested to use the same WandB name for all the experiments (use --exp_name argument to pass the WandB name), so they will all be shown together. It is also suggested to store the results in the same folder in different sub-folders, and the name of the root folder to store the experiment folders is passed by --run_path argument. **In other words, it is suggested to use the same --run_path and --exp_name arguments for all experiments.**

In the main experiments, there are 5 networks to train. In the paper, they are named OursFull, OursPart, OursDecoupled, Baseline, and Classical. We train each network with a small and a big capacity (marked in the names with superscript - or +), and with or without data augmentation (marked in the names with or without Aug). Thus we end up with 20 experiments. The names of the experiments are the same as in the paper.

To train OursFull<sup>- </sup> or OursFull<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_full```

To train OursFullAug<sup>- </sup> or OursFullAug<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_full --data_aug```

To train OursPart<sup>- </sup> or OursPart<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_part```

To train OursPartAug<sup>- </sup> or OursPartAug<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_part --data_aug```

To train OursDecoupled<sup>- </sup> or OursDecoupled<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_decoupled```

To train OursDecoupledAug<sup>- </sup> or OursDecoupledAug<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_decoupled --data_aug```

To train Baseline<sup>- </sup> or Baseline<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 1 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network baseline```

To train BaselineAug<sup>- </sup> or BaselineAug<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 1 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network baseline --data_aug```

To train Classical<sup>- </sup> or Classical<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network classical```

To train ClassicalAug<sup>- </sup> or ClassicalAug<sup>+</sup>, run ```python train_drivedata.py --path [your path to the root folder] --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network classical --data_aug```

After the main experiments, we also provided 2 small example networks that use Classical CNN with augmented training (different from data augmentation). Of these 2 networks, the models themselves encode some augmentation, one network is fully augmented and another is partly augmented. 

To train the fully augmented network, run ```python train_drivedata.py --path [your path to the root folder] --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network classical_augment_full```

To train the partly augmented network, run ```python train_drivedata.py --path [your path to the root folder] --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network classical_augment_part```

The SE(3) network that is used to compare with an existing method in literature has the same theoretical formulation as OursFull, but is configured in a way such that it has similar capacity to the method that it is compared to. To run this SE(3) network, run ```python train_drivedata.py --path [your path to the root folder] --interoplate --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --network ours_compare```

### Testing

To get test results from all models, the script equiv_test.py is used. This script generates test results from both the original test set and a randomly rotated test set. To make sure the random rotations that are used for all experiments are the same, we generate these random rotations first and use these rotations for all trained models. To generate the rotations, run ```python gen_cube_rotation_inds.py --path [your path to the root folder]```

After the random rotations are generated, we can generate the test results from all models that were trained, using both the original and the rotated test set. The accuracies and dices scores of each scan will be stored in a folder named ```images/``` in the experiment folder, and the predicted segmentation will also be generated and stored in ```images/```.

After running ```equiv_test.py``` for all experiments using the commands below, ```gather_results.ipynb``` gathers the results together and calculates the statistics for all experiments.

To generate the results from OursFull<sup>- </sup> or OursFull<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_full```

To generate the results from OursFullAug<sup>- </sup> or OursFullAug<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_full --data_aug```

To generate the results from OursPart<sup>- </sup> or OursPart<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_part```

To generate the results from OursPartAug<sup>- </sup> or OursPartAug<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_part --data_aug```

To generate the results from OursDecoupled<sup>- </sup> or OursDecoupled<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_decoupled```

To generate the results from OursDecoupledAug<sup>- </sup> or OursDecoupledAug<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_decoupled --data_aug```

To generate the results from Baseline<sup>- </sup> or Baseline<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 1 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network baseline```

To generate the results from BaselineAug<sup>- </sup> or BaselineAug<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 1 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network baseline --data_aug```

To generate the results from Classical<sup>- </sup> or Classical<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network classical```

To generate the results from ClassicalAug<sup>- </sup> or ClassicalAug<sup>+</sup>, run ```python equiv_test.py --path [your path to the root folder] --grid_size 7 --model_capacity [small or big] --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network classical --data_aug```

For the augmented Classical CNNs, o generate the results from the fully augmented network, run ```python equiv_test.py --path [your path to the root folder] --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network classical_augment_full```

To generate the results from the partly augmented network, run ```python equiv_test.py --path [your path to the root folder] --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network classical_augment_part```

To generate the results from the SE(3) network that is used to compare with an existing method, run ```python equiv_test.py --path [your path to the root folder] --interoplate --grid_size 7 --exp_name [your wandb name] --run_path [name of the folder to be created to store the results] --epoch [the epoch you choose] --network ours_compare```

