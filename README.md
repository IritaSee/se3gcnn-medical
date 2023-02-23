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

In this work, Human-connectome Project (HCP) data are used. The data can be downloaded from ...

After registering and downloading the data (you can download one scan at a time), each scan is a folder named by an ID (e.g. 100408) containing all the data needed. All scan folders are contained in a root folder. The structure of the folders should be:
```
Folder that contains the root folder
├──	HCP(root folder)
	├──	100206 (scan folder)
	├──	100307 (scan folder)
	...	
	...
```
## Steps to run the networks

The data need to be pre-processed before being fed into the networks. The script to process the data is in datagen.py.

### Data pre-processing and generation

There are 3 types of pre-processed data that need to be generated. The brain mask is used to extract meaningful voxels that are in the brain, and for all the networks, either a grid of voxels centered at a meaningful voxel or just the voxel itself is used, based what network it is. Details are explained below.

* Data grids of size 7x7x7 in which each voxel contains a spherical function, interpolated using the directional signals in the original scan. This type of data is used to train the SE(3) group CNN. In the ablation study, this type of data is also used to train the T<sup>3</sup> x SO(3) group CNN. In the paper, the networks that use this type of data are called Ours (including OursFull and OursPart) and OursDecoupled. To generate this type of data, run ```python datagen.py --path [your path to the data folder] --interpolate --grid_size 7```

* Data grids of size 7x7x7 in which each voxel contains flattened signals with no directional information. This type of data is used to train classical CNNs. To generate this type of data, run ```python datagen.py --path [your path to the data folder] --grid_size 7```

* Single voxels. Each voxel is a spherical function interpolated using the directional signals in the original scan. To generate this typd of data, run ```python datagen.py --path [your path to the data folder] --interpolate --grid_size 1```

### Train the networks

In the main experiments, there are 5 networks to train. In the paper, they are named OursFull, OursPart, OursDecoupled, Baseline, and Classical. We train each network with a small and a big capacity, and with or without data augmentation. Thus we end up with 20 experiments