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

## Steps to run the networks

The data need to be pre-processed before being fed into the networks. The script to process the data is in datagen.py.

### Data pre-processing and generation

There are 3 types of pre-processed data that need to be generated. For the 