# Sample Code of Precision Gating (PG)

This repo contains the sample code of the precision gating project submitted to ICLR 2020. The code trains and tests ShuffleNet V2 0.5x on ImageNet dataset.

## Installation

We use the following tensorflow version for building up the code:

```
tensorflow-gpu 1.9.0
```

Though other versions of tensorflow are not suggested, we have also tested the code in v1.14.0. It can run except for some warnings. However, performance variance in other versions remains unknown.

## Folder Structure

The folder is organized as follows:

```
plain-precision-gating/
    --- quantization/
        --- layers.py (PG conv and FC layers)
        --- pact.py
        --- quantize.py (quantization function)
        --- util_layers.py
    --- resnet/
        --- imagenet_shufflenet.py (main file)
        --- shufflenet_model.py (shufflenet model file)
        --- imagenet_preprocessing.py
        --- common.py (common functions for other models)
        --- imagenet_utils.py
        --- lr_schedule.py
        --- optimize.py
        --- train_shufflenet.py (train file)
    --- utils/ (some util files)
    --- train_shufflenet.sh (the train script)
```

## Prerequisite

Before running the code, imagenet dataset needs to be downloaded somewhere. We refer readers to 
[Tensorflow ImageNet Dataset.](https://www.tensorflow.org/datasets/catalog/imagenet2012)

## Usage

To train or test PG, run

```
cd /path/to/plain-precision-gating
source train_shufflenet.sh
```

The train script ```train_shufflenet.sh``` contains the bash command running ```train_shufflenet.py``` and passes some arguments. Typically, we adjust the following arguments:

```
--data_dir  (Point to the path that stores tensorflow imagenet dataset.)
--a         (High-precision activation bitwidth. The same as B in the paper. Example: 6.)
--t         (Low-precision activation bitwidth. The same as Bhb in the paper. Example: 4.)
--g         (Gating target. Example: 0.1.)
--gpus      (Which gpus to use. Example: 0,1.)
--pact      (If added, use PACT during training.)
--test_only (If added, load the pretrained model and do test only.)
```