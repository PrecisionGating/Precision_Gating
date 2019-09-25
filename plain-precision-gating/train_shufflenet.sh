#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR

python $PWD/resnet/train_shufflenet.py --data_dir=/scratch-x2/datasets/imagenet-tensorflow/ --a 6 --t 4 --g 0.0 --gpus 0,1 --pact
