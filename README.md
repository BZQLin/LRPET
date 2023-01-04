# Compact Model Training by Low-Rank Projection with Energy Transfer
## Introduction
This is the PyTorch implementation of our paper "Compact Model Training by Low-Rank Projection with Energy Transfer"

## Prerequisites


## Usage
### CIFAR-10 dataset
Use python cifar10_train.py to train a new model. Here is some example settings:
```
CUDA_VISIBLE_DEVICES=0 nohup python cifar10_train.py  --model resnet56 --prun_goal 0.80 --redu_fac 0 --epochs 400 >SVD/CNN/cifar10/save_log/resnet56_prun0.80.log 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=0 nohup python cifar10_train.py  --model resnet110 --prun_goal 0.62 --epochs 400 >SVD/CNN/cifar10/save_log/resnet110_prun0.62.log 2>&1 &
```

Use python cifar10_train.py to search a model. Here is a example setting:
```
CUDA_VISIBLE_DEVICES=0 nohup python search.py  --model resnet56  --redu_fac 0 --epochs 60 >SVD/CNN/cifar10/save_log/resnet56_prun0.7_search.log 2>&1 &
```
### ImageNet dataset
```
CUDA_VISIBLE_DEVICES=0 nohup python imagenet_resnet_trans_train.py  -a resnet34 --prun_goal 0.58 >SVD/CNN/imagenet/save_log/imagenet_resnet34_prun0.58.log 2>&1 &
```
```
CUDA_VISIBLE_DEVICES=0 nohup python imagenet_resnet_trans_train.py  -a resnet50 --prun_goal 0.62 >SVD/CNN/imagenet/save_log/imagenet_resnet50_prun0.62.log 2>&1 &
```

