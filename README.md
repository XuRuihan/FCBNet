# FCBNet: Improving Convolutional Neural Networks with Fourier Convolution Block

This repo is the official PyTorch implementation of **FCBNet** proposed by our paper "ParCNetV2: Oversized Kernel with Enhanced Attention (to be released)".


## Introduction

Convolutional Neural Networks (CNNs) usually comprise locally connected convolution layers. Although reinforced by spatial pooling and deep architecture, small convolution kernels limit their ability in inefficient modeling of large spatial contexts. This paper aims to enlarge the receptive field of vanilla CNNs without altering their architectures. We introduce a novel Fourier Convolution Block (FCB) that explicitly extends the receptive field of each convolutional layer through the Fourier transform. In contrast to the vanilla CNNs that rely on stacking small kernels (*e.g.*, 3×3) to reach a large receptive field, the Fourier Convolution Block is composed of a spatial convolution layer, a residual Fourier convolution layer, and pointwise convolution layers. The spatial convolution extracts local features, while the Fourier Convolution Block focuses on modeling global clues. Combining both modules in the pointwise convolution layer effectively models both local and long-range dependencies. Our Fourier Convolution Block is computationally efficient and compatible with various vanilla CNNs, producing more discriminative representations. Experiments demonstrate its significant improvement in various vision tasks, including image recognition, object detection, and semantic segmentation. We conclude that the proposed approach consistently enhances performance and exhibits promising generalization capabilities.


## Requirements

### Packages

```
torch<2.0.0
pyyaml
timm==0.6.12
mmdetection==3.2.0
mmsegmentation==1.2.1
```


Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train

We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.


```bash

DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MASTER_PORT=29501

cd $CODE_PATH && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
--master_port=$MASTER_PORT \
train.py $DATA_PATH \
--model fcbnet_s18 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0 \
> log/parcnetv2_tiny.log 2>&1
```
Training scripts of other models are shown in [scripts](/scripts/).


## License

This project is released under the MIT license. Please see the [LICENSE](/LICENSE) file for more information.

<!-- ## Citation

If you find this repository helpful, please consider citing:

```
``` -->

## Acknowledgement

This repository is built using the following libraries and repositories.

1. [Timm](https://github.com/rwightman/pytorch-image-models)
2. [DeiT](https://github.com/facebookresearch/deit)
3. [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
4. [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
5. [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
6. [MetaFormer](https://github.com/sail-sg/metaformer)