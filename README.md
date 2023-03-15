# [CVPR2023] StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields

## [Project page](TODO) |  [Paper](TODO)
This repository contains a pytorch implementation for the paper: [StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields](TODO). StyleRF is an innovative 3D style transfer technique that achieves superior 3D stylization quality with precise geometry reconstruction and it can generalize to various new styles in a zero-shot manner. <br><br>

## Installation
#### Tested on Ubuntu 20.04 + Pytorch 1.12.1

Install environment:
```
conda create -n StyleRF python=3.9
conda activate StyleRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```

## Datasets
Please put the datasets in `./data`. You can put the datasets elsewhere if you modify the corresponding paths in configs.

### 3D scene datasets
* [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [llff](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
### Style image dataset
* [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan)

## Quick Start
We provide some pretrained checkpoints in: TODO

Then modify `scripts/test_style.sh`:
* `--config`: choose `configs/llff_style.txt` or `configs/nerf_synthetic_style.txt` according to which type of dataset is being used
* `--datadir`: dataset's path
* `--ckpt`: checkpoint's path
* `--style_img`: reference style image's path


To generate stylized novel view:
```
bash scripts/test_style.sh [GPU ID]
```

## Training
We follow the following 3 steps of training:
### 1. Train original TensoRF
This step is for reconstructing the density field, which contains more precise geometry details compared to mesh-based methods. You can skip this step by directly downloading pre-trained checkpoints provided by [TensoRF](https://github.com/apchenstu/TensoRF).

The configs are stored in `configs/llff.txt` and `configs/nerf_synthetic.txt`. For the details of the settings, please also refer to [TensoRF](https://github.com/apchenstu/TensoRF).

You can train the original TensoRF by:
```
bash script/train.sh [GPU ID]
```

### 2. Feature grid training stage
The configs are stored in `configs/llff_feature.txt` and `configs/nerf_synthetic_feature.txt`, in which `ckpt` specifies the checkpoints trained in the **first** step.
```
bash script/train_feature.sh [GPU ID]
```

### 3. Stylization training stage 
The configs are stored in `configs/llff_style.txt` and `configs/nerf_synthetic_style.txt`, in which `ckpt` specifies the checkpoints trained in the **second** step.
```
bash script/train_style.sh [GPU ID]
```

## Acknowledgments
This repo is heavily based on the [TensoRF](https://github.com/apchenstu/TensoRF). Thank them for sharing their amazing work!

## Citation
If you find our code or paper helps, please consider citing:
```
TODO
```

