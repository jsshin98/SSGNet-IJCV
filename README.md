# Scene Structure Guidance Network: Unfolding Graph Partitioning into Spatial Feature Learning

**IJCV Under Review**

Official PyTorch implementation of SSGNet, a lightweight (~56K parameters) unsupervised module that
unfolds spectral graph partitioning into a spatial feature learning network and provides scene structure
guidance for low-level vision tasks.

This repository contains the SSGNet model and its pretraining / evaluation / ablation code only.
Downstream task networks (depth upsampling, image denoising, etc.) are not included.

## Requirements

```
pip install -r requirements.txt
```

The NYU Depth V2 labeled dataset (`nyu_depth_v2_labeled.mat`) is used for pretraining.
Place it under `./data/` or pass `--data-root`.

## Repository structure

```
SSGNet-IJCV/
├── model.py                 # SSGNet (and LN / grayscale variants)
├── train.py                 # pretraining on NYU Depth V2
├── test.py                  # eigenvector visualisation with a pretrained checkpoint
├── train_ablation.py        # pretraining with ablation flags
├── datasets/                # NYU Depth V2 loader, KNN-matting / Gaussian affinity, Laplacian
├── options/                 # argument parsers (train / test / ablation)
├── utils/losses.py          # eigen loss + spatial loss
└── tools/
    ├── compare_eigh.py      # compare SSGNet output with Laplacian eigenvectors (scipy eigsh)
    ├── profile_affinity.py  # timing / memory profile of the affinity construction step
    └── flowlib.py
```

## Pretraining

```
python train.py --data-root ./data --save_dir ckpt/ssgnet
```

## Test

```
python test.py --data-root ./data --ssgnet_pretrained ckpt/ssgnet/<iter>.tar --save_result result/ssgnet
```

## Ablation studies

```
# loss weight sweep
python train_ablation.py --exp_name lambda_100 --lambda_s 100

# eigen-only / spatial-only
python train_ablation.py --exp_name eigen_only   --lambda_s 0
python train_ablation.py --exp_name spatial_only --lambda_e 0

# KNN neighbourhood / distance weights
python train_ablation.py --exp_name knn_K10_5 --knn-neighbors 10,5 --knn-distance-weights 2.0,0.1

# Gaussian-weighted affinity
python train_ablation.py --exp_name gauss_s1 --affinity-type gaussian --gaussian-sigma 1.0
```

Each run writes checkpoints, a `config.json` snapshot and TensorBoard logs to `ckpt/ablation/<exp_name>/`.

## Analysis tools

```
python tools/compare_eigh.py --ckpt ckpt/ssgnet/<iter>.tar --data-root ./data --num 5 --out result/compare_eigh
python tools/profile_affinity.py --data-root ./data --num 32
```

## Citation

The conference version of this work:

```
@inproceedings{shin2023task,
  title     = {Task-Specific Scene Structure Representations},
  author    = {Shin, Jisu and Shin, Seunghyun and Jeon, Hae-Gon},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}
}
```
