"""Ablation-ready SSGNet pretraining.

Mirrors train.py but exposes flags for the ablation studies:
  - lambda sweep                : --lambda_e --lambda_s
  - eigen-only / spatial-only   : set --lambda_e 0 or --lambda_s 0
  - KNN K & distance-weight     : --knn-neighbors --knn-distance-weights
  - Gaussian affinity           : --affinity-type gaussian --gaussian-sigma
  - reproducible seed           : --seed

Outputs land in {--save-root}/{--exp_name}/ along with a `config.json` snapshot.
The original train.py is left untouched for the reproduce baseline.

Example: full ablation matrix entry
    python train_ablation.py --exp_name lambda_100 --lambda_s 100
    python train_ablation.py --exp_name eigen_only --lambda_s 0
    python train_ablation.py --exp_name knn_K10_5 --knn-neighbors 10,5
    python train_ablation.py --exp_name gauss_s1 --affinity-type gaussian --gaussian-sigma 1.0
"""
import json
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from datasets import nyu
from utils import losses
from options.ablation_options import AblationOptions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    opt = AblationOptions().parse
    set_seed(opt.seed)

    save_dir = os.path.join(opt.save_root, opt.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # snapshot the config so each run is self-describing
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        cfg = vars(opt).copy()
        for k in ('knn_neighbors', 'knn_distance_weights'):
            cfg[k] = list(cfg[k])
        json.dump(cfg, f, indent=2)

    eval_border = 6 if opt.eval_border < 0 else opt.eval_border
    lowres_mode = opt.lowres_mode if opt.lowres_mode else 'center'

    affinity_kwargs = dict(
        affinity_type=opt.affinity_type,
        n_neighbors=opt.knn_neighbors,
        distance_weights=opt.knn_distance_weights,
        gaussian_sigma=opt.gaussian_sigma,
    )

    train_tfm = nyu.AssembleJointUpsamplingInputs(
        opt.factor, flip=True, lowres_mode=lowres_mode,
        zero_guidance=opt.zero_guidance, output_crop=eval_border,
        crop=(None if opt.train_crop <= 0 else opt.train_crop),
        affinity_kwargs=affinity_kwargs,
    )

    train_dset = nyu.NYUDepthV2(
        opt.data_root, transform=train_tfm, download=False,
        split=(opt.train_split or 'train'),
        val_ratio=opt.val_ratio, cache_all=bool(opt.cache_all),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers,
    )

    print(f'[ablation] exp={opt.exp_name} | train samples={len(train_dset)}')
    print(f'[ablation] affinity={affinity_kwargs} | λ_e={opt.lambda_e} λ_s={opt.lambda_s}')

    assert torch.cuda.is_available()
    torch.cuda.set_device(torch.device('cuda:0'))

    net = getattr(model, opt.model)().cuda()
    net = torch.nn.DataParallel(net)
    if opt.resume:
        state = torch.load(opt.resume, map_location='cpu')['state_dict']
        net.load_state_dict(state, strict=False)
        print(f'[ablation] resumed from {opt.resume}')

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[ablation] params={n_params}')

    optz = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    writer = SummaryWriter(save_dir)
    loss_fn = losses.get_loss_fn('our_loss')()
    net.train()

    start = 0
    for epoch in range(opt.epoch):
        print(f'[ablation] Epoch {epoch}')
        for i, (image, affinity) in enumerate(tqdm(train_loader)):
            current_iter = start + epoch * len(train_loader) + i + 1
            if opt.is_cuda:
                image, affinity = image.cuda(), affinity.cuda()

            eigen_vs = net(image)
            e_loss, s_loss = loss_fn(eigen_vs, affinity)
            loss = opt.lambda_e * e_loss + opt.lambda_s * s_loss

            optz.zero_grad()
            loss.backward()
            optz.step()

            if current_iter % opt.print_freq == 0:
                writer.add_scalar('eigen_loss', e_loss.item(), current_iter)
                writer.add_scalar('spatial_loss', s_loss.item(), current_iter)
                writer.add_scalar('total_loss', loss.item(), current_iter)

                # log per-channel std as a degenerate-collapse diagnostic
                with torch.no_grad():
                    ev = eigen_vs.detach()
                    per_ch_std = ev.flatten(2).std(dim=2).mean(dim=0)
                    for c in range(per_ch_std.shape[0]):
                        writer.add_scalar(f'channel_std/ch{c}', per_ch_std[c].item(), current_iter)
                    writer.add_scalar('channel_std/mean', per_ch_std.mean().item(), current_iter)

                if not opt.save_every_iter:
                    torch.save({'epoch': current_iter, 'state_dict': net.state_dict()},
                               os.path.join(save_dir, f'{current_iter}.tar'))

            if opt.save_every_iter:
                torch.save({'epoch': current_iter, 'state_dict': net.state_dict()},
                           os.path.join(save_dir, f'{current_iter}.tar'))

        torch.save({'epoch': epoch, 'state_dict': net.state_dict()},
                   os.path.join(save_dir, f'epoch{epoch}.tar'))


if __name__ == '__main__':
    main()
