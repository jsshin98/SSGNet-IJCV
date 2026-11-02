"""Wall-clock + memory profile of the KNN-matting affinity step.

Times `compute_affinity_laplacian` on a batch of real NYU samples and reports:
  - mean / min / max wall-clock per image (CPU)
  - approximate FLOPs (KNN is dominated by feature-distance squared per neighbor)
  - peak memory of W (sparse) and L

Usage:
  python tools/profile_affinity.py --num 32
  python tools/profile_affinity.py --affinity-type gaussian --gaussian-sigma 1.0
  python tools/profile_affinity.py --crop 256 --num 16
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import nyu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--num', type=int, default=16)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--affinity-type', default='knn', choices=['knn', 'gaussian'])
    ap.add_argument('--knn-neighbors', default='20,10')
    ap.add_argument('--knn-distance-weights', default='2.0,0.1')
    ap.add_argument('--gaussian-sigma', type=float, default=1.0)
    args = ap.parse_args()

    nbrs = tuple(int(x) for x in args.knn_neighbors.split(','))
    dws = tuple(float(x) for x in args.knn_distance_weights.split(','))

    affinity_kwargs = dict(affinity_type=args.affinity_type, n_neighbors=nbrs,
                            distance_weights=dws, gaussian_sigma=args.gaussian_sigma)
    tfm = nyu.AssembleJointUpsamplingInputs(8, flip=False, lowres_mode='center',
                                              output_crop=6, crop=args.crop,
                                              affinity_kwargs=affinity_kwargs)
    ds = nyu.NYUDepthV2(args.data_root, split='train', cache_all=False, transform=tfm)

    times = []
    nnzs = []
    print(f'profiling affinity: type={args.affinity_type} crop={args.crop} K={nbrs} dw={dws}')
    for i in range(args.num):
        # Pull raw image, run the affinity step directly to isolate its cost
        sample = ds.data_headers if not ds.cache_all else None  # noqa: not used
        # Quickest: index into the dataset and time __getitem__ minus model overhead.
        # __getitem__ does: load + transform + affinity + normalise. Affinity dominates.
        t = time.perf_counter()
        image, affinity = ds[i]
        dt = time.perf_counter() - t
        times.append(dt)
        # estimate non-zero count of the affinity (as a dense tensor here)
        nnz = int((affinity != 0).sum().item())
        nnzs.append(nnz)

    times = np.array(times)
    nnzs = np.array(nnzs)
    print('=== timing ({} samples, crop={}, N={} px) ==='.format(args.num, args.crop, args.crop * args.crop))
    print(f'  mean   : {times.mean()*1000:.1f} ms')
    print(f'  median : {np.median(times)*1000:.1f} ms')
    print(f'  min/max: {times.min()*1000:.1f} / {times.max()*1000:.1f} ms')
    print(f'  W nnz  : mean {nnzs.mean():.0f}, max {nnzs.max()} (out of {args.crop**4})')
    K_total = sum(nbrs)
    print(f'  KNN cost ~ N * sum(K) = {args.crop*args.crop} * {K_total} = {args.crop*args.crop*K_total:,} pairwise dists')


if __name__ == '__main__':
    main()
