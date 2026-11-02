"""Compare SSGNet output Y with ground-truth Laplacian eigenvectors.

Eigenvectors are determined up to sign + permutation, and SSGNet's softmax-
normalized Y lives in a different parametric space than the bottom-k Laplacian
eigenvectors. To make a fair comparison we:

  1. Build the same affinity / Laplacian used in training.
  2. Get bottom-k eigenvectors V (k = #channels of Y) via scipy eigsh.
  3. Centre + L2-normalise both V and Y per channel.
  4. Solve a Hungarian assignment on |cosine similarity| to match channels.
  5. Resolve sign ambiguity per matched pair.
  6. Report cosine sim per channel + the mean, and write side-by-side PNGs.

Usage:
  python tools/compare_eigh.py \
      --ckpt ckpt/reproduce/1000.tar \
      --data-root ./data --num 5 --out result/compare_eigh
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torchvision.utils as tvu
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model
from datasets import nyu


def normalise(x):
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def align_and_score(Y, V):
    """Hungarian-match Y channels to V channels by |cosine|, then sign-flip.

    Y, V: (K, N) arrays. Returns (matched_V_indices, cosines_after_alignment, signs).
    """
    K = Y.shape[0]
    Yn = np.stack([normalise(y) for y in Y])
    Vn = np.stack([normalise(v) for v in V])
    cost = -np.abs(Yn @ Vn.T)        # maximise |cos|
    row_ind, col_ind = linear_sum_assignment(cost)
    cos_matched = Yn[row_ind] @ Vn[col_ind].T
    signs = np.sign(np.diag(cos_matched))
    signs[signs == 0] = 1.0
    final_cos = np.diag(cos_matched) * signs
    return col_ind, final_cos, signs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--num', type=int, default=5, help='number of test images to evaluate')
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--k', type=int, default=3, help='# eigenvectors to compare (= SSGNet output channels)')
    ap.add_argument('--affinity-type', default='knn', choices=['knn', 'gaussian'])
    ap.add_argument('--knn-neighbors', default='20,10')
    ap.add_argument('--knn-distance-weights', default='2.0,0.1')
    ap.add_argument('--gaussian-sigma', type=float, default=1.0)
    ap.add_argument('--out', default='result/compare_eigh')
    args = ap.parse_args()

    nbrs = tuple(int(x) for x in args.knn_neighbors.split(','))
    dws = tuple(float(x) for x in args.knn_distance_weights.split(','))

    os.makedirs(args.out, exist_ok=True)

    # build dataset
    affinity_kwargs = dict(affinity_type=args.affinity_type, n_neighbors=nbrs,
                            distance_weights=dws, gaussian_sigma=args.gaussian_sigma)
    tfm = nyu.AssembleJointUpsamplingInputs(
        8, flip=False, lowres_mode='center', output_crop=6,
        crop=args.crop, affinity_kwargs=affinity_kwargs,
    )
    ds = nyu.NYUDepthV2(args.data_root, split=args.split, cache_all=False, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # load model
    net = model.SSGNet().cuda()
    net = torch.nn.DataParallel(net)
    state = torch.load(args.ckpt, map_location='cpu')['state_dict']
    net.load_state_dict(state, strict=False)
    net.eval()

    all_cos = []
    for idx, (image, affinity) in enumerate(loader):
        if idx >= args.num:
            break
        with torch.no_grad():
            Y = net(image.cuda())[0].detach().cpu().numpy()       # (K, H, W)
        K, H, W = Y.shape
        N = H * W
        # Rebuild sparse L for eigsh (todense in the loader is too big to eig on)
        rgb = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        L_sparse, _ = nyu.compute_affinity_laplacian(rgb, **affinity_kwargs)
        # Bottom-k eigenvectors of L (smallest eigenvalues, ascending)
        try:
            eigvals, V = sla.eigsh(L_sparse.astype(np.float64), k=args.k, sigma=0.0, which='LM')
        except Exception:
            eigvals, V = sla.eigsh(L_sparse.astype(np.float64), k=args.k, which='SM')
        order = np.argsort(eigvals)
        eigvals, V = eigvals[order], V[:, order]
        V = V.T                                                    # (K, N)
        Yflat = Y.reshape(K, N)

        col_ind, cos, signs = align_and_score(Yflat, V)
        all_cos.append(cos)
        print(f'[{idx}] eig={eigvals} | matched cos = {cos.round(3)} | mean |cos| = {np.abs(cos).mean():.3f}')

        # save side-by-side PNG: row1 = image, rowK = Y (matched, sign-flipped), rowK = GT V
        def to_uint8(z):
            z = z.reshape(H, W)
            z = (z - z.min()) / (z.max() - z.min() + 1e-12)
            return (z * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb)
        rows = [np.array(rgb_img.convert('L'))]
        for c in range(K):
            rows.append(to_uint8(Yflat[c] * signs[c]))
        for c in range(K):
            rows.append(to_uint8(V[col_ind[c]]))
        grid = np.concatenate(rows, axis=0)
        Image.fromarray(grid).save(os.path.join(args.out, f'cmp_{idx:03d}.png'))

    all_cos = np.stack(all_cos)
    print(f'\n=== summary over {len(all_cos)} samples ===')
    print(f'mean cos per channel        : {all_cos.mean(0).round(3)}')
    print(f'mean |cos| per channel      : {np.abs(all_cos).mean(0).round(3)}')
    print(f'overall mean |cos|          : {np.abs(all_cos).mean():.3f}')
    np.save(os.path.join(args.out, 'cosines.npy'), all_cos)


if __name__ == '__main__':
    main()
