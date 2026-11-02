import argparse
import pprint
from collections import OrderedDict


def _parse_csv_floats(s):
    return tuple(float(x) for x in s.split(',') if x.strip())


def _parse_csv_ints(s):
    return tuple(int(x) for x in s.split(',') if x.strip())


class AblationOptions:
    """Options for train_ablation.py.

    Covers the ablation studies:
      lambda sweep: --lambda_s (replaces hardcoded 40)
      loss ablation: --lambda_e + --lambda_s (set one to 0 for eigen-only / spatial-only)
      KNN sensitivity: --knn-neighbors, --knn-distance-weights (K and weight sensitivity)
      affinity type: --affinity-type (knn|gaussian), --gaussian-sigma
    """
    def __init__(self):
        p = argparse.ArgumentParser()

        # dataset
        p.add_argument('--data-root', type=str, default='./data',
                       help='dir containing nyu_depth_v2_labeled.mat')
        p.add_argument('--train-split', type=str, default='train')
        p.add_argument('--test-split', type=str, default='test')
        p.add_argument('--lowres-mode', type=str, default='center')
        p.add_argument('--factor', type=int, default=8)
        p.add_argument('--train-crop', type=int, default=128)
        p.add_argument('--val-ratio', type=float, default=0.0)
        p.add_argument('--zero-guidance', action='store_true')
        p.add_argument('--eval-border', type=int, default=-1)

        # learning
        p.add_argument('--batch_size', type=int, default=4)
        p.add_argument('--test-batch-size', type=int, default=1)
        p.add_argument('--lr', type=float, default=1e-4)
        p.add_argument('--momentum', type=float, default=0.9)
        p.add_argument('--weight_decay', type=float, default=0.0)
        p.add_argument('--epoch', type=int, default=100)
        p.add_argument('--print_freq', type=int, default=10)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--is_cuda', type=bool, default=True)
        p.add_argument('--num_workers', type=int, default=0)
        p.add_argument('--cache-all', type=int, default=1,
                       help='1 = load entire mat into RAM (faithful to official); 0 = lazy per-sample reads')

        # logging / checkpointing
        p.add_argument('--exp_name', type=str, required=True,
                       help='subfolder under --save-root; one experiment per ablation config')
        p.add_argument('--save-root', type=str, default='ckpt/ablation')
        p.add_argument('--save-every-iter', type=int, default=0,
                       help='1 = mimic official (save every iter); 0 = save only on print_freq')

        # loss balancing
        p.add_argument('--lambda_e', type=float, default=1.0,
                       help='weight for eigen loss; set 0 for spatial-only ablation')
        p.add_argument('--lambda_s', type=float, default=40.0,
                       help='weight for spatial loss; set 0 for eigen-only ablation')

        # affinity construction
        p.add_argument('--affinity-type', type=str, default='knn', choices=['knn', 'gaussian'])
        p.add_argument('--knn-neighbors', type=str, default='20,10',
                       help='comma-separated KNN sizes per stage')
        p.add_argument('--knn-distance-weights', type=str, default='2.0,0.1',
                       help='comma-separated spatial weights per stage')
        p.add_argument('--gaussian-sigma', type=float, default=1.0,
                       help='sigma for Gaussian affinity')

        # model variant
        p.add_argument('--model', type=str, default='SSGNet',
                       choices=['SSGNet', 'SSGNet_Gray'],
                       help='base SSGNet variant')

        # resume
        p.add_argument('--resume', type=str, default='',
                       help='checkpoint .tar to resume from (empty = train from scratch)')

        self.opts = p.parse_args()

        # post-process: convert csv strings to tuples
        self.opts.knn_neighbors = _parse_csv_ints(self.opts.knn_neighbors)
        self.opts.knn_distance_weights = _parse_csv_floats(self.opts.knn_distance_weights)
        assert len(self.opts.knn_neighbors) == len(self.opts.knn_distance_weights), \
            "--knn-neighbors and --knn-distance-weights must have same length"

    @property
    def parse(self):
        d = OrderedDict(vars(self.opts))
        pprint.pprint(d)
        return self.opts
