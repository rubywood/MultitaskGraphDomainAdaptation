import argparse
import TrainStudent

from implementation_utils import str2bool

parser = argparse.ArgumentParser()

parser.add_argument('--root-dir', type=str, default='checkpoint/')
parser.add_argument('--base-name', type=str, default='CTransPath')
parser.add_argument('--base-version', type=float, default=5.0)
parser.add_argument('--graph-version', type=float, default=5.0)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--save-folder', type=str,
                    default='Clustering')
parser.add_argument('--clinical-file', type=str,
                default='Metadata/GASPatchLabelsInclNAsTm20TGFbEpi.csv')

parser.add_argument('--model-name', type=str)
parser.add_argument('--graph-name', type=str,
                    default='superpixel_5X_compactness_20_scaleslic_2_patch_scaled_connectivity_800')

parser.add_argument('--train-cohorts', type=str, nargs='+', default=['GRAMPIAN', 'ARISTOTLE'])
parser.add_argument('--test-cohort', type=str, default='SALZBURG')
parser.add_argument('--preproc', type=str2bool, default=True)

parser.add_argument('--dim-features', type=int, default=768)
parser.add_argument('--dim-target', type=int, default=1)
parser.add_argument('--layers', default=[64, 32, 16, 8], nargs='+', type=int, help='Layer dimensions in GNN')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--conv', type=str, default='GINConv')
parser.add_argument('--aggr', type=str, default='min')
parser.add_argument('--gembed', type=str2bool, default=False)
parser.add_argument('--scaler', type=str2bool, default=False)
parser.add_argument('--use-mlp', type=str2bool, default=True)
parser.add_argument('--temper', type=float, default=1.5)
parser.add_argument('--mlp-version', type=int, default=3)
parser.add_argument('--mlp-dropout', type=float, default=0.5)

parser.add_argument('--responses', default=['response_cr_nocr', 'CMS4', 'epithelium'], nargs='+', type=str)

parser.add_argument('--seed', type=int, default=0, help='Seed for features and Kmeans')
parser.add_argument('--training-seeds', type=int, nargs='+', default=[0,1,2,3,4],
                    help='Seeds for training student model')
#parser.add_argument('--load-kmeans', type=str2bool, default=True,
#                    help='Whether to load KMeans instead of recomputing')
parser.add_argument('--use-scaled-features', type=str2bool, default=True)
parser.add_argument('--scale-centroids', type=str2bool, default=False)
parser.add_argument('--nclusters', type=int, default=6)
parser.add_argument('--best-nclusters', type=str2bool, default=True)
parser.add_argument('--centre-loss', type=str2bool, default=False)

parser.add_argument('--num-epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--model-version', type=str, default='Student_001')

args = parser.parse_args()
#args = parser.parse_known_args()[0]


if __name__ == '__main__':
    for seed in args.training_seeds:
        #parser.add_argument('--training-seed', type=int, const=seed, action='store_const')
        #seed_args = parser.parse_args()
        setattr(args, 'training_seed', seed)
        TrainStudent.main(args)
