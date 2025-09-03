import argparse
import os
import pandas as pd
import torch
import pickle

# Local utils
import sys
from utils.utils import mkdir, str2bool

from Clustering.implementation_utils import cohort_from_name, predict_node_features_multiple,\
    train_cohorts_from_model_name, torch_scaler

parser = argparse.ArgumentParser()

parser.add_argument('--root-dir', type=str, default='checkpoint/')
parser.add_argument('--base-name', type=str, default='CTransPath')
parser.add_argument('--base-version', type=float, default=5.0)
parser.add_argument('--fold', type=int, default=0)  # used?

parser.add_argument('--save-folder', type=str,
                    default='Clustering')

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

parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()


def setup(args):

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    arch_kwargs = dict(
        dim_features=args.dim_features,
        dim_target=args.dim_target,
        layers=args.layers,
        dropout=args.dropout,
        pooling=args.pooling,
        conv=args.conv,
        aggr=args.aggr,
        gembed=args.gembed,
        scaler=args.scaler,
        temper=args.temper,
        use_mlp=args.use_mlp,
        mlp_version=args.mlp_version,
        mlp_dropout=args.mlp_dropout
    )

    root_output_dir = os.path.join(args.root_dir, f"{args.base_name}Base{args.base_version}")
    graph_dir = os.path.join(root_output_dir, 'graph/epithelium', args.graph_name, 'Validation')
    train_graph_dir = os.path.join(root_output_dir, 'graph/epithelium', args.graph_name, 'Train')

    model_dir = os.path.join(root_output_dir, 'model', args.model_name)
    # If extracting features need teacher
    assert os.path.exists(model_dir), f"Model directory does not exist at {model_dir}"

    scaler_path = f"{root_output_dir}/{args.model_name}_node_scaler.dat"
    results_save_path = os.path.join(root_output_dir, 'results', args.model_name)

    preds = pd.read_csv(os.path.join(results_save_path, 'fold_predictions'), index_col=0)
    best_epochs = preds.best_epoch

    feature_save_dir = os.path.join(args.save_folder, 'checkpoint', 'features', args.model_name)
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)

    return best_epochs, graph_dir, train_graph_dir, model_dir, arch_kwargs, scaler_path, feature_save_dir


def define_slides(graph_dir, args):
    wsi_names = [wsi.split('.')[0] for wsi in os.listdir(graph_dir)]

    test_slides = [wsi for wsi in wsi_names if cohort_from_name(wsi) == args.test_cohort]
    print(f'{len(test_slides)} test slides from test cohort {args.test_cohort}')

    train_slides = [wsi for wsi in wsi_names if cohort_from_name(wsi) in args.train_cohorts]
    print(f'{len(train_slides)} test slides from training cohorts')

    return test_slides, train_slides


def main(args):
    best_epochs, graph_dir, train_graph_dir, model_dir, arch_kwargs, scaler_path, feature_save_dir = setup(args)
    test_slides, train_slides = define_slides(graph_dir, args)

    print('\nExtracting source features')
    wsi_feats = predict_node_features_multiple(train_slides, args.responses, train_graph_dir, model_dir,
                                               best_epochs[0], arch_kwargs, fold=args.fold, preproc=args.preproc,
                                               scaler_path=scaler_path)
    all_feats = torch.vstack(list(wsi_feats.values()))
    print('Extracted source features of size:', all_feats.shape)

    #train_cohort_initials = train_cohorts_from_model_name(args.model_name)
    train_cohort_initials = ''.join([c[0] for c in args.train_cohorts])

    raw_feats_savename = os.path.join(feature_save_dir, f'feats_{train_cohort_initials}_full_seed_{args.seed}.pt')
    print(f'\nSaving raw source features to {raw_feats_savename}')
    torch.save(wsi_feats, raw_feats_savename)

    print('\nScaling source features')
    X_transformed, X_mean, X_std = torch_scaler(all_feats)
    print('X transformed:', X_transformed.shape)
    print('X mean:', X_mean.shape)
    print('X std:', X_std.shape)
    scaler = {'mean': X_mean, 'std': X_std}

    # should work for numpy or tensor version. Tensor version will be dict, numpy version is scaler itself.
    scaler_savename = os.path.join(feature_save_dir, f'scaler_{train_cohort_initials}_full_seed_{args.seed}.p')
    print(f'\nSaving scaler to {scaler_savename}')
    pickle.dump(scaler, open(scaler_savename, "wb"))

    scaled_feats_savename = os.path.join(feature_save_dir,
                                         f'feats_{train_cohort_initials}_full_scaled_seed_{args.seed}.pt')
    print(f'\nSaving scaled source features to {scaled_feats_savename}')
    torch.save(X_transformed, scaled_feats_savename)


if __name__ == '__main__':
    main(args)
