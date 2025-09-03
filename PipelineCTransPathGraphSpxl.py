import warnings
import os
import logging
import random
import pickle
import datetime
import copy

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# Need this for node scaler calculation but gives errors afterwards
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch_geometric
import argparse

warnings.filterwarnings("ignore")

#mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook, default 100

# local utils
from utils.visualisation import mkdir, recur_find_ext, rm_n_mkdir, load_json
from utils.spxl_graph import construct_superpixel_graph
from utils.data import SlideGraphEpiDataset, load_patch_labels, filter_wsis_by_epi_graphs, split_train_val, \
    make_label_df_with_slide_labels, find_base_data, get_mask_dir, get_epi_mask_dir, get_wsi_dir, filter_wsis, \
    dual_upsample, filter_wsis_by_mode_graphs, upsample_multiclass
from utils.model import select_checkpoints, SlideGraphArch
from utils.helper import reset_logging
from utils.metrics import create_resp_metric_dict, find_optimal_cutoff, threshold_predictions, metric_str_thresh_all, \
    create_multiclass_resp_metric_dict
from utils.plot import plot_confusion_matrix, density_plot
from utils.utils import str2bool

from superpixels import superpixel_feats_for_one_slide, patch_feats_for_one_slide
from graphs import construct_slidegraph
from train import run_once
from validate import validation_metrics, multiclass_validation_metrics, get_val_wsis_from_slide_df


########## Arguments ##########

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=3, choices=[0,1,2,3],
                    help='GPU number to use')
parser.add_argument('--seed', type=int, default=4, #choices=[1, 2, 3, 4],
                    help='set the seed for training and data split/fold. ')
parser.add_argument('--train-val-split', type=float, default=0.7, help='Propn of train cases vs val cases')

parser.add_argument('--clinical-file', type=str,
                    default="Metadata/PatchLabelsInclNAsTm20TGFb.csv",
                    help='CSV file where patch labels and other metadata is defined')

parser.add_argument('--mag', type=str, default='20X', choices=['5X', '10X', '20X'],
                    help='Magnification of patches')
parser.add_argument('--resp', nargs='+', default=['response_cr_nocr', 'CMS4', 'epithelium'],
                    help='List of response variables')
# use like python script.py --resp response_cr_nocr CMS4 epithelium -- etc.
parser.add_argument('--label-dim', type=int, nargs='+', default=[1, 1, 1],
                    help='Dimension of response labels e.g. 1 for binary, 4 for CMS')


#parser.add_argument('--cohorts', nargs='+', default=['GRAMPIAN', 'ARISTOTLE'], # SALZBURG
#                    help='List of cohorts to train and validate on')
parser.add_argument('--train-cohorts', nargs='+', default=['GRAMPIAN', 'ARISTOTLE'],
                    help='List of cohorts to train on')
parser.add_argument('--val-cohorts', nargs='+', default=[],  # ['SALZBURG']
                    help='Cohort(s) to validate on, default empty mean use training cohorts')

# Added new args
#parser.add_argument('--filter-epi', type=str2bool, default=True,
#                    help='Filter WSIs by those which have saved epithelial graphs already created') # do automatically
parser.add_argument('--upsample', type=str2bool, default=True,
                    help='Whether to upsample WSIs from minority classes. Generally always true.')
parser.add_argument('--shuffle-splits', type=str2bool, default=True,
                    help='Whether to shuffle WSIs in training/validation splits. Generally true.')
parser.add_argument('--resolution', default=5.0, type=float,
                    help='Resolution/magnification for graph generation')
parser.add_argument('--compactness', default=20.0, type=float,
                    help='Compactness parameter for SLIC algorithm')
parser.add_argument('--generate-graphs', default=False, type=str2bool,
                    help='Whether to generate graphs or use saved graphs. May depend on other parameters.')
parser.add_argument('--generate-superpixels', default=False, type=str2bool,
                    help='Whether to generate superpixels or use saved features. May depend on other parameters.')
parser.add_argument('--set-max-clusters', default=False, type=str2bool,
                    help='Whether to set max number of clusters in WSI graph')
parser.add_argument('--num-clusters', type=int, default=None, help='Number of clusters if setting maximum for graph')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for GNN training')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size for GNN training')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate for GNN')
parser.add_argument('--weight-decay', default=1.0e-4, type=float, help='Weight decay for learning for GNN')
###############

parser.add_argument('--base-name', default='CTransPath', type=str, choices=['CTransPath', 'DINO'],
                    help='Baseline model for patch/node features')
parser.add_argument('--base-version', default='5.1', type=str,
                    help='Baseline model version for patch/node features')

parser.add_argument('--scaler', default=False, type=str2bool,
                    help='True for trainable logistic regression (upside down results), False for nonparametric sigmoid')
parser.add_argument('--temper', type=float, default=1.5, help='Tempering output; 1.5 used for MICCAI; alt 0.1')

parser.add_argument('--connectivity-scale', help='Graph connectivity scale relative to size', type=int) # e.g. 8, 16, 20
parser.add_argument('--connectivity-dist', help='Graph connectivity absolute distance', type=int) # e.g. 800, 1000
parser.add_argument('--gembed', type=str2bool, default=False, help='Whether to gembed the GNN')
parser.add_argument('--superpixel', type=str2bool, default=True, help='True for MICCAI')
parser.add_argument('--scale-slic', type=int, default=2, help='Scale for SLIC algorithm, 8 for Salzburg, 2 otherwise')
parser.add_argument('--spxl-by-patch', type=str2bool, default=False, 
                    help='Number of superpixels ~ patches. False for MICCAI. Implemented after v5.x')
parser.add_argument('--with-stride', type=str2bool, default=False, 
                    help='To determine number of patches for calculating number of superpixels')
parser.add_argument('--remove-background', type=str2bool, default=False, 
                    help='Removing white background superpixels. Implemented after v5.x')


parser.add_argument('--mlp', type=str2bool, default=True, help='MLP layer for output')
parser.add_argument('--mlp-version', type=int, default=1,
                    help='MLP layer version for output. 1 is MICCAI version. 2 is ops.MLP applied earlier.')
parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'slidegraph'], help='Loss function')
parser.add_argument('--loss-weights', nargs='+', type=float, default=[1., 1., 1.],
                    help='Weights on respective response variables')

parser.add_argument('--remove-unclassified-cms4', type=str2bool, default=False,
                    help='Remove unclassified CMS4 WSIs from analysis (usually treated as not CMS4)')
parser.add_argument('--remove-unmatched-cms4', type=str2bool, default=False,
                    help='Remove unmatched CMS4 WSIs from analysis (usually treated as not CMS4)')

parser.add_argument('--preproc', type=str2bool, default=True,
                    help='Whether to preprocess and normalize the node features prior to GNN training')

parser.add_argument('--log', default=False, type=str2bool, help='Whether to log training in Tensorboard')
parser.add_argument('--dev-mode', default=False, type=str2bool, help='Whether to run reduced analysis in dev mode')

parser.add_argument('--epi-graph-dir-root', type=str,
                    default='checkpoint/',
                    help='Root directory where graphs for epithelium are saved. Base model details will be added.')
parser.add_argument('--root-dir', type=str,
                    default='checkpoint/',
                    help='Root directory where everything is saved. Base model details will be added.')

parser.add_argument('--conv', default='GINConv', type=str, choices=['GINConv', 'EdgeConv', 'GATConv'])
parser.add_argument('--layer-dims', default=[64, 32, 16], nargs='+', type=int, help='Layer dimensions in GNN')
parser.add_argument('--graph-agg', default='min', type=str, choices=['mean', 'max', 'min', 'sum', 'mul'],
                    help='Aggregation method for GNN')
parser.add_argument('--graph-pool', default='mean', type=str, choices=['mean', 'max', 'add'],
                    help='Pooling method for GNN')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout probability for GNN')
parser.add_argument('--mlp-dropout', default=0.1, type=float, help='Dropout probability for MLP heads')
parser.add_argument('--graph-cache-name', default='new', type=str)

parser.add_argument('--overwrite', default=False, type=str2bool,
                    help='Whether to write over existing model checkpoints')
parser.add_argument('--train-model', default=True, type=str2bool, help='Whether to train the GNN')
parser.add_argument('--feature-version', default=1, type=int, help='Directory for spxl features and graphs')


# 'slidegraph'
#'superpixel_5X_compactness_20_scaleslic_2' # for MICCAI
#'superpixel_upsample_connectivity_range_.125_gembed_true_temper_1.5_ginconv_scaleslic_2' # None

args = parser.parse_args()

########## Add defined arguments ##########
#setattr(args, 'base_version', f'4.0{args.seed}')
setattr(args, 'root_output_dir', os.path.join(args.root_dir, f"{args.base_name}Base{args.base_version}"))
#setattr(args, 'epi_graph_dir', os.path.join(args.root_dir,
#                                     f'{args.base_name}Base{args.base_version}/graph/epithelium'))

#if args.graph_cache_name == 'None':
#    setattr(args, 'graph_cache_name', None)
#elif args.graph_cache_name == 'default':
#    setattr(args, 'graph_cache_name', f'superpixel_5X_compactness_20_scaleslic_{args.scale_slic}')
if args.superpixel:
    spxl_feature_name = f'superpixel_{int(args.resolution)}X_compactness_{int(args.compactness)}_scaleslic_{args.scale_slic}'
    if args.spxl_by_patch:
        spxl_feature_name += '_patch_scaled'
    if args.remove_background:
        spxl_feature_name += '_filtered'
else:
    spxl_feature_name = 'patches'

if args.graph_cache_name == 'original':
    graph_name = spxl_feature_name
    connectivity_str = ""
elif args.connectivity_scale is not None:
    connectivity_str = f'_connect_scale_{str(args.connectivity_scale)}'
    graph_name = f'{spxl_feature_name}_connectivity_{args.connectivity_scale}'
elif args.connectivity_dist is not None:
    connectivity_str = f'_connect_dist_{str(args.connectivity_dist)}'
    graph_name = f'{spxl_feature_name}_connectivity_{args.connectivity_dist}'
else:
    connectivity_str = ""

if args.feature_version > 1:
    graph_name = f'{graph_name}_v{args.feature_version}'
    spxl_feature_name = f'{spxl_feature_name}_v{args.feature_version}'

setattr(args, 'graph_name', graph_name)

WSI_FEATURE_DIR = os.path.join(args.root_output_dir, 'features', spxl_feature_name)                             
#if float(args.base_version) >= 5.0:
    #WSI_FEATURE_DIR = os.path.join(WSI_FEATURE_DIR, f'seed_{args.seed}')
    # need seed for train/val split as features have diff augs
# replaced seed with train/val folders

# Set graph dir
GRAPH_DIR = f"{args.root_output_dir}/graph/epithelium/{args.graph_name}" #graph_cache_name
#if float(args.base_version) >= 5.0:
#    GRAPH_DIR = os.path.join(GRAPH_DIR, f'seed_{args.seed}')
    # need seed for train/val split as features have diff augs
# replaced seed with train/val folders
print('Graph dir:', GRAPH_DIR)

if args.set_max_clusters:
    print('Setting max number of clusters')
    GRAPH_DIR = os.path.join(f"{args.root_output_dir}/graph", f'{args.num_clusters}_clusters')
    CLUSTER_DIR = f"{args.root_output_dir}/clusters/{args.graph_name}"
    
setattr(args, 'epi_graph_dir', GRAPH_DIR)

setattr(args, 'cohorts', args.train_cohorts + args.val_cohorts)


loss_weights_str = 'weight_' + '_'.join(str(num) for num in args.loss_weights)
mlp_str = f"_mlp_{args.mlp_version}_dropout_{str(args.mlp_dropout).lstrip('0')}" if args.mlp else "" #_dropout_{str(args.mlp_dropout).lstrip('0')}"
print('args.layer_dims:', args.layer_dims)

setattr(args, 'layer_dims', list(args.layer_dims))
layer_str = 'layers_' + '_'.join(str(num) for num in args.layer_dims)
#if args.layer_dims==[64, 32, 16]:
#    layer_str += "_xlarge"
#elif args.layer_dims==[128, 64, 32, 16]:
#    layer_str += "_xxlarge"
#elif args.layer_dims==[32, 16]:
#    layer_str += "_large"
#else:
#    layer_str = ""

cohort_str = f'train_{"_".join(args.train_cohorts)}'
if len(args.val_cohorts) > 0:
    cohort_str = f'{cohort_str}_val_{"_".join(args.val_cohorts)}'

setattr(args, 'model_name', os.path.join("_".join(args.resp),
                                  cohort_str,
                                  f'{"superpixel" if args.superpixel else "patches"}_' +
                                  f'{"patch_scaled_" if args.spxl_by_patch else ""}' +
                                  f'{"filtered_" if args.remove_background else ""}' +
                                  f'{"rm_unmatched_" if args.remove_unmatched_cms4 else ""}' +
                                  f'{"rm_unclassified_" if args.remove_unclassified_cms4 else ""}' +
                                  f'{"upsample_" if args.upsample else ""}' +
                                  f'{"scaler_" if args.scaler else ""}' +
                                  f'{"preproc_false" if args.preproc == False else "normalize_train"}' +
                                  #f'_connectivity_range_{str(1/args.connectivity_scale).lstrip("0")[:4]}' +
                                  connectivity_str +
                                  f'_gembed_{str(args.gembed).lower()}_' +
                                  f'temper_{args.temper}_{args.conv.lower()}_dropout_{str(args.dropout).lstrip("0")}' +
                                  mlp_str +
                                  f'_{args.loss}' +
                                  f'{layer_str}_' +
                                  f'{args.graph_agg}_aggr_{args.graph_pool}_pool' +
                                  f'{loss_weights_str if (not all(it == 1 for it in args.loss_weights)) else ""}' +
                                  f'_fold_{args.seed}'))

########## Assert parameters as expected ##########
if args.superpixel and not args.spxl_by_patch:
    if 'SALZBURG' in args.cohorts:
        assert args.scale_slic == 8, f"Scale SLIC parameter ({args.scale_slic}) should be 8 for SALZBURG"
    else:
        assert args.scale_slic == 2, f"Scale SLIC parameter ({args.scale_slic}) should be 2 for GRAMPIAN/ARISTOTLE"
    
if args.with_stride and 'nostride' in args.clinical_file.lower():
    raise Exception("with-stride set to True but no stride metadata used")

########## Check if model already exists ##########
MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/", args.model_name)
if args.set_max_clusters:
    MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/{args.num_clusters}_clusters", args.model_name)
print('Model dir:', MODEL_DIR)

if os.path.exists(MODEL_DIR) and args.overwrite:
    print('WARNING: model directory already exists, set to overwrite results')
if not args.overwrite:
    while os.path.exists(MODEL_DIR):
        #if not args.overwrite:
        print('WARNING: overwrite set to False. Set --overwrite True to overwrite previous model.')
        model_version = args.model_name.split('_')[-1]
        if model_version.startswith('v'):
            version_number = model_version[1:]
            new_version_number = int(version_number) + 1
            new_model_name = args.model_name.replace(f'_v{version_number}', f'_v{new_version_number}')
        else:
            new_model_name = args.model_name + '_v1'
        # check extists again
        #setattr(args, 'model_name', new_model_name)
        #print(f'Model name updated to {new_model_name}')
        MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/", new_model_name)
        setattr(args, 'model_name', new_model_name)
        
        #else:
    #    pass

        print(f'Model name updated to {new_model_name}')


setattr(args, 'save_img_path', os.path.join(args.root_output_dir, 'visualisations', str(args.model_name)))

########## Set GPU ##########
torch.cuda.set_device(args.gpu)


########## Logging ##########
if args.log:
    sub_dir = 'tensorboard/graphspxl'
    if args.dev_mode:
        sub_dir = 'tensorboard_dev'
    tensorboard_dir = os.path.join(f'/well/rittscher/users/axs296/Code/SlideGraph/logs/{sub_dir}',
                                   f'{args.base_name}{args.base_version}', args.model_name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    current_time = str(datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S"))
    train_log_dir = tensorboard_dir + '/train/' + current_time
    val_log_dir = tensorboard_dir + '/val/' + current_time
    train_summary_writer = SummaryWriter(log_dir=train_log_dir)
    val_summary_writer = SummaryWriter(log_dir=val_log_dir)
else:
    train_summary_writer, val_summary_writer = None, None

########## Set seed ##########
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

########## Make directories ##########
if not args.dev_mode:
    mkdir(args.save_img_path)
    mkdir(args.root_output_dir)
    mkdir(f'{args.root_output_dir}/{"_".join(args.resp)}')


########## Load data ##########

wsi_dirs, msk_dirs, epi_msk_dirs = [], [], []
for cohort in args.cohorts:
    cohort_wsi_dir = get_wsi_dir(cohort)
    wsi_dirs.append(cohort_wsi_dir)
    setattr(args, f'WSI_DIR_{cohort.upper()}', cohort_wsi_dir)
    cohort_mask_dir = get_mask_dir(cohort)
    msk_dirs.append(cohort_mask_dir)
    setattr(args, f'MSK_DIR_{cohort.upper()}', cohort_mask_dir)
    cohort_epi_mask_dir = get_epi_mask_dir(cohort)
    epi_msk_dirs.append(cohort_epi_mask_dir)
    setattr(args, f'EPI_MSK_DIR_{cohort.upper()}', cohort_epi_mask_dir)


####### Cohort agnostic #######

# Load WSIs for which we have patch features saved
# for >v5.x, test=T/F doesn't matter and neither does seed
wsi_names, wsi_paths, msk_paths, epi_msk_paths, base_feature_dir = find_base_data(wsi_dirs, msk_dirs,
                                                                                  base_name=args.base_name,
                                                                                  base_version=args.base_version,
                                                                                  seed=args.seed,
                                                                                  epi_msk_dirs=epi_msk_dirs,
                                                                    test=False if args.cohorts!=['SALZBURG'] else True)
print('Wsi names:', len(wsi_names))
print('Wsi paths:', len(wsi_paths))
print('Mask paths:', len(msk_paths))
print('Epi mask paths:', len(epi_msk_paths))

### Now do this later on each train/val set instead of all at once
# Remove WSIs which don't have a graph saved, if already generated graphs
#if not args.generate_graphs:
#    wsi_names = filter_wsis_by_epi_graphs(wsi_names, args.filter_epi, args.epi_graph_dir,
#                                          args.graph_name if float(args.base_version) < 5.0 else None)


patch_labels = load_patch_labels(args.clinical_file, args.mag, args.resp, args.cohorts)
# set slide column to type string - for Salzburg int IDs
patch_labels.slide = patch_labels.slide.astype('str')

slide_df = patch_labels.groupby('slide').first().drop(['patch'], axis=1).reset_index()
if args.remove_unclassified_cms4:
    print(f'Removing {slide_df.CMS_matching.value_counts()["Unclassified"]} unclassified CMS from dataset')
    slide_df = slide_df[slide_df.CMS_matching != 'Unclassified'].reset_index(drop=True)
if args.remove_unmatched_cms4:
    print(f'Removing {slide_df.CMS_matching.value_counts()["Unmatched"]} unmatched CMS from dataset')
    slide_df = slide_df[slide_df.CMS_matching != 'Unmatched'].reset_index(drop=True)
# Select response columns, without dealing with possible epithelium response here
label_df, slide_responses = make_label_df_with_slide_labels(slide_df, responses=args.resp)


# Filter label_df based on WSIs we have features for
our_sel = np.where([wsi in wsi_names for wsi in label_df['WSI-CODE']])[0]
label_df = label_df.loc[our_sel].reset_index(drop=True)
# Drop Na labels
label_df = label_df.dropna().reset_index(drop=True)
print('Labels:', len(label_df))

# Redo wsi_names for normalizer and superpixels
wsi_names = label_df['WSI-CODE'].values

if args.dev_mode:
    idx = random.sample(range(len(wsi_names)), k=100)
    wsi_names = [wsi_names[i] for i in idx]
    print(wsi_names)
    label_df = label_df[label_df['WSI-CODE'].isin(wsi_names)]
    #wsi_paths = [wsi_paths[i] for i in idx]
    #msk_paths = [msk_paths[i] for i in idx]

assert len(label_df) > 0, "Problem loading WSI labels, none found"

########## Generating superpixels ##########
if args.base_name == 'CTransPath':
    NUM_NODE_FEATURES = 768
elif args.base_name == 'DINO':
    NUM_NODE_FEATURES = 384
else:
    raise Exception("Number of features per node not defined for this base model")

if args.generate_superpixels:
    print('WSI feature dir:', WSI_FEATURE_DIR)
    if args.superpixel:

        patch_labels.slide = patch_labels.slide.astype('str')
        patch_counts_per_slide = patch_labels.groupby('slide')['patch'].count()
        num_patches = None
        
        failed_spxl = []
        for wsi in sorted(wsi_names, reverse=True):
            print(f'Generating superpixels for {wsi}')
            if args.spxl_by_patch:
                # divide by 4 if 50% stride overlap
                num_patches = patch_counts_per_slide[wsi] / 4 if args.with_stride else patch_counts_per_slide[wsi]
            
            try:
                _, _ = superpixel_feats_for_one_slide(wsi, wsi_paths=wsi_paths, mask_paths=msk_paths, 
                                                      epi_msk_paths=epi_msk_paths,
                                                      wsi_feature_dir=WSI_FEATURE_DIR,
                                                      scale_slic=args.scale_slic,
                                                      base_name=args.base_name,
                                                      base_version=args.base_version,
                                                      seed=args.seed, # not used for >=v5.0
                                                      num_node_features=NUM_NODE_FEATURES,
                                                      train_or_val='both',
                                                      num_patches=num_patches,
                                                      remove_background=args.remove_background,
                                                      resolution=args.resolution, mag=args.mag,
                                                      compactness=args.compactness,
                                                      save_feats=True, jit=False)

            except Exception as e:
                print(f'Couldn\'t generate superpixels for slide {wsi}. \nError: {e}')
                failed_spxl.append(wsi)

        #for wsi in failed_spxl:
        #    wsi_names = list(wsi_names)
        #    wsi_names.remove(wsi)
    else:
        failed_spxl = []
        for wsi in sorted(wsi_names, reverse=True):
            print(f'Generating patch features for {wsi}')
            try:
                _, _ = patch_feats_for_one_slide(wsi, wsi_paths, epi_msk_paths, WSI_FEATURE_DIR, base_name=args.base_name,
                                                 base_version=args.base_version, seed=args.seed, train_or_val='both',
                                                 resolution=args.resolution, mag=args.mag, save_feats=True)
            except Exception as e:
                print(f'Couldn\'t generate patch features for slide {wsi}. \nError: {e}')
                failed_spxl.append(wsi)

    for wsi in failed_spxl:
        wsi_names = list(wsi_names)
        wsi_names.remove(wsi)


########## Generating graphs ##########
if args.generate_graphs:
    # check train and val dirs separately
    train_wsis = np.unique([file.split('.')[0] for file in os.listdir(os.path.join(WSI_FEATURE_DIR, 'Train'))]).tolist()
    val_wsis = np.unique([file.split('.')[0] for file in os.listdir(os.path.join(WSI_FEATURE_DIR,
                                                                                 'Validation'))]).tolist()
    wsi_names = np.unique(train_wsis + val_wsis)
    #wsi_names = np.unique([file.split('.')[0] for file in os.listdir(WSI_FEATURE_DIR)])


    print(f'Generating graphs for {len(wsi_names)} slides')
    mkdir(args.epi_graph_dir)
    
    if not args.set_max_clusters:
        for mode in ['Train', 'Validation']:
            mkdir(os.path.join(args.epi_graph_dir, mode))
            for wsi in wsi_names:
                construct_superpixel_graph(wsi, save_path=f"{args.epi_graph_dir}/{mode}/{wsi}.json",
                                           connectivity_scale=args.connectivity_scale,
                                           connectivity_dist=args.connectivity_dist,
                                           wsi_feature_dir=os.path.join(WSI_FEATURE_DIR, mode),
                                           add_epi=True)  # assumes binary_epi_labels exist in feature dir
    else:
        coords_clusters_dict, coords_max_clusters_dict = {}, {}
        for mode in ['Train', 'Validation']:
            mkdir(os.path.join(args.epi_graph_dir, mode))
            for wsi in wsi_names:
                coords_clusters, coords_max_clusters = construct_slidegraph(wsi,
                                                                    save_path=f"{args.epi_graph_dir}/{mode}/{wsi}.json",
                                                                            base_name=args.base_name,
                                                                            base_version=args.base_version,
                                                                            seed=args.seed,
                                                                            set_max_clusters=args.set_max_clusters,
                                                                            train_or_val=mode)


                if coords_clusters is None:
                    continue
                coords_clusters_dict[wsi] = coords_clusters
                coords_max_clusters_dict[wsi] = coords_max_clusters

            mkdir(os.path.join(CLUSTER_DIR, mode))
            pickle.dump(coords_clusters_dict, open(f'{CLUSTER_DIR}/{mode}/graph_clusters.p', 'wb'))
            if args.set_max_clusters:
                pickle.dump(coords_max_clusters_dict,
                            open(f'{CLUSTER_DIR}/{mode}/graph_{args.num_clusters}_clusters.p', 'wb'))


# Exit if not training GNN
if not args.train_model:
    exit()


########## Create training data splits ##########

# splits is list of length 1 (num_folds). In list is dictionary with keys ['train', 'valid', 'test'].
# Each dict value is list of tuples, tuples of length two, with slide name and response value.

#split_cache_path = f"{args.root_output_dir}/shuffle_splits.dat"

# Define
mkdir(f"{args.root_output_dir}/{args.model_name}")
SPLIT_PATH = os.path.join(f"{args.root_output_dir}/{args.model_name}",
                          f"{'shuffle_' if args.shuffle_splits else ''}splits.dat")

NUM_FOLDS = 1

if float(args.base_version) >= 5.0:
    if len(args.val_cohorts) == 0:
        # use train_cohorts and split all cohorts by case in training set for both train and val
        train_wsis, val_wsis = split_train_val(label_df, train_val_split=args.train_val_split, seed=args.seed)

    else:
        # use train_cohorts for training and val_cohorts for val, no ratios required
        train_wsis = label_df[label_df.cohort.isin(args.train_cohorts)]['WSI-CODE']
        val_wsis = label_df[label_df.cohort.isin(args.val_cohorts)]['WSI-CODE']
else:
    if args.train_cohorts == ['SALZBURG']:
         train_wsis, val_wsis = split_train_val(label_df, train_val_split=args.train_val_split, seed=args.seed)
    else:
         train_wsis = sorted(os.listdir(os.path.join(base_feature_dir, 'Train')))
         val_wsis = sorted(os.listdir(os.path.join(base_feature_dir, 'Validation')))

# Filter WSIs for those which we have labels
train_wsis = filter_wsis(train_wsis, label_df)
val_wsis = filter_wsis(val_wsis, label_df)

# Filter WSIs for those which we have graphs
train_wsis = filter_wsis_by_mode_graphs(train_wsis, args.epi_graph_dir, 'Train')
val_wsis = filter_wsis_by_mode_graphs(val_wsis, args.epi_graph_dir, 'Validation')

#wsi_names = filter_wsis_by_epi_graphs(wsi_names, args.filter_epi, args.epi_graph_dir,
#                                          args.graph_name if float(args.base_version) < 5.0 else None)

if args.shuffle_splits:
    random.seed(args.seed)  # changed from 0 after DINO1.11 first two models
    random.shuffle(train_wsis)
    random.shuffle(val_wsis)
    print('Shuffled wsis')

train_labels = [
    label_df[label_df['WSI-CODE'] == slide][[f'LABEL_{i}' for i in range(len(slide_responses))]].values.tolist()[0] for
    slide in train_wsis]
val_labels = [
    label_df[label_df['WSI-CODE'] == slide][[f'LABEL_{i}' for i in range(len(slide_responses))]].values.tolist()[0] for
    slide in val_wsis]

print('Train and validation set response distribution (across both labels):')
print(f'    {np.unique(train_labels, return_counts=True)}')
print(f'    {np.unique(val_labels, return_counts=True)}')

if not args.dev_mode:
    for i in range(len(slide_responses)):
        print(
            f'There are {np.unique(np.array(val_labels)[:, i], return_counts=True)[1][1]} positive {slide_responses[i]} slides in' +
            ' the validation set')

########## Scale graph features before upsampling ##########


SCALER_PATH = f"{args.root_output_dir}/{args.model_name}_{'clusters_' if args.set_max_clusters else ''}node_scaler.dat"

if args.preproc:
    print('Checking if path exists:', SCALER_PATH)
    if os.path.exists(SCALER_PATH):
        print('Using existing node scaler')
        node_scaler = joblib.load(SCALER_PATH)
    else:
        #if NODE_PREDICTION:
        # Use Train graphs (all WSIs with train augmentations) for scaler
        dataset = SlideGraphEpiDataset(train_wsis, graph_dir=os.path.join(args.epi_graph_dir, 'Train'), mode="infer")  # no labels
        loader = torch_geometric.loader.DataLoader(
            dataset, num_workers=8, batch_size=1, shuffle=False, drop_last=False
        )
        node_features = [v[0]["graph"].x for idx, v in enumerate(tqdm(loader))]
        node_features = torch.cat(node_features, dim=0)
        print('Node features before scaling:', node_features.shape)
        # line errors if >960 slides
        #node_features = [v[0]["graph"].x.numpy() for idx, v in enumerate(tqdm(loader))]

        node_scaler = StandardScaler(copy=False) # Standardize features by removing the mean and scaling to unit variance.
        node_scaler.fit(node_features)
        if not args.dev_mode:
            print(f'Saving node scaler to {SCALER_PATH}')
            joblib.dump(node_scaler, SCALER_PATH)

    # we must define the function after training/loading
    def nodes_preproc_func(node_feats):
        return node_scaler.transform(node_feats)

########## Finish defining data splits ##########

if args.upsample:
    multiclass_idx = np.where([d > 1 for d in args.label_dim])[0]
    if len(multiclass_idx) > 0:
        for i in multiclass_idx:
            train_wsis, train_labels = upsample_multiclass(train_wsis, train_labels, i, seed=args.seed)
    else:
        train_wsis, train_labels = dual_upsample(train_wsis, train_labels, slide_responses)
    #print('train_labels:', train_labels)
    #print()
    print(f'{len(train_wsis)} slides in training set after upsampling')
    print('Train set response distribution (across both labels) after upsampling:')
    print(f'    {np.unique(train_labels, return_counts=True)}')

print('Number of train slides:', len(train_wsis))
print('Number of validation slides:', len(val_wsis))

assert len(set(val_wsis).intersection(set(train_wsis))) == 0, \
    f"Train and Validation overlap by {len(set(val_wsis).intersection(set(train_wsis)))} slides"

# Redo wsi_names based on train and val
wsi_names = list(set(val_wsis).union(set(train_wsis)))

splits = []
splits.append(
    {
        "train": list(zip(train_wsis, train_labels)),
        "valid": list(zip(val_wsis, val_labels)),
        # "test": list(zip(val_wsis, val_labels)),
    }
)

# Save splits
if not args.dev_mode:
    joblib.dump(splits, SPLIT_PATH)


########## Train model ##########
NUM_EPOCHS = 5 if args.dev_mode else args.epochs

torch.autograd.set_detect_anomaly(True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not args.dev_mode:
    splits = joblib.load(SPLIT_PATH)
if args.preproc:
    if not args.dev_mode:
        node_scaler = joblib.load(SCALER_PATH)

    def nodes_preproc_func(node_feats):
        return node_scaler.transform(node_feats)
else:
    nodes_preproc_func = None

loader_kwargs = dict(
    num_workers=8,
    batch_size=args.batch_size if not args.dev_mode else 16,  # RW: can't have batch_size bigger than dataset. changed from 16 to 4.
)

arch_kwargs = dict(
    dim_features=NUM_NODE_FEATURES,
    dim_target=max(args.label_dim),  # RW: changed from 1 to 4
    layers=args.layer_dims,  # changed from [16, 16, 8], xlarge is [64, 32, 16], xxlarge is [128, 64, 32, 16]
    dropout=args.dropout,  # changed from 0.5 to 0.3
    pooling=args.graph_pool,  # changed from mean to max
    conv=args.conv,
    aggr=args.graph_agg,  # changed from max to min
    gembed=args.gembed,
    scaler=args.scaler,
    temper=args.temper,
    use_mlp=args.mlp,
    mlp_version=args.mlp_version,
    mlp_dropout=args.mlp_dropout,
    label_dim=args.label_dim, # added to GNNMLPv4
)
optim_kwargs = dict(
    lr=args.lr,  # RW: changed from 1e-3 to 1e-1
    weight_decay=args.weight_decay,
)

logging.basicConfig(
    level=logging.INFO,
)

for split_idx, split in enumerate(splits):
    new_split = {"train": split["train"]}
    if args.scaler:
        new_split.update({"infer-train": split["train"]})  # adding copy of training dataset called infer-train
    new_split.update({"infer-valid-A": split["valid"]})
    # "infer-valid-B": split["test"], # Same as validation for now

    split_save_dir = f"{MODEL_DIR}/{split_idx:02d}/"
    rm_n_mkdir(split_save_dir)
    reset_logging(split_save_dir)
    out, wsis = run_once(
        resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
        preproc=args.preproc, temper=args.temper,
        dataset_dict=new_split,
        num_epochs=NUM_EPOCHS,
        graph_dir=args.epi_graph_dir,
        save_dir=split_save_dir,
        nodes_preproc_func=nodes_preproc_func,
        dev_mode=args.dev_mode,
        train_summary_writer=train_summary_writer,
        val_summary_writer=val_summary_writer,
        pretrained=None,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        optim_kwargs=optim_kwargs,
    )

if args.log:
    train_summary_writer.close()

########## Save losses ##########
if not args.dev_mode:
    for split_idx, split in enumerate(splits):

        stats_dict = load_json(recur_find_ext(f"{MODEL_DIR}/{split_idx:02d}/", [".json"])[0])
        # keys are strings of epochs. Each value contains dict of loss and metrics.
        train_losses = [d['train-EMA-loss'] for d in stats_dict.values()]
        val_losses = [d['infer-valid-A-loss'] for d in stats_dict.values()]
        np.save(f"{MODEL_DIR}/{split_idx:02d}/train_losses.npy", train_losses)
        np.save(f"{MODEL_DIR}/{split_idx:02d}/val_losses.npy", val_losses)

        mpl.rcParams["figure.dpi"] = 100
        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(MODEL_DIR, f'{split_idx:02d}', 'loss_plot.png'))
        #plt.show()


########## Inference ##########
if args.dev_mode:
    exit()

TOP_K = 1
#metric_name = f"{RESP[0]}-infer-valid-A-auroc" # choose best model based on first response only
metric_name = 'infer-valid-A-auroc' # choose based on all responses

PRETRAINED_DIR = MODEL_DIR

splits = joblib.load(SPLIT_PATH)
if args.preproc:
    node_scaler = joblib.load(SCALER_PATH)
    print('Loading node scaler')
    def nodes_preproc_func(node_feats):
        return node_scaler.transform(node_feats)

# need loader_kwargs and arch_kwargs defined, usually from training in same go

cum_stats, cum_preds = [], []
for split_idx, split in enumerate(splits):
    new_split = {'valid': split["valid"]}  # want valid to return epi label

    stat_files = recur_find_ext(f"{PRETRAINED_DIR}/{split_idx:02d}/", [".json"])
    print(stat_files)
    stat_files = [v for v in stat_files if ".old.json" not in v]
    print(stat_files)
    assert len(stat_files) == 1
    chkpts, chkpt_stats_list, best_epoch = select_checkpoints(
        stat_files[0], top_k=TOP_K, metrics=[metric_name]
    )

    # Perform ensembling by averaging probabilities across checkpoint predictions
    cum_results = []
    for chkpt_info in chkpts:
        chkpt_results, wsis = run_once(
            resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
            preproc=args.preproc, temper=args.temper,
            dataset_dict=new_split,
            num_epochs=1,
            graph_dir=args.epi_graph_dir,
            save_dir=None,
            nodes_preproc_func=nodes_preproc_func,
            dev_mode=args.dev_mode,
            val_summary_writer=val_summary_writer,
            pretrained=chkpt_info,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs
        )

        # * re-calibrate logit to probabilities
        chkpt_results = np.array(chkpt_results)
        chkpt_results = np.squeeze(chkpt_results)

        cum_results.append(chkpt_results)
    cum_results = np.array(cum_results)
    if len(args.resp) > 1:
        cum_results = np.squeeze(cum_results)
    
    
    # Generalize for different number of responses with node predictions always last (but check)
    ####################
    metric_dict = {}
    pred_dict = {
        "fold": split_idx, "best_epoch": best_epoch[split_idx],
    }
    all_mets = []
    for i in range(len(args.resp)):
        #node_level = False
        #if 'epithelium' in args.resp[i]:
        #    node_level = True
        
        output_logit, output_true = [], []

        for out in cum_results:
            if 'epithelium' in args.resp[i]:
                output_logit.extend([out_[0] for out_ in out[i:]])
                output_true.extend([out_[1] for out_ in out[i:]])
            elif args.resp[i] in ['CMS', 'CMS_matching']:
                output_logit.extend([out_[0][0] for out_ in out[i:]])
                output_true.extend([out_[0][1] for out_ in out[i:]])
            else:
                output_logit.append(out[i][0])
                output_true.append(out[i][1])
        ###############  these are scalers ? ###############
                
        output_logit = np.array(output_logit, dtype=np.float16)
        output_true = np.array(output_true)

        if args.resp[i] in ['CMS', 'CMS_matching']:
            resp_mets = create_multiclass_resp_metric_dict(args.resp[i], output_true, output_logit,
                                                                  best_epoch[split_idx])
            metric_dict.update(resp_mets)
            probs = torch.nn.functional.softmax(torch.Tensor(output_logit), dim=1).numpy()
            pred = np.argmax(probs, axis=1)
            # define preds as argmax, logits as probs
            pred_dict.update({f"{args.resp[i]}_preds": pred, f"{args.resp[i]}_true": output_true,
                              f"{args.resp[i]}_probs": probs})

        else:
            metric_dict.update(create_resp_metric_dict(args.resp[i], output_true, output_logit, best_epoch[split_idx]))
        
            # Add thresholded metrics
            print('    Using thresholding from all cohorts')
            threshold = find_optimal_cutoff(output_true, output_logit)
            resp_mets = create_resp_metric_dict(args.resp[i], output_true, output_logit, best_epoch[split_idx],
                                                cutoff=threshold)
            resp_mets = {'threshold-' + k: v for k, v in resp_mets.items() if not k == 'best_epoch'}
            resp_mets[f'{args.resp[i]}-threshold'] = threshold[0]
            metric_dict.update(resp_mets)

            pred_dict.update({f'{args.resp[i]}-threshold': threshold[0]})
            pred_dict.update({f"{args.resp[i]}_preds": output_logit, f"{args.resp[i]}_true": output_true})

        
        # Print metrics in table format
        all_mets.append(resp_mets)
        


    cum_stats.append(metric_dict)
    if args.log:
        hparams = vars(args).copy()
        hparams['layer_dims'] = '_'.join(str(num) for num in hparams['layer_dims'])
        hparams['cohorts'] = '_'.join(str(cohort) for cohort in hparams['cohorts'])
        hparams['train_cohorts'] = '_'.join(str(cohort) for cohort in hparams['train_cohorts'])
        hparams['val_cohorts'] = '_'.join(str(cohort) for cohort in hparams['val_cohorts'])
        hparams['loss_weights'] = '_'.join(str(num) for num in hparams['loss_weights'])
        hparams['resp'] = '_'.join(str(response) for response in hparams['resp'])
        hparams['label_dim'] = '_'.join(str(lbl_dim) for lbl_dim in hparams['label_dim'])

    cum_preds.append(pred_dict)


# Save metrics
print(args.base_name, args.base_version, args.model_name)
stat_df = pd.DataFrame(cum_stats)
for metric in stat_df.columns:
    vals = stat_df[metric]
    mu = np.mean(vals)
    va = np.std(vals)
    print(f"- {metric}: {mu:0.4f}±{va:0.4f}")

results_save_path = os.path.join(args.root_output_dir, 'results', args.model_name)
if not os.path.exists(results_save_path):
    mkdir(results_save_path)
stat_df.to_csv(os.path.join(results_save_path, 'mean_best_metrics_over_folds'), index=False)

preds_df = pd.DataFrame(cum_preds)
preds_df.to_csv(os.path.join(results_save_path, 'fold_predictions'), index=False)

# Save confusion matrices and prediction density plots
viz_fold = 0
viz_epoch = best_epoch[int(viz_fold)]

met_args = [resp for resp in args.resp if resp!='cohort_cls'] # exclude cohort_cls
#met_args = args.resp[:2] + [args.resp[-1]]  # exclude any third value in case is cohort_cls

for response in list(met_args):
    print(response)
    resp_true = preds_df[f'{response}_true'][0]
    resp_preds = preds_df[f'{response}_preds'][0]

    if response in ['CMS', 'CMS_matching']:
        conf_preds = resp_preds
    else:
        conf_preds = threshold_predictions(resp_true, resp_preds)
    confusion_fig = plot_confusion_matrix(resp_true, conf_preds, response, viz_fold, viz_epoch, save=True,
                                          save_img_path=args.save_img_path, thresh='CMS' not in response)
    density_fig = density_plot(resp_true, resp_preds, response, viz_fold, viz_epoch, save=True,
                               save_img_path=args.save_img_path)
    if args.log:
        val_summary_writer.add_figure(f'Validation Confusion Matrix{f" with Threshold" if "CMS" not in response else ""} - {response}', confusion_fig)
        val_summary_writer.add_figure(f'Validation Density Plot - {response}', density_fig)

# Print metrics in table format
#all_mets = [resp_0_mets, resp_1_mets, resp_2_mets]
print('(Thresholded) metrics printed below - can be used in Notebook table')
print()
to_threshold = not any(v in met_args for v in ['CMS', 'CMS_matching'])
print(f'| {args.base_name} {args.base_version} | {"/".join(args.model_name.split("/")[1:2])} |' +\
      metric_str_thresh_all(all_mets, met_args, 'auroc', threshold=to_threshold) +
      metric_str_thresh_all(all_mets, met_args, 'balanced_acc', threshold=to_threshold) +
      metric_str_thresh_all(all_mets, met_args, 'weighted_f1', threshold=to_threshold))
print()


# Check validation metrics on different cohorts

if len(args.val_cohorts) > 0:
    cohorts_in_val = args.val_cohorts
else:
    cohorts_in_val = args.train_cohorts
for cohort in cohorts_in_val:
    cohort_split = get_val_wsis_from_slide_df(cohort, new_split, slide_df)
    print(cohort)
    if args.resp in ['CMS', 'CMS_matching']:
        cohort_metric_dict = multiclass_validation_metrics(cohort_split, chkpts[0], best_epoch[0], arch_kwargs,
                       loader_kwargs, args, nodes_preproc_func, val_summary_writer)
    else:
        cohort_metric_dict = validation_metrics(cohort_split, chkpts[0], best_epoch[0], arch_kwargs,
                       loader_kwargs, args, nodes_preproc_func, val_summary_writer)
    cohort_metric_dict = {f'{cohort}-{k}': v for k, v in cohort_metric_dict.items()}
    metric_dict.update(cohort_metric_dict)

val_summary_writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)
