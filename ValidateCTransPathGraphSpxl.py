import os
import joblib
import numpy as np
import pandas as pd
import argparse
import random
import datetime

# Local utils
from utils.utils import str2bool
from utils.visualisation import mkdir, recur_find_ext
from utils.model import select_checkpoints, SlideGraphArch
from utils.metrics import create_resp_metric_dict, find_optimal_cutoff, threshold_predictions, metric_str_thresh_all, \
    create_multiclass_resp_metric_dict
from utils.plot import plot_confusion_matrix, density_plot
from utils.data import get_mask_dir, get_epi_mask_dir, get_wsi_dir, find_base_data, load_patch_labels, \
    make_label_df_with_slide_labels, filter_wsis, filter_wsis_by_mode_graphs


from train import run_once
from validate import validation_metrics, multiclass_validation_metrics, get_val_wsis_from_slide_df

import torch
from torch.utils.tensorboard import SummaryWriter


########## Arguments ##########
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=3, choices=[0, 1, 2, 3], help='GPU number to use')
parser.add_argument('--seed', type=int, default=4, #choices=[1, 2, 3, 4],
                    help='set the seed for training and data split/fold. ')

parser.add_argument('--resp', nargs='+', default=['response_cr_nocr', 'CMS4', 'epithelium'],
                    help='List of response variables')
# use like python script.py --resp response_cr_nocr CMS4 epithelium -- etc.
parser.add_argument('--label-dim', type=int, nargs='+', default=[1, 1, 1],
                    help='Dimension of response labels e.g. 1 for binary, 4 for CMS')

parser.add_argument('--clinical-file', type=str,
                    default="Metadata/PatchLabelsInclNAsTm20TGFb.csv",
                    help='CSV file where patch labels and other metadata is defined'
                    )
parser.add_argument('--mag', type=str, default='20X', choices=['5X', '10X', '20X'],
                    help='Magnification of patches')

parser.add_argument('--train-cohorts', nargs='+', default=['GRAMPIAN', 'ARISTOTLE'],
                    help='List of cohorts to train on')
parser.add_argument('--val-cohorts', nargs='+', default=[],  # ['SALZBURG']
                    help='Cohort(s) to validate on, default empty means splitting training cohorts')
parser.add_argument('--test-cohorts', nargs='+', default=[],  # ['SALZBURG']
                    help='Unseen cohort(s) to test on, default empty means evaluating validation cohorts')

parser.add_argument('--upsample', type=str2bool, default=True,
                    help='Whether to upsample WSIs from minority classes. Generally always true.')
parser.add_argument('--shuffle-splits', type=str2bool, default=True,
                    help='Whether to shuffle WSIs in training/validation splits. Generally true.')
parser.add_argument('--resolution', default=5.0, type=float,
                    help='Resolution/magnification for graph generation')
parser.add_argument('--compactness', default=20.0, type=float,
                    help='Compactness parameter for SLIC algorithm')

parser.add_argument('--set-max-clusters', default=False, type=str2bool,
                    help='Whether to set max number of clusters in WSI graph')
parser.add_argument('--num-clusters', type=int, default=None, help='Number of clusters if setting maximum for graph')

parser.add_argument('--base-name', default='CTransPath', type=str, choices=['CTransPath', 'DINO'],
                    help='Baseline model for patch/node features')
parser.add_argument('--base-version', default='4.04', type=str,
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
parser.add_argument('--remove-background', type=str2bool, default=False,
                    help='Removing white background superpixels. Implemented after v5.x')


parser.add_argument('--mlp', type=str2bool, default=True, help='MLP layer for output')
parser.add_argument('--mlp-version', type=int, default=1,
                    help='MLP layer version for output. 1 is MICCAI version. 2 is ops.MLP applied earlier.')
# For model name
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

parser.add_argument('--model-version', default=1, type=int,
                    help='Version of saved model to validate')
parser.add_argument('--feature-version', default=1, type=int, help='Directory for spxl features and graphs')
parser.add_argument('--metric-name', default='infer-valid-A-auroc', type=str, help='Metric to determine best epoch',
                    choices=['infer-valid-A-auroc', 'response_cr_nocr-infer-valid-A-auroc',
                             'response_cr_nocr-infer-valid-A-balanced-acc'])
parser.add_argument('--save-predictions', default=True, type=str2bool, help='Save node predictions (T/F)')


args = parser.parse_args()
########## Add defined arguments ##########
setattr(args, 'root_output_dir', os.path.join(args.root_dir, f"{args.base_name}Base{args.base_version}"))

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


# Set graph dir
GRAPH_DIR = f"{args.root_output_dir}/graph/epithelium/{args.graph_name}"  # graph_cache_name
print('Graph dir:', GRAPH_DIR)

if args.set_max_clusters:
    print('Setting max number of clusters')
    GRAPH_DIR = os.path.join(f"{args.root_output_dir}/graph", f'{args.num_clusters}_clusters')
    CLUSTER_DIR = f"{args.root_output_dir}/clusters/{args.graph_name}"

setattr(args, 'epi_graph_dir', GRAPH_DIR)

# just train and validation cohorts for model name identification
setattr(args, 'cohorts', args.train_cohorts + args.val_cohorts)

loss_weights_str = 'weight_' + '_'.join(str(num) for num in args.loss_weights)
mlp_str = f"_mlp_{args.mlp_version}_dropout_{str(args.mlp_dropout).lstrip('0')}" if args.mlp else ""  # _dropout_{str(args.mlp_dropout).lstrip('0')}"
print('args.layer_dims:', args.layer_dims)

setattr(args, 'layer_dims', list(args.layer_dims))
layer_str = 'layers_' + '_'.join(str(num) for num in args.layer_dims)
#if args.layer_dims == [64, 32, 16]:
#    layer_str += "_xlarge"
#elif args.layer_dims == [128, 64, 32, 16]:
#    layer_str += "_xxlarge"
#elif args.layer_dims == [32, 16]:
#    layer_str += "_large"
# else:
#    layer_str = ""

cohort_str = f'train_{"_".join(args.train_cohorts)}'
if len(args.val_cohorts) > 0:
    cohort_str = f'{cohort_str}_val_{"_".join(args.val_cohorts)}'
#if len(args.test_cohorts) > 0:
#    cohort_str = f'{cohort_str}_test_{"_".join(args.test_cohorts)}'

setattr(args, 'model_name', os.path.join("_".join(args.resp),
                                         cohort_str,
                                         #"_".join(args.cohorts) + '_' +
                                         f'{"superpixel" if args.superpixel else "patches"}_' +
                                         f'{"patch_scaled_" if args.spxl_by_patch else ""}' +
                                         f'{"filtered_" if args.remove_background else ""}' +
                                         f'{"rm_unmatched_" if args.remove_unmatched_cms4 else ""}' +
                                         f'{"rm_unclassified_" if args.remove_unclassified_cms4 else ""}' +
                                         f'{"upsample_" if args.upsample else ""}' +
                                         f'{"scaler_" if args.scaler else ""}' +
                                         f'{"preproc_false" if args.preproc == False else "normalize_train"}' +
                                         # f'_connectivity_range_{str(1/args.connectivity_scale).lstrip("0")[:4]}' +
                                         connectivity_str +
                                         f'_gembed_{str(args.gembed).lower()}_' +
                                         f'temper_{args.temper}_{args.conv.lower()}_dropout_{str(args.dropout).lstrip("0")}' +
                                         mlp_str +
                                         #f'temper_{args.temper}_ginconv{mlp_str}' +
                                         f'_{args.loss}' +
                                         f'{layer_str}_' +
                                         f'{args.graph_agg}_aggr_{args.graph_pool}_pool' +
                                         f'{loss_weights_str if (not all(it == 1 for it in args.loss_weights)) else ""}' +
                                         f'_fold_{args.seed}' +
                                         f'{f"_v{args.model_version}" if args.model_version > 1 else ""}'
                                         ))  # add to model_name with _v{model_version}

########## Assert parameters as expected ##########
# Set model directory
MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/", args.model_name)
if args.set_max_clusters:
    MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/{args.num_clusters}_clusters", args.model_name)
print('Model dir:', MODEL_DIR)

setattr(args, 'save_img_path', os.path.join(args.root_output_dir, 'visualisations', str(args.model_name)))

########## Set GPU ##########
torch.cuda.set_device(args.gpu)

########## Inference ##########

## Can't update validation summary writer log, so make new postval one
if args.log:
    sub_dir = 'tensorboard/graphspxl'
    if args.dev_mode:
        sub_dir = 'tensorboard_dev'
    tensorboard_dir = os.path.join(f'logs/{sub_dir}',
                                   f'{args.base_name}{args.base_version}', args.model_name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    current_time = str(datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S"))
    val_log_dir = os.path.join(tensorboard_dir, 'postval')
    if len(args.test_cohorts) > 0:
        val_log_dir = os.path.join(val_log_dir,
                                   f'test_{"_".join(args.test_cohorts)}' if len(args.test_cohorts) > 0 else '')
    val_log_dir = os.path.join(val_log_dir, current_time)
    val_summary_writer = SummaryWriter(log_dir=val_log_dir)
else:
    val_summary_writer = None

########## Set seed ##########
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

########## Load data ##########

TOP_K = 1
#metric_name = f"{RESP[0]}-infer-valid-A-auroc" # choose best model based on first response only
#metric_name = 'infer-valid-A-auroc' # choose based on all responses

PRETRAINED_DIR = MODEL_DIR

SPLIT_PATH = os.path.join(f"{args.root_output_dir}/{args.model_name}",
                          f"{'shuffle_' if args.shuffle_splits else ''}splits.dat")

splits = joblib.load(SPLIT_PATH)

SCALER_PATH = f"{args.root_output_dir}/{args.model_name}_{'clusters_' if args.set_max_clusters else ''}node_scaler.dat"
if args.preproc:
    node_scaler = joblib.load(SCALER_PATH)

    def nodes_preproc_func(node_feats):
        return node_scaler.transform(node_feats)
else:
    nodes_preproc_func = None

if args.base_name == 'CTransPath':
    NUM_NODE_FEATURES = 768
elif args.base_name == 'DINO':
    NUM_NODE_FEATURES = 384
else:
    raise Exception("Number of features per node not defined for this base model")


# need loader_kwargs and arch_kwargs defined, usually from training in same go
loader_kwargs = dict(
    num_workers=8,
    batch_size=1,
)

arch_kwargs = dict(
    dim_features=NUM_NODE_FEATURES,
    dim_target=1,  # RW: changed from 1 to 4
    layers=args.layer_dims,  # changed from [16, 16, 8], xlarge is [64, 32, 16]
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


cum_stats, cum_preds = [], []
for split_idx, split in enumerate(splits):
    new_split = {'valid': split["valid"]}  # want valid to return epi label

    stat_files = recur_find_ext(f"{PRETRAINED_DIR}/{split_idx:02d}/", [".json"])
    print(stat_files)
    stat_files = [v for v in stat_files if ".old.json" not in v]
    print(stat_files)
    assert len(stat_files) == 1
    chkpts, chkpt_stats_list, best_epoch = select_checkpoints(
        stat_files[0], top_k=TOP_K, metrics=[args.metric_name]
    )

    # Perform ensembling by averaging probabilities across checkpoint predictions
    cum_results = []
    for chkpt_info in chkpts:
        # TODO: replace with validation_metrics function
        #validation_metrics(new_split, ..., val_summary_writer=None, chkpt_info=chkpt_info,
        #                   arch_kwargs=arch_kwargs, loader_kwargs=loader_kwargs,
        #                   epoch=best_epoch[split_idx])

        chkpt_results, wsis = run_once(
            resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
            preproc=args.preproc, temper=args.temper,
            dataset_dict=new_split,
            num_epochs=1,
            graph_dir=args.epi_graph_dir,
            save_dir=None,
            nodes_preproc_func=nodes_preproc_func,
            dev_mode=False,
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
            #if node_level:
            if 'epithelium' in args.resp[i]:
                output_logit.extend([out_[0] for out_ in out[i:]])
                output_true.extend([out_[1] for out_ in out[i:]])
            elif args.resp[i] in ['CMS', 'CMS_matching']:
                output_logit.extend([out_[0][0] for out_ in out[i:]])
                output_true.extend([out_[0][1] for out_ in out[i:]])
            else:
                output_logit.append(out[i][0])
                output_true.append(out[i][1])

        output_logit = np.array(output_logit, dtype=np.float16)
        print('output_logit.shape:', output_logit.shape)
        output_true = np.array(output_true)
        print('output_true.shape:', output_true.shape)

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
        hparams['test_cohorts'] = '_'.join(str(cohort) for cohort in hparams['test_cohorts'])
        hparams['loss_weights'] = '_'.join(str(num) for num in hparams['loss_weights'])
        hparams['resp'] = '_'.join(str(response) for response in hparams['resp'])
        hparams['label_dim'] = '_'.join(str(lbl_dim) for lbl_dim in hparams['label_dim'])
        print('hparams')
        print(hparams)
        print('\nmetric_dict')
        print(metric_dict)

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

for response in list(met_args):
    print(response)
    resp_true = preds_df[f'{response}_true'][0]
    resp_preds = preds_df[f'{response}_preds'][0]

    if response in ['CMS', 'CMS_matching']:
        conf_preds = resp_preds
    else:
        conf_preds = threshold_predictions(resp_true, resp_preds)
    confusion_fig = plot_confusion_matrix(resp_true, conf_preds, response,
                          viz_fold, viz_epoch, save=True, save_img_path=args.save_img_path, thresh=response not in ['CMS', 'CMS_matching'])
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
print(f'| {args.base_name} {args.base_version} | {args.model_name.split("/")[1]} |' +\
      f' {args.model_name.split("/")[2]} |' +\
      metric_str_thresh_all(all_mets, met_args, 'auroc', threshold=to_threshold) +
      metric_str_thresh_all(all_mets, met_args, 'balanced_acc', threshold=to_threshold) +
      metric_str_thresh_all(all_mets, met_args, 'weighted_f1', threshold=to_threshold))
print()



# TODO: offer test option on unseen test set not in splits

if len(args.test_cohorts) > 0:
    wsi_dirs, msk_dirs, epi_msk_dirs = [], [], []
    for cohort in args.test_cohorts:
        cohort_wsi_dir = get_wsi_dir(cohort)
        wsi_dirs.append(cohort_wsi_dir)
        cohort_mask_dir = get_mask_dir(cohort)
        msk_dirs.append(cohort_mask_dir)
        cohort_epi_mask_dir = get_epi_mask_dir(cohort)
        epi_msk_dirs.append(cohort_epi_mask_dir)

    wsi_names, wsi_paths, msk_paths, epi_msk_paths, base_feature_dir = find_base_data(wsi_dirs, msk_dirs,
                                                                                      base_name=args.base_name,
                                                                                      base_version=args.base_version,
                                                                                      seed=args.seed,
                                                                                      epi_msk_dirs=epi_msk_dirs,
                                                                                      test=False if args.test_cohorts != [
                                                                                          'SALZBURG'] else True)
    patch_labels = load_patch_labels(args.clinical_file, args.mag, args.resp, args.test_cohorts)
    # set slide column to type string - for Salzburg int IDs
    patch_labels.slide = patch_labels.slide.astype('str')

    slide_df = patch_labels.groupby('slide').first().drop(['patch'], axis=1).reset_index()
    #if args.remove_unclassified_cms4:
    #    print(f'Removing {slide_df.CMS_matching.value_counts()["Unclassified"]} unclassified CMS from dataset')
    #    slide_df = slide_df[slide_df.CMS_matching != 'Unclassified'].reset_index(drop=True)
    #if args.remove_unmatched_cms4:
    #    print(f'Removing {slide_df.CMS_matching.value_counts()["Unmatched"]} unmatched CMS from dataset')
    #    slide_df = slide_df[slide_df.CMS_matching != 'Unmatched'].reset_index(drop=True)
    # Select response columns, without dealing with possible epithelium response here
    label_df, slide_responses = make_label_df_with_slide_labels(slide_df, responses=args.resp)

    # Filter label_df based on WSIs we have features for
    our_sel = np.where([wsi in wsi_names for wsi in label_df['WSI-CODE']])[0]
    label_df = label_df.loc[our_sel].reset_index(drop=True)
    # Drop Na labels
    label_df = label_df.dropna().reset_index(drop=True)
    print('Labels:', len(label_df))
    assert len(label_df) > 0, "Problem loading WSI labels, none found"

    # Redo wsi_names for normalizer and superpixels
    wsi_names = label_df['WSI-CODE'].values

    # Load test data, can't get from splits
    if float(args.base_version) >= 5.0:
        test_wsis = label_df[label_df.cohort.isin(args.test_cohorts)]['WSI-CODE']
    else:
        test_wsis = sorted(os.listdir(os.path.join(base_feature_dir, 'Test')))

    # Filter WSIs for those which we have labels
    test_wsis = filter_wsis(test_wsis, label_df)

    # Filter WSIs for those which we have graphs - use Validation augmentations
    test_wsis = filter_wsis_by_mode_graphs(test_wsis, args.epi_graph_dir, 'Validation')

    random.seed(args.seed)  # changed from 0 after DINO1.11 first two models
    random.shuffle(test_wsis)
    print('Shuffled wsis')

    test_labels = [
        label_df[label_df['WSI-CODE'] == slide][[f'LABEL_{i}' for i in range(len(slide_responses))]].values.tolist()[0]
        for
        slide in test_wsis]

    print('Test set response distribution (across both labels):')
    print(f'    {np.unique(test_labels, return_counts=True)}')

    #if not args.dev_mode:
    for i in range(len(slide_responses)):
        print(
            f'There are {np.unique(np.array(test_labels)[:, i], return_counts=True)[1][1]} positive {slide_responses[i]} slides in' +
            ' the validation set')
    print('Number of test slides:', len(test_wsis))

    # Redo wsi_names based on train and val
    wsi_names = test_wsis #list(set(val_wsis).union(set(train_wsis)))

    new_split = {'valid': list(zip(test_wsis, test_labels))}  # want valid to return epi label



# use val wsis from splits saved under model name
if len(args.test_cohorts) > 0:
    cohorts_to_eval = args.test_cohorts
elif len(args.val_cohorts) > 0:
    cohorts_to_eval = args.val_cohorts
else:
    cohorts_to_eval = args.train_cohorts
for cohort in cohorts_to_eval:
    cohort_split = get_val_wsis_from_slide_df(cohort, new_split, slide_df)
    print(cohort)
    if args.resp == ['CMS'] or args.resp == ['CMS_matching']:
        cohort_metric_dict = multiclass_validation_metrics(cohort_split, chkpts[0], best_epoch[0], arch_kwargs,
                       loader_kwargs, args, nodes_preproc_func, val_summary_writer)
    else:
        thresholds_from_val = [metric_dict[f'{resp}-threshold'] for resp in args.resp]

        cohort_metric_dict = validation_metrics(cohort_split, chkpts[0], best_epoch[0], arch_kwargs,
                       loader_kwargs, args, nodes_preproc_func, val_summary_writer,
                                                thresholds=thresholds_from_val)
    cohort_metric_dict = {f'{cohort}-{k}': v for k, v in cohort_metric_dict.items()}
    metric_dict.update(cohort_metric_dict)

val_summary_writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)
