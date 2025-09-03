import os
import joblib
import numpy as np
import pandas as pd
import argparse

# Local utils
from utils.utils import str2bool
from utils.visualisation import mkdir, recur_find_ext
from utils.model import select_checkpoints, SlideGraphArch
from utils.metrics import create_resp_metric_dict, find_optimal_cutoff, threshold_predictions, metric_str_thresh_all
from utils.plot import plot_confusion_matrix, density_plot

from train import run_once



########## Arguments ##########

parser = argparse.ArgumentParser()

parser.add_argument('--root-dir', type=str,
                    default='checkpoint/',
                    help='Root directory where everything is saved. Base model details will be added.')
parser.add_argument('--base-name', default='CTransPath', type=str, choices=['CTransPath', 'DINO'],
                    help='Baseline model for patch/node features')
parser.add_argument('--seed', type=int, default=4, choices=[1, 2, 3, 4],
                    help='set the seed for training and data split. '
                         'should match with baseline model to separate train/val datasets')
parser.add_argument('--gpu', type=int, default=3, choices=[0, 1, 2, 3], help='GPU number to use')

parser.add_argument('--set-max-clusters', default=False, type=str2bool,
                    help='Whether to set max number of clusters in WSI graph')
parser.add_argument('--num-clusters', default=None, help='Number of clusters if setting maximum for graph')
parser.add_argument('--resp', nargs='+', default=['response_cr_nocr', 'CMS4', 'epithelium'],
                    help='List of response variables')
# use like python script.py --resp response_cr_nocr CMS4 epithelium -- etc.
parser.add_argument('--superpixel', type=str2bool, default=True, help='True for MICCAI')
parser.add_argument('--cohorts', nargs='+', default=['GRAMPIAN', 'ARISTOTLE'], # SALZBURG
                    help='List of cohorts to train and validate on')

parser.add_argument('--remove-unclassified-cms4', type=str2bool, default=False,
                    help='Remove unclassified CMS4 WSIs from analysis (usually treated as not CMS4)')
parser.add_argument('--remove-unmatched-cms4', type=str2bool, default=False,
                    help='Remove unmatched CMS4 WSIs from analysis (usually treated as not CMS4)')

parser.add_argument('--preproc', type=str2bool, default=True,
                    help='Whether to preprocess and normalize the node features prior to GNN training')
parser.add_argument('--connectivity-scale', default=8, help='Graph connectivity', type=int)
parser.add_argument('--gembed', type=str2bool, default=False, help='Whether to gembed the GNN')
parser.add_argument('--temper', type=float, default=1.5, help='Tempering output; 1.5 used for MICCAI; alt 0.1')
parser.add_argument('--loss-weights', nargs='+', default=[1., 1., 1.], help='Weights on respective response variables')
parser.add_argument('--mlp', type=str2bool, default=True, help='MLP layer for output')
parser.add_argument('--mlp-version', type=int, default=1,
                    help='MLP layer version for output. 1 is MICCAI version. 2 is ops.MLP applied earlier.')
parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'slidegraph'], help='Loss function')
parser.add_argument('--layer-dims', default=[64, 32, 16], nargs='+', help='Layer dimensions in GNN')
parser.add_argument('--graph-agg', default='min', type=str, choices=['mean', 'max', 'min'],
                    help='Aggregation method for GNN')
parser.add_argument('--graph-pool', default='mean', type=str, choices=['mean', 'max', 'min'],
                    help='Pooling method for GNN')
parser.add_argument('--resolution', default=5.0, type=float,
                    help='Resolution/magnification for graph generation')
parser.add_argument('--compactness', default=20.0, type=float,
                    help='Compactness parameter for SLIC algorithm')
parser.add_argument('--scale-slic', type=int, default=2, help='Scale for SLIC algorithm, 8 for Salzburg, 2 otherwise')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout probability for GNN')
parser.add_argument('--scaler', default=False, type=str2bool,
                    help='True for trainable logistic regression (upside down results), False for nonparametric sigmoid')


parser.add_argument('--shuffle-splits', type=str2bool, default=True,
                    help='Whether to shuffle WSIs in training/validation splits. Generally tru.')
#parser.add_argument('--batch-size', default=64, type=int, help='Batch size for GNN training')


args = parser.parse_args()

setattr(args, 'base_version', f'4.0{args.seed}')
setattr(args, 'root_output_dir', os.path.join(args.root_dir, f"{args.base_name}Base{args.base_version}"))

loss_weights_str = 'weight_' + '_'.join(str(num) for num in args.loss_weights)
mlp_str = f"_mlp_v{args.mlp_version}" if args.mlp else ""
setattr(args, 'model_name', os.path.join("_".join(args.resp),
                                  "_".join(args.cohorts) +
                                  f'_{"superpixel" if args.superpixel else "slidegraph"}_' +
                                  f'{"rm_unmatched_" if args.remove_unmatched_cms4 else ""}' +
                                  f'{"rm_unclassified_" if args.remove_unclassified_cms4 else ""}' +
                                  f'upsample_{"preproc_false" if args.preproc == False else "normalize_train"}' +
                                  f'_connectivity_range_{str(1/args.connectivity_scale).lstrip("0")[:4]}' +
                                  f'_gembed_{str(args.gembed).lower()}_' +
                                  f'temper_{args.temper}_ginconv{mlp_str}' +
                                  f'_{args.loss}' +
                                  f'{"_xlarge" if args.layer_dims==[64, 32, 16] else ""}_' +
                                  f'{args.graph_agg}_aggr_{args.graph_pool}_pool' +
                                  f'{loss_weights_str if args.loss_weights != [1, 1, 1] else ""}'))
setattr(args, 'save_img_path', os.path.join(args.root_output_dir, 'visualisations', str(args.model_name)))


########## Inference ##########

## Can't update validation summary writer log

TOP_K = 1
#metric_name = f"{RESP[0]}-infer-valid-A-auroc" # choose best model based on first response only
metric_name = 'infer-valid-A-auroc' # choose based on all responses

MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/", args.model_name)
if args.set_max_clusters:
    MODEL_DIR = os.path.join(f"{args.root_output_dir}/model/{args.num_clusters}_clusters", args.model_name)
print('Model dir:', MODEL_DIR)
PRETRAINED_DIR = MODEL_DIR

GRAPH_NAME = f'superpixel_{int(args.resolution)}X_compactness_{int(args.compactness)}_scaleslic_{args.scale_slic}'
GRAPH_DIR = f"{args.root_output_dir}/graph/epithelium/{GRAPH_NAME}"

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
    conv="GINConv",
    aggr=args.graph_agg,  # changed from max to min
    gembed=args.gembed,
    scaler=args.scaler,
    temper=args.temper,
    use_mlp=args.mlp,
    mlp_version=args.mlp_version
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
        stat_files[0], top_k=TOP_K, metrics=[metric_name]
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
            graph_dir=GRAPH_DIR,
            save_dir=None,
            nodes_preproc_func=nodes_preproc_func,
            dev_mode=False,
            val_summary_writer=None,
            pretrained=chkpt_info,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs
        )

        # * re-calibrate logit to probabilities
        chkpt_results = np.array(chkpt_results)
        chkpt_results = np.squeeze(chkpt_results)

        if args.scaler:
            model = SlideGraphArch(responses=args.resp, **arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results = scaler.predict_proba(np.array(chkpt_results, ndmin=2).T)[:, 0]

        cum_results.append(chkpt_results)
    cum_results = np.array(cum_results)
    cum_results = np.squeeze(cum_results)

    output_1_logit, output_1_true = [], []
    output_2_logit, output_2_true = [], []
    node_output_logit, node_output_true = [], []

    if 'cohort_cls' in args.resp:
        epi_idx = 3
    else:
        epi_idx = 2

    for out in cum_results:
        output_1_logit.append(out[0][0])
        output_1_true.append(out[0][1])

        output_2_logit.append(out[1][0])
        output_2_true.append(out[1][1])

        node_output_logit.extend([out_[0] for out_ in out[epi_idx:]])
        node_output_true.extend([out_[1] for out_ in out[epi_idx:]])

    output_1_logit = np.array(output_1_logit)
    output_1_true = np.array(output_1_true)
    output_2_logit = np.array(output_2_logit)
    output_2_true = np.array(output_2_true)
    node_output_logit = np.array(node_output_logit)
    node_output_true = np.array(node_output_true)

    metric_dict = {}
    metric_dict.update(create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, best_epoch[split_idx]))
    metric_dict.update(create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, best_epoch[split_idx]))
    metric_dict.update(create_resp_metric_dict(args.resp[2], node_output_true, node_output_logit,
                                               best_epoch[split_idx]))

    # Add thresholded metrics
    print('    Using thresholding from joint cohorts')
    threshold_0 = find_optimal_cutoff(output_1_true, output_1_logit)
    resp_0_mets = create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, best_epoch[split_idx],
                                          cutoff=threshold_0)
    resp_0_mets = {'threshold-' + k: v for k, v in resp_0_mets.items() if not k == 'best_epoch'}
    resp_0_mets[f'{args.resp[0]}-threshold'] = threshold_0[0]
    metric_dict.update(resp_0_mets)

    threshold_1 = find_optimal_cutoff(output_2_true, output_2_logit)
    resp_1_mets = create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, best_epoch[split_idx],
                                          cutoff=threshold_1)
    resp_1_mets = {'threshold-' + k: v for k, v in resp_1_mets.items() if not k == 'best_epoch'}
    resp_1_mets[f'{args.resp[1]}-threshold'] = threshold_1[0]
    metric_dict.update(resp_1_mets)

    threshold_2 = find_optimal_cutoff(node_output_true, node_output_logit)
    resp_2_mets = create_resp_metric_dict(args.resp[-1], node_output_true, node_output_logit, best_epoch[split_idx],
                                          cutoff=threshold_2)
    resp_2_mets = {'threshold-' + k: v for k, v in resp_2_mets.items() if not k == 'best_epoch'}
    resp_2_mets[f'{args.resp[-1]}-threshold'] = threshold_2[0]
    metric_dict.update(resp_2_mets)

    cum_stats.append(metric_dict)
    #if args.log:
    #    hparams = vars(args).copy()
    #    hparams['layer_dims'] = '_'.join(str(num) for num in hparams['layer_dims'])
    #    hparams['cohorts'] = '_'.join(str(cohort) for cohort in hparams['cohorts'])
    #    hparams['loss_weights'] = '_'.join(str(num) for num in hparams['loss_weights'])
    #    hparams['resp'] = '_'.join(str(response) for response in hparams['resp'])
    #    #print('hparams')
    #    #print(hparams)
    #    #print('\nmetric_dict')
    #    #print(metric_dict)
    #    val_summary_writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)

    cum_preds.append(
        {"fold": split_idx, "best_epoch": best_epoch[split_idx],
         f"{args.resp[0]}_preds": output_1_logit, f"{args.resp[0]}_true": output_1_true,
         f"{args.resp[1]}_preds": output_2_logit, f"{args.resp[1]}_true": output_2_true,
         f"{args.resp[2]}_preds": node_output_logit, f"{args.resp[2]}_true": node_output_true
         }
    )


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

for response in list(args.resp):
    print(response)
    resp_true = preds_df[f'{response}_true'][0]
    resp_preds = preds_df[f'{response}_preds'][0]

    confusion_fig = plot_confusion_matrix(resp_true, threshold_predictions(resp_true, resp_preds), response,
                          viz_fold, viz_epoch, save=True, save_img_path=args.save_img_path, thresh=True)
    density_fig = density_plot(resp_true, resp_preds, response, viz_fold, viz_epoch, save=True,
                               save_img_path=args.save_img_path)
    #if args.log:
    #    val_summary_writer.add_figure(f'Validation Confusion Matrix with Threshold - {response}', confusion_fig)
    #    val_summary_writer.add_figure(f'Validation Density Plot - {response}', density_fig)


# Print metrics in table format
all_mets = [resp_0_mets, resp_1_mets, resp_2_mets]
print('Thresholded metrics printed below - can be used in Notebook table')
print()
print(f'| {args.base_name} {args.base_version} | {args.model_name.split("/")[1]} |' +\
      metric_str_thresh_all(all_mets, args.resp, 'auroc', threshold=True) +
      metric_str_thresh_all(all_mets, args.resp, 'balanced_acc', threshold=True) +
      metric_str_thresh_all(all_mets, args.resp, 'weighted_f1', threshold=True))
print()

# Check validation metrics on different cohorts
def validation_metrics(split, chkpt_info=chkpts[0], epoch=best_epoch[0], arch_kwargs=arch_kwargs,
                       loader_kwargs=loader_kwargs):
    chkpt_results, wsis = run_once(
        resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
        preproc=args.preproc, temper=args.temper,
        dataset_dict=split,
        num_epochs=1,
        graph_dir=GRAPH_DIR,
        save_dir=None,
        nodes_preproc_func=nodes_preproc_func,
        dev_mode=False,
        val_summary_writer=None,
        pretrained=chkpt_info,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs
    )

    # * re-calibrate logit to probabilities
    chkpt_results = np.array(chkpt_results)
    chkpt_results = np.squeeze(chkpt_results)

    if args.scaler:
        model = SlideGraphArch(responses=args.resp, **arch_kwargs)
        model.load(*chkpt_info)
        scaler = model.aux_model["scaler"]
        chkpt_results = scaler.predict_proba(np.array(chkpt_results, ndmin=2).T)[:, 0]
    cum_results = chkpt_results
    cum_results = np.array(cum_results)
    cum_results = np.squeeze(cum_results)

    output_1_logit, output_1_true = [], []
    output_2_logit, output_2_true = [], []
    node_output_logit, node_output_true = [], []

    if 'cohort_cls' in args.resp:
        epi_idx = 3
    else:
        epi_idx = 2

    for out in cum_results:
        output_1_logit.append(out[0][0])
        output_1_true.append(out[0][1])

        output_2_logit.append(out[1][0])
        output_2_true.append(out[1][1])

        node_output_logit.extend([out_[0] for out_ in out[epi_idx:]])
        node_output_true.extend([out_[1] for out_ in out[epi_idx:]])

    output_1_logit = np.array(output_1_logit)
    output_1_true = np.array(output_1_true)
    output_2_logit = np.array(output_2_logit)
    output_2_true = np.array(output_2_true)
    node_output_logit = np.array(node_output_logit)
    node_output_true = np.array(node_output_true)

    print('Without thresholding')
    metric_dict = {}
    print(args.resp[0])
    metric_dict.update(create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, epoch))
    print(args.resp[1])
    metric_dict.update(create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, epoch))
    print(args.resp[2])
    metric_dict.update(create_resp_metric_dict(args.resp[-1], node_output_true, node_output_logit, epoch))

    print('Using thresholding from joint cohorts')
    print(args.resp[0])
    resp_0_mets = create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, epoch,
                                          cutoff=threshold_0)
    resp_0_mets = {'threshold-' + k: v for k, v in resp_0_mets.items() if not k == 'best_epoch'}
    resp_0_mets[f'{args.resp[0]}-threshold'] = threshold_0[0]
    metric_dict.update(resp_0_mets)
    print(args.resp[1])
    resp_1_mets = create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, epoch,
                                          cutoff=threshold_1)
    resp_1_mets = {'threshold-' + k: v for k, v in resp_1_mets.items() if not k == 'best_epoch'}
    resp_1_mets[f'{args.resp[1]}-threshold'] = threshold_1[0]
    metric_dict.update(resp_1_mets)
    print(args.resp[2])
    resp_2_mets = create_resp_metric_dict(args.resp[-1], node_output_true, node_output_logit, epoch,
                                          cutoff=threshold_2)
    resp_2_mets = {'threshold-' + k: v for k, v in resp_2_mets.items() if not k == 'best_epoch'}
    resp_2_mets[f'{args.resp[-1]}-threshold'] = threshold_2[0]
    metric_dict.update(resp_2_mets)

    return metric_dict


if args.cohorts == ['GRAMPIAN', 'ARISTOTLE']:
    gramp_split = {'valid': list(filter(lambda wsi: int(wsi[0][3]) < 2, new_split['valid']))}
    arist_split = {'valid': list(filter(lambda wsi: int(wsi[0][3]) >= 2, new_split['valid']))}

    print('GRAMPIAN')
    gramp_metric_dict = validation_metrics(gramp_split)

    print('ARISTOTLE')
    arist_metric_dict = validation_metrics(arist_split)
