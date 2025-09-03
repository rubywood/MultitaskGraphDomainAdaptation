import argparse
import os
import pandas as pd
import torch
import pickle
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.disable(logging.DEBUG)

# Local utils
from utils.utils import mkdir, str2bool
from utils.data import SlideGraphEpiDataset, collate_fn_pad

from implementation_utils import cohort_from_name, predict_node_features_multiple, load_labels, \
    train_cohorts_from_model_name, torch_scaler, define_slides, load_model, create_model, predict_from_model,\
    calc_metrics
from loss import ClusterTripletLoss, CentreTripletLoss

def define_args():
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
    parser.add_argument('--training-seed', type=int, default=0, help='Seed for training student model')
    #parser.add_argument('--training-seeds', type=int, nargs='+',
    #                    help='Seeds for training student model from TrainStudentMultiple.py')
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
    return args


def setup(args):
    torch.cuda.set_device(args.gpu)

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

    graph_root_dir = os.path.join(args.root_dir, f"{args.base_name}Base{args.graph_version}")
    graph_dir = os.path.join(graph_root_dir, 'graph/epithelium', args.graph_name, 'Validation')
    train_graph_dir = os.path.join(graph_root_dir, 'graph/epithelium', args.graph_name, 'Train')

    model_dir = os.path.join(root_output_dir, 'model', args.model_name)
    # If extracting features need teacher
    assert os.path.exists(model_dir), f"Model directory does not exist at {model_dir}"

    scaler_path = f"{root_output_dir}/{args.model_name}_node_scaler.dat"
    results_save_path = os.path.join(root_output_dir, 'results', args.model_name)

    preds = pd.read_csv(os.path.join(results_save_path, 'fold_predictions'), index_col=0)
    best_epochs = preds.best_epoch

    feature_save_dir = os.path.join(args.save_folder, 'checkpoint', 'features', args.model_name)
    metric_save_dir = os.path.join(args.save_folder, 'checkpoint', 'metrics', args.model_name)

    return best_epochs, graph_dir, train_graph_dir, model_dir, arch_kwargs, scaler_path, feature_save_dir, \
           metric_save_dir, root_output_dir


def train(student, dataloader, num_epochs, optimizer, criterion, student_save_dir, scaler=None):
    epoch_losses = []
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        epoch_loss = 0
        for step, batch_data in enumerate(dataloader):
            # print(batch_data)

            wsi_graphs = batch_data[0]["graph"].to("cuda")
            #wsi_names = batch_data[1]

            # Data type conversion
            wsi_graphs.x = wsi_graphs.x.type(torch.float32)
            # print(wsi_graphs)

            student.train()
            optimizer.zero_grad()

            output_dict = student(wsi_graphs)
            # print(output_dict.keys())

            features = output_dict['features']  # (n, 96)

            # Scale features using same scaler as in training - torch version
            # scaled_features = scaler.transform(features) - numpy version
            if scaler:
                features = torch_scaler(features, data_mean=scaler['mean'], data_std=scaler['std'])

            loss = criterion(features)  # .type(torch.FloatTensor).cuda())

            # Backprop and update
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print('  Loss:', epoch_loss)

        student.save(f"{student_save_dir}/epoch={epoch:03d}.weights.pth")

        epoch_losses.append(epoch_loss)
    return epoch_losses


def load_student(model_dir, epoch, arch_kwargs, responses, eval_mode=True):
    model_weights_path = f'{model_dir}/epoch={epoch:03d}.weights.pth'

    model = create_model(arch_kwargs['mlp_version'], responses, arch_kwargs)
    model.mlp_heads = None
    model.load(path=model_weights_path)
    model = model.to("cuda")
    if eval_mode:
        model.eval()
    return model


def evaluate(epoch_losses, student_save_dir, arch_kwargs, responses, test_slides, graph_dir, root_output_dir, args):
    best_student_epoch = min(range(len(epoch_losses)), key=epoch_losses.__getitem__)
    print('Best epoch:', best_student_epoch)
    # manual
    # best_student_epoch = 9

    student = load_student(student_save_dir, best_student_epoch, arch_kwargs, responses, eval_mode=True)

    datalist = load_labels(test_slides, clinical_file=args.clinical_file, resp=responses,
                           cohorts=[args.test_cohort],  mag='20X')

    dataloader = load_data(datalist, graph_dir, 'train', batch_size=1, root_output_dir=root_output_dir,
                           args=args)

    student_results = predict_from_model(dataloader, student, responses)

    metrics = calc_metrics(student_results['response_cr_nocr_true'].astype(int).values,
                           student_results['response_cr_nocr_pred'].astype(float).values)

    return metrics, best_student_epoch


def load_data(datalist, graph_dir, mode, batch_size, root_output_dir, args):
    SCALER_PATH = f"{root_output_dir}/{args.model_name}_node_scaler.dat"
    if args.preproc:
        node_scaler = joblib.load(SCALER_PATH)

    def nodes_preproc_func(node_feats):
        return node_scaler.transform(node_feats)

    ds = SlideGraphEpiDataset(datalist, graph_dir=graph_dir, mode=mode, preproc=nodes_preproc_func)
    # infer mode doesn't return labels

    dataloader = torch.utils.data.DataLoader(  # changed from geometric to normal dataloader
        ds,
        collate_fn=collate_fn_pad,
        batch_sampler=None,  # or can stratify by label
        drop_last=False,
        shuffle=True,
        num_workers=8,
        batch_size=batch_size  # 1 for validation
    )
    return dataloader


def main(args):
    print(f'Training student model with seed {args.training_seed}\n')

    best_epochs, graph_dir, train_graph_dir, model_dir, arch_kwargs, scaler_path, feature_save_dir, metric_save_dir, root_output_dir = setup(args)
    test_slides, train_slides = define_slides(graph_dir, args)
    #train_cs = train_cohorts_from_model_name(args.model_name)
    train_cs = ''.join([c[0] for c in args.train_cohorts])

    # Load scaler if using
    scaler_savename = os.path.join(feature_save_dir, f'scaler_{train_cs}_full_seed_{args.seed}.p')
    if args.use_scaled_features:
        scaler = pickle.load(open(scaler_savename, "rb"))
    else:
        scaler = None

    # Load KMeans
    kmeans_save_dir = os.path.join(args.save_folder, 'checkpoint', 'kmeans', args.model_name)
    kmeans_name = f'kmeans_{"best_" if args.best_nclusters else ""}k{args.nclusters}_{train_cs}_full{"_scaled" if args.use_scaled_features else ""}_seed_{args.seed}.p'
    with open(os.path.join(kmeans_save_dir, kmeans_name), 'rb') as fp:
        kmeans = pickle.load(fp)
    assert int(args.nclusters) == len(kmeans.cluster_centers_), \
        f"{len(kmeans.cluster_centers_)} clusters in Kmeans, not {args.nclusters} as expected"

    nclusters = len(kmeans.cluster_centers_)
    print('Number of clusters from loaded KMeans:', nclusters)

    # Load Student Model
    student = load_model(model_dir, best_epochs[0], arch_kwargs, args.responses, fold=0, eval_mode=False)
    student.mlp_heads = None
    student.train()

    cluster_centres = kmeans.cluster_centers_
    if args.scale_centroids:
        scaled_cluster_centres = MinMaxScaler().fit_transform(cluster_centres)
        centroids = torch.tensor(scaled_cluster_centres, device='cuda', dtype=torch.float32, requires_grad=True)
    else:
        centroids = torch.tensor(cluster_centres, device='cuda', dtype=torch.float32, requires_grad=True)

    dataloader = load_data(test_slides, train_graph_dir, 'infer', batch_size=32, root_output_dir=root_output_dir,
                           args=args)

    student_save_dir = os.path.join(args.save_folder, 'checkpoint', 'student', args.model_name,
                                    f'{args.model_version}_seed_{args.training_seed}')
    if not os.path.exists(student_save_dir):
        os.makedirs(student_save_dir)

    optim_kwargs = dict(
        lr=args.lr,
        weight_decay=args.wd,
    )
    optimizer = torch.optim.Adam(student.parameters(), **optim_kwargs)
    if args.centre_loss:
        # Ablation study
        criterion = CentreTripletLoss(centroids=centroids)
    else:
        criterion = ClusterTripletLoss(centroids=centroids)

    ########## Set seed ##########
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed(args.training_seed)

    # Train student
    print('\nTraining Student Model')
    epoch_losses = train(student, dataloader, args.num_epochs, optimizer, criterion, student_save_dir, scaler=scaler)

    fig = plt.figure()
    plt.plot(epoch_losses)
    plt.title('Training loss')
    fig.savefig(os.path.join(student_save_dir, 'TrainingLoss.png'))
    print(f'Saved loss plot to {os.path.join(student_save_dir, "TrainingLoss.png")}')

    # Evaluate student
    print('\nEvaluating Student Model')
    metrics, best_epoch = evaluate(epoch_losses, student_save_dir, arch_kwargs, args.responses, test_slides, graph_dir,
                                   root_output_dir, args)
    slide_level_auc, slide_level_acc, weighted_acc, f1, precision, recall = metrics
    print(f'AUC: {slide_level_auc:.4f}, Balanced Acc: {weighted_acc:.4f}, Weighted F1: {f1:.4f}')

    # Save metrics in pd dataframe, saved under model_name
    metrics_csv = os.path.join(metric_save_dir, 'metrics.csv')
    if os.path.exists(metrics_csv):
        metrics_df = pd.read_csv(metrics_csv)
        print('\nLoading metrics dataframe')
    else:
        print('\nCreating new metrics dataframe')
        mkdir(metric_save_dir)
        metrics_df = pd.DataFrame(columns=['model_version', 'training_seed', 'cluster_seed', 'clustering', 'loss',
                                           'nclusters', 'best_nclusters', 'target_source', 'source_cohorts', 'scaled_features',
                                           'scaled_centroids', 'epoch_best', 'lr', 'wd',
                                           'CR_AUC', 'CR_Acc', 'CR_Balanced_Acc', 'CR_Weighted_F1',
                                           'CR_Weighted_Precision', 'CR_Weighted_Recall'])

    metrics_series = pd.DataFrame({'model_version': args.model_version, 'training_seed': args.training_seed,
                                    'cluster_seed': args.seed, 'clustering': 'KMeans',
                                    'loss': 'CentreTripletLoss' if args.centre_loss else 'ClusterTripletLoss',
                                    'nclusters': args.nclusters, 'best_nclusters': args.best_nclusters,
                                    'target_source': args.test_cohort,
                                    'source_cohorts': train_cohorts_from_model_name(args.model_name, initials=False),
                                    'scaled_features': args.use_scaled_features, 'scaled_centroids': args.scale_centroids,
                                    'epoch_best': f'{best_epoch}/{args.num_epochs-1}', 'lr': args.lr, 'wd': args.wd,
                                    'CR_AUC': slide_level_auc, 'CR_Acc': slide_level_acc, 'CR_Balanced_Acc': weighted_acc,
                                    'CR_Weighted_F1': f1, 'CR_Weighted_Precision': precision, 'CR_Weighted_Recall': recall},
                                  index=[0])
    print('Metrics:', metrics_series)
    metrics_df = pd.concat([metrics_df, metrics_series], axis=0, ignore_index=True)
    print(f'Added metrics, saving to {metrics_csv}')
    metrics_df.to_csv(metrics_csv, index=False)


if __name__ == '__main__':
    args = define_args()
    main(args)
