import pandas as pd
import os
import json
import joblib
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import pickle
from tqdm import tqdm
import argparse

from skimage.transform import pyramid_expand, rescale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import silhouette_score

from sompy.sompy import SOMFactory

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, Batch

# Local utils
from utils.model import SlideGraphArch, SlideGraphArchMLPv2, GNNMLPv3
from utils.visualisation import plot_node_activations, plot_slide_prediction_hist
from utils.utils import mkdir
from utils.data import SlideGraphEpiDataset, collate_fn_pad, load_patch_labels, make_label_df_with_slide_labels
from utils.metrics import calc_metrics

from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader
from tiatoolbox.utils.visualization import plot_graph

import consensusClustering as cc
from rcc import RccCluster

import plotly.express as px

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.metrics import average_precision_score as auprc_scorer


def predict_node_features(slide, all_responses, graph_dir, model_dir, epoch, arch_kwargs,
                          fold=0, preproc=True, scaler_path=None):
    graph_path = f"{graph_dir}/{slide}.json"

    model_weights_path = f'{model_dir}/{fold:02d}/epoch={epoch:03d}.weights.pth'
    model_aux_path = None  # f'{model_dir}/{fold:02d}/epoch={epoch:03d}.aux.dat'

    with open(graph_path, "r") as fptr:
        graph_dict = json.load(fptr)
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

    if preproc:
        node_scaler = joblib.load(scaler_path)
        graph_dict["x"] = node_scaler.transform(graph_dict["x"])

    graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
    graph = Data(**graph_dict)
    batch = Batch.from_data_list([graph])

    model = create_model(arch_kwargs['mlp_version'], all_responses, arch_kwargs)
    model.load(path=model_weights_path)
    model = model.to("cuda")
    model.eval()

    # Data type conversion
    batch = batch.to("cuda")
    batch.x = batch.x.type(torch.float32)
    with torch.no_grad():
        output_dict = model(batch)
        # predictions, node_activations = model(batch)
    node_features = output_dict['features']
    # predictions, node_activations = output_dict[resp]
    print('node_features:', node_features.shape)
    # node_features = node_features.detach().cpu().numpy()

    return node_features


def load_model(model_dir, epoch, arch_kwargs, responses, fold=0, eval_mode=True):
    model_weights_path = f'{model_dir}/{fold:02d}/epoch={epoch:03d}.weights.pth'
    model = create_model(arch_kwargs['mlp_version'], responses, arch_kwargs)
    model.load(path=model_weights_path)
    print(f'Loaded model weights from {model_weights_path}')
    model = model.to("cuda")
    if eval_mode:
        model.eval()
    return model


def load_student(model_dir, epoch, arch_kwargs, responses, eval_mode=True):
    model_weights_path = f'{model_dir}/epoch={epoch:03d}.weights.pth'

    model = create_model(arch_kwargs['mlp_version'], responses, arch_kwargs)
    model.mlp_heads = None
    model.load(path=model_weights_path)
    model = model.to("cuda")
    if eval_mode:
        model.eval()
    return model


def predict_node_features_multiple(slides, all_responses, graph_dir, model_dir, epoch, arch_kwargs,
                                   fold=0, preproc=True, scaler_path=None, sample=0):
    # model_weights_path = f'{model_dir}/{fold:02d}/epoch={epoch:03d}.weights.pth'
    # model_aux_path = None #f'{model_dir}/{fold:02d}/epoch={epoch:03d}.aux.dat'
    #
    # model = create_model(arch_kwargs['mlp_version'], all_responses, arch_kwargs)
    # model.load(path=model_weights_path)
    # model = model.to("cuda")
    # model.eval()

    model = load_model(model_dir, epoch, arch_kwargs, all_responses, fold=fold, eval_mode=True)

    wsi_feats = {key: [] for key in slides}

    for slide in slides:

        graph_path = f"{graph_dir}/{slide}.json"

        with open(graph_path, "r") as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if preproc:
            node_scaler = joblib.load(scaler_path)
            graph_dict["x"] = node_scaler.transform(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        graph = Data(**graph_dict)
        batch = Batch.from_data_list([graph])

        # Data type conversion
        batch = batch.to("cuda")
        batch.x = batch.x.type(torch.float32)
        with torch.no_grad():
            output_dict = model(batch)
            # predictions, node_activations = model(batch)
        node_features = output_dict['features']
        # predictions, node_activations = output_dict[resp]
        if sample != 0:
            indices = torch.randperm(len(node_features))[:sample]
            node_features = node_features[indices]

        print('node_features:', node_features.shape)
        # node_features = node_features.detach().cpu().numpy()

        wsi_feats[slide] = node_features

    return wsi_feats


def create_model(version, responses, arch_kwargs):
    if version == 3:
        print('Loading GNNMLPv3')
        model = GNNMLPv3(responses=responses, **arch_kwargs)
    elif version == 2:
        print('Loading SlideGraphArchMLPv2')
        model = SlideGraphArchMLPv2(responses=responses, **arch_kwargs)
    else:
        print('Loading SlideGraphArch')
        model = SlideGraphArch(responses=responses, **arch_kwargs)
    return model


# this won't work with other cohorts
def cohort_from_name(slide_name):
    if not 'SC0' in slide_name:
        return 'SALZBURG'
    else:
        if slide_name[3] in ['0', '1']:
            return 'GRAMPIAN'
        elif slide_name[3] in ['2', '3']:
            return 'ARISTOTLE'
        else:
            raise Exception


def train_cohorts_from_model_name(model_name, initials=True):
    train_cohorts = model_name.split('train_')[1].split('_val')[0].split('/')[0]
    # return initials as string
    if initials:
        return ''.join([c[0] for c in train_cohorts.split('_')])
    else:
        return train_cohorts


def torch_scaler(tensor_data, data_mean=None, data_std=None):
    if data_mean is None and data_std is None:
        # calculate scaler if not given
        data_mean = tensor_data.mean(0, keepdim=True)
        data_std = tensor_data.std(0, unbiased=False, keepdim=True)
        shifted = tensor_data - data_mean
        scaled = shifted / data_std
        return scaled, data_mean, data_std
    else:
        # if scaler given, apply to data
        shifted = tensor_data - data_mean.to("cuda")
        scaled = shifted / data_std.to("cuda")
        return scaled


def define_slides(graph_dir, args):
    wsi_names = [wsi.split('.')[0] for wsi in os.listdir(graph_dir)]

    test_slides = [wsi for wsi in wsi_names if cohort_from_name(wsi) == args.test_cohort]
    print(f'{len(test_slides)} test slides from test cohort {args.test_cohort}')

    train_slides = [wsi for wsi in wsi_names if cohort_from_name(wsi) != args.test_cohort]
    print(f'{len(train_slides)} train slides from training cohorts')

    return test_slides, train_slides


def load_labels(wsi_names, clinical_file, resp, cohorts, mag='20X'):
    patch_labels = load_patch_labels(clinical_file, mag, resp, cohorts)
    patch_labels.slide = patch_labels.slide.astype('str')

    slide_df = patch_labels.groupby('slide').first().drop(['patch'], axis=1).reset_index()
    del patch_labels
    # Select response columns, without dealing with possible epithelium response here
    label_df, slide_responses = make_label_df_with_slide_labels(slide_df, responses=resp)

    # Filter label_df based on WSIs we have features for
    our_sel = np.where([wsi in wsi_names for wsi in label_df['WSI-CODE']])[0]
    label_df = label_df.loc[our_sel].reset_index(drop=True)
    # Drop Na labels
    label_df = label_df.dropna().reset_index(drop=True)
    print('Labels:', len(label_df))
    assert len(label_df) > 0, "Problem loading WSI labels, none found"

    # Redo wsi_names for normalizer and superpixels
    wsi_names = label_df['WSI-CODE'].values

    labels = [
        label_df[label_df['WSI-CODE'] == slide][[f'LABEL_{i}' for i in range(len(slide_responses))]].values.tolist()[0]
        for
        slide in wsi_names]

    return list(zip(wsi_names, labels))


def predict_from_model(dataloader, model, responses):
    predict_slides = [info[0] for info in dataloader.dataset.info_list]
    result_dict = {wsi: {} for wsi in predict_slides}

    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(dataloader)):
            # print('Step', step)

            # Assume one WSI in each batch

            wsi_graphs = batch_data[0]["graph"].to("cuda")
            wsi_names = batch_data[1]

            # Data type conversion
            wsi_graphs.x = wsi_graphs.x.type(torch.float32)

            # if "label" in batch_data:
            wsi_labels = batch_data[0]["label"][0]  # second [0] since batch size is 1
            wsi_labels = wsi_labels.cpu().numpy()
            # first label is cr, second label is cms4, rest are epithelium

            output_dict = model(wsi_graphs)

            for i in range(len(responses)):
                resp = responses[i]

                # for each label have wsi prediction [0] and node prediction [1]
                if resp == 'epithelium':
                    # last entry is node-level epithelium
                    prediction = output_dict[resp][1].squeeze().detach().cpu().numpy()
                    label = wsi_labels[i:]
                else:
                    prediction = output_dict[resp][0].item()
                    label = wsi_labels[i]

                result_dict[wsi_names[0]][f'{resp}_pred'] = prediction
                result_dict[wsi_names[0]][f'{resp}_true'] = label

    model_results = pd.DataFrame.from_dict(result_dict).transpose()
    return model_results


def calc_metrics(targets, outputs, cutoff=None):
    if cutoff:
        predictions = np.where(outputs > cutoff, 1, 0)
    else:
        predictions = np.round(outputs)

    # slide-level accuracy and AUC - should be same as saved best from training/validation
    slide_level_auc = roc_auc_score(targets, outputs, average='weighted')
    slide_level_acc = accuracy_score(targets, predictions)
    weighted_acc = balanced_accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')

    print('- Weighted AUC', slide_level_auc)
    print('- Accuracy', slide_level_acc)
    print('- Balanced accuracy', weighted_acc)
    print('- Weighted F1 score', f1)
    print('- Weighted Precision', precision)
    print('- Weighted Recall', recall)
    print()

    return slide_level_auc, slide_level_acc, weighted_acc, f1, precision, recall


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
