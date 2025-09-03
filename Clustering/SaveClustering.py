import argparse
import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.disable(logging.DEBUG)

# Local utils
import sys
from utils.utils import mkdir, str2bool
from Clustering.implementation_utils import cohort_from_name, predict_node_features_multiple, \
    train_cohorts_from_model_name, torch_scaler

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='Clustering')
parser.add_argument('--model-name', type=str)
parser.add_argument('--train-cohorts', type=str, nargs='+', default=['GRAMPIAN', 'ARISTOTLE'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-scaled-features', type=str2bool, default=True)
parser.add_argument('--k-from', type=int, default=2, help='min number of clusters to try')
parser.add_argument('--k-to', type=int, default=20, help='max number of clusters to try')
parser.add_argument('--nclusters', type=int, default=None, help='define exact number of clusters')

args = parser.parse_args()


def main(args):

    # for naming
    #train_cs = train_cohorts_from_model_name(args.model_name)
    train_cs = ''.join([c[0] for c in args.train_cohorts])

    # Load features
    feature_save_dir = os.path.join(args.save_folder, 'checkpoint', 'features', args.model_name)
    feats_savename = os.path.join(feature_save_dir,
                            f'feats_{train_cs}_full{"_scaled" if args.use_scaled_features else ""}_seed_{args.seed}.pt')
    features = torch.load(feats_savename) # should be on CPU
    np_features = features.detach().cpu().numpy()

    # Fit Kmeans
    kmeans_save_dir = os.path.join(args.save_folder, 'checkpoint', 'kmeans', args.model_name)
    if not os.path.exists(kmeans_save_dir):
        os.makedirs(kmeans_save_dir)

    if args.nclusters:
        nclusters = args.nclusters
        print(f'Using user-defined k={args.nclusters} clusters')
    else:
        scores = []
        for k in range(args.k_from, args.k_to):
            kmeans = MiniBatchKMeans(n_clusters=k,
                                     random_state=args.seed,
                                     batch_size=256,
                                     max_iter=10).fit(np_features)

            labels = kmeans.predict(np_features)

            random_idx = random.sample(range(len(np_features)), 10000)  # 1000 quicker, same results
            sample_X = np_features[random_idx]
            sample_labels = labels[random_idx]

            sil_score = silhouette_score(sample_X, sample_labels)
            scores.append(sil_score)

        # Plot source
        fig = plt.figure()
        plt.plot(range(args.k_from, args.k_to), scores)
        print(f'Saving silhouette scores to {os.path.join(kmeans_save_dir, "silhouette_widths.png")}')
        fig.savefig(os.path.join(kmeans_save_dir, 'silhouette_widths.png'))

        nclusters = np.argmax(scores) + args.k_from
        print('Number of clusters from best KMeans:', nclusters)

    kmeans = MiniBatchKMeans(n_clusters=nclusters,
                             random_state=args.seed,
                             batch_size=256,
                             max_iter=10).fit(np_features)

    print('Cluster centres have shape:', kmeans.cluster_centers_.shape)

    # Plot source distribution of clusters
    labels = kmeans.predict(np_features)
    len(labels)
    parent_distn = np.bincount(labels)  # or P
    fig = plt.figure()
    plt.bar(x=range(nclusters), height=parent_distn)
    plt.title('Teacher training cluster distribution')
    plt.xlabel('cluster')
    plt.ylabel('frequency')
    fig.savefig(os.path.join(kmeans_save_dir, 'source_clusters.png'))

    save_name = f'kmeans_{"best_" if args.nclusters is None else ""}k{nclusters}_{train_cs}_full{"_scaled" if args.use_scaled_features else ""}_seed_{args.seed}.p'
    with open(os.path.join(kmeans_save_dir, save_name), "wb") as f:
        pickle.dump(kmeans, f)

    # TODO: implement clustering type options? e.g. consensus for comparison


if __name__ == '__main__':
    main(args)
