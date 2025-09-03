import numpy as np
import os

# Use ujson as replacement for default json because it's faster for large JSON
import ujson as json

# TIAToolbox imports
from tiatoolbox.tools.graph import SlideGraphConstructor

# Local imports
from utils.data import load_slide_features, patch_corner_coordinates


# Constructing graph using SlideGraph clustering method, not superpixels
def construct_slidegraph(wsi_name, save_path, base_name, base_version, seed, set_max_clusters, train_or_val):
    """Construct graph for one WSI and save to file."""
    slide_features_paths = load_slide_features(wsi_name, base_name=base_name, base_version=base_version, seed=seed,
                                               train_or_val=train_or_val)
    features = slide_features_paths['slide_features'].cpu().numpy()
    positions = np.array([patch_corner_coordinates(path) for path in slide_features_paths['patch_paths']])
    print(wsi_name)

    graph_dict, clusters, max_clusters = SlideGraphConstructor.build(
        positions[:, :2], features, feature_range_thresh=None, connectivity_distance=1000, # default 4000
        # num_clust=NUM_CLUSTERS,
        set_max_cluster=set_max_clusters, lambda_f=1e-2,  # lambda_h=0.75,
    )
    if graph_dict is None:
        return None, None
    # lambda_f default 1.0e-3
    # lambda_h default 0.8, in [0,1] - lower means more clusters
    # connectivity default 4000

    # One cluster label for each node/patch/position
    coords_clusters = list(zip(positions, clusters))
    # coords_clusters_dict[wsi_name] = coords_clusters

    # One cluster label for each node/patch/position
    coords_max_clusters = list(zip(positions, max_clusters))
    # coords_max_clusters_dict[wsi_name] = coords_max_clusters

    # Write a graph to a JSON file
    with open(save_path, "w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(graph_dict, handle)

    return coords_clusters, coords_max_clusters
