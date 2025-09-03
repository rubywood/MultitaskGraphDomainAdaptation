from typing import Dict
from numpy.typing import ArrayLike
from numbers import Number
import numpy as np

# Use ujson as replacement for default json because it's faster for large JSON
import ujson as json

# ! save_yaml, save_as_json => need same name, need to factor out jsonify
from tiatoolbox.tools.graph import SlideGraphConstructor, delaunay_adjacency, affinity_to_edge_index

from .data import *

def build_graph_without_clustering(
        points: ArrayLike,
        features: ArrayLike,
        connectivity_distance: Number = 4000,
) -> Dict[str, ArrayLike]:
    adjacency_matrix = delaunay_adjacency(
        points=points,  # replaced point_centroids with superpixel centers (positions)
        dthresh=connectivity_distance,
    )

    if adjacency_matrix is None:
        return None
    edge_index = affinity_to_edge_index(adjacency_matrix)

    return {
        "x": features,  # replaced with input features from superpixels
        "edge_index": edge_index,
        "coords": points,
    }


def average_range(posns):
    x_range = np.max(posns[:, 0]) - np.min(posns[:, 0])
    y_range = np.max(posns[:, 1]) - np.min(posns[:, 1])
    return np.mean([x_range, y_range])


# Will work for patches too
def construct_superpixel_graph(wsi_name, save_path, connectivity_scale, connectivity_dist, wsi_feature_dir,
                               add_epi=False):
    positions = np.load(f"{wsi_feature_dir}/{wsi_name}.position.npy")  # superpixel centres
    features = np.load(f"{wsi_feature_dir}/{wsi_name}.features.npy")
    # Remove for 2.xx and epi ratios onwards - no need to add patch corners to superpixel centres
    #positions = np.array([position_corners(coords) for coords in positions])
    if add_epi:
        epi_labels = np.load(f"{wsi_feature_dir}/{wsi_name}.binary_epi_labels.npy")
    #print(wsi_name)

    if connectivity_scale is not None:
        connect_dist = int(average_range(positions) / connectivity_scale)  # e.g. 2081
    elif connectivity_dist is not None:
        connect_dist = int(connectivity_dist)
    else:
        raise Exception("Either connectivity_dist or connectivity_scale must be supplied in arguments")

    graph_dict = build_graph_without_clustering(
        positions[:, :2], features, connectivity_distance=connect_dist  # [, :2] redundant
    )
    if graph_dict is None:
        raise Exception('Graph dict is None')
        
    if add_epi:
        graph_dict['epi_label'] = epi_labels

    # Write a graph to a JSON file
    with open(save_path, "w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(graph_dict, handle)
