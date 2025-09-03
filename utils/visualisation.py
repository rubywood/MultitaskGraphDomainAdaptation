# Python standard library imports
import os
import warnings

# Third party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__() # allows to load in large mask images
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS


from skimage.transform import pyramid_expand, rescale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from skimage.exposure import equalize_hist

from torch_geometric.data import Data


warnings.filterwarnings("ignore")

from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader
from tiatoolbox.utils.visualization import plot_graph

# local utils
from .utils import *
from .data import path_for_wsi, label_from_splits, mask_for_wsi


        
def to_shape(a, shape):
    # shape may be larger or smaller. if larger, pad, else trim.
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    if y_pad > 0:
        a = np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), (0, 0)), mode = 'constant')
    elif y_pad < 0:
        a = a[:shape[0],:]
    if x_pad > 0:
        a = np.pad(a,((0, 0), (x_pad//2, x_pad//2 + x_pad%2)), mode = 'constant')
    elif x_pad < 0:
        a = a[:, :shape[1]]
    return a


def load_thumb(slide, wsi_paths, resolution=5.0):
    # Load WSI
    wsi_path = path_for_wsi(slide, wsi_paths)
    reader = WSIReader.open(wsi_path)
    thumb = reader.slide_thumbnail(resolution=resolution, units="power")
    return thumb


def load_mask(slide, msk_paths, wsi_paths=None, thumb=None):#, upscale=False, downscale=False):
    # Load mask
    mask_path = mask_for_wsi(slide, msk_paths)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if len(mask.shape) > 2: 
        mask = mask[...,0] # assume 3rd dimension all the same
    
    if thumb is None:
        thumb = load_thumb(slide, wsi_paths)
        
    if mask.shape[0] > 1.5*thumb.shape[0]:
        # downsample
        rescale_factor = thumb.shape[0] / mask.shape[0]
        print(f'mask rescale_factor: {rescale_factor:.5f}')
        mask = np.round(rescale(mask, rescale_factor)).astype(np.uint8)
    elif thumb.shape[0] > 1.5*mask.shape[0]:
        # upsample
        upscale_factor = int(np.round(thumb.shape[0]/mask.shape[0]))
        mask = pyramid_expand(mask, upscale=upscale_factor)
        
    #if downscale:
    #    rescale_factor = thumb.shape[0] / mask.shape[0]
    #    print(f'mask rescale_factor: {rescale_factor:.5f}')
    #    mask = rescale(mask, rescale_factor).astype(np.uint8)
    #    
    #if upscale:
    #    mask = pyramid_expand(mask, upscale=2)
    
    if mask.shape != thumb.shape[:2]:
        print(f'Fitting mask of size {mask.shape} to thumbnail of size {thumb.shape[:2]}')
        mask = to_shape(mask, thumb.shape[:2])
        assert mask.shape == thumb.shape[:2], 'Sizes of images do not exactly match'
    return mask
    

def mask_image(mask, thumb):
    # Mask the WSI
    bool_mask = mask == 0
    masked = thumb.copy()
    masked[bool_mask] = 0
    return masked


def mask_image_alpha(mask, thumb, alpha=1.0):
    # Ensure alpha is in the range [0, 1]
    alpha = max(0.0, min(alpha, 1.0))
    # Apply the mask with transparency
    bool_mask = mask == 0
    masked = thumb.copy()
    masked[bool_mask] = masked[bool_mask] * (1 - alpha)
    return masked


def plot_image(image, large=False):
    if large:
        mpl.rcParams["figure.dpi"] = 300
    else:
        mpl.rcParams["figure.dpi"] = 100
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def plot_images(images: List, large=False):
    if large:
        mpl.rcParams["figure.dpi"] = 300
    else:
        mpl.rcParams["figure.dpi"] = 100
    
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


# Added msk_paths parameter
def plot_graph_overlay_with_mask(viz_slide, msk_paths, graph_dir, mag='20X', resolution=5.0, node_size=6):

    graph_path = f"{graph_dir}/{viz_slide}.json"
    wsi_path = path_for_wsi(viz_slide)
    
    mag2mpp = {'40X': .25, '20X': .5, '10X': 1.}
    node_res = mag2mpp[mag]
    
    node_resolution_dict = dict(resolution=node_res, units="mpp") # changed from 0.5
    plot_resolution_dict = dict(resolution=resolution, units='power') # res = 5.0
    
    graph_dict = load_json(graph_path)
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph = Data(**graph_dict)
    
    # deriving node colors via projecting n-d features down to 3-d
    graph.x = StandardScaler().fit_transform(graph.x)
    # .c for node colors
    node_colors = FastICA(n_components=3).fit_transform(graph.x)[:, [1, 0, 2]] # FastICA faster than PCA
    for channel in range(node_colors.shape[-1]):
        node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
    node_colors = (node_colors * 255).astype(np.uint8)
    
    reader = WSIReader.open(wsi_path)
    #thumb = reader.slide_thumbnail(4.0, "mpp")
    
    node_resolution = reader.slide_dimensions(**node_resolution_dict)
    plot_resolution = reader.slide_dimensions(**plot_resolution_dict)
    scale_adjust = np.array(node_resolution) / np.array(plot_resolution)
    if mag == '10X':
        scale_adjust = scale_adjust*2
    print(scale_adjust)
    
    node_coordinates = np.array(graph.coords) / scale_adjust
    edges = graph.edge_index.T
    
    thumb = reader.slide_thumbnail(**plot_resolution_dict)

    thumb_overlaid = plot_graph(
        thumb.copy(), node_coordinates, edges, node_colors=node_colors, node_size=node_size, edge_size=2
    )
    
    mask = load_mask(viz_slide, msk_paths, thumb=thumb)#, downscale=True)
    plot_images([thumb, thumb_overlaid, mask_image(mask, thumb)], large=True)
    

# Added msk_paths parameter
def plot_graph_mask_overlay(viz_slide, msk_paths, wsi_paths, graph_dir, mag='20X', resolution=5.0, node_size=4):

    graph_path = f"{graph_dir}/{viz_slide}.json"
    wsi_path = path_for_wsi(viz_slide, wsi_paths)
    
    mag2mpp = {'40X': .25, '20X': .5, '10X': 1.}
    node_res = mag2mpp[mag]
    
    node_resolution_dict = dict(resolution=node_res, units="mpp") # changed from 0.5
    #PLOT_RESOLUTION = dict(resolution=4.0, units="mpp")
    plot_resolution_dict = dict(resolution=resolution, units='power')
    
    graph_dict = load_json(graph_path)
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph = Data(**graph_dict)
    
    # deriving node colors via projecting n-d features down to 3-d
    graph.x = StandardScaler().fit_transform(graph.x)
    # .c for node colors
    node_colors = FastICA(n_components=3).fit_transform(graph.x)[:, [1, 0, 2]] # FastICA faster than PCA
    for channel in range(node_colors.shape[-1]):
        node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
    node_colors = (node_colors * 255).astype(np.uint8)
    
    reader = WSIReader.open(wsi_path)
    #thumb = reader.slide_thumbnail(4.0, "mpp")
    
    node_resolution = reader.slide_dimensions(**node_resolution_dict)
    plot_resolution = reader.slide_dimensions(**plot_resolution_dict)
    fx = np.array(node_resolution) / np.array(plot_resolution)
    if mag == '10X':
        fx = fx*2
    
    node_coordinates = np.array(graph.coords) / fx
    edges = graph.edge_index.T
    
    thumb = reader.slide_thumbnail(**plot_resolution_dict)
    thumb_overlaid = plot_graph(
        thumb.copy(), node_coordinates, edges, node_colors=node_colors, node_size=node_size, edge_size=2
    )
    
    mask = load_mask(viz_slide, msk_paths, thumb=thumb)  #, downscale=True)
    
    graph_overlaid = plot_graph(
        mask_image(mask, thumb).copy(), node_coordinates, edges, node_colors=node_colors, node_size=node_size, 
        edge_size=2
    )
    plot_images([thumb, thumb_overlaid, mask_image(mask, thumb),
                 mask_image(mask, thumb_overlaid), graph_overlaid], large=True)
    
    return mask, node_coordinates


def plot_graph_overlay(viz_slide, graph_dir, wsi_paths, mag='20X', save=True, node_size=8, edge_size=2, plot_edges=True,
                       save_image_path=None, num_clusters=None, set_max_clusters=False, plot_thumb=True):
    graph_path = f"{graph_dir}/{viz_slide}.json"
    wsi_path = path_for_wsi(viz_slide, wsi_paths)

    mag2mpp = {'40X': .25, '20X': .5, '10X': 1.}
    node_res = mag2mpp[mag]

    #NODE_RESOLUTION = dict(resolution=node_res, units="mpp")  # changed from 0.5
    #PLOT_RESOLUTION = dict(resolution=4.0, units="mpp")

    mpl.rcParams["figure.dpi"] = 300

    graph_dict = load_json(graph_path)
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph = Data(**graph_dict)

    # deriving node colors via projecting n-d features down to 3-d
    graph.x = StandardScaler().fit_transform(graph.x)
    # .c for node colors
    node_colors = FastICA(n_components=3).fit_transform(graph.x)[:, [1, 0, 2]] # FastICA faster than PCA
    for channel in range(node_colors.shape[-1]):
        node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
    node_colors = (node_colors * 255).astype(np.uint8)

    reader = WSIReader.open(wsi_path)
    # thumb = reader.slide_thumbnail(4.0, "mpp")

    #node_resolution = reader.slide_dimensions(**NODE_RESOLUTION)
    #plot_resolution = reader.slide_dimensions(**PLOT_RESOLUTION)
    #fx = np.array(node_resolution) / np.array(plot_resolution)

    PLOT_RESOLUTION = dict(resolution=5.0, units='power')
    node_resolution = 5.0
    power_scale = {1.25: 1, 5.0: 4, 20.0: 16}
    fx = power_scale[node_resolution]
    if mag == '10X':
        fx = fx * 2

    node_coordinates = np.array(graph.coords) / fx
    if plot_edges:
        edges = graph.edge_index.T
    else:
        edges = []

    thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
    thumb_overlaid = plot_graph(
        thumb.copy(), node_coordinates, edges, node_colors=node_colors, node_size=node_size, edge_size=edge_size
    )

    plt.figure(figsize=(20,20))
    if plot_thumb:
        plt.subplot(1, 2, 1)
        plt.imshow(thumb)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(thumb_overlaid)
        plt.axis("off")
    else:
        plt.imshow(thumb_overlaid)
        plt.axis("off")

    if save:
        fig = plt.gcf()
        fig.savefig(os.path.join(save_image_path,
                                 f'graph{f"_max_{num_clusters}" if set_max_clusters else ""}_{viz_slide}.png'))
    plt.show()


def plot_node_activations(viz_slide, viz_cohort, node_activations, prediction, graph, resp, viz_fold,
                          viz_epoch, mag='20X', node_size=8, edge_size=2, plot_edges=True, col_map="jet", large=False,
                          splits=None, responses=None, save_img_path=None, num_clusters=None, set_max_clusters=None):
    WSI_PATH = f'/well/rittscher/projects/imRSS/Data/{viz_cohort}/WSI/{viz_slide}.svs'

    mpl.rcParams["figure.dpi"] = 300

    mag2mpp = {'40X': .25, '20X': .5, '10X': 1.}

    node_resolution_dict = dict(resolution=mag2mpp[mag], units="mpp")  # changed from 0.5
    plot_resolution_dict = dict(resolution=4.0, units="mpp")  # changed from 4

    reader = OpenSlideWSIReader(WSI_PATH)
    node_resolution = reader.slide_dimensions(**node_resolution_dict)
    plot_resolution = reader.slide_dimensions(**plot_resolution_dict)
    fx = np.array(node_resolution) / np.array(plot_resolution)
    if mag == '10X':
        fx = fx * 2

    cmap = plt.get_cmap(col_map) # autumn, inferno, jet
    graph = graph.to("cpu")

    node_coordinates = np.array(graph.coords) / fx
    node_colors = (cmap(np.squeeze(node_activations))[..., :3] * 255).astype(np.uint8)
    if plot_edges:
        edges = graph.edge_index.T
    else:
        edges = []

    thumb = reader.slide_thumbnail(**plot_resolution_dict)
    thumb_overlaid = plot_graph(
        thumb.copy(), node_coordinates, edges, node_colors=node_colors, node_size=node_size, edge_size=edge_size
    )

    if large:
        fig, ax = plt.subplots(1, 1, figsize=(50, 50))
    else:
        ax = plt.subplot(1, 1, 1)
    plt.imshow(thumb_overlaid)
    plt.axis("off")
    # Add minorticks on the colorbar to make it easy to read the
    # values off the colorbar.
    fig = plt.gcf()
    norm = mpl.colors.Normalize(
        vmin=np.min(node_activations), vmax=np.max(node_activations)
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, extend="both")
    cbar.minorticks_on()

    if resp != 'epithelium':
        plt.title(f'Prediction: {prediction:.3f}. True: {label_from_splits(viz_slide, splits, resp, responses)}.')

    fig = plt.gcf()
    mkdir(os.path.join(save_img_path, resp))
    if save_img_path is not None:
        fig.savefig(os.path.join(save_img_path, resp,
                                 f'node_activation_{viz_slide}_{resp}_fold{viz_fold:02d}_epoch{viz_epoch}' + \
                                 f'{f"_max_{num_clusters}" if set_max_clusters else ""}' + \
                                 f'_{col_map}.png'))
    plt.show()


def plot_slide_prediction_hist(viz_slide, node_activations, resp, viz_epoch, viz_fold, responses, splits,
                               save_img_path=None):
    mpl.rcParams["figure.dpi"] = 100

    plt.figure(figsize=(10, 2))
    plt.hist(node_activations, bins=20)
    plt.xlim([0, 1])
    if resp != 'epithelium':
        plt.title(f'Response for {viz_slide}: {label_from_splits(viz_slide, splits, resp, responses)}')

    fig = plt.gcf()
    mkdir(os.path.join(save_img_path, resp))
    fig.savefig(os.path.join(save_img_path, resp,
                             f'node_activation_hist_{viz_slide}_{resp}_fold{viz_fold:02d}_epoch{viz_epoch}.png'))
    plt.show()

