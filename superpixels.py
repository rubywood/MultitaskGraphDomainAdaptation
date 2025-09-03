import numpy as np
import cv2
import os
import time

from skimage.segmentation import slic, mark_boundaries
from skimage.transform import pyramid_expand
from skimage.color import rgb2gray
from skimage.measure import regionprops

# TIAToolbox imports
from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader

# Local imports
from utils.data import path_for_wsi, mask_for_wsi, load_slide_features, patch_corner_coordinates
from utils.visualisation import to_shape, load_mask, mask_image
from utils.jit_utils import weight_slide_spxl_feats, mean_centers
from utils.utils import delete_multiple_element, mkdir


def weight_slide_spxl_feats_no_jit(spxls, scaled_coords, scaled_delta, features):

    uniq_spxls = np.delete(np.unique(spxls), 0)

    spxl_dict = {key: [] for key in uniq_spxls}
    slide_spxl_feats = []

    # search over patches in supxl mask for spxl propn in each patch
    for x ,y in scaled_coords:
        segment_patch_mask = spxls[y: y+ scaled_delta, x: x + scaled_delta]
        for spxl in np.unique(segment_patch_mask):  # 0,27,36
            if spxl == 0:
                continue
            propn = len(segment_patch_mask[segment_patch_mask == spxl]) / (scaled_delta * scaled_delta)
            spxl_dict[spxl].append([x, y, propn])

    unused_spxl = []
    # then take weighted mean for each spxl
    for spxl in uniq_spxls:
        if spxl_dict[spxl] == []:
            unused_spxl.append(spxl)
            continue
        spxl_weighted_feats = []
        for coords_propn in spxl_dict[spxl]:
            feat_idx = np.flatnonzero((scaled_coords == coords_propn[:2]).all(1))
            patch_feat_propn = features[feat_idx] * coords_propn[-1]
            spxl_weighted_feats.append(patch_feat_propn)
        spxl_feats = np.mean(spxl_weighted_feats, axis=0)
        slide_spxl_feats.extend(spxl_feats)

    if len(unused_spxl) > 0:
        print(
            f'WARNING: possible that patches weren\'t generated correctly originally. {len(unused_spxl)} superpixels unused.')

    return np.vstack(slide_spxl_feats), unused_spxl, uniq_spxls  # returns array (segs, dim)


### At low resolution

def assign_epithelium_labels(node_coordinates, mask):
    node_epi_labels = []
    for node in node_coordinates:
        node_epi_labels.append(int(mask[int(np.round(node[1])), int(np.round(node[0]))]))
    assert len(node_epi_labels) == len(node_coordinates)
    return node_epi_labels


def remove_background_spxls(segments, thumb, min_tissue_ratio=0.6, background_cutoff=220):
    # greyscale background cutoff [0,1]
    background_cutoff = background_cutoff / 256
    
    for spxl in np.unique(segments):
        if spxl == 0:
            # background, excluded
            continue
        
        # Use as mask
        spxl_mask = np.where(segments==spxl, 1, 0)
        spxl_thumb = mask_image(spxl_mask, thumb)
        spxl_arr = spxl_thumb[..., :3]
        # remove rgb channels to compare white/gray/black pixels
        grey_spxl_arr = rgb2gray(spxl_arr)
        
        # Count all pixels in spxl
        nz = np.nonzero(grey_spxl_arr)
        num_spxl_pixels = len(nz[0])
        
        # Count background pixels
        num_white_pixels = np.sum(grey_spxl_arr[nz] > background_cutoff)
        
        # Compare background to normal ratio
        if not num_white_pixels <= min_tissue_ratio * num_spxl_pixels:
            # delete superpixel - replace superpixel values with 0 in segments
            segments[segments == spxl] = 0
        
    return segments


def get_scale_factor(wsi, wsi_paths, mag='20X', resolution=5.0):

    mag2mpp = {'40X': .25, '20X': .5, '10X': 1.}
    node_res = mag2mpp[mag]
    #print('Node res in mpp:', node_res)
    
    node_resolution_dict = dict(resolution=node_res, units="mpp") # changed from 0.5
    plot_resolution_dict = dict(resolution=resolution, units='power') # e.g. 5X magnfiication
    
    wsi_path = path_for_wsi(wsi, wsi_paths=wsi_paths)
    reader = WSIReader.open(wsi_path)
    
    node_resolution = reader.slide_dimensions(**node_resolution_dict)
    #print('Node resolution:', node_resolution)
    plot_resolution = reader.slide_dimensions(**plot_resolution_dict)
    #print('Plot resolution:', plot_resolution)
    fx = np.array(node_resolution) / np.array(plot_resolution)
    
    if mag == '10X':
        fx = fx*2
        
    print(f'Rescaling coords by {fx}')
    
    return fx
    
#node_coordinates = np.array(centers) / fx


# TODO: optimise below function, seems to get slower after many iterations

# @timeout(3*60)
def superpixel_feats_for_one_slide(slide, wsi_paths, mask_paths, epi_msk_paths, wsi_feature_dir,
                                   scale_slic, base_name, base_version, seed, num_node_features,  train_or_val,
                                   num_patches=None, remove_background=False,
                                   resolution=5.0, mag='20X',
                                   compactness=20.0, save_feats=True, jit=False):
    if train_or_val.lower() == 'both':
        train_or_val = ['Train', 'Validation']
    else:
        train_or_val = [train_or_val]

    # If already exists, don't repeat
    if os.path.exists(os.path.join(wsi_feature_dir, train_or_val[-1], f'{slide}.features.npy')):
        print(f'Superpixel features already exist for {slide}. Skipping.')
        return None, None

    power_scale = {1.25: 1, 5.0: 4, 20.0: 16}
    # scale_factors = {2.0 : 4, 4.0 : 2, 8.0 : 1} # when units = mpp
    position_scale = power_scale[resolution]

    #if mag == '10X':
    #    position_scale = power_scale[resolution] / 2
    #else:
    #    position_scale = power_scale[resolution]
        
    exact_epi_scale_factors = get_scale_factor(slide, wsi_paths, mag=mag, resolution=resolution)

    # Load WSI
    wsi_path = path_for_wsi(slide, wsi_paths=wsi_paths)
    reader = WSIReader.open(wsi_path)
    thumb = reader.slide_thumbnail(resolution=resolution, units="power")

    # Load mask and scale to WSI
    mask_path = mask_for_wsi(slide, msk_paths=mask_paths)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # For Salzburg
    if len(mask.shape)>2:
        mask = mask[..., 0]

    upsampled_mask = pyramid_expand(mask, upscale=power_scale[resolution]) # mask size relative
    if upsampled_mask.shape != thumb.shape[:2]:
        print(f'Fitting mask of size {upsampled_mask.shape} to thumbnail of size {thumb.shape[:2]}')
        upsampled_mask = to_shape(upsampled_mask, thumb.shape[:2])

    # SLIC
    if scale_slic is None:
        num_segments = 100 # default in sklearn
    elif num_patches is not None:
        num_segments = int(num_patches / scale_slic) 
    else:
        num_segments = int(np.mean(thumb.shape[:2]) / scale_slic)  # segments relative to size of thumbnail
    segments = slic(thumb, n_segments=num_segments, slic_zero=False, compactness=compactness, mask=upsampled_mask)
    
    print(f'Found {len(np.unique(segments)) - 1} segments')
    
    if remove_background:
        segments = remove_background_spxls(segments, thumb)
        print(f'Found {len(np.unique(segments)) - 1} segments after removing background segments')

    # Load slide patch features

    # Epithelium binary labels - just uses spxl centres so much quicker than finding ratio
    epi_mask = load_mask(slide, epi_msk_paths, thumb=thumb)

    for mode in train_or_val:
        print('Mode:', mode)
        slide_features_paths = load_slide_features(slide, base_name=base_name, base_version=base_version, seed=seed,
                                                   train_or_val=mode)
        features = slide_features_paths['slide_features'].cpu().numpy()
        positions = np.array([patch_corner_coordinates(path) for path in slide_features_paths['patch_paths']])

        # Top left corner
        scaled_positions = np.round(positions[:, :2] / position_scale).astype(int) # replaced position_scale

        patch_size = 256
        scaled_delta = np.round(patch_size / position_scale).astype(int) # replaced position_scale

        # Find mean features per superpixel shape
        if jit:
            slide_spxl_feats, unused_spxl = weight_slide_spxl_feats(segments, scaled_positions, scaled_delta,
                                                                    features, num_node_features=num_node_features)
            centers = mean_centers(segments)
            centers = centers * position_scale #power_scale[resolution]
            ## swap x and y
            centers = np.array([list(v[::-1]) for v in centers])
        else:
            slide_spxl_feats, unused_spxl, uniq_spxls = weight_slide_spxl_feats_no_jit(segments, scaled_positions,
                                                                           scaled_delta, features)
            # uniq_spxl doesn't include 0

            #centers = np.array(
            #    [np.round(np.mean(np.nonzero(segments == i), axis=1)).astype(int) for i in np.unique(segments)])
            #centers = {i: list(reversed(np.round(np.mean(np.nonzero(segments == i),
            #                                 axis=1)).astype(int) * position_scale)) for i in uniq_spxls}

            # Compute region properties - fast
            props = regionprops(segments)
            # Initialize centers dictionary
            centers = {}
            # Iterate over regions
            for prop in props:
                label = prop.label
                centroid = prop.centroid
                # Calculate rounded centroid and multiply by position_scale
                rounded_centroid = np.round(centroid).astype(int) * position_scale
                # Add to centers dictionary
                centers[label] = list(reversed(rounded_centroid))
            # no zero label with regionprops approach

        # remove unused superpixels
        for spxl in unused_spxl:
            del centers[spxl]

        centers = list(centers.values())

        #centers = centers * np.mean(exact_scale_factors) #power_scale[resolution]
        ## swap x and y
        #centers = np.array([list(v[::-1]) for v in centers])
        ## remove unused superpixels
        #centers = delete_multiple_element(centers, unused_spxl)
        #if not jit:
        #    # remove 0th segment
        #    centers = np.delete(centers, 0, axis=0)

        assert slide_spxl_feats.shape[0] == len(centers), \
            f'The positions ({len(centers)}) and the features ({slide_spxl_feats.shape[0]}) are not the same shape'

        # Epithelium binary labels
        node_coordinates = np.array(centers) / exact_epi_scale_factors #position_scale
        #epi_mask = load_mask(slide, epi_msk_paths, thumb=thumb)
        epi_labels = assign_epithelium_labels(node_coordinates, epi_mask)

        ########## Epithelial ratio labels for each superpixel segment ##########
        # epi_mask = load_mask(slide, epi_msk_paths, thumb=thumb)
        # epi_labels = slide_epi_labels(segments, thumb, epi_mask)
        ## remove unused superpixels
        # epi_labels = delete_multiple_element(epi_labels, unused_spxl)
        # if not jit:
        #    # remove 0th segment
        #    epi_labels = np.delete(epi_labels, 0, axis=0)
        #
        # assert len(epi_labels) == len(centers), \
        #    f'The epi labels ({len(epi_labels)}) and positions ({len(centers)}) are not the same length'
        ####################

        if save_feats:
            mkdir(os.path.join(wsi_feature_dir, mode))
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.position.npy", centers)
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.features.npy", slide_spxl_feats)
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.binary_epi_labels.npy", epi_labels)

    return thumb, segments  # , centers, slide_spxl_feats


def patch_feats_for_one_slide(slide, wsi_paths, epi_msk_paths, wsi_feature_dir,
                                   base_name, base_version, seed, train_or_val,
                                   resolution=5.0, mag='20X', save_feats=True):
    # Load WSI
    wsi_path = path_for_wsi(slide, wsi_paths=wsi_paths)
    reader = WSIReader.open(wsi_path)
    thumb = reader.slide_thumbnail(resolution=resolution, units="power")

    # Load epi mask
    epi_mask = load_mask(slide, epi_msk_paths, thumb=thumb)

    exact_epi_scale_factors = get_scale_factor(slide, wsi_paths, mag=mag, resolution=resolution)
    #power_scale = {1.25: 1, 5.0: 4, 20.0: 16}
    #position_scale = power_scale[resolution]

    # Load slide patch features
    if train_or_val.lower() == 'both':
        train_or_val = ['Train', 'Validation']
    else:
        train_or_val = [train_or_val]

    for i in range(len(train_or_val)):
        mode = train_or_val[i]
        print('Mode:', mode)

        slide_features_paths = load_slide_features(slide, base_name=base_name, base_version=base_version, seed=seed,
                                                   train_or_val=mode)
        features = slide_features_paths['slide_features']
        # don't need to repeat positions across modes, only features
        if i == 0:
            positions = np.array([patch_corner_coordinates(path) for path in slide_features_paths['patch_paths']])
            del slide_features_paths
            
            # Patch centres
            # all x coords
            x_mean = np.mean([positions[:, 0], positions[:, 2]], axis=0)
            # all y coords
            y_mean = np.mean([positions[:, 1], positions[:, 3]], axis=0)
            centers = list(zip(x_mean, y_mean))
            
            # Epithelium binary labels - just uses spxl centres so much quicker than finding ratio
            node_coordinates = np.array(centers) / exact_epi_scale_factors
            epi_labels = assign_epithelium_labels(node_coordinates, epi_mask)

        if save_feats:
            mkdir(os.path.join(wsi_feature_dir, mode))
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.position.npy", centers)
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.features.npy", features.cpu().numpy())
            np.save(f"{os.path.join(wsi_feature_dir, mode, slide)}.binary_epi_labels.npy", epi_labels)
        del features

    return thumb, centers
