import errno
import os
import signal
import functools
from numba import jit, prange
import numpy as np

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# * Using Jit to find superpixel patch feature proportions

@jit(nopython=True)
def superpixel_patch_proportions(spxls, scaled_coords, scaled_delta):
    uniq_spxls = np.delete(np.unique(spxls), 0)

    # spxl_dict = {key: numba.typed.List.empty_list(numba.f8) for key in uniq_spxls} # change back to list [] throws numba error
    spxl_dict_x = {key: np.empty(0) for key in uniq_spxls}
    spxl_dict_y = {key: np.empty(0) for key in uniq_spxls}
    spxl_dict_propn = {key: np.empty(0) for key in uniq_spxls}

    for x, y in scaled_coords:
        segment_patch_mask = spxls[y: y + scaled_delta, x: x + scaled_delta]
        for spxl in np.unique(segment_patch_mask):  # 0,27,36
            if spxl == 0:
                continue
            propn = np.count_nonzero(segment_patch_mask == spxl) / (scaled_delta * scaled_delta)
            # spxl_dict[spxl] = np.append(spxl_dict[spxl], [[x, y, propn]])
            # spxl_dict[spxl].append(numba.typed.List([x, y, propn]))
            spxl_dict_x[spxl] = np.append(spxl_dict_x[spxl], x)
            spxl_dict_y[spxl] = np.append(spxl_dict_y[spxl], y)
            spxl_dict_propn[spxl] = np.append(spxl_dict_propn[spxl], propn)

    return spxl_dict_x, spxl_dict_y, spxl_dict_propn


@jit(nopython=True)
def weighted_mean_superpixel_features(spxls, scaled_coords, features, spxl_dict_x, spxl_dict_y, spxl_dict_propn,
                                      num_node_features=384):
    uniq_spxls = np.delete(np.unique(spxls), 0)

    # determine shape of matrix to save features to i.e. (number of spxls, feature dim)
    count_unused_spxl = 0
    for spxl in uniq_spxls:
        if spxl_dict_x[spxl].size == 0:
            count_unused_spxl += 1
    total_spxls = len(uniq_spxls) - count_unused_spxl
    # np.matrix(total_spxls)
    print('Total superpixels:', total_spxls)

    slide_spxl_feats = np.zeros(shape=(total_spxls, num_node_features))

    unused_spxl = np.empty(0)
    # then take weighted mean for each spxl
    for j in range(len(uniq_spxls)):
        spxl = uniq_spxls[j]
        if spxl_dict_x[spxl].size == 0:  # was spxl_dict[spxl] == []
            # print(spxl)
            unused_spxl = np.append(unused_spxl, spxl)
            ##unused_spxl.append(spxl)
            continue

        x_vals = spxl_dict_x[spxl]
        y_vals = spxl_dict_y[spxl]
        propn_vals = spxl_dict_propn[spxl]

        spxl_weighted_feats = np.zeros(shape=(len(x_vals), num_node_features))

        for i in range(len(x_vals)):
            x = x_vals[i]
            y = y_vals[i]
            propn = propn_vals[i]

            feat_idx = None
            for c_idx in range(len(scaled_coords)):
                coords_x = scaled_coords[c_idx, 0]
                coords_y = scaled_coords[c_idx, 1]

                # for coords_x, coords_y in scaled_coords:
                # print('coords_x:', coords_x)
                # print('coords_y:', coords_y)
                # print('Matching to x:', x)
                # print('Matching to y:', y)
                if x == coords_x and y == coords_y:
                    feat_idx = c_idx
                    # print('Feat idx set to:', feat_idx)
                    break
            # print('Calculating features')
            if feat_idx is None:
                raise Exception(f'No matching coordinates found for x={x}, y={y}')
            patch_feat_propn = features[feat_idx] * propn
            spxl_weighted_feats[i] = patch_feat_propn
            # spxl_weighted_feats = np.append(spxl_weighted_feats, patch_feat_propn)

        # for coords_propn in spxl_dict[spxl]:
        #    #print(scaled_coords == coords_propn[:2])
        #    idx = 0
        #    for x,y in scaled_coords:
        #        if x == coords_propn[:2][0] and y == coords_propn[:2][1]:
        #        #if ([x,y] == coords_propn[:2]).all():
        #            feat_idx = idx
        #        #    break
        #        idx += 1
        #    #print(feat_idx)
        #    feat_idx = np.flatnonzero((scaled_coords == coords_propn[:2]).all(1))
        #    patch_feat_propn = features[feat_idx] * coords_propn[-1]
        #    spxl_weighted_feats = np.append(spxl_weighted_feats, patch_feat_propn)
        spxl_feats = mean_numba(spxl_weighted_feats)
        # slide_spxl_feats.extend(spxl_feats)
        slide_spxl_feats[j] = spxl_feats

    if len(unused_spxl) > 0:
        print(
            f'WARNING: possible that patches weren\'t generated correctly originally. {len(unused_spxl)} superpixels unused.')

    return slide_spxl_feats, unused_spxl
    # return np.vstack(slide_spxl_feats), unused_spxl # returns array (segs, dim)


@jit
def mean_numba(a):
    res = []
    for i in prange(a.shape[-1]):
        res.append(a[:, i].mean())

    return np.array(res)


#@jit(nopython=True)
def mean_centers(spxls):
    uniq_spxls = np.unique(spxls)
    centers = np.zeros(shape=(len(uniq_spxls) - 1, 2))
    for i in range(len(uniq_spxls)):
        if i == 0:
            continue  # don't do for zero as not a superpixel, just background
        spxl_region = np.nonzero(spxls == uniq_spxls[i])
        center = [int(np.round(spxl_region[0].mean())), int(np.round(spxl_region[1].mean()))]
        #    center = np.round(mean_numba_axis_1(spxl_region)).astype(int)
        centers[i - 1] = center
    return centers


def weight_slide_spxl_feats(spxls, scaled_coords, scaled_delta, features, num_node_features=384):
    spxl_dict_x, spxl_dict_y, spxl_dict_propn = superpixel_patch_proportions(spxls, scaled_coords, scaled_delta)
    return weighted_mean_superpixel_features(spxls, scaled_coords, features,
                                             spxl_dict_x, spxl_dict_y, spxl_dict_propn, num_node_features)