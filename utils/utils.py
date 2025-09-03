import os
# Use ujson as replacement for default json because it's faster for large JSON
import ujson as json
from typing import Callable, List, Tuple, Dict, Union
import pathlib
import shutil
import numpy as np
import copy
import argparse


def load_json(path: str):
    """Load JSON from a file path."""
    with open(path, "r") as fptr:
        json_dict = json.load(fptr)
    return json_dict


def rmdir(dir_path: str):
    """Remove a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def rm_n_mkdir(dir_path: str):
    """Remove then re-create a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return


def mkdir(dir_path: str):
    """Create a directory if it does not exist."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def recur_find_ext(root_dir: str, exts: List[str]) -> List[str]:
    """Recursively find files with an extension in `exts`.

    This is much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (str):
            Root directory for searching.
        exts (list):
            List of extensions to match.

    Returns:
        List of full paths with matched extension in sorted order.

    """
    assert isinstance(exts, list)
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in exts:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object = np.delete(list_object, idx, axis=0)
    return list_object


# Four functions below from tiatoolbox.utils.misc.py

def __walk_list_dict(in_list_dict):
    """Recursive walk and jsonify in place.

    Args:
        in_list_dict (list or dict):  input list or a dictionary.

    Returns:
        list or dict

    """
    if isinstance(in_list_dict, dict):
        __walk_dict(in_list_dict)
    elif isinstance(in_list_dict, list):
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.ndarray):
        in_list_dict = in_list_dict.tolist()
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.generic):
        in_list_dict = in_list_dict.item()
    elif in_list_dict is not None and not isinstance(
        in_list_dict, (int, float, str, bool)
    ):
        raise ValueError(
            f"Value type `{type(in_list_dict)}` `{in_list_dict}` is not jsonified."
        )
    return in_list_dict


def __walk_list(lst):
    """Recursive walk and jsonify a list in place.

    Args:
        lst (list):  input list.

    """
    for i, v in enumerate(lst):
        lst[i] = __walk_list_dict(v)


def __walk_dict(dct):
    """Recursive walk and jsonify a dictionary in place.

    Args:
        dct (dict):  input dictionary.

    """
    for k, v in dct.items():
        if not isinstance(k, (int, float, str, bool)):
            raise ValueError(f"Key type `{type(k)}` `{k}` is not jsonified.")
        dct[k] = __walk_list_dict(v)


def save_as_json(
    data: Union[dict, list],
    save_path: Union[str, pathlib.Path],
    parents: bool = False,
    exist_ok: bool = False,
):
    """Save data to a json file.

    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.

    Args:
        data (dict or list):
            Input data to save.
        save_path (str):
            Output to save the json of `input`.
        parents (bool):
            Make parent directories if they do not exist. Default is
            False.
        exist_ok (bool):
            Overwrite the output file if it exists. Default is False.


    """
    shadow_data = copy.deepcopy(data)  # make a copy of source input
    if not isinstance(shadow_data, (dict, list)):
        raise ValueError(f"Type of `data` ({type(data)}) must be in (dict, list).")

    if isinstance(shadow_data, dict):
        __walk_dict(shadow_data)
    else:
        __walk_list(shadow_data)

    save_path = pathlib.Path(save_path)
    if save_path.exists() and not exist_ok:
        raise FileExistsError("File already exists.")
    if parents:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as handle:  # skipcq: PTC-W6004
        json.dump(shadow_data, handle)


def str2bool(v):
    if isinstance(v, list):
        return [str2bool(elt) for elt in v]
    elif isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

