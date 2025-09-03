import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data, Dataset, Batch
import ujson as json
import random
from collections import Counter

from .utils import *


# * Determine WSI and mask paths


def collect_wsi_names(WSI_DIR, MSK_DIR):
    wsi_paths = recur_find_ext(WSI_DIR, [".svs", ".ndpi"])
    wsi_names = [pathlib.Path(v).stem for v in wsi_paths]
    msk_paths = None if MSK_DIR is None else [f"{MSK_DIR}/{v}.png" for v in wsi_names]
    assert len(wsi_paths) > 0, "No files found."
    return wsi_paths, wsi_names, msk_paths


def epi_paths(epi_msk_dir):
    return recur_find_ext(epi_msk_dir, [".png", ".jpeg", ".jpg"])


def slide_name_from_path(path, split_by='WSI'):
    return path.split(f'{split_by}/')[1].split('.')[0]


def wsi_feature_paths(wsi_dir, msk_dir, wsi_names):
    all_wsi_paths = recur_find_ext(wsi_dir, [".svs", ".ndpi"]) # added list(map(str, wsi_names)) below
    wsi_feat_paths = list(filter(lambda path: slide_name_from_path(path) in list(map(str, wsi_names)), all_wsi_paths))
    msk_paths = [f"{msk_dir}/{slide_name_from_path(path)}.png" for path in wsi_feat_paths]
    return wsi_feat_paths, msk_paths


def path_for_wsi(wsi, wsi_paths):
    for path in wsi_paths:
        if wsi == path.split('/')[-1].split('.')[0]:
            return path


def mask_for_wsi(wsi, msk_paths):
    for path in msk_paths:
        if wsi == path.split('/')[-1].split('.')[0]:
            return path


def get_mask_dir(cohort):
    mask_dir_str = f'Data/{cohort}/mask'
    if cohort == 'SALZBURG':
        mask_dir_str += 'QC'
    return mask_dir_str


def get_epi_mask_dir(cohort):
    mask_dir_str = f'Data/{cohort}/epithelial-mask'
    return mask_dir_str


def get_wsi_dir(cohort):
    return f'Data/{cohort}/WSI'


def find_base_data(wsi_dirs, msk_dirs, base_name, base_version, seed, epi_msk_dirs=None, test=False):
    
    base_feature_dir = os.path.join(f'checkpoint/{base_name}'
                                    + str(base_version), 'Features')
    # Find WSI names with saved patch features
    if float(base_version) < 5.0:
        base_feature_dir = os.path.join(base_feature_dir,  f'Round_{seed}')
        
        if test:
            wsi_names = sorted(os.listdir(os.path.join(base_feature_dir, 'Test')))
        else:
            wsi_names = sorted(os.listdir(os.path.join(base_feature_dir, 'Train')) +
                               os.listdir(os.path.join(base_feature_dir, 'Validation')))
    else:
            # have applied train and test transforms to all slides
            train_wsis = set(os.listdir(os.path.join(base_feature_dir, 'Train')))
            val_wsis = set(os.listdir(os.path.join(base_feature_dir, 'Validation')))
            # can also use validation slides for test, as have same augmentations
            wsi_names = sorted(list(train_wsis.intersection(val_wsis)))
    
    assert len(wsi_dirs) == len(msk_dirs)

    # Find paths for WSI names
    wsi_paths, msk_paths, epi_msk_paths = [], [], []
    for i in range(len(wsi_dirs)):
        cohort_wsi_paths, cohort_msk_paths = wsi_feature_paths(wsi_dirs[i], msk_dirs[i], wsi_names)
        wsi_paths += cohort_wsi_paths
        msk_paths += cohort_msk_paths

        if epi_msk_dirs:
            cohort_epi_msk_paths = epi_paths(epi_msk_dirs[i])
            epi_msk_paths += cohort_epi_msk_paths

    #assert len(wsi_names) == len(wsi_paths)
    #assert len(msk_paths) == len(wsi_names)

    return wsi_names, wsi_paths, msk_paths, epi_msk_paths, base_feature_dir


def find_wsi_path(wsi_name, seed_path):
    if os.path.exists(os.path.join(seed_path, 'Train', wsi_name)):
        return os.path.join(seed_path, 'Train', wsi_name)
    elif os.path.exists(os.path.join(seed_path, 'Validation', wsi_name)):
        return os.path.join(seed_path, 'Validation', wsi_name)
    elif os.path.exists(os.path.join(seed_path, 'Test', wsi_name)):
        return os.path.join(seed_path, 'Test', wsi_name)
    else:
        raise IOError(f"No features saved in Train, Validation or Test dirs at {seed_path} for slide {wsi_name}.")


# * Load patch metadata including labels

def load_patch_labels(clinical_file, mag, responses, cohorts):
    patch_labels = pd.read_csv(clinical_file, index_col=0)
    patch_labels = patch_labels[patch_labels.magnification == mag]
    for response in responses:
        if 'CMS' in response:
            # Salzburg doesn't have CMS_matching column so dropped here
            #patch_labels = patch_labels.dropna(subset=['CMS_matching'])
            if response not in patch_labels.columns:
                patch_labels = patch_labels.dropna(subset=['CMS_matching'])
                patch_labels = add_CMS_column(patch_labels, response)
            else:
                patch_labels = patch_labels.dropna(subset=[response])
        if response == 'response_cr_nocr':
            patch_labels = patch_labels.dropna(subset=[response])
            print('Filtering out responses with Standard Cap RT treatment only')
            patch_labels = patch_labels.dropna(subset=['Treatment'])
            patch_labels = patch_labels[patch_labels.Treatment.str.contains('Standard Cap RT')]
    patch_labels = select_cohort(patch_labels, cohorts)
    return patch_labels


def select_cohort(patch_labels, cohorts):
    #if cohort is not None:
    #if cohort in ['GRAMPIAN', 'ARISTOTLE']:
    patch_labels = patch_labels[patch_labels.cohort.isin(cohorts)]
    patch_labels.reset_index(drop=True, inplace=True)
    print('Total number of slides:', len(patch_labels.slide.unique()))
    return patch_labels


def add_CMS_column(patch_labels_df, response):
    patch_labels_df[response] = np.where(patch_labels_df.CMS_matching == response, 1, 0)
    return patch_labels_df


#def label_from_splits(slide, splits, resp, responses):
#    fold_train = np.array(splits[0]['train'])
#    fold_valid = np.array(splits[0]['valid'])
#    resp_index = responses.index(resp)
#    if slide in fold_train[:,0]:
#        return fold_train[np.where(fold_train[:,0]==slide)[0][0]][1][resp_index]
#    elif slide in fold_valid[:,0]:
#        return fold_valid[np.where(fold_valid[:,0]==slide)[0][0]][1][resp_index]
#    else:
#        return 'ERROR: No label found'

def label_from_splits(slide, splits, resp, responses):
    resp_index = responses.index(resp)
    for split_name in list(splits[0].keys()):
        fold_split = np.array(splits[0][split_name])
        if slide in fold_split[:,0]:
            return fold_split[np.where(fold_split[:,0]==slide)[0][0]][1][resp_index]
    return 'ERROR: No label found'


def filter_wsis_by_mode_graphs(wsi_names, epi_graph_dir, mode):
    mode_graph_paths = recur_find_ext(os.path.join(epi_graph_dir, mode), [".json"])
    mode_wsi_names = list(map(lambda path: path.split('/')[-1].split('.json')[0], mode_graph_paths))

    # those wsis in our list which we have graphs for
    return list(filter(lambda wsi: wsi in mode_wsi_names, wsi_names))


# train_wsis = np.unique([file.split('.')[0] for file in os.listdir(os.path.join(WSI_FEATURE_DIR, 'Train'))])
#    val_wsis = np.unique([file.split('.')[0] for file in os.listdir(os.path.join(WSI_FEATURE_DIR, 'Validation'))])
#    wsi_names = np.unique(train_wsis + val_wsis)

def filter_wsis_by_epi_graphs(wsi_names, filter_epi, epi_graph_dir, graph_cache_name=None):
    if filter_epi:
        if graph_cache_name is None:
            # From 5.0 onwards, graph_dir includes graph_name
            epi_graph_paths = recur_find_ext(epi_graph_dir, [".json"])
            epi_wsi_names = list(map(lambda path: path.split('/')[-1].split('.json')[0], epi_graph_paths))
        else:
            epi_graph_paths = recur_find_ext(os.path.join(epi_graph_dir, graph_cache_name), [".json"])
            epi_wsi_names = list(map(lambda path: slide_name_from_path(path, split_by=graph_cache_name),
                                     epi_graph_paths))
        # those which we have epi masks for
        wsi_names = list(filter(lambda wsi: wsi in epi_wsi_names, wsi_names))
    else:
        wsi_names = list(wsi_names)
    return wsi_names


def make_label_df_with_slide_labels(slide_df, responses):
    # epithelium is node-level label, dealt with differently
    responses_to_add = list(filter(lambda response: response != 'epithelium', responses))
    # wasn't splitting by case before
    slide_label_df = slide_df[['slide', 'case', 'cohort'] + responses_to_add]
    slide_label_df.slide = slide_label_df.slide.astype('str')  # for Salzburg int IDs
    slide_label_df.case = slide_label_df.case.astype('str')  # for Salzburg int IDs
    slide_label_df.rename(columns={responses_to_add[i]: f'LABEL_{i}' for i in range(len(responses_to_add))},
                          inplace=True)
    slide_label_df.rename(columns={'slide': 'WSI-CODE'}, inplace=True)
    return slide_label_df, responses_to_add


def split_train_val(label_df, train_val_split=0.7, seed=0):
    # Splitting by case (patient), not slide
    label_df.case = label_df.case.astype('str')
    cases = np.unique(label_df['case'].values)
    num_train_cases = int(np.ceil(len(cases) * train_val_split))
    random.seed(seed)
    random.shuffle(cases)
    train_cases = cases[:num_train_cases]
    val_cases = cases[num_train_cases:]
    train_slides = label_df[label_df['case'].isin(train_cases)]['WSI-CODE']
    val_slides = label_df[label_df['case'].isin(val_cases)]['WSI-CODE']
    print('Number of train slides:', len(train_slides))
    print('Number of validation slides:', len(val_slides))
    return sorted(train_slides), sorted(val_slides)


def filter_wsis(wsi_dataset, label_df):
    # filter train/val datasets by those which have a matching label
    return [wsi for wsi in wsi_dataset if wsi in label_df['WSI-CODE'].tolist()]


def upsample(wsis, labels):
    label_counts = np.unique(labels, return_counts=True)[1]
    upsample_factor = int(round(max(label_counts) / min(label_counts)))
    print('Upsample training set by a factor of', upsample_factor)

    upsampled_slides = list(wsis.copy())
    upsampled_labels = list(labels.copy())
    min_class = label_counts.argmin()

    min_label_slides = [wsis[i] for i in np.where(labels == min_class)[0]]
    min_labels = [min_class for i in np.where(labels == min_class)[0]]
    for l in range(upsample_factor - 1):
        upsampled_slides += min_label_slides
        upsampled_labels += min_labels

    return upsampled_slides, upsampled_labels

# one response at a time
def upsample_multiclass(wsi, all_labels, label_i, seed=0):
    random.seed(seed)
    all_labels = np.array(all_labels)
    wsi = np.array(wsi)

    upsampled_slides = wsi.copy()
    #print('all_labels:', all_labels)
    #print('label_i:', label_i)
    labels = all_labels[:, label_i]  # from [[1],[2],...] to [1,2,...]
    upsampled_labels = all_labels.copy()

    label_counts = Counter(labels)
    mode_label_count = label_counts.most_common(1)[0]
    #print('Mode label count:', mode_label_count)

    for label in np.unique(labels):
        #print(f'Label {label}')
        if label == mode_label_count[0]:
            continue
        else:
            # find wsis with corresponding label
            label_idx = np.where(labels == label)[0]
            #print(f'label_idx: {label_idx}')
            #label_slides = wsi[label_idx]
            #label_slides = np.array([wsi[i] for i in label_idx])
            sample_n = mode_label_count[1] - label_counts[label]
            #print(f'sample_n: {sample_n}')

            sample_idx = random.choices(label_idx, k=sample_n)
            #print('sample_idx:', sample_idx)
            sample_slides = wsi[sample_idx]
            #print(f'sample_slides: {sample_slides}')
            #sample_slides = random.choices(label_slides, k=sample_n)


            sample_labels = all_labels[sample_idx]
            #print(f'sample_labels: {sample_labels}')
            assert all(sample_labels[:,label_i] == label), f'Sample labels not {label}: {sample_labels}'

            #print('upsampled_slides.shape:', upsampled_slides.shape)
            #print('sample_slides.shape:', sample_slides.shape)
            upsampled_slides = np.append(upsampled_slides, sample_slides)
            #print('upsampled_labels.shape:', upsampled_labels.shape)
            #print('sample_labels.shape:', sample_labels.shape)
            upsampled_labels = np.vstack((upsampled_labels, sample_labels))
    #print(f'upsampled_labels: {upsampled_labels}')

    #upsampled_labels = [[lbl] for lbl in upsampled_labels]
    assert len(upsampled_slides) == len(upsampled_labels), \
        f'len(upsampled_slides) {len(upsampled_slides)} != len(upsampled_labels) {len(upsampled_labels)}'

    return upsampled_slides.tolist(), upsampled_labels.tolist()

def dual_upsample(wsi, labels, responses):
    upsampled_slides = list(wsi.copy())
    upsampled_labels = list(labels.copy())

    # Order of upsampling matters.
    upsample_factors = []
    for label_i in range(len(responses)):
        first_labels = np.array(upsampled_labels)[:, label_i]
        label_counts = np.unique(first_labels, return_counts=True)[1]
        print(label_counts)
        upsample_factor = int(round(max(label_counts) / min(label_counts)))
        upsample_factors.append(upsample_factor)

    # updated 16/01/23 to calculate factors first before loop round again instead of cumulative adjustments
    for label_i in range(len(responses)):
        upsample_factor = upsample_factors[label_i]
        print('Upsample training set by a factor of', upsample_factor)

        first_labels = np.array(upsampled_labels)[:, label_i]
        min_class = np.unique(first_labels, return_counts=True)[1].argmin()

        min_label_slides = [upsampled_slides[i] for i in np.where(first_labels == min_class)[0]]
        min_labels = pd.Series(upsampled_labels).loc[np.where(first_labels == min_class)[0]].tolist()

        for l in range(upsample_factor - 1):
            upsampled_slides += min_label_slides
            upsampled_labels += min_labels

    return upsampled_slides, upsampled_labels


# * Deal with features and patch coordinates

def load_slide_features(wsi_name, base_name, base_version, seed, train_or_val):
    
    seed_path = os.path.join(f'checkpoint/{base_name}'
                             + str(base_version), 'Features')
    if float(base_version) < 5.0:
        seed_path = os.path.join(seed_path,  f'Round_{seed}')
        wsi_feature_path = find_wsi_path(wsi_name, seed_path)
    else:
        wsi_feature_path = os.path.join(seed_path, train_or_val, wsi_name)
    return torch.load(wsi_feature_path, map_location=torch.device('cuda')) # dictionary of feats and paths


def position_corners(coords, patch_size=256):
    return [coords[0], coords[1], coords[0]+patch_size, coords[1]+patch_size]


def patch_corner_coordinates(patch_path, patch_size=256):
    patch_path = patch_path.replace('0.0x', '0x')  # some patches have 20.0x in name but later split by .
    path_list = patch_path.split('/')[-1].split('.')[0].split('_')
    try:
        x_index = path_list.index('x')
        y_index = path_list.index('y')
    except IndexError:
        raise ValueError("x or y is missing")
    x_coord = int(path_list[x_index + 1])
    y_coord = int(path_list[y_index + 1])
    return position_corners([x_coord, y_coord], patch_size)
    #return [x_coord, y_coord, x_coord+patch_size, y_coord+patch_size]


# * Define sampler class

class StratifiedSampler(Sampler):
    """Sampling the dataset such that the batch contains stratified samples.

    Args:
        labels (list): List of labels, must be in the same ordering as input
            samples provided to the `SlideGraphDataset` object.
        batch_size (int): Size of the batch.
    Returns:
        List of indices to query from the `SlideGraphDataset` object.

    """

    def __init__(self, labels, batch_size=10):
        self.batch_size = batch_size
        self.num_splits = int(len(labels) / self.batch_size)
        self.labels = labels
        self.num_steps = self.num_splits

    def _sampling(self):
        # do we want to control randomness here
        skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
        indices = np.arange(len(self.labels))  # idx holder
        # create single label indicating at least one 1 in label set
        indicator_label = np.ceil(np.mean(self.labels, axis=1))
        # return array of arrays of indices in each batch
        return [tidx for _, tidx in skf.split(indices, indicator_label)]

    def __iter__(self):
        return iter(self._sampling())

    def __len__(self):
        """The length of the sampler.

        This value actually corresponds to the number of steps to query
        sampled batch indices. Thus, to maintain epoch and steps hierarchy,
        this should be equal to the number of expected steps as in usual
        sampling: `steps=dataset_size / batch_size`.

        """
        return self.num_steps


# * Define Pytorch dataset classes

### This version of SlideGraphDataset changes graph dir depending on cohort i.e. _8 for Salzburg, _2 for G+A
class SlideGraphDataset(Dataset):
    """Handling loading graph data from disk.

    Args:
        info_list (list): In case of `train` or `valid` is in `mode`,
            this is expected to be a list of `[uid, label]` . Otherwise,
            it is a list of `uid`. Here, `uid` is used to construct
            `f"{GRAPH_DIR}/{wsi_code}.json"` which is a path points to
            a `.json` file containing the graph structure. By `label`, we mean
            the label of the graph. The format within the `.json` file comes
            from `tiatoolbox.tools.graph`.
        mode (str): This denotes which data mode the `info_list` is in.
        preproc (callable): The preprocessing function for each node
            within the graph.

    """

    def __init__(self, info_list, graph_dir, mode="train", preproc=None, scale_slic=2):
        super().__init__(None, transform=None, pre_transform=None)  # add for latest pyg version
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.graph_dir = graph_dir
        self.scale_slic = scale_slic

    def get(self, idx):
        info = self.info_list[idx]
        if self.mode == 'infer':  # then info is wsi_name only
            wsi_code = info  # [0] # remove [0] for preproc func and validation
        else:
            wsi_code, label = info
            # torch.Tensor will create 1-d vector not scalar
            labels = torch.tensor(label)

        # ADDED CODE TO DISTINGUISH BETWEEN SALZBURG AND OTHER GRAPH DIRS WITH DIFF SCALE SLIC
        # print(wsi_code)
        if wsi_code.isnumeric():
            graph_dir = self.graph_dir[:-1] + str(self.scale_slic * 4)
        else:
            graph_dir = self.graph_dir[:-1] + str(self.scale_slic)

        with open(f"{graph_dir}/{wsi_code}.json", "r") as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        # epi_label = graph_dict.pop('epi_label')
        # print('This many epi labels:', len(epi_label))
        graph = Data(**graph_dict)

        if graph_dict['edge_index'].shape[1] == 0:
            print(f'WSI graph {wsi_code} has no edges.')

        if any(v in self.mode for v in ["train", "valid", "test"]):
            # print(labels)
            # labels = torch.hstack((labels, epi_label))
            # labels[:2] is first 2 labels, labels[2:] is epi labels
            return dict(graph=graph, label=labels), wsi_code
        return dict(graph=graph), wsi_code

    def len(self):
        return len(self.info_list)


# Use for epithelial prediction on each node, not one epithelial label per slide (e.g. ratio)
class SlideGraphEpiDataset(Dataset):
    """Handling loading graph data from disk.

    Args:
        info_list (list): In case of `train` or `valid` is in `mode`,
            this is expected to be a list of `[uid, label]` . Otherwise,
            it is a list of `uid`. Here, `uid` is used to construct
            `f"{GRAPH_DIR}/{wsi_code}.json"` which is a path points to
            a `.json` file containing the graph structure. By `label`, we mean
            the label of the graph. The format within the `.json` file comes
            from `tiatoolbox.tools.graph`.
        mode (str): This denotes which data mode the `info_list` is in.
        preproc (callable): The preprocessing function for each node
            within the graph.

    """

    def __init__(self, info_list, graph_dir, mode="train", preproc=None):
        super().__init__(None, transform=None, pre_transform=None) # add for latest pyg version
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.graph_dir = graph_dir

    def get(self, idx):
        info = self.info_list[idx]
        if self.mode == 'infer':  # then info is wsi_name only
            wsi_code = info  # [0] # remove [0] for preproc func and validation
        else:
            wsi_code, label = info
            # torch.Tensor will create 1-d vector not scalar
            labels = torch.tensor(label)
            # wsi_code = info[0]

        with open(f"{self.graph_dir}/{wsi_code}.json", "r") as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        epi_label = graph_dict.pop('epi_label')
        # print('This many epi labels:', len(epi_label))
        graph = Data(**graph_dict)

        if graph_dict['edge_index'].shape[1] == 0:
            print(f'WSI graph {wsi_code} has no edges.')

        if any(v in self.mode for v in ["train", "valid", "test"]):
            # print(labels)
            labels = torch.hstack((labels, epi_label))
            # labels[:2] is first 2 labels, labels[2:] is epi labels
            return dict(graph=graph, label=labels), wsi_code
        return dict(graph=graph), wsi_code

    def len(self):
        return len(self.info_list)


def collate_fn_pad(batch):
    # print('COLLATING!')
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print('Batch:', batch)

    graph_dicts = [b[0] for b in batch]
    # print('Graph dicts:', graph_dicts)
    geo_data = [g_d['graph'] for g_d in graph_dicts]
    # print('Geo data:', geo_data)
    if 'label' in graph_dicts[0]:
        label_present = True
        labels = [g_d['label'] for g_d in graph_dicts]
    else:
        label_present = False
    # print('Labels:', labels)

    batched_geo_data = Batch.from_data_list(geo_data)
    # print('Batched geo data:', batched_geo_data)

    wsi_str = [b[1] for b in batch]
    # print('Wsi names:', wsi_str)
    batched_wsis = torch.utils.data.default_collate(wsi_str)

    if label_present:
        ## get sequence lengths
        lengths = torch.tensor([t.shape[0] for t in labels])  # .to(device)
        # print('Lengths:', lengths)
        ## padd
        batched_labels = [torch.Tensor(t) for t in labels]  # add .to(device) after tensor
        # print('Labels as tensors:', batch)
        batched_labels = torch.nn.utils.rnn.pad_sequence(batched_labels, batch_first=True, padding_value=-1.)
        # print('Labels after padding:', batched_labels)
        ## compute mask
        # masks = (batched_labels != -1)#.to(device)
        # return batch, lengths, masks # must apply masks piecewise i.e. not to whole batch at once

    # recombine batch data
    if label_present:
        batch_graph_dict = {'graph': batched_geo_data, 'label': batched_labels, 'length': lengths}
    else:
        batch_graph_dict = {'graph': batched_geo_data}
    return (batch_graph_dict, batched_wsis)

# TODO: return batched_geo_data, batched_labels, batched_wsis
# Return as tuple, first element dictionary with 'graph' and 'label' (and lengths, mask?) keys,
#   second element WSI name
