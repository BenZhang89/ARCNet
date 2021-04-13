from __future__ import absolute_import, print_function
import os
import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import preprocessor
import util.preprocessor as preprocessor
from itertools import groupby
import gc
from memory_profiler import profile
# transform_train = transforms.Compose([
#     transforms.RandomCrop(200, padding=56),
#     transforms.ToTensor(),
# ])


class ImdbData(data.Dataset):
    def __init__(self, X, y, w, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.transforms = transforms

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        weight = torch.from_numpy(self.w[index])
        return img, label, weight

    def __len__(self):
        return len(self.y)

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
    """
    def __init__(self, X_path, y_path, w_path, transforms=None):
        super().__init__()
        self.X_path = X_path
        self.y_path = y_path
        self.w_path = w_path
        self.data_info = []
        self.X = None
        self.y = None
        self.w = None
        self.transforms = transforms
        #for sample output in logWriter
        self.sample_index = None
        self.sample_X = None
        self.sample_y = None
        files = [self.X_path, self.y_path, self.w_path]
        for h5dataset_fp in files:
            self._add_data_infos(h5dataset_fp)           
    
    def __getitem__(self, index):
        # get data
        self.get_data("data", index)
        self.X = self.X if len(self.X.shape) == 3 else self.X[np.newaxis, :, :]
        self.X = torch.from_numpy(self.X)

        # get label
        self.get_data("label", index)
        self.y = torch.from_numpy(self.y)

        # get weight
        self.get_data("class_weights", index)
        self.w = torch.from_numpy(self.w)  
        
        return self.X, self.y, self.w

    def __len__(self):
        return self.get_data_infos('data')[0]['shape']
    
    def _add_data_infos(self, file_path):
        with h5py.File(file_path,"r") as h5_file:
            # Walk through all groups, extracting datasets
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                data_num = ds.shape[0]
                if not self.sample_index:
                    self.sample_index = list(np.random.choice(data_num, 3, replace=False))
                    self.sample_index.sort()
                    print(f'Selecting sample index for logWrigter of {file_path} is {self.sample_index}')
                if dname == "data":
                    self.sample_X = ds[self.sample_index]
                    self.sample_X = self.sample_X if len(self.sample_X.shape) == 4 else self.sample_X[:,np.newaxis, :, :]
                if dname == "label":
                    self.sample_y = ds[self.sample_index] 
                # just load the idx for data, label and weight, not the realdata              
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                #print({'file_path': file_path, 'type': dname, 'shape': data_num})
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': data_num})

    def _load_data(self, data_info, i):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(data_info['file_path'],"r") as h5_file:
            for dname, ds in h5_file.items():
                # add data to the data cache and retrieve
                # the cache index
                #print(f'Load data index {i}, data type is {data_info["type"]}, data shape is {ds[i].shape}')
                #self.data_cache[dname] = ds.value[i]
                if dname == "data":
                    self.X = ds[i]
                elif dname == "label":
                    self.y = ds[i]
                else:
                    self.w = ds[i]
    def get_data_infos(self, data_type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == data_type]
        return data_info_type

    def get_data(self, data_type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        di = self.get_data_infos(data_type)[0]
        self._load_data(di, i)

def get_imdb_dataset(data_params):
#    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
#    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
#    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
#    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')
#
#    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
#    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
#    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
#    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')
#
#    return (ImdbData(data_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()]),
#            ImdbData(data_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()]))

    data_train = os.path.join(data_params['data_dir'], data_params['train_data_file'])
    label_train = os.path.join(data_params['data_dir'], data_params['train_label_file'])
    class_weight_train = os.path.join(data_params['data_dir'], data_params['train_class_weights_file'])
    weight_train = os.path.join(data_params['data_dir'], data_params['train_weights_file'])

    data_test = os.path.join(data_params['data_dir'], data_params['test_data_file'])
    label_test = os.path.join(data_params['data_dir'], data_params['test_label_file'])
    class_weight_test = os.path.join(data_params['data_dir'], data_params['test_class_weights_file'])
    weight_test = os.path.join(data_params['data_dir'], data_params['test_weights_file'])


    return (HDF5Dataset(data_train, label_train, class_weight_train),
            HDF5Dataset(data_test, label_test, class_weight_test))

def load_dataset(file_paths,
                 orientation,
                 remap_config,
                 return_weights=False,
                 reduce_slices=False,
                 remove_black=False):
    print("Loading and preprocessing data...")
    volume_list, labelmap_list, headers, class_weights_list, weights_list = [], [], [], [], []

    for file_path in file_paths:
        try:
            volume, labelmap, class_weights, weights, header = load_and_preprocess(file_path, orientation,
                                                                               remap_config=remap_config,
                                                                               reduce_slices=reduce_slices,
                                                                               remove_black=remove_black,
                                                                               return_weights=return_weights)

            volume_list.append(volume)
            labelmap_list.append(labelmap)

            if return_weights:
                class_weights_list.append(class_weights)
                weights_list.append(weights)

            headers.append(header)

            print("#", end='', flush=True)
        except:
            print(f'\nFailed to convert {file_path}', flush=True)
            continue 
    print("100%", flush=True)
    if return_weights:
        return volume_list, labelmap_list, class_weights_list, weights_list, headers
    else:
        return volume_list, labelmap_list, headers


def load_and_preprocess(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False):
    volume, labelmap, header = load_data(file_path, orientation)

    volume, labelmap, class_weights, weights = preprocess(volume, labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume, labelmap, class_weights, weights, header


def load_and_preprocess_eval(file_path, orientation, notlabel=True):
    volume_nifty = nb.load(file_path[0])
    header = volume_nifty.header
    volume = volume_nifty.get_fdata()
    if notlabel:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    else:
        volume = np.round(volume)
    if orientation == "COR":
        volume = volume.transpose((2, 0, 1))
    elif orientation == "AXI":
        volume = volume.transpose((1, 2, 0))
    return volume, header


def load_data(file_path, orientation):
    volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
    volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, orientation)
    return volume, labelmap, volume_nifty.header


def preprocess(volume, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume, labelmap = preprocessor.reduce_slices(volume, labelmap)

    if remap_config:
        labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, labelmap = preprocessor.remove_black(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, labelmap, class_weights, weights
    else:
        return volume, labelmap, None, None


# def load_file_paths(data_dir, label_dir, volumes_txt_file=None):
#     """
#     This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
#     It should be modified to suit the need of the project
#     :param data_dir: Directory which contains the data files
#     :param label_dir: Directory which contains the label files
#     :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
#     :return: list of file paths as string
#     """
#
#     volume_exclude_list = ['IXI290', 'IXI423']
#     if volumes_txt_file:
#         with open(volumes_txt_file) as file_handle:
#             volumes_to_use = file_handle.read().splitlines()
#     else:
#         volumes_to_use = [name for name in os.listdir(data_dir) if
#                           name.startswith('IXI') and name not in volume_exclude_list]
#
#     file_paths = [
#         [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol, 'mri/aseg.auto_noCCseg.mgz')]
#         for
#         vol in volumes_to_use]
#     return file_paths


def load_file_paths(data_dir, label_dir, data_id, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param data_id: A flag indicates the name of Dataset for proper file reading
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]

    if data_id == "MALC":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol + '_glm.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "ADNI":
        file_paths = [
            [os.path.join(data_dir, vol, 'orig.mgz'), os.path.join(label_dir, vol, 'Lab_con.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "CANDI":
        file_paths = [
            [os.path.join(data_dir, vol + '/' + vol + '_1.mgz'),
             os.path.join(label_dir, vol + '/' + vol + '_1_seg.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "IBSR":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol + '_map.nii.gz')]
            for
            vol in volumes_to_use]
    #add for HCP-YA
    elif data_id == "HCP_YA":
        file_paths = [            
            [os.path.join(data_dir, vol, 'mri/T1.mgz'), os.path.join(label_dir, vol, 'mri/aparc+aseg.mgz')] 
            for
            vol in volumes_to_use]
    #add for ADNI, patient wise
    elif data_id == "ADNI_patient":
        volumes_to_use_patient = [list(i) for j, i in groupby(volumes_to_use, lambda a: a.split('_')[0])]
        file_paths = []
        for i,patient in enumerate(volumes_to_use_patient):
            file_paths.append([])
            for vol in patient:
                file_paths[i].append([os.path.join(data_dir, vol, 'mri/T1.mgz'), os.path.join(label_dir, vol, 'mri/aparc+aseg.mgz')])

    #add for NACC
    elif data_id == "NACC":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/T1.mgz'), os.path.join(label_dir, vol, 'mri/aparc+aseg.mgz')]
            for
            vol in volumes_to_use]
    #add for MINDBOGGLE
    elif data_id == "MINDBOGGLE":
    #example path for mindboogle:
    #/gpfs/data/cbi/hcp/hcp_seg/data_manual/manual_data_3/Extra-18_volumes/HLN-12-6/, vol is Extra-18_volumes/HLN-12-6
        file_paths = [
            [os.path.join(data_dir, vol, 't1weighted.MNI152.nii.gz'), os.path.join(label_dir, vol, 'labels.DKT31.manual+aseg.MNI152.nii.gz')]
            for
            vol in volumes_to_use]    

    else:
        raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    return file_paths


def load_file_paths_eval(data_dir, volumes_txt_file, dir_struct):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param volumes_txt_file:  Path to the a csv file, when provided only these data points will be read
    :param dir_struct: If the id_list is in FreeSurfer style or normal
    :return: list of file paths as string
    """

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()
    if dir_struct == "FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/T1.nii')]   #orig.mgz is the original, T1.nii is for DARTS
            for
            vol in volumes_to_use]
    elif dir_struct == "Linear":
        file_paths = [
            [os.path.join(data_dir, vol)]
            for
            vol in volumes_to_use]
    elif dir_struct == "part_FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'orig.mgz')]
            for
            vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are FS and Linear")
    return file_paths
