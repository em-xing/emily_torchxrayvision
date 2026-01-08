import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile
import pdb
import itertools

import imageio
import numpy as np
import pandas as pd
import skimage
from typing import Dict
import skimage.transform
from skimage.io import imread
import torch
from torchvision import transforms
from .windowing import apply_random_window_width, apply_window
from .constants import default_pathologies

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

# this is for caching small things for speed
_cache_dict = {}

__all__ = [
    'NIH_Dataset', 'PC_Dataset', 'CheX_Dataset', 'RSNA_Pneumonia_Dataset',
    'SIIM_Pneumothorax_Dataset', 'COVID19_Dataset', 'NLMT_Dataset',
    'VinBrain_Dataset', 'ObjectCXR_Dataset', 'PadChest_Dataset',
    'MIMIC_Dataset', 'XRayCenterCrop', 'XRayResizer', 'RandomZoom', 'apply_transforms'
]


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).format(img.max(), maxval)")

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


def relabel_dataset(pathologies, dataset, silent=False):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        if not silent:
            print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


def calculate_multifactorial_sample_idxs(class_values_df, class_balancing_proportions):
    possible_keys = []
    idxs_per_factor_vals = {}
    for factor_num, tup in enumerate(class_balancing_proportions):
        factor = tup[0]
        specs = tup[1]
        if factor == 'pathology':
            possible_paths = specs.keys()
            possible_vals = []
            for this_path in possible_paths:
                for v in ['0', '1']:
                    possible_vals.append(this_path + ' -- ' + v)
        else:
            if isinstance(specs, list):
                possible_vals = specs
            else: # dict
                possible_vals = list(specs.keys())
        possible_keys.append(possible_vals)

        idxs_per_factor_vals[factor_num] = {}
        if factor == 'pathology':
            for v in possible_vals:
                path_vals = v.split(' -- ')  # e.g 'No Finding -- 0'
                pathology = path_vals[0]
                path_label = int(path_vals[1])
                idxs_per_factor_vals[factor_num][v] = class_values_df[pathology] == path_label
        else:
            for v in possible_vals:
                idxs_per_factor_vals[factor_num][v] = class_values_df[factor] == v

    all_sample_keys = list(itertools.product(*possible_keys))
    print('number of sample combinations:', len(all_sample_keys))

    idxs_per_sample_key = {}
    counts_per_sample_key = {}
    for sample_key in all_sample_keys:
        for factor_num, factor_val in enumerate(sample_key):
            this_idx = idxs_per_factor_vals[factor_num][factor_val]
            if factor_num == 0:
                sample_idx = this_idx
            else:
                sample_idx = sample_idx & this_idx
        idxs_per_sample_key[sample_key] = np.where(sample_idx.values)[0].tolist()
        counts_per_sample_key[sample_key] = len(idxs_per_sample_key[sample_key])
        if not counts_per_sample_key[sample_key]:
            print('no examples for', sample_key)

    return idxs_per_sample_key


def sample_idx_multifactorial(class_balancing_proportions, idxs_per_sample_key):
    idx = None
    while idx is None:
        sample_key = []
        for factor_num, tup in enumerate(class_balancing_proportions):
            # tup looks like (factor, dict or list)
            factor = tup[0]
            specs = tup[1]
            if factor == 'pathology':
                sampled_path = np.random.choice(list(specs.keys()))  # nested dict
                path_label = np.random.choice(list(specs[sampled_path].keys()),
                                              p=list(specs[sampled_path].values()))
                factor_val = f'{sampled_path} -- {path_label}'
            else:
                if isinstance(specs, list):
                    factor_val = np.random.choice(specs)
                else:
                    factor_val = np.random.choice(list(specs.keys()), p=list(specs.values()))
            sample_key.append(factor_val)

        sample_key = tuple(sample_key)
        if len(idxs_per_sample_key[sample_key]):
            idx = np.random.choice(idxs_per_sample_key[sample_key])
    return idx


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown - fix pandas compatibility
        if 'view' in self.csv.columns:
            self.csv['view'].fillna("UNKNOWN", inplace=True)
            if "*" not in views:
                self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view
        else:
            print(f"Warning: 'view' column not found in CSV. Skipping view filtering for {views}")
            # Don't filter by view if column doesn't exist


class MergeDataset(Dataset):
    def __init__(self, datasets, seed=0, label_concat=False):
        super(MergeDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")

        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")

        self.which_dataset = self.which_dataset.astype(int)

        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1] * len(datasets)]) * np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i, shift * size:shift * size + size] = self.labels[i]
            self.labels = new_labels

        try:
            self.csv = pd.concat([d.csv for d in datasets])
        except:
            print("Could not merge dataframes (.csv not available):", sys.exc_info()[0])

        self.csv = self.csv.reset_index(drop=True)

    def __setattr__(self, name, value):
        if name == "transform":
            raise NotImplementedError("Cannot set transform on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal datasets which could have unexpected side effects")
        if name == "data_aug":
            raise NotImplementedError("Cannot set data_aug on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal datasets which could have unexpected side effects")

        object.__setattr__(self, name, value)

    def string(self):
        s = self.__class__.__name__ + " num_samples={}\n".format(len(self))
        for i, d in enumerate(self.datasets):
            if i < len(self.datasets) - 1:
                s += "├{} ".format(i) + d.string().replace("\n", "\n|  ") + "\n"
            else:
                s += "└{} ".format(i) + d.string().replace("\n", "\n   ") + "\n"
        return s

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx - int(self.offset[idx])]
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]
        return item


# alias so it is backwards compatible
Merge_Dataset = MergeDataset


class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)

                self.idxs += list(np.where(dataset.labels[:, list(dataset.pathologies).index(label)] == 1)[0])

        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]

    def __setattr__(self, name, value):
        if name == "transform":
            raise NotImplementedError("Cannot set transform on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects")
        if name == "data_aug":
            raise NotImplementedError("Cannot set data_aug on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects")

        object.__setattr__(self, name, value)

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class NIH_Dataset(Dataset):
    """NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz"),
                 bbox_list_path=os.path.join(datapath, "BBox_List_2017.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False
                 ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(bbox_list_path,
                                              names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                                              skiprows=1)

        # change label name to match
        self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        self.csv['age_years'] = self.csv['Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Sex'] == 'Male'
        self.csv['sex_female'] = self.csv['Sex'] == 'Female'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        imgid = imgid.split('/', 1)[1]  # Remove the version prefix from Path
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask


class RSNA_Pneumonia_Dataset(Dataset):
    """RSNA Pneumonia Detection Challenge

    Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
    Annotations of Possible Pneumonia.
    Shih, George, Wu, Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica, Galperin-Aizenberg,
    Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs, Stephen, Jeudy, Jean, Laroia, Archana,
    Shah, Palmi N., Vummidi, Dharshan, Yaddanapudi, Kavitha, and Stein, Anouk.
    Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.

    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip"),
                 dicomcsvpath=os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False,
                 extension=".jpg"
                 ):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "Lung Opacity"]

        self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")

        # Load data
        self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()

        self.dicomcsvpath = dicomcsvpath
        self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")

        self.csv = self.csv.join(self.dicomcsv, on="patientId")

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['ViewPosition']
        self.limit_to_selected_views(views)

        self.csv = self.csv.reset_index()

        # Get our classes.
        self.labels = []
        self.labels.append(self.csv["Target"].values)
        self.labels.append(self.csv["Target"].values)  # same labels for both

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # TODO: merge with NIH metadata to get dates for images

        # patientid
        self.csv["patientid"] = self.csv["patientId"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['patientId'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception("Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:

                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask


class NIH_Google_Dataset(Dataset):
    """A relabelling of a subset of images from the NIH dataset.  The data tables should
    be applied against an NIH download.  A test and validation split are provided in the
    original.  They are combined here, but one or the other can be used by providing
    the original csv to the csvpath argument.

    Chest Radiograph Interpretation with Deep Learning Models: Assessment with
    Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
    Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J. Reicher, Scott Mayer
    McKinney, Gavin E. Duggan, Krish Eswaran, Po-Hsuan Cameron Chen, Yun Liu,
    Sreenivasa Raju Kalidindi, Alexander Ding, Greg S. Corrado, Daniel Tse, and
    Shravya Shetty. Radiology 2020

    https://pubs.rsna.org/doi/10.1148/radiol.2019191293

    NIH data can be downloaded here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "google2019_nih-chest-xray-labels.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(NIH_Google_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Fracture", "Pneumothorax", "Airspace opacity",
                            "Nodule or mass"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Airspace opacity", "Lung Opacity")
        self.pathologies = np.char.replace(self.pathologies, "Nodule or mass", "Nodule/Mass")
        self.pathologies = list(self.pathologies)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class PC_Dataset(Dataset):
    """PadChest dataset
    Hospital San Juan de Alicante - University of Alicante

    Note that images with null labels (as opposed to normal), and images that cannot
    be properly loaded (listed as 'missing' in the code) are excluded, which makes
    the total number of available images slightly less than the total number of image
    files.

    PadChest: A large chest x-ray image dataset with multi-label annotated reports.
    Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá.
    arXiv preprint, 2019. https://arxiv.org/abs/1901.07441

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(PC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Fracture",
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis",
                            "Hilar Enlargement", "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation",
                            "Support Devices", "Tube'"]  # the Tube' is intentional

        self.pathologies = sorted(self.pathologies)

        mapping = dict()

        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern",
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Tube'"] = ["stent'"]  # the ' is to select findings which end in that word

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        self.csvpath = csvpath

        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # Standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]), "Projection"] = "AP Supine"

        self.csv["view"] = self.csv['Projection']
        self.limit_to_selected_views(views)

        # Remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]

        # Remove missing files
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
                   "216840111366964012989926673512011074122523403_00-163-058.png",
                   "216840111366964012959786098432011033083840143_00-176-115.png",
                   "216840111366964012558082906712009327122220177_00-102-064.png",
                   "216840111366964012339356563862009072111404053_00-043-192.png",
                   "216840111366964013076187734852011291090445391_00-196-188.png",
                   "216840111366964012373310883942009117084022290_00-064-025.png",
                   "216840111366964012283393834152009033102258826_00-059-087.png",
                   "216840111366964012373310883942009170084120009_00-097-074.png",
                   "216840111366964012819207061112010315104455352_04-024-184.png",
                   "216840111366964012819207061112010306085429121_04-020-102.png"]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        self.pathologies[self.pathologies.index("Tube'")] = "Tube"

        # add consistent csv values

        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(int) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        self.csv['age_years'] = (2017 - self.csv['PatientBirth'])

        # sex
        self.csv['sex_male'] = self.csv['PatientSex_DICOM'] == 'M'
        self.csv['sex_female'] = self.csv['PatientSex_DICOM'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['patientId'].iloc[idx]
        imgid = imgid.split('/', 1)[1]  # Remove the version prefix from Path
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception("Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:

                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask


class CheX_Dataset(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison
    https://arxiv.org/abs/1901.07031

    Dataset is not meant to be used directly, please use the other dataset classes.

    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 views=["PA"],
                 extension=".jpg",
                 use_pydicom=False,
                 **kwargs):

        super(CheX_Dataset, self).__init__()
        
        np.random.seed(seed)
        
        self.views = views

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"
        ]
        self.pathologies = sorted(self.pathologies)
        
        self.extension = extension
        self.use_pydicom = use_pydicom

        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        
        # Normalize image paths stored in CSV: some CSVs include the dataset
        # version directory (e.g. `CheXpert-v1.0-small/...`) while `imgpath`
        # may already point to that directory. Strip the leading component when
        # it duplicates the imgpath basename or starts with 'CheXpert' to avoid
        # duplicating the folder when joining paths later.
        imgpath_basename = os.path.basename(self.imgpath.rstrip(os.sep))
        def _normalize_path(p):
            p = str(p)
            parts = p.split('/', 1)
            if len(parts) > 1 and (parts[0] == imgpath_basename or parts[0].lower().startswith('chexpert')):
                return parts[1]
            return p
        self.csv['Path'] = self.csv['Path'].apply(_normalize_path)

        # Extract PatientID from Path
        self.csv['PatientID'] = self.csv['Path'].str.split('/').str[2]
        
        if 'patientId' not in self.csv.columns and 'Patient' in self.csv.columns:
            self.csv['patientId'] = self.csv['Patient']

        # Remove images with view position other than specified
        if type(views) is list:
            self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
            self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]
            self.csv = self.csv[self.csv["view"].isin(views)]
        else:
            #if all views are kept, we still need to create the view column
            self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
            self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
            else:
                # in case a pathology is not in this dataset
                mask = np.zeros(len(self.csv))
            
            # handle uncertain and unmentioned cases
            mask[mask == -1] = 0
            mask[np.isnan(mask)] = 0
            
            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        self.csv['age_years'] = self.csv['Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Sex'] == 'Male'
        self.csv['sex_female'] = self.csv['Sex'] == 'Female'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset (modeled after CheX_Dataset, with windowing support)
    """
    def __init__(self,
                 imgpath,
                 csvpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True,
                 min_window_width=None,
                 labels_to_use=None,
                 use_class_balancing=False,
                 use_no_finding=False,
                 class_balancing_proportions=None,
                 class_balancing_labels_df=None,
                 window_nbins=None
                 ):
        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.views = views
        self.min_window_width = min_window_width
        self.use_class_balancing = use_class_balancing
        self.class_balancing_proportions = class_balancing_proportions
        self.flat_dir = flat_dir
        self.spline_window = None
        if window_nbins is not None:
            self.spline_window = SplineWindowingFunction(n_bins=window_nbins)

        # Load CSV
        self.csv = pd.read_csv(self.csvpath)
        
        # If dicom_id is missing, we need to merge with metadata CSV to get it
        if 'dicom_id' not in self.csv.columns:
            # Try to find and merge with metadata CSV
            base_path = os.path.dirname(self.csvpath)
            metadata_csv_path = os.path.join(base_path, "mimic-cxr-2.0.0-metadata.csv")
            
            if os.path.exists(metadata_csv_path):
                print(f"  🔧 Loading metadata CSV to get dicom_id: {metadata_csv_path}")
                metadata_df = pd.read_csv(metadata_csv_path)
                
                # Merge on subject_id and study_id to get dicom_id
                if 'subject_id' in metadata_df.columns and 'study_id' in metadata_df.columns and 'dicom_id' in metadata_df.columns:
                    self.csv = pd.merge(self.csv, metadata_df[['subject_id', 'study_id', 'dicom_id', 'ViewPosition']], 
                                       on=['subject_id', 'study_id'], how='inner')
                    print(f"  ✅ Merged with metadata: {len(self.csv)} samples after merge")
                    
                    # Set view column from ViewPosition
                    if 'ViewPosition' in self.csv.columns:
                        self.csv['view'] = self.csv['ViewPosition']
                        print("  ✅ Added view column from ViewPosition")
                else:
                    print("  ❌ Metadata CSV missing required columns")
            else:
                print(f"  ❌ Metadata CSV not found at {metadata_csv_path}")
        
        # Set view column if needed (adjust as per your CSV structure)
        if 'view' not in self.csv.columns:
            if 'ViewPosition' in self.csv.columns:
                self.csv['view'] = self.csv['ViewPosition']
            elif 'Frontal/Lateral' in self.csv.columns:
                self.csv['view'] = self.csv['Frontal/Lateral']
        self.limit_to_selected_views(views)
        if unique_patients and 'subject_id' in self.csv.columns:
            self.csv = self.csv.groupby('subject_id').first().reset_index()

        # Pathologies (adjust as needed)
        self.pathologies = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]
        if use_no_finding:
            self.pathologies.append("No Finding")
        self.pathologies = sorted(self.pathologies)

        # Labels (adjust as needed)
        healthy = self.csv["No Finding"] == 1 if "No Finding" in self.csv.columns else None
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology == "No Finding" and healthy is not None:
                    for idx, row in self.csv.iterrows():
                        if pd.isna(row['No Finding']):
                            if (row[6:18] == 1).sum():
                                self.csv.loc[idx, 'No Finding'] = 0
                elif pathology != "Support Devices" and healthy is not None:
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                self.labels.append(mask.values)
            else:
                self.labels.append(np.zeros(len(self.csv)))
        self.labels = np.asarray(self.labels).T.astype(np.float32)
        if "No Finding" in self.csv.columns:
            self.labels[self.labels == -1] = np.nan

        # Patient ID
        if 'subject_id' in self.csv.columns:
            self.csv['patientid'] = self.csv['subject_id'].astype(str)
        elif 'Path' in self.csv.columns:
            patientid = self.csv.Path.str.extract(r'(patient\\d+)')
            self.csv['patientid'] = patientid

    def string(self):
        return self.__class__.__name__ + f" num_samples={len(self)} views={self.views} data_aug={self.data_aug}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        # Image path logic (adjust as needed for your MIMIC file structure)
        row = self.csv.iloc[idx]
        if 'subject_id' in row and 'study_id' in row and 'dicom_id' in row:
            subjectid = str(row["subject_id"])
            studyid = str(row["study_id"])
            dicom_id = str(row["dicom_id"])
            img_path = os.path.join(self.imgpath, f"p{subjectid[:2]}", f"p{subjectid}", f"s{studyid}", f"{dicom_id}.jpg")
        elif 'Path' in row:
            img_path = os.path.join(self.imgpath, row['Path'])
        else:
            raise RuntimeError("Cannot determine image path for MIMIC sample.")
        img = imread(img_path)
        # Windowing
        if self.min_window_width:
            img = apply_random_window_width(img, self.min_window_width, max_width=256)
        if self.spline_window is not None:
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0) if img.ndim == 2 else torch.from_numpy(img)
            img = self.spline_window(img).numpy()
            img = img * 255.0
        sample["img"] = normalize(img, maxval=255, reshape=True)
        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)
        return sample


def apply_transforms(sample, transforms_obj):
    """Apply transforms to a sample dict.

    - If transforms_obj is None: return sample unchanged.
    - If it's a list/tuple, apply each in sequence.
    - If it's callable, first try calling it with the whole sample (some custom
      transforms expect a dict and return a dict). If that fails, try calling
      it with just the image and assign the returned image back to sample['img'].
    """
    if transforms_obj is None:
        return sample

    # If a sequence of transforms, apply sequentially
    if isinstance(transforms_obj, (list, tuple)):
        for t in transforms_obj:
            sample = apply_transforms(sample, t)
        return sample

    # If callable, try applying to whole sample, then fallback to image-only
    if callable(transforms_obj):
        try:
            out = transforms_obj(sample)
            if isinstance(out, dict):
                return out
            # otherwise assume it returned an image
            sample["img"] = out
            return sample
        except Exception:
            # fallback: try applying to image only
            img = sample.get("img")
            if img is None:
                raise
            out_img = transforms_obj(img)
            sample["img"] = out_img
            return sample

    # If we get here, transforms_obj type is unsupported — return sample unchanged
    return sample


class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return self.to_pil(x[0])


class XRayResizer(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        # if 'cv2' in sys.modules:
        #     print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


class XRayCenterCrop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class CovariateDataset(Dataset):
    """Dataset which will correlate the dataset with a specific label.

    Viviano et al. Saliency is a Possible Red Herring When Diagnosing Poor Generalization
    https://arxiv.org/abs/1910.00199
    """

    def __init__(self,
                 d1, d1_target,
                 d2, d2_target,
                 ratio=0.5,
                 mode="train",
                 seed=0,
                 nsamples=None,
                 splits=[0.5, 0.25, 0.25],
                 verbose=False
                 ):
        super(CovariateDataset, self).__init__()

        self.splits = np.array(splits)
        self.d1 = d1
        self.d1_target = d1_target
        self.d2 = d2
        self.d2_target = d2_target

        assert mode in ['train', 'valid', 'test']
        assert np.sum(self.splits) == 1.0

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        all_imageids = np.concatenate([np.arange(len(self.d1)),
                                       np.arange(len(self.d2))]).astype(int)

        all_idx = np.arange(len(all_imageids)).astype(int)

        all_labels = np.concatenate([d1_target,
                                     d2_target]).astype(int)

        all_site = np.concatenate([np.zeros(len(self.d1)),
                                   np.ones(len(self.d2))]).astype(int)

        idx_sick = all_labels == 1
        n_per_category = np.min([sum(idx_sick[all_site == 0]),
                                 sum(idx_sick[all_site == 1]),
                                 sum(~idx_sick[all_site == 0]),
                                 sum(~idx_sick[all_site == 1])])

        all_csv = pd.concat([d1.csv, d2.csv])
        all_csv['site'] = all_site
        all_csv['label'] = all_labels

        if verbose:
            print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site == 0) & (all_labels == 0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site == 0) & (all_labels == 1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site == 1) & (all_labels == 0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site == 1) & (all_labels == 1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * ratio * splits[0] * 2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * ratio * splits[0] * 2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        if verbose:
            print("TRAIN (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                ratio,
                len(train_0_neg) + len(train_1_neg),
                len(train_0_pos) + len(train_1_pos),
                len(train_0_pos),
                len(train_0_neg),
                len(train_1_pos),
                len(train_1_neg)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        if verbose:
            print("VALID (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(valid_0_neg) + len(valid_1_neg),
                len(valid_0_pos) + len(valid_1_pos),
                len(valid_0_pos),
                len(valid_0_neg),
                len(valid_1_pos),
                len(valid_1_neg)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        if verbose:
            print("TEST (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(test_0_neg) + len(test_1_neg),
                len(test_0_pos) + len(test_1_pos),
                len(test_0_pos),
                len(test_0_neg),
                len(test_1_pos),
                len(test_1_neg)))

        def _reduce_nsamples(nsamples, a, b, c, d):
            if nsamples:
                a = a[:int(np.floor(nsamples / 4))]
                b = b[:int(np.ceil(nsamples / 4))]
                c = c[:int(np.ceil(nsamples / 4))]
                d = d[:int(np.floor(nsamples / 4))]

            return (a, b, c, d)

        if mode == "train":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, train_0_neg, train_0_pos, train_1_neg, train_1_pos)
        elif mode == "valid":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, valid_0_neg, valid_0_pos, valid_1_neg, valid_1_pos)
        elif mode == "test":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, test_0_neg, test_0_pos, test_1_neg, test_1_pos)
        else:
            raise Exception("unknown mode")

        self.select_idx = np.concatenate([a, b, c, d])
        self.imageids = all_imageids[self.select_idx]
        self.pathologies = ["Custom"]
        self.labels = all_labels[self.select_idx].reshape(-1, 1)
        self.site = all_site[self.select_idx]
        self.csv = all_csv.iloc[self.select_idx]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.d1
        else:
            dataset = self.d2

        sample = dataset[self.imageids[idx]]

        #

        sample["site"] = self.site[idx]

        return sample
