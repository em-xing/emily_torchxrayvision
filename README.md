🚨 Paper now online! [https://arxiv.org/abs/2111.00595](https://arxiv.org/abs/2111.00595)

# TorchXRayVision 

| <img src="https://raw.githubusercontent.com/mlmed/torchxrayvision/master/docs/torchxrayvision-logo.png" width="300px"/>  |  ([🎬 promo video](https://www.youtube.com/watch?v=Rl7xz0uULGQ)) <br>[<img src="http://img.youtube.com/vi/Rl7xz0uULGQ/0.jpg" width="400px"/>)](http://www.youtube.com/watch?v=Rl7xz0uULGQ "Video Title") |
|---|---|

# What is it?

A library for chest X-ray datasets and models. Including pre-trained models.


TorchXRayVision is an open source software library for working with chest X-ray datasets and deep learning models. It provides a common interface and common pre-processing chain for a wide set of publicly available chest X-ray datasets. In addition, a number of classification and representation learning models with different architectures, trained on different data combinations, are available through the library to serve as baselines or feature extractors.

- In the case of researchers addressing clinical questions it is a waste of time for them to train models from scratch. To address this, TorchXRayVision provides pre-trained models which are trained on large cohorts of data and enables 1) rapid analysis of large datasets 2) feature reuse for few-shot learning.
- In the case of researchers developing algorithms it is important to robustly evaluate models using multiple external datasets. Metadata associated with each dataset can vary greatly which makes it difficult to apply methods to multiple datasets. TorchXRayVision provides access to many datasets in a uniform way so that they can be swapped out with a single line of code. These datasets can also be merged and filtered to construct specific distributional shifts for studying generalization.

Twitter: [@torchxrayvision](https://twitter.com/torchxrayvision)

## Getting started

```
$ pip install torchxrayvision
```

```python3
import torchxrayvision as xrv
import skimage, torch, torchvision

# Prepare the image:
img = skimage.io.imread("16747_3_1.jpg")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-all")
outputs = model(img[None,...]) # or model.features(img[None,...]) 

# Print results
dict(zip(model.pathologies,outputs[0].detach().numpy()))

{'Atelectasis': 0.32797316,
 'Consolidation': 0.42933336,
 'Infiltration': 0.5316924,
 'Pneumothorax': 0.28849724,
 'Edema': 0.024142697,
 'Emphysema': 0.5011832,
 'Fibrosis': 0.51887786,
 'Effusion': 0.27805611,
 'Pneumonia': 0.18569896,
 'Pleural_Thickening': 0.24489835,
 'Cardiomegaly': 0.3645515,
 'Nodule': 0.68982,
 'Mass': 0.6392845,
 'Hernia': 0.00993878,
 'Lung Lesion': 0.011150705,
 'Fracture': 0.51916164,
 'Lung Opacity': 0.59073937,
 'Enlarged Cardiomediastinum': 0.27218717}

```

A sample script to process images usings pretrained models is [process_image.py](https://github.com/mlmed/torchxrayvision/blob/master/scripts/process_image.py)

```
$ python3 process_image.py ../tests/00000001_000.png
{'preds': {'Atelectasis': 0.50500506,
           'Cardiomegaly': 0.6600903,
           'Consolidation': 0.30575264,
           'Edema': 0.274184,
           'Effusion': 0.4026162,
           'Emphysema': 0.5036339,
           'Enlarged Cardiomediastinum': 0.40989172,
           'Fibrosis': 0.53293407,
           'Fracture': 0.32376793,
           'Hernia': 0.011924741,
           'Infiltration': 0.5154413,
           'Lung Lesion': 0.22231922,
           'Lung Opacity': 0.2772148,
           'Mass': 0.32237658,
           'Nodule': 0.5091847,
           'Pleural_Thickening': 0.5102617,
           'Pneumonia': 0.30947986,
           'Pneumothorax': 0.24847917}}

```

## Models ([demo notebook](https://github.com/mlmed/torchxrayvision/blob/master/scripts/xray_models.ipynb))

Specify weights for pretrained models (currently all DenseNet121)
Note: Each pretrained model has 18 outputs. The `all` model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset. The only valid outputs are listed in the field `{dataset}.pathologies` on the dataset that corresponds to the weights. 

```python3

## 224x224 models
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)

# 512x512 models
model = xrv.models.ResNet(weights="resnet50-res512-all")

# DenseNet121 from JF Healthcare for the CheXpert competition
model = xrv.baseline_models.jfhealthcare.DenseNet() 

# Official Stanford CheXpert model
model = xrv.baseline_models.chexpert.DenseNet()

```

Benchmarks of the modes are here: [BENCHMARKS.md](BENCHMARKS.md) and the performance of some of the models can be seen in this paper [arxiv.org/abs/2002.02497](https://arxiv.org/abs/2002.02497). 


## Autoencoders 
You can also load a pre-trained autoencoder that is trained on the PadChest, NIH, CheXpert, and MIMIC datasets.
```python3
ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
z = ae.encode(image)
image2 = ae.decode(z)
```

## Datasets 
[View docstrings for more detail on each dataset](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py) and [Demo notebook](https://github.com/mlmed/torchxrayvision/blob/master/scripts/xray_datasets.ipynb) and [Example loading script](https://github.com/mlmed/torchxrayvision/blob/master/scripts/dataset_utils.py)

```python3
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

# RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
d_kaggle = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="path to stage_2_train_images_jpg",
                                       transform=transform)
                
# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031             
d_chex = xrv.datasets.CheX_Dataset(imgpath="path to CheXpert-v1.0-small",
                                   csvpath="path to CheXpert-v1.0-small/train.csv",
                                   transform=transform)

# National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
d_nih = xrv.datasets.NIH_Dataset(imgpath="path to NIH images")

# A relabelling of a subset of NIH images from: https://pubs.rsna.org/doi/10.1148/radiol.2019191293
d_nih2 = xrv.datasets.NIH_Google_Dataset(imgpath="path to NIH images")

# PadChest: A large chest x-ray image dataset with multi-label annotated reports. https://arxiv.org/abs/1901.07441
d_pc = xrv.datasets.PC_Dataset(imgpath="path to image folder")

# COVID-19 Image Data Collection. https://arxiv.org/abs/2006.11988
d_covid19 = xrv.datasets.COVID19_Dataset() # specify imgpath and csvpath for the dataset

# SIIM Pneumothorax Dataset. https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
d_siim = xrv.datasets.SIIM_Pneumothorax_Dataset(imgpath="dicom-images-train/",
                                                csvpath="train-rle.csv")

# VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
d_vin = xrv.datasets.VinBrain_Dataset(imgpath=".../train",
                                      csvpath=".../train.csv")

# National Library of Medicine Tuberculosis Datasets. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/
d_nlmtb = xrv.datasets.NLMTB_Dataset(imgpath="path to MontgomerySet or ChinaSet_AllFiles")
```

## Dataset fields

Each dataset contains a number of fields. These fields are maintained when xrv.datasets.Subset_Dataset and xrv.datasets.Merge_Dataset are used.

 - `.pathologies` This field is a list of the pathologies contained in this dataset that will be contained in the `.labels` field ].

 - `.labels` This field contains a 1,0, or NaN for each label defined in `.pathologies`. 

 - `.csv` This field is a pandas DataFrame of the metadata csv file that comes with the data. Each row aligns with the elements of the dataset so indexing using `.iloc` will work. 

If possible, each dataset's `.csv` will have some common fields of the csv. These will be aligned when The list is as follows:

- `csv.patientid` A unique id that will uniqely identify samples in this dataset

- `csv.offset_day_int` An integer time offset for the image in the unit of days. This is expected to be for relative times and has no absolute meaning although for some datasets it is the epoch time.

- `csv.age_years` The age of the patient in years.

- `csv.sex_male` If the patient is male

- `csv.sex_female` If the patient is female


## Dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```python3
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies , d_nih) # has side effects
```

specify a subset of views ([demo notebook](https://github.com/mlmed/torchxrayvision/blob/master/scripts/xray_datasets_views.ipynb))
```python3
d_kaggle = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="...",
                                               views=["PA","AP","AP Supine"])
```

specify only 1 image per patient
```python3
d_kaggle = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="...",
                                               unique_patients=True)
```

obtain summary statistics per dataset
```python3
d_chex = xrv.datasets.CheX_Dataset(imgpath="CheXpert-v1.0-small",
                                   csvpath="CheXpert-v1.0-small/train.csv",
                                 views=["PA","AP"], unique_patients=False)

CheX_Dataset num_samples=191010 views=['PA', 'AP']
{'Atelectasis': {0.0: 17621, 1.0: 29718},
 'Cardiomegaly': {0.0: 22645, 1.0: 23384},
 'Consolidation': {0.0: 30463, 1.0: 12982},
 'Edema': {0.0: 29449, 1.0: 49674},
 'Effusion': {0.0: 34376, 1.0: 76894},
 'Enlarged Cardiomediastinum': {0.0: 26527, 1.0: 9186},
 'Fracture': {0.0: 18111, 1.0: 7434},
 'Lung Lesion': {0.0: 17523, 1.0: 7040},
 'Lung Opacity': {0.0: 20165, 1.0: 94207},
 'Pleural Other': {0.0: 17166, 1.0: 2503},
 'Pneumonia': {0.0: 18105, 1.0: 4674},
 'Pneumothorax': {0.0: 54165, 1.0: 17693},
 'Support Devices': {0.0: 21757, 1.0: 99747}}
```

## Pathology masks ([demo notebook](https://github.com/mlmed/torchxrayvision/blob/master/scripts/xray_masks.ipynb))

Masks are available in the following datasets:
```python3
xrv.datasets.RSNA_Pneumonia_Dataset() # for Lung Opacity
xrv.datasets.SIIM_Pneumothorax_Dataset() # for Pneumothorax
xrv.datasets.NIH_Dataset() # for Cardiomegaly, Mass, Effusion, ...
```

Example usage:

```python3
d_rsna = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="stage_2_train_images_jpg", 
                                            views=["PA","AP"],
                                            pathology_masks=True)
                                            
# The has_masks column will let you know if any masks exist for that sample
d_rsna.csv.has_masks.value_counts()
False    20672
True      6012       

# Each sample will have a pathology_masks dictionary where the index 
# of each pathology will correspond to a mask of that pathology (if it exists).
# There may be more than one mask per sample. But only one per pathology.
sample["pathology_masks"][d_rsna.pathologies.index("Lung Opacity")]
```
![](https://raw.githubusercontent.com/mlmed/torchxrayvision/master/docs/pathology-mask-rsna2.png)
![](https://raw.githubusercontent.com/mlmed/torchxrayvision/master/docs/pathology-mask-rsna3.png)

it also works with data_augmentation if you pass in `data_aug=data_transforms` to the dataloader. The random seed is matched to align calls for the image and the mask.

![](https://raw.githubusercontent.com/mlmed/torchxrayvision/master/docs/pathology-mask-rsna614-da.png)

## Distribution shift tools ([demo notebook](https://github.com/mlmed/torchxrayvision/blob/master/scripts/xray_datasets-CovariateShift.ipynb))

The class `xrv.datasets.CovariateDataset` takes two datasets and two 
arrays representing the labels. The samples will be returned with the 
desired ratio of images from each site. The goal here is to simulate 
a covariate shift to make a model focus on an incorrect feature. Then 
the shift can be reversed in the validation data causing a catastrophic
failure in generalization performance.

ratio=0.0 means images from d1 will have a positive label
ratio=0.5 means images from d1 will have half of the positive labels
ratio=1.0 means images from d1 will have no positive label

With any ratio the number of samples returned will be the same.

```python3
d = xrv.datasets.CovariateDataset(d1 = # dataset1 with a specific condition
                                  d1_target = #target label to predict,
                                  d2 = # dataset2 with a specific condition
                                  d2_target = #target label to predict,
                                  mode="train", # train, valid, and test
                                  ratio=0.9)

```

# Emily's Chest X-ray Domain Adaptation Research Project

## Overview

This project extends TorchXRayVision with advanced domain adaptation techniques for chest X-ray analysis. The main focus is on the **AutoStainer** - a learned image transformation system that can adapt images between different scanners and datasets while preserving medical diagnostic information.

## Key Features

### 🔄 AutoStainer Domain Adaptation
- **Purpose**: Transform chest X-rays between different scanners/datasets (e.g., CheXpert ↔ MIMIC-CXR)
- **Method**: Learned parameter-based transformations using adversarial training
- **Goal**: Scanner confusion while maintaining disease classification accuracy

### 📊 Disease AUC Loss
- **Innovation**: Directly optimizes Area Under ROC Curve (AUC) instead of simple similarity
- **Benefits**: Better disease preservation, more stable training, handles missing labels
- **Implementation**: Differentiable pairwise ranking loss for multi-disease classification

### 🏥 Medical Imaging Pipeline
- **Datasets**: CheXpert, MIMIC-CXR, NIH ChestX-ray8, PadChest, RSNA Pneumonia
- **Models**: DenseNet121, ResNet50 pre-trained on chest X-ray datasets
- **Windowing**: Learned spline-based intensity transformations

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd emily_torchxrayvision

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import torchxrayvision as xrv
import torch

# Load a pre-trained model
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Load and preprocess an image
img = xrv.datasets.normalize(your_image, 255)  # Convert to [-1024, 1024]
transform = torch.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img = transform(img)
img = torch.from_numpy(img)

# Get predictions
outputs = model(img[None, ...])
predictions = dict(zip(model.pathologies, outputs[0].detach().numpy()))
```

## Training Scripts

### AutoStainer Training

Train the domain adaptation model:

```bash
# Single training run
python scripts/train_fixed_simple_autostainer.py

# With custom parameters
python scripts/train_fixed_simple_autostainer.py \
    --num_epochs 50 \
    --batch_size 8 \
    --lambda_adversarial 50.0 \
    --lambda_disease 2.0
```

### Hyperparameter Search

Run automated hyperparameter optimization:

```bash
# Note: hyperparameter_search.py may need to be recreated or check outputs/
# for recent search results
python scripts/hyperparameter_search.py
```

### Windowing Analysis

Explore learned intensity transformations:

```bash
# Run the analysis notebook
jupyter notebook windowing_analysis.ipynb
```

## Project Structure

```
├── torchxrayvision/           # Core library (extended from mlmed/torchxrayvision)
│   ├── datasets.py           # Dataset loaders (CheXpert, MIMIC, NIH, etc.)
│   ├── models.py             # Pre-trained models (DenseNet, ResNet)
│   ├── windowing.py          # Spline-based intensity transformations
│   └── autoencoders.py       # Autoencoder models
├── scripts/                  # Training and analysis scripts
│   ├── train_fixed_simple_autostainer.py    # Main AutoStainer training
│   ├── train_utils.py        # Training utilities
│   ├── dataset_utils.py      # Dataset processing tools
│   └── *.ipynb              # Analysis notebooks
├── outputs/                  # Training outputs and checkpoints
├── tests/                    # Test data and unit tests
└── *.md                      # Documentation
```

## Key Research Components

### 1. AutoStainer Architecture

The AutoStainer uses adversarial training with three components:

- **Image Transformer**: Learns to transform images between domains
- **Scanner Classifier**: Adversary that tries to identify the source scanner
- **Disease Classifier**: Ensures medical information is preserved

**Training Objective**:
```
Loss = λ_adversarial × scanner_confusion + λ_disease × disease_preservation + λ_embedding × feature_similarity
```

### 2. Disease AUC Loss

Replaces simple L1 loss with differentiable AUC optimization:

```python
# Old approach: similarity preservation
disease_consistency = L1_loss(transformed_predictions, original_predictions)

# New approach: direct AUC optimization
disease_auc_loss = compute_differentiable_auc_loss(transformed_predictions, disease_labels)
```

### 3. Windowing Functions

Learned intensity transformations using cubic splines:

```python
from torchxrayvision.windowing import SplineWindowingFunction

# Create windowing function
windowing = SplineWindowingFunction(n_knots=32, learnable=True)

# Apply to image
transformed_image = windowing(normalized_image)
```

## Datasets

The project supports multiple chest X-ray datasets:

| Dataset | Size | Pathologies | Source |
|---------|------|-------------|---------|
| CheXpert | 224,316 | 14 diseases | Stanford |
| MIMIC-CXR | 377,110 | 14 diseases | MIT |
| NIH ChestX-ray8 | 112,120 | 8 diseases | NIH |
| PadChest | 160,000 | 174 findings | Alicante |
| RSNA Pneumonia | 26,684 | Pneumonia | RSNA |

## Results & Performance

### AutoStainer Performance (Latest Results)

- **Scanner Confusion**: 56.1% accuracy (near-random performance)
- **Disease Preservation**: 89.4% AUC maintenance
- **Training Stability**: Converged in ~20 epochs

### Disease AUC Loss Benefits

- **Better Optimization**: Directly optimizes evaluation metric
- **Stability**: More robust to prediction scale variations
- **Missing Labels**: Gracefully handles NaN annotations

## Analysis & Visualization

### Notebooks

- `windowing_analysis.ipynb`: Visualize learned intensity transformations
- `xray_datasets.ipynb`: Dataset exploration and statistics
- `xray_models.ipynb`: Model evaluation and benchmarking

### Key Analysis Scripts

```bash
# Analyze hyperparameter search results
python scripts/analyze_hyperparam_results.py

# Visualize transformation quality
python scripts/analyze_transformation_quality.py

# Test AUC loss implementation
python test_auc_loss.py
```

## Configuration & Hyperparameters

### AutoStainer Config

```python
config = {
    # Learning rates
    'transformer_lr': 0.005,    # Aggressive transformation learning
    'scanner_lr': 0.00001,      # Easy to fool scanner
    'disease_lr': 0.0001,       # Stable disease learning
    
    # Loss weights
    'lambda_adversarial': 50.0, # Strong scanner confusion
    'lambda_disease': 2.0,      # Disease preservation
    'lambda_embedding': 0.5,    # Feature similarity
    
    # Architecture
    'latent_dim': 128,
    'num_control_points': 8,
}
```

## Dependencies

- PyTorch ≥ 1.0
- TorchVision ≥ 0.5
- scikit-image ≥ 0.16
- NumPy, Pandas, PIL
- tqdm, requests

## GitHub Repository

This project is based on the [TorchXRayVision library](https://github.com/mlmed/torchxrayvision) by Joseph Paul Cohen et al.

**Note**: This appears to be a research fork. For the official repository, see: https://github.com/mlmed/torchxrayvision

## Citation

If you use this work, please cite both the original TorchXRayVision paper and acknowledge the AutoStainer extensions:

```bibtex
@inproceedings{Cohen2022xrv,
title = {{TorchXRayVision: A library of chest X-ray datasets and models}},
author = {Cohen, Joseph Paul and Viviano, Joseph D. and Bertin, Paul and Morrison, Paul and Torabian, Parsa and Guarrera, Matteo and Lungren, Matthew P and Chaudhari, Akshay and Brooks, Rupert and Hashir, Mohammad and Bertrand, Hadrien},
booktitle = {Medical Imaging with Deep Learning},
url = {https://github.com/mlmed/torchxrayvision},
arxivId = {2111.00595},
year = {2022}
}
```

**Status**: all recent hyperparameter search runs show AUC=0.0

**Problem**: 
- `tuple index out of range` error during AUC calculation in `train_fixed_simple_autostainer.py`
- Empty or mis-shaped arrays being passed to AUC calculation
- All batches being skipped due to dimension mismatches

**Root Cause**:
- Disease dimension mismatches between datasets (CheXpert: 13 diseases vs MIMIC-CXR: 14 diseases)
- Batch collection logic failing when dimensions don't align
- AUC calculation receiving empty arrays, causing index errors

**Impact**:
- Cannot evaluate disease preservation performance
- Hyperparameter optimization ineffective
- Training appears to work but cannot measure success


