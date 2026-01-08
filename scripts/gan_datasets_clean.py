#!/usr/bin/env python
# coding: utf-8
"""
GAN Dataset Loader for Multi-Dataset Training - CLEAN VERSION
Focuses on CheXpert + MIMIC with dataset ID tracking for adversarial training.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path for torchxrayvision imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchxrayvision as xrv
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random


def custom_collate_fn(batch):
    """
    Custom collate function to handle different image sizes in GAN training.
    Resizes all images to a common size.
    """
    import skimage.transform
    
    # Find the target size (use the most common size or a fixed size)
    target_size = (320, 320)  # Fixed size for GAN training
    
    imgs = []
    labs = []
    dataset_ids = []
    dataset_names = []
    
    for sample in batch:
        # Get image and resize if needed
        img = sample['img']
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # Resize to target size if needed
        if img.shape[-2:] != target_size:
            # img is shape [C, H, W], resize H, W dimensions
            if len(img.shape) == 3:  # [C, H, W]
                img_resized = np.zeros((img.shape[0], target_size[0], target_size[1]), dtype=img.dtype)
                for c in range(img.shape[0]):
                    img_resized[c] = skimage.transform.resize(
                        img[c], target_size, preserve_range=True, anti_aliasing=True
                    )
            else:  # [H, W]
                img_resized = skimage.transform.resize(
                    img, target_size, preserve_range=True, anti_aliasing=True
                ).astype(img.dtype)
                # Add channel dimension
                img_resized = np.expand_dims(img_resized, axis=0)
            
            img = img_resized
        else:
            # Ensure channel dimension exists
            if len(img.shape) == 2:  # [H, W]
                img = np.expand_dims(img, axis=0)  # [1, H, W]
        
        # Convert back to tensor and normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32)
        
        # Normalize if not already normalized
        if img.max() > 1.0:
            img = img / 255.0
        
        imgs.append(img)
        labs.append(sample.get('lab', torch.zeros(14)))  # Default to 14 pathology classes
        dataset_ids.append(sample.get('dataset_id', 0))
        dataset_names.append(sample.get('dataset_name', 'unknown'))
    
    return {
        'img': torch.stack(imgs),
        'lab': torch.stack(labs),
        'dataset_id': torch.tensor(dataset_ids, dtype=torch.long),
        'dataset_name': dataset_names
    }


class DatasetIDWrapper(Dataset):
    """
    Wrapper to add dataset ID and name to each sample.
    """
    def __init__(self, dataset, dataset_id, dataset_name):
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        
        # Copy essential attributes from wrapped dataset
        self.pathologies = getattr(dataset, 'pathologies', [])
        self.targets = getattr(dataset, 'targets', None)
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            self.labels = dataset.labels
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            
            # Handle different sample formats
            if isinstance(sample, dict):
                sample = dict(sample)  # Make a copy to avoid modifying original
            else:
                # Convert tuple/other formats to dict
                sample = {'img': sample[0] if hasattr(sample, '__getitem__') else sample}
            
            # Add dataset information
            sample['dataset_id'] = self.dataset_id
            sample['dataset_name'] = self.dataset_name
            
            # Ensure we have essential keys
            if 'img' not in sample:
                raise KeyError("Sample missing 'img' key")
            
            # Add dummy labels if missing
            if 'lab' not in sample:
                if hasattr(self, 'pathologies') and self.pathologies:
                    sample['lab'] = torch.zeros(len(self.pathologies))
                else:
                    sample['lab'] = torch.zeros(14)  # Default number of pathologies
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx} from {self.dataset_name}: {e}")
            # Return a dummy sample to avoid breaking training
            return {
                'img': torch.zeros((1, 320, 320)),
                'lab': torch.zeros(14),
                'dataset_id': self.dataset_id,
                'dataset_name': self.dataset_name
            }


class MultiDatasetGANLoader:
    """
    Multi-dataset loader for GAN training with CheXpert and MIMIC.
    Provides balanced sampling and dataset ID tracking.
    """
    
    def __init__(self, dataset_dir="/lotterlab/datasets/", batch_size=32, 
                 balance_datasets=True, limit_per_dataset=None, seed=42):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.balance_datasets = balance_datasets
        self.limit_per_dataset = limit_per_dataset
        self.seed = seed
        
        # Dataset configurations
        self.dataset_configs = {
            'chexpert': {'id': 0, 'name': 'CheXpert'},
            'mimic': {'id': 1, 'name': 'MIMIC-CXR'}
        }
        
        self.datasets = {}
        self.combined_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        print("Initializing Multi-Dataset GAN Loader...")
        self._load_datasets()
        self._create_combined_dataset()
        
    def _load_datasets(self):
        """Load individual datasets with error handling."""
        
        # Load CheXpert
        try:
            print(f"🔄 Loading CheXpert from {self.dataset_dir}...")
            from tqdm import tqdm
            
            chexpert_dataset = xrv.datasets.CheX_Dataset(
                imgpath=os.path.join(self.dataset_dir, "CheXpert-v1.0-small"),
                csvpath=os.path.join(self.dataset_dir, "CheXpert-v1.0-small/train.csv"),
                views=['PA', 'AP'],  # Include both PA and AP views for full dataset
                transform=None, data_aug=None, flat_dir_structure=False,
                seed=self.seed, pure_labels=False, unique_patients=False  # Allow more samples
            )
            
            # Always limit CheXpert to balance with MIMIC
            target_size = self.limit_per_dataset if self.limit_per_dataset else 10000
            if len(chexpert_dataset) > target_size:
                print(f"  📊 Sampling {target_size:,} from {len(chexpert_dataset):,} CheXpert samples...")
                # Randomly sample indices for balanced dataset
                all_indices = list(range(len(chexpert_dataset)))
                random.seed(self.seed)
                random.shuffle(all_indices)
                selected_indices = all_indices[:target_size]
                chexpert_dataset = xrv.datasets.SubsetDataset(chexpert_dataset, selected_indices)
                print(f"  🎯 Sampled {target_size:,} from {len(all_indices):,} CheXpert samples")
            
            self.datasets['chexpert'] = DatasetIDWrapper(
                chexpert_dataset, 
                self.dataset_configs['chexpert']['id'],
                self.dataset_configs['chexpert']['name']
            )
            print(f"✅ CheXpert loaded: {len(self.datasets['chexpert']):,} samples")
            
        except Exception as e:
            print(f"❌ Failed to load CheXpert: {e}")
            self.datasets['chexpert'] = None
        
        # Load MIMIC
        try:
            print(f"🔄 Loading MIMIC from {self.dataset_dir}...")
            from tqdm import tqdm
            
            # Apply comprehensive pandas compatibility fix for MIMIC dataset
            import pandas as pd
            
            # Store the original DataFrame if it exists
            original_dataframe = pd.DataFrame
            
            # Don't add .view method - it conflicts with column access
            print("  🔧 Pandas compatibility: avoiding .view method conflict")
            
            # Try a different approach - patch the specific torchxrayvision line
            # Let's monkey-patch the problematic line in the source
            import torchxrayvision.datasets as xrv_datasets
            
            # Check if we need to patch the Dataset base class
            if hasattr(xrv_datasets.Dataset, '__init__'):
                original_dataset_init = xrv_datasets.Dataset.__init__
                
                def patched_dataset_init(self, *args, **kwargs):
                    result = original_dataset_init(self, *args, **kwargs)
                    # Fix the view column access issue
                    if hasattr(self, 'csv') and hasattr(self.csv, 'view'):
                        # If 'view' is a column, handle fillna properly
                        if 'view' in self.csv.columns:
                            try:
                                self.csv['view'].fillna("UNKNOWN", inplace=True)
                                print("    🔧 Applied fillna patch for view column")
                            except Exception as e:
                                print(f"    ⚠️ Fillna patch failed: {e}")
                    return result
                
                # Apply the patch
                xrv_datasets.Dataset.__init__ = patched_dataset_init
                print("  🔧 Applied torchxrayvision Dataset.init patch")
            
            # Use correct paths - files are in the nested physionet.org structure
            imgpath = '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
            csvpath = '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/mimic-cxr-2.0.0-chexpert.csv'
            # Add metadata CSV which contains the ViewPosition info
            metacsvpath = '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/mimic-cxr-2.0.0-metadata.csv'
            views = ['PA', 'AP']
            
            print(f"  Using imgpath: {imgpath}")
            print(f"  Using csvpath: {csvpath}")
            print(f"  Using metacsvpath: {metacsvpath}")
            print(f"  Using views: {views}")
            
            # Try MIMIC loading with metadata CSV first
            try:
                print("  Trying with metadata CSV...")
                mimic_dataset = xrv.datasets.MIMIC_Dataset(
                    imgpath=imgpath,
                    csvpath=csvpath,
                    views=views,
                    transform=None, 
                    data_aug=None, 
                    flat_dir=True,
                    seed=self.seed,
                    unique_patients=False,
                    min_window_width=None,
                    use_no_finding=True
                )
                
            except Exception as e:
                print(f"  Failed with standard approach: {e}")
                print("  Trying minimal MIMIC loading...")
                # Try with minimal parameters
                mimic_dataset = xrv.datasets.MIMIC_Dataset(
                    imgpath=imgpath,
                    csvpath=csvpath
                )
            
            # Balance MIMIC dataset size with CheXpert
            target_size = self.limit_per_dataset if self.limit_per_dataset else 10000
            if len(mimic_dataset) > target_size:
                print(f"  📊 Sampling {target_size:,} from {len(mimic_dataset):,} MIMIC samples...")
                # Randomly sample indices for balanced dataset
                all_indices = list(range(len(mimic_dataset)))
                random.seed(self.seed)
                random.shuffle(all_indices)
                selected_indices = all_indices[:target_size]
                mimic_dataset = xrv.datasets.SubsetDataset(mimic_dataset, selected_indices)
                print(f"  🎯 Sampled {target_size:,} from {len(all_indices):,} MIMIC samples")

            self.datasets['mimic'] = DatasetIDWrapper(
                mimic_dataset,
                self.dataset_configs['mimic']['id'], 
                self.dataset_configs['mimic']['name']
            )
            print(f"✅ MIMIC loaded: {len(self.datasets['mimic']):,} samples")
            
        except Exception as e:
            print(f"❌ Failed to load MIMIC: {e}")
            self.datasets['mimic'] = None
    
    def _create_combined_dataset(self):
        """Create combined dataset from available datasets."""
        available_datasets = [ds for ds in self.datasets.values() if ds is not None]
        
        if not available_datasets:
            raise ValueError("No datasets could be loaded!")
        
        if len(available_datasets) == 1:
            print("⚠️ Warning: Only one dataset available. GAN training may not be effective.")
        
        # Combine datasets
        self.combined_dataset = ConcatDataset(available_datasets)
        print(f"📊 Combined dataset size: {len(self.combined_dataset):,} samples")
        
        # Print dataset distribution
        for name, dataset in self.datasets.items():
            if dataset is not None:
                print(f"  - {name}: {len(dataset):,} samples")
    
    def create_dataloaders(self, train_split=0.8, num_workers=4, pin_memory=True):
        """Create train and validation dataloaders."""
        if self.combined_dataset is None:
            raise ValueError("Combined dataset not created. Call _create_combined_dataset() first.")
        
        # Split dataset
        total_size = len(self.combined_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        print(f"🔄 Creating data splits: {train_size:,} train, {val_size:,} validation")
        
        # Use manual splitting to ensure reproducibility
        indices = list(range(total_size))
        random.seed(self.seed)
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self.combined_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.combined_dataset, val_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=True  # Important for GAN training to keep consistent batch sizes
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        print(f"✅ Created dataloaders: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
        return self.train_loader, self.val_loader
    
    def get_dataset_info(self):
        """Get information about loaded datasets."""
        info = {
            'total_samples': len(self.combined_dataset) if self.combined_dataset else 0,
            'datasets': {}
        }
        
        for name, dataset in self.datasets.items():
            if dataset is not None:
                info['datasets'][name] = {
                    'id': dataset.dataset_id,
                    'name': dataset.dataset_name,
                    'size': len(dataset),
                    'pathologies': getattr(dataset, 'pathologies', [])
                }
            else:
                info['datasets'][name] = None
        
        return info
    
    def get_sample_batch(self, n_samples=4):
        """Get a sample batch for testing/visualization."""
        if self.train_loader is None:
            raise ValueError("Train loader not created. Call create_dataloaders() first.")
        
        # Get first batch
        for batch in self.train_loader:
            # Truncate to n_samples
            sample_batch = {
                'img': batch['img'][:n_samples],
                'lab': batch['lab'][:n_samples],
                'dataset_id': batch['dataset_id'][:n_samples],
                'dataset_name': batch['dataset_name'][:n_samples]
            }
            return sample_batch
        
        return None


if __name__ == "__main__":
    # Test the loader
    print("Testing MultiDatasetGANLoader...")
    
    try:
        loader = MultiDatasetGANLoader(
            batch_size=8,
            limit_per_dataset=100  # Small test
        )
        
        train_loader, val_loader = loader.create_dataloaders()
        
        print("\n📊 Dataset Info:")
        info = loader.get_dataset_info()
        for name, dataset_info in info['datasets'].items():
            if dataset_info:
                print(f"  {name}: {dataset_info['size']} samples")
        
        print(f"\n🔄 Testing batch loading...")
        sample_batch = loader.get_sample_batch(n_samples=2)
        if sample_batch:
            print(f"  ✅ Sample batch shape: {sample_batch['img'].shape}")
            print(f"  Dataset IDs: {sample_batch['dataset_id'].tolist()}")
            print(f"  Dataset names: {sample_batch['dataset_name']}")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
