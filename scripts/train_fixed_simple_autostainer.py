#!/usr/bin/env python
"""
Fixed Simple AutoStainer that properly handles disease dimension mismatches
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import sys
import json
from sklearn.metrics import roc_auc_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchxrayvision as xrv
from scripts.gan_datasets import MultiDatasetGANLoader

class SplineHyperNetwork(nn.Module):
    """
    Hypernetwork that generates spline control points conditioned on scanner/dataset
    Learns dataset-specific intensity transformations via cubic splines
    """
    def __init__(self, num_datasets=2, num_control_points=8, latent_dim=128):
        super().__init__()
        self.num_datasets = num_datasets
        self.num_control_points = num_control_points
        
        # Dataset embedding
        self.dataset_embedding = nn.Embedding(num_datasets, latent_dim)
        
        # Hypernetwork: dataset_id -> spline control points
        self.hyper_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_control_points * 2)  # (x, y) pairs for control points
        )
        
        # Initialize control points near identity mapping (y = x)
        with torch.no_grad():
            self.hyper_net[-1].weight.fill_(0.0)
            identity_points = torch.linspace(0, 1, num_control_points)
            self.hyper_net[-1].bias.copy_(
                torch.stack([identity_points, identity_points]).t().flatten()
            )
    
    def forward(self, x, dataset_ids):
        """
        x: [B, 1, H, W] images
        dataset_ids: [B] dataset indices
        Returns: transformed images, control points
        """
        batch_size = x.shape[0]
        
        # Get dataset embeddings
        dataset_emb = self.dataset_embedding(dataset_ids)  # [B, latent_dim]
        
        # Generate control points for each sample
        control_points = self.hyper_net(dataset_emb)  # [B, num_control_points * 2]
        control_points = control_points.view(batch_size, self.num_control_points, 2)
        
        # Sort x coordinates and clamp to [0, 1]
        control_points[..., 0], _ = torch.sort(control_points[..., 0], dim=1)
        control_points = torch.clamp(control_points, 0.0, 1.0)
        
        # Apply spline transformation to each image
        transformed = self.apply_spline_batch(x, control_points)
        
        return transformed, control_points
    
    def apply_spline_batch(self, images, control_points):
        """Apply piecewise linear interpolation using control points (fast GPU operation)"""
        batch_size = images.shape[0]
        
        # Use piecewise linear interpolation (differentiable)
        # For each pixel value in [0, 1], find which control point segment it falls into
        # and linearly interpolate
        
        # Get control point coordinates
        cp_x = control_points[..., 0]  # [B, num_cp]
        cp_y = control_points[..., 1]  # [B, num_cp]
        
        # Flatten images for vectorized processing
        img_flat = images.view(batch_size, -1)  # [B, H*W]
        
        # For each pixel value, find the interpolated output
        transformed_flat = torch.zeros_like(img_flat)
        
        for i in range(self.num_control_points - 1):
            # Get current segment
            x1 = cp_x[:, i:i+1]  # [B, 1]
            x2 = cp_x[:, i+1:i+2]  # [B, 1]
            y1 = cp_y[:, i:i+1]  # [B, 1]
            y2 = cp_y[:, i+1:i+2]  # [B, 1]
            
            # Find pixels in this segment
            mask = (img_flat >= x1) & (img_flat < x2)
            
            # Linear interpolation: y = y1 + (y2-y1)/(x2-x1) * (x-x1)
            slope = (y2 - y1) / (x2 - x1 + 1e-8)
            interpolated = y1 + slope * (img_flat - x1)
            
            transformed_flat = torch.where(mask, interpolated, transformed_flat)
        
        # Handle values >= last control point
        mask = img_flat >= cp_x[:, -1:]
        transformed_flat = torch.where(mask, cp_y[:, -1:], transformed_flat)
        
        # Handle values < first control point
        mask = img_flat < cp_x[:, 0:1]
        transformed_flat = torch.where(mask, cp_y[:, 0:1], transformed_flat)
        
        # Reshape back to image shape
        transformed = transformed_flat.view_as(images)
        transformed = torch.clamp(transformed, 0.0, 1.0)
        
        return transformed

class ScannerClassifier(nn.Module):
    """Simple scanner classifier for adversarial training"""
    
    def __init__(self, input_channels=1, num_scanners=2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Increased from 0.3 to make scanner weaker
            nn.Linear(32, num_scanners)
        )
        
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class RealFakeDiscriminator(nn.Module):
    """
    Discriminator to distinguish original vs transformed images
    Ensures transformations look realistic
    """
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)  # Real (1) vs Fake (0)
        )
    
    def forward(self, x):
        return self.model(x)

class DiseaseClassifier(nn.Module):
    """
    Pretrained disease classifier from torchxrayvision (90%+ accuracy)
    Uses DenseNet121 trained on multiple chest X-ray datasets
    """
    def __init__(self, input_channels=1, num_diseases=14):
        super().__init__()
        self.num_diseases = num_diseases
        
        # Load pretrained model from torchxrayvision
        print("📦 Loading pretrained disease classifier (DenseNet121-All)...")
        try:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        except Exception as e:
            print(f"⚠️  Standard loading failed: {e}")
            print("🔄 Trying with weights_only=False...")
            # Monkey-patch torch.load temporarily for compatibility
            import torch
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            torch.load = original_load
        
        self.model.eval()  # Set to eval mode by default for feature extraction
        
        # Freeze the pretrained model (we don't want to retrain it)
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✅ Loaded pretrained model with {len(self.model.pathologies)} pathologies")
        print(f"   Pathologies: {self.model.pathologies}")
        
        # The model expects 224x224 images, we'll handle resizing
        self.target_size = 224
    
    def forward(self, x, return_features=False):
        """
        x: [B, 1, H, W] input images (any size, will be resized to 224x224)
        return_features: if True, also return intermediate features
        """
        batch_size = x.shape[0]
        
        # Resize to 224x224 if needed
        if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
            x = torch.nn.functional.interpolate(x, size=(self.target_size, self.target_size), 
                                               mode='bilinear', align_corners=False)
        
        # The pretrained model expects images in specific range
        # Our images are in [0, 1], need to normalize properly
        # torchxrayvision models expect images normalized with mean=0.5, std=0.25 in [0,1] range
        
        # Get predictions from pretrained model
        predictions = self.model(x)  # [B, num_pathologies]
        
        # Extract features for perceptual loss
        if return_features:
            # Get intermediate features from DenseNet
            # The xrv.models.DenseNet has a 'features' attribute that contains the CNN
            with torch.no_grad():
                features = []
                x_feat = x
                
                # Extract features from different layers
                try:
                    x_feat = self.model.features.conv0(x_feat)
                    x_feat = self.model.features.norm0(x_feat)
                    x_feat = self.model.features.relu0(x_feat)
                    features.append(x_feat)  # Early features
                    
                    x_feat = self.model.features.pool0(x_feat)
                    x_feat = self.model.features.denseblock1(x_feat)
                    features.append(x_feat)  # Block 1
                    
                    x_feat = self.model.features.transition1(x_feat)
                    x_feat = self.model.features.denseblock2(x_feat)
                    features.append(x_feat)  # Block 2
                    
                    x_feat = self.model.features.transition2(x_feat)
                    x_feat = self.model.features.denseblock3(x_feat)
                    features.append(x_feat)  # Block 3
                except AttributeError:
                    # Fallback: just use the predictions as features
                    features = [predictions, predictions, predictions, predictions]
            
            return predictions, features
        
        # For backward compatibility, return predictions and dummy feature
        return predictions, None

class FixedSimpleAutoStainer:
    """Fixed Simple AutoStainer with proper disease dimension handling"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.spline_hypernet = SplineHyperNetwork(
            num_datasets=2,
            num_control_points=config.get('num_control_points', 8),
            latent_dim=config.get('latent_dim', 128)
        ).to(self.device)
        
        self.scanner_classifier = ScannerClassifier(input_channels=1, num_scanners=2).to(self.device)
        self.disease_classifier = DiseaseClassifier(input_channels=1, num_diseases=14).to(self.device)
        self.realfake_discriminator = RealFakeDiscriminator(input_channels=1).to(self.device)
        
        # Initialize optimizers
        self.hypernet_optimizer = optim.Adam(
            self.spline_hypernet.parameters(),
            lr=config['hypernet_lr'],
            betas=(0.5, 0.999)
        )
        
        self.scanner_optimizer = optim.Adam(
            self.scanner_classifier.parameters(),
            lr=config['scanner_lr'],
            betas=(0.5, 0.999)
        )
        
        # NOTE: Disease classifier is pretrained and frozen - no optimizer needed
        self.disease_optimizer = None
        
        self.realfake_optimizer = optim.Adam(
            self.realfake_discriminator.parameters(),
            lr=config['realfake_lr'],
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.scanner_criterion = nn.CrossEntropyLoss()
        self.disease_criterion = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.metrics = {
            'epochs': [],
            'hypernet_loss': [],
            'realfake_loss': [],
            'scanner_accuracy': [],
            'disease_preservation': [],
            'disease_auc_original': [],  # AUC of original images
            'disease_auc_transformed': [],  # AUC of transformed images
            'transform_magnitude': [],
            'control_points': []
        }
    
    def align_disease_tensors(self, predictions, labels):
        """
        Align disease predictions and labels to handle dimension mismatches
        predictions: [batch, 14] - model always predicts 14 diseases
        labels: [batch, N] - where N might be 13 or 14 depending on dataset
        """
        batch_size = predictions.shape[0]
        pred_diseases = predictions.shape[1]  # Should be 14
        label_diseases = labels.shape[1]
        
        if label_diseases == pred_diseases:
            # Perfect match - no alignment needed
            return predictions, labels
        elif label_diseases < pred_diseases:
            # Labels have fewer diseases (e.g., 13 vs 14)
            # Truncate predictions to match labels
            aligned_predictions = predictions[:, :label_diseases]
            return aligned_predictions, labels
        else:
            # Labels have more diseases than predictions (unusual)
            # Pad predictions with zeros
            padding = torch.zeros(batch_size, label_diseases - pred_diseases, 
                                device=predictions.device)
            aligned_predictions = torch.cat([predictions, padding], dim=1)
            return aligned_predictions, labels
    
    def compute_differentiable_auc_loss(self, predictions, labels):
        """
        Compute differentiable AUC loss using pairwise ranking approach
        
        AUC measures ranking quality: positive samples should be ranked higher than negative samples.
        We approximate this with a differentiable pairwise ranking loss.
        
        Args:
            predictions: [N, num_diseases] disease prediction logits
            labels: [N, num_diseases] ground truth labels (0 or 1)
        
        Returns:
            loss: Scalar loss where lower = better AUC
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute AUC loss per disease, then average
        auc_losses = []
        
        for disease_idx in range(labels.shape[1]):
            disease_probs = probs[:, disease_idx]
            disease_labels = labels[:, disease_idx]
            
            # Find positive and negative samples
            pos_mask = (disease_labels == 1) & (~torch.isnan(disease_labels))
            neg_mask = (disease_labels == 0) & (~torch.isnan(disease_labels))
            
            n_pos = pos_mask.sum()
            n_neg = neg_mask.sum()
            
            # Need at least 1 positive and 1 negative to compute AUC
            if n_pos > 0 and n_neg > 0:
                pos_probs = disease_probs[pos_mask]  # [n_pos]
                neg_probs = disease_probs[neg_mask]  # [n_neg]
                
                # Pairwise ranking loss: for each (pos, neg) pair,
                # we want pos_prob > neg_prob
                # Use sigmoid to make it differentiable: sigmoid(pos - neg)
                # Loss = -log(sigmoid(pos - neg)) = log(1 + exp(neg - pos))
                
                # Expand to all pairs: [n_pos, 1] vs [1, n_neg]
                pos_expanded = pos_probs.unsqueeze(1)  # [n_pos, 1]
                neg_expanded = neg_probs.unsqueeze(0)  # [1, n_neg]
                
                # Margin-based ranking loss with smooth approximation
                # We want: pos_prob > neg_prob + margin
                margin = 0.0  # Can adjust if needed
                pairwise_diff = pos_expanded - neg_expanded - margin  # [n_pos, n_neg]
                
                # Smooth approximation: loss = log(1 + exp(-diff))
                # This is numerically stable version of -log(sigmoid(diff))
                pairwise_loss = torch.nn.functional.softplus(-pairwise_diff)
                
                # Average over all pairs
                auc_loss = pairwise_loss.mean()
                auc_losses.append(auc_loss)
        
        # Return average AUC loss across all diseases
        if len(auc_losses) > 0:
            return torch.stack(auc_losses).mean()
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with spline hypernetwork and multi-objective losses"""
        self.spline_hypernet.train()
        self.scanner_classifier.train() 
        self.disease_classifier.train()
        self.realfake_discriminator.train()
        
        epoch_hypernet_loss = 0.0
        epoch_disease_loss = 0.0
        epoch_scanner_loss = 0.0
        epoch_realfake_loss = 0.0
        scanner_correct = 0
        scanner_total = 0
        disease_preservation_scores = []
        transform_magnitudes = []
        control_points_list = []
        
        # For AUC calculation
        all_orig_disease_probs = []
        all_trans_disease_probs = []
        all_disease_labels = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch_size = batch['img'].shape[0]
                images = batch['img'].to(self.device)
                
                # CRITICAL: Normalize images to [0, 1] range
                img_min = images.min()
                img_max = images.max()
                
                if img_max > 1.0 or img_min < 0.0:
                    if batch_idx == 0:  # Only print once per epoch
                        print(f"\n🔧 Normalizing images from [{img_min.item():.3f}, {img_max.item():.3f}] to [0, 1]")
                    
                    # Normalize to [0, 1]
                    if img_max > img_min:
                        images = (images - img_min) / (img_max - img_min)
                    else:
                        images = torch.zeros_like(images)
                
                disease_labels = batch['lab'].to(self.device)
                scanner_labels = batch['dataset_id'].to(self.device)
                
                # =====================================
                # 1. Train Real/Fake Discriminator
                # =====================================
                self.realfake_optimizer.zero_grad()
                
                # Real images
                real_preds = self.realfake_discriminator(images)
                real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
                
                # Fake (transformed) images
                with torch.no_grad():
                    transformed_images, control_points = self.spline_hypernet(images, scanner_labels)
                fake_preds = self.realfake_discriminator(transformed_images)
                fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))
                
                realfake_loss = (real_loss + fake_loss) / 2
                realfake_loss.backward()
                self.realfake_optimizer.step()
                epoch_realfake_loss += realfake_loss.item()
                
                # =====================================
                # 2. Extract Disease Features (No Training - Pretrained & Frozen)
                # =====================================
                # The disease classifier is pretrained and frozen
                # We extract features for use in HyperNet training (perceptual loss)
                with torch.no_grad():
                    disease_logits, orig_features = self.disease_classifier(images, return_features=True)
                    
                    # Align tensors to handle dimension mismatches
                    aligned_disease_logits, aligned_disease_labels = self.align_disease_tensors(
                        disease_logits, disease_labels
                    )
                    
                    # Create mask for valid disease labels
                    disease_mask = ~torch.isnan(aligned_disease_labels)
                
                # =====================================
                # 3. Train Scanner Classifier
                # =====================================
                self.scanner_optimizer.zero_grad()
                
                # On original images
                scanner_logits_real = self.scanner_classifier(images)
                scanner_loss_real = self.scanner_criterion(scanner_logits_real, scanner_labels)
                
                # On transformed images
                with torch.no_grad():
                    transformed_images, control_points = self.spline_hypernet(images, scanner_labels)
                
                scanner_logits_fake = self.scanner_classifier(transformed_images)
                scanner_loss_fake = self.scanner_criterion(scanner_logits_fake, scanner_labels)
                
                scanner_loss = scanner_loss_real + scanner_loss_fake
                scanner_loss.backward()
                self.scanner_optimizer.step()
                epoch_scanner_loss += scanner_loss.item()
                
                # Track scanner accuracy on transformed images
                with torch.no_grad():
                    scanner_preds = torch.argmax(scanner_logits_fake, dim=1)
                    scanner_correct += (scanner_preds == scanner_labels).sum().item()
                    scanner_total += batch_size
                
                # =====================================
                # 4. Train Spline Hypernetwork
                # =====================================
                self.hypernet_optimizer.zero_grad()
                
                # Generate transformations
                transformed_images, control_points = self.spline_hypernet(images, scanner_labels)
                
                # Loss 1: Fool the real/fake discriminator
                fake_preds = self.realfake_discriminator(transformed_images)
                adversarial_realfake_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
                
                # Loss 2: Fool the scanner classifier
                scanner_logits_gen = self.scanner_classifier(transformed_images)
                adversarial_scanner_loss = self.scanner_criterion(scanner_logits_gen, scanner_labels)
                
                # Loss 3: Preserve disease features (multi-scale perceptual loss)
                disease_logits_trans, trans_features = self.disease_classifier(
                    transformed_images, return_features=True
                )
                
                # Perceptual loss across multiple scales
                perceptual_loss = 0
                for orig_feat, trans_feat in zip(orig_features, trans_features):
                    perceptual_loss += self.l1_loss(orig_feat.detach(), trans_feat)
                perceptual_loss = perceptual_loss / len(orig_features)
                
                # Loss 4: Disease AUC loss (optimize for actual prediction accuracy)
                aligned_orig, aligned_labels = self.align_disease_tensors(disease_logits.detach(), disease_labels)
                aligned_trans, _ = self.align_disease_tensors(disease_logits_trans, disease_labels)
                
                if disease_mask.any() and aligned_orig.numel() > 0:
                    # Use differentiable AUC approximation instead of simple L1 consistency
                    disease_auc_loss = self.compute_differentiable_auc_loss(
                        aligned_trans[disease_mask], 
                        aligned_labels[disease_mask]
                    )
                else:
                    disease_auc_loss = torch.tensor(0.0, device=self.device)
                
                # Loss 5: Spline smoothness regularization
                spline_smoothness = torch.mean(torch.abs(control_points[:, 1:, :] - control_points[:, :-1, :]))
                
                # Transformation magnitude (normalized to [0, 1] range)
                transform_magnitude = self.l1_loss(transformed_images, images)
                
                # Sanity check - if transform magnitude is too high, something is wrong
                if transform_magnitude.item() > 1.0:
                    print(f"\n⚠️ WARNING: Transform magnitude = {transform_magnitude.item():.2f}")
                    print(f"   Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                    print(f"   Transformed range: [{transformed_images.min().item():.3f}, {transformed_images.max().item():.3f}]")
                    print(f"   Control points range: [{control_points.min().item():.3f}, {control_points.max().item():.3f}]")
                
                # Combined hypernetwork loss
                lambda_realfake = self.config['lambda_realfake']
                lambda_scanner = self.config['lambda_scanner']
                lambda_perceptual = self.config['lambda_perceptual']
                lambda_disease = self.config['lambda_disease']
                lambda_smooth = self.config['lambda_smooth']
                
                hypernet_loss = (
                    lambda_realfake * adversarial_realfake_loss +
                    lambda_scanner * adversarial_scanner_loss +
                    lambda_perceptual * perceptual_loss +
                    lambda_disease * disease_auc_loss +
                    lambda_smooth * spline_smoothness
                )
                
                hypernet_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.spline_hypernet.parameters(), max_norm=1.0)
                self.hypernet_optimizer.step()
                epoch_hypernet_loss += hypernet_loss.item()
                
                # =====================================
                # 5. Metrics Collection
                # =====================================
                with torch.no_grad():
                    # Disease preservation (correlation between original and transformed predictions)
                    # NOTE: High correlation means predictions are similar, not necessarily accurate!
                    if disease_mask.any() and aligned_orig.numel() > 0:
                        orig_probs = torch.sigmoid(aligned_orig[disease_mask])
                        trans_probs = torch.sigmoid(aligned_trans[disease_mask])
                        
                        if orig_probs.numel() > 1 and trans_probs.numel() > 1:
                            try:
                                # Correlation: measures if predictions move together (high = preserved relationship)
                                correlation_matrix = torch.corrcoef(torch.stack([orig_probs.flatten(), trans_probs.flatten()]))
                                preservation = correlation_matrix[0,1]
                                if not torch.isnan(preservation):
                                    disease_preservation_scores.append(preservation.item())
                            except:
                                # Fallback to L1 similarity
                                similarity = 1.0 - self.l1_loss(orig_probs, trans_probs).item()
                                disease_preservation_scores.append(similarity)
                    
                    # Transform magnitude and control points
                    transform_magnitudes.append(transform_magnitude.item())
                    control_points_list.append(control_points[0].cpu().numpy())
                    
                    # Collect predictions for AUC calculation (per-sample, not masked)
                    # We need to keep the batch structure [batch, num_diseases] for proper AUC calculation
                    if aligned_orig.numel() > 0:
                        # Verify tensors are 2D before storing
                        orig_probs_np = torch.sigmoid(aligned_orig).cpu().numpy()
                        trans_probs_np = torch.sigmoid(aligned_trans).cpu().numpy()
                        labels_np = aligned_disease_labels.cpu().numpy()
                        
                        # Debug: check if we're getting flattened tensors
                        if orig_probs_np.ndim != 2 or labels_np.ndim != 2:
                            if epoch == 1 and batch_idx < 3:
                                print(f"\n⚠️ Batch {batch_idx} has unexpected tensor dimensions:")
                                print(f"   orig_probs: {orig_probs_np.shape}, labels: {labels_np.shape}")
                                print(f"   aligned_orig shape: {aligned_orig.shape}")
                                print(f"   aligned_disease_labels shape: {aligned_disease_labels.shape}")
                        
                        # Only store if tensors are properly shaped (2D: [batch, num_diseases])
                        if orig_probs_np.ndim == 2 and trans_probs_np.ndim == 2 and labels_np.ndim == 2:
                            all_orig_disease_probs.append(orig_probs_np)
                            all_trans_disease_probs.append(trans_probs_np)
                            all_disease_labels.append(labels_np)
                        elif epoch == 1 and batch_idx < 3:
                            print(f"   Skipping batch {batch_idx} due to dimension mismatch")
                
                # Update progress bar
                scanner_acc = scanner_correct / scanner_total if scanner_total > 0 else 1.0
                avg_disease_pres = np.mean(disease_preservation_scores) if disease_preservation_scores else 0.0
                
                # AGGRESSIVE ADAPTIVE LEARNING RATE ADJUSTMENT
                if scanner_acc > 0.60:  # Scanner too strong - BOOST hypernet, WEAKEN scanner
                    for param_group in self.hypernet_optimizer.param_groups:
                        param_group['lr'] = min(param_group['lr'] * 1.5, 0.001)  # Boost more aggressively
                    for param_group in self.scanner_optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.5, 1e-8)  # Cut scanner LR in half
                    if batch_idx % 100 == 0:
                        print(f"\n📉 Scanner too strong ({scanner_acc*100:.1f}%) - Boosting hypernet LR, cutting scanner LR")
                        
                elif scanner_acc < 0.45:  # Scanner too weak - strengthen it
                    for param_group in self.scanner_optimizer.param_groups:
                        param_group['lr'] = min(param_group['lr'] * 1.1, 0.0001)
                    if batch_idx % 100 == 0:
                        print(f"\n📈 Scanner too weak ({scanner_acc*100:.1f}%) - Increasing scanner LR")
                
                progress_bar.set_postfix({
                    'HyperLoss': f'{hypernet_loss.item():.3f}',
                    'ScanAcc': f'{scanner_acc*100:.1f}%',
                    'TransMag': f'{transform_magnitude.item():.3f}',
                    'RealFake': f'{realfake_loss.item():.3f}'
                })
                
            except Exception as e:
                print(f"\n⚠️ Batch {batch_idx} error: {e}")
                continue
        
        # Calculate epoch metrics
        avg_hypernet_loss = epoch_hypernet_loss / len(dataloader)
        scanner_accuracy = scanner_correct / scanner_total if scanner_total > 0 else 0.0
        avg_disease_preservation = np.mean(disease_preservation_scores) if disease_preservation_scores else 0.0
        avg_transform_magnitude = np.mean(transform_magnitudes) if transform_magnitudes else 0.0
        avg_control_points = np.mean(control_points_list, axis=0) if control_points_list else np.zeros((8, 2))
        
        # Calculate AUC for disease predictions
        disease_auc_orig = 0.0
        disease_auc_trans = 0.0
        
        if all_orig_disease_probs and all_disease_labels:
            try:
                # Debug: Check shapes before concatenation
                if epoch == 1:
                    print(f"\n🔍 AUC Debug - Before concatenation:")
                    print(f"   Number of batches collected: {len(all_orig_disease_probs)}")
                    if len(all_orig_disease_probs) > 0:
                        for i, (probs, labels) in enumerate(zip(all_orig_disease_probs[:3], all_disease_labels[:3])):
                            print(f"   Batch {i}: probs shape={probs.shape}, labels shape={labels.shape}")
                
                # Concatenate all predictions and labels
                y_pred_orig = np.concatenate(all_orig_disease_probs, axis=0)
                y_pred_trans = np.concatenate(all_trans_disease_probs, axis=0)
                y_true = np.concatenate(all_disease_labels, axis=0)
                
                # Debug: print shapes to understand the data
                if epoch == 1:
                    print(f"\n🔍 AUC Debug - After concatenation:")
                    print(f"   y_pred_orig shape: {y_pred_orig.shape}")
                    print(f"   y_pred_trans shape: {y_pred_trans.shape}")
                    print(f"   y_true shape: {y_true.shape}")
                
                # Verify we have proper 2D arrays before calculating AUC
                if y_pred_orig.ndim != 2 or y_true.ndim != 2:
                    print(f"\n⚠️ Invalid tensor dimensions after concatenation:")
                    print(f"   y_pred_orig: {y_pred_orig.shape} (expected 2D)")
                    print(f"   y_pred_trans: {y_pred_trans.shape} (expected 2D)")
                    print(f"   y_true: {y_true.shape} (expected 2D)")
                    print(f"   Skipping AUC calculation for this epoch")
                elif y_true.shape[1] == 0:
                    print(f"\n⚠️ y_true has 0 diseases, cannot calculate AUC")
                else:
                    # Calculate AUC per disease, then average (macro-average AUC)
                    auc_scores_orig = []
                    auc_scores_trans = []
                    
                    for disease_idx in range(y_true.shape[1]):
                        # Get valid labels for this disease (not NaN)
                        valid_mask = ~np.isnan(y_true[:, disease_idx])
                        
                        if valid_mask.sum() > 0 and len(np.unique(y_true[valid_mask, disease_idx])) > 1:
                            # Need at least 2 classes to calculate AUC
                            try:
                                auc_orig = roc_auc_score(y_true[valid_mask, disease_idx], 
                                                         y_pred_orig[valid_mask, disease_idx])
                                auc_trans = roc_auc_score(y_true[valid_mask, disease_idx], 
                                                          y_pred_trans[valid_mask, disease_idx])
                                auc_scores_orig.append(auc_orig)
                                auc_scores_trans.append(auc_trans)
                            except Exception as e:
                                if epoch == 1:
                                    print(f"\n⚠️ Disease {disease_idx} AUC calculation failed: {e}")
                    
                    # Average AUC across diseases
                    if auc_scores_orig:
                        disease_auc_orig = np.mean(auc_scores_orig)
                        disease_auc_trans = np.mean(auc_scores_trans)
                        if epoch == 1:
                            print(f"   ✅ Successfully calculated AUC for {len(auc_scores_orig)} diseases")
                    else:
                        if epoch == 1:
                            print(f"   ⚠️ No valid AUC scores calculated")
            except Exception as e:
                print(f"\n⚠️ AUC calculation error: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'hypernet_loss': avg_hypernet_loss,
            'scanner_accuracy': scanner_accuracy,
            'disease_preservation': avg_disease_preservation,
            'disease_auc_original': disease_auc_orig,
            'disease_auc_transformed': disease_auc_trans,
            'transform_magnitude': avg_transform_magnitude,
            'control_points': avg_control_points,
            'realfake_loss': epoch_realfake_loss / len(dataloader)
        }
    
    def train(self, train_dataloader, val_dataloader, num_epochs, output_dir, 
              early_stop_threshold=0.85, early_stop_patience=3):
        """
        Full training loop with early stopping
        
        Args:
            early_stop_threshold: Stop if scanner accuracy stays above this for patience epochs
            early_stop_patience: Number of consecutive epochs above threshold before stopping
        """
        
        print(f"🚀 Starting Fixed Simple AutoStainer training for {num_epochs} epochs")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖥️ Device: {self.device}")
        print(f"🎯 Goal: Scanner accuracy → 50-60%, Disease preservation > 70%")
        
        if early_stop_threshold < 1.0:
            print(f"⏹️  Early stopping enabled: Stop if scanner acc > {early_stop_threshold*100:.0f}% for {early_stop_patience} epochs")
        
        consecutive_high_scanner = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            metrics = self.train_epoch(train_dataloader, epoch)
            
            # Store metrics
            self.metrics['epochs'].append(epoch)
            for key, value in metrics.items():
                if key == 'control_points':
                    self.metrics[key].append(value.tolist())
                else:
                    self.metrics[key].append(value)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch}/{num_epochs} Summary:")
            print(f"   Hypernetwork Loss: {metrics['hypernet_loss']:.4f}")
            print(f"   Real/Fake Loss: {metrics['realfake_loss']:.4f}")
            print(f"   Scanner Accuracy: {metrics['scanner_accuracy']*100:.1f}% (Target: ~50%)")
            print(f"   Disease Prediction AUC (actual accuracy):")
            print(f"      Original Images: {metrics['disease_auc_original']*100:.1f}%")
            print(f"      Transformed Images: {metrics['disease_auc_transformed']*100:.1f}%")
            auc_drop = metrics['disease_auc_original'] - metrics['disease_auc_transformed']
            print(f"      AUC Drop: {auc_drop*100:.1f}% (Target: <5%)")
            print(f"   Transform Magnitude: {metrics['transform_magnitude']:.4f}")
            print(f"   Spline Control Points Shape: {metrics['control_points'].shape}")
            
            # Early stopping check
            if metrics['scanner_accuracy'] > early_stop_threshold:
                consecutive_high_scanner += 1
                print(f"   ⚠️  Scanner accuracy too high ({consecutive_high_scanner}/{early_stop_patience})")
                
                if consecutive_high_scanner >= early_stop_patience:
                    print(f"\n🛑 Early stopping: Scanner accuracy stayed above {early_stop_threshold*100:.0f}% for {early_stop_patience} epochs")
                    print(f"   This configuration is not achieving good scanner confusion.")
                    break
            else:
                consecutive_high_scanner = 0  # Reset counter
            
            # Interpret results
            scanner_acc = metrics['scanner_accuracy']
            disease_pres = metrics['disease_preservation']
            
            if 0.45 <= scanner_acc <= 0.65 and disease_pres > 0.7:
                print("   ✅ EXCELLENT: Good scanner confusion + disease preservation!")
            elif 0.45 <= scanner_acc <= 0.65:
                print("   📈 GOOD: Scanner confusion in range")
            elif disease_pres > 0.7:
                print("   🏥 GOOD: Disease preservation maintained")
            else:
                print("   📈 LEARNING: Still optimizing balance")
            
            # Save samples and checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_samples(train_dataloader, epoch, output_dir)
                self.save_checkpoint(epoch, output_dir)
        
        # Save final metrics
        self.save_metrics(output_dir)
        print(f"\n🎯 Fixed Simple AutoStainer training completed!")
        return self.metrics
    
    def save_samples(self, dataloader, epoch, output_dir):
        """Save sample transformations with spline control points"""
        self.spline_hypernet.eval()
        self.disease_classifier.eval()
        
        with torch.no_grad():
            # Collect samples from multiple batches for better statistics
            all_images = []
            all_transformed = []
            all_control_points = []
            all_disease_labels = []
            all_dataset_names = []
            all_scanner_labels = []
            all_orig_probs = []
            all_trans_probs = []
            
            # Sample from first 4 batches for diversity (32 samples total)
            for i, batch in enumerate(dataloader):
                if i >= 4:  # 4 batches × 8 samples = 32 samples
                    break
                    
                images = batch['img'].to(self.device)
                disease_labels = batch['lab'].to(self.device)
                scanner_labels = batch['dataset_id'].to(self.device)
                dataset_names = [batch['dataset_name'][j] for j in range(len(batch['dataset_name']))]
                
                # Generate transformations
                transformed_images, control_points = self.spline_hypernet(images, scanner_labels)
                
                # Get disease predictions
                orig_disease_logits, _ = self.disease_classifier(images)
                trans_disease_logits, _ = self.disease_classifier(transformed_images)
                
                # Align tensors
                aligned_orig_logits, aligned_labels = self.align_disease_tensors(orig_disease_logits, disease_labels)
                aligned_trans_logits, _ = self.align_disease_tensors(trans_disease_logits, disease_labels)
                
                orig_disease_probs = torch.sigmoid(aligned_orig_logits)
                trans_disease_probs = torch.sigmoid(aligned_trans_logits)
                
                # Collect data
                all_images.append(images.cpu())
                all_transformed.append(transformed_images.cpu())
                all_control_points.append(control_points.cpu())
                all_disease_labels.append(aligned_labels.cpu())
                all_scanner_labels.append(scanner_labels.cpu())
                all_dataset_names.extend(dataset_names)
                all_orig_probs.append(orig_disease_probs.cpu())
                all_trans_probs.append(trans_disease_probs.cpu())
            
            # Concatenate all data
            if all_images:
                sample_data = {
                    'original_images': torch.cat(all_images, dim=0).numpy(),
                    'transformed_images': torch.cat(all_transformed, dim=0).numpy(),
                    'control_points': torch.cat(all_control_points, dim=0).numpy(),
                    'scanner_labels': torch.cat(all_scanner_labels, dim=0).numpy(),
                    'dataset_names': all_dataset_names,
                    'original_disease_probs': torch.cat(all_orig_probs, dim=0).numpy(),
                    'transformed_disease_probs': torch.cat(all_trans_probs, dim=0).numpy(),
                    'true_labels': torch.cat(all_disease_labels, dim=0).numpy()
                }
                
                # Save numpy file
                samples_dir = os.path.join(output_dir, 'samples')
                os.makedirs(samples_dir, exist_ok=True)
                sample_file = os.path.join(samples_dir, f'samples_epoch_{epoch}.npy')
                np.save(sample_file, sample_data)
                
                print(f"💾 Saved {sample_data['original_images'].shape[0]} samples: {sample_file}")
    
    def save_checkpoint(self, epoch, output_dir):
        """Save model checkpoint"""
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'spline_hypernet_state_dict': self.spline_hypernet.state_dict(),
            'scanner_classifier_state_dict': self.scanner_classifier.state_dict(),
            'disease_classifier_state_dict': self.disease_classifier.state_dict(),
            'realfake_discriminator_state_dict': self.realfake_discriminator.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        
        checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    
    def save_metrics(self, output_dir):
        """Save training metrics"""
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = os.path.join(analysis_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def main():
    """Main training function"""
    
    # Spline Hypernetwork AutoStainer configuration
    config = {
        # Hypernetwork config
        'num_control_points': 8,
        'latent_dim': 128,
        
        # Learning rates - REDUCED for stability
        'hypernet_lr': 0.0001,        # Much lower to prevent instability
        'scanner_lr': 0.00001,        # Lower scanner LR
        'disease_lr': 0.0001,
        'realfake_lr': 0.0001,        # Lower discriminator LR
        
        # Loss weights - REBALANCED to prevent exploding loss
        'lambda_realfake': 1.0,       # Reduced from 10.0
        'lambda_scanner': 5.0,        # Reduced from 30.0 (was dominating)
        'lambda_perceptual': 1.0,     # Reduced from 5.0
        'lambda_disease': 2.0,        # Reduced from 10.0
        'lambda_smooth': 0.01,        # Reduced from 0.1
        
        'batch_size': 8,
        'num_epochs': 50,
        'num_samples_per_dataset': 20000  # Start with 20k for faster iteration
    }
    
    print("🎯 Spline Hypernetwork AutoStainer: Dataset-specific intensity transformations")
    print("🎯 Goal: Scanner accuracy → 50-60%, Disease preservation > 70%")
    print("🎯 Uses: Spline control points + Real/Fake discriminator + Multi-scale perceptual loss")
    print(f"📊 Training with {config['num_samples_per_dataset']:,} samples per dataset")
    
    # Load datasets
    print("\n📊 Loading datasets...")
    loader = MultiDatasetGANLoader(
        limit_per_dataset=config['num_samples_per_dataset'],
        batch_size=config['batch_size']
    )
    train_dataloader, val_dataloader = loader.create_dataloaders(
        train_split=0.8,
        num_workers=2  # Reduced for stability
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'/lotterlab/emily_torchxrayvision/outputs/fixed_simple_autostainer_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and train
    autostainer = FixedSimpleAutoStainer(config)
    metrics = autostainer.train(train_dataloader, val_dataloader, 
                               config['num_epochs'], output_dir)
    
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
