#!/usr/bin/env python
# coding: utf-8
"""
Dataset Discriminator Networks for GAN-based Windowing
Implements neural networks that classify dataset sources to enable adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class DatasetDiscriminator(nn.Module):
    """
    Dataset Discriminator Network - tries to classify which dataset an image came from.
    This is the core component for adversarial windowing training.
    
    The windowing function will try to fool this discriminator by making images 
    from different datasets indistinguishable.
    """
    
    def __init__(self, num_datasets=2, input_channels=1, architecture='resnet18', 
                 dropout_rate=0.3, use_spectral_norm=False):
        """
        Initialize the Dataset Discriminator.
        
        Args:
            num_datasets: Number of datasets to classify (2 for CheXpert + MIMIC)
            input_channels: Number of input channels (1 for grayscale X-rays)
            architecture: Backbone architecture ('resnet18', 'resnet50', 'densenet121')
            dropout_rate: Dropout rate for regularization
            use_spectral_norm: Whether to use spectral normalization for stability
        """
        super(DatasetDiscriminator, self).__init__()
        
        self.num_datasets = num_datasets
        self.input_channels = input_channels
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.use_spectral_norm = use_spectral_norm
        
        print(f"🏗️ Initializing Dataset Discriminator:")
        print(f"  - Architecture: {architecture}")
        print(f"  - Number of datasets: {num_datasets}")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Spectral norm: {use_spectral_norm}")
        
        # Build the backbone
        self.backbone = self._build_backbone()
        
        # Build the classifier head
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ Dataset Discriminator initialized with {self._count_parameters():,} parameters")
    
    def _build_backbone(self):
        """Build the feature extraction backbone."""
        
        if self.architecture == 'resnet18':
            backbone = models.resnet18(pretrained=False)
            # Modify first conv layer for grayscale input
            if self.input_channels != 3:
                backbone.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, 
                                         padding=3, bias=False)
            # Remove the final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 512
            
        elif self.architecture == 'resnet50':
            backbone = models.resnet50(pretrained=False)
            if self.input_channels != 3:
                backbone.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2,
                                         padding=3, bias=False)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
            
        elif self.architecture == 'densenet121':
            backbone = models.densenet121(pretrained=False)
            if self.input_channels != 3:
                # DenseNet uses features.conv0 instead of conv1
                backbone.features.conv0 = nn.Conv2d(self.input_channels, 64, kernel_size=7, 
                                                   stride=2, padding=3, bias=False)
            # Remove classifier
            backbone = backbone.features
            self.feature_dim = 1024
            
        elif self.architecture == 'simple_cnn':
            # Simple CNN for faster training/debugging
            backbone = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 128
            
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        return backbone
    
    def _build_classifier(self):
        """Build the classification head."""
        
        layers = []
        
        # Global average pooling if needed (for ResNet/DenseNet)
        if self.architecture in ['resnet18', 'resnet50']:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif self.architecture == 'densenet121':
            layers.extend([
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            ])
        
        # Flatten
        layers.append(nn.Flatten())
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Classification layers
        hidden_dim = max(128, self.feature_dim // 4)
        
        # First hidden layer
        linear1 = nn.Linear(self.feature_dim, hidden_dim)
        if self.use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
        layers.extend([
            linear1,
            nn.ReLU(inplace=True),
        ])
        
        # Dropout
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        linear2 = nn.Linear(hidden_dim, self.num_datasets)
        if self.use_spectral_norm:
            linear2 = nn.utils.spectral_norm(linear2)
        layers.append(linear2)
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Dataset classification logits [batch_size, num_datasets]
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_probabilities(self, x):
        """
        Get dataset probabilities instead of raw logits.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Dataset probabilities [batch_size, num_datasets]
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def get_features(self, x):
        """
        Extract intermediate features (useful for analysis).
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Extracted features [batch_size, feature_dim]
        """
        features = self.backbone(x)
        if len(features.shape) > 2:  # If not flattened yet
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return features


class LightweightDatasetDiscriminator(nn.Module):
    """
    Lightweight version of Dataset Discriminator for faster training.
    Suitable for quick experiments and debugging.
    """
    
    def __init__(self, num_datasets=2, input_channels=1):
        super(LightweightDatasetDiscriminator, self).__init__()
        
        self.num_datasets = num_datasets
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block  
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_datasets)
        )
        
        print(f"✅ Lightweight Dataset Discriminator initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def get_probabilities(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def test_dataset_discriminators():
    """Test function for the dataset discriminator networks."""
    print("🧪 Testing Dataset Discriminator Networks...")
    
    # Test input (batch_size=4, channels=1, height=320, width=320)
    test_input = torch.randn(4, 1, 320, 320)
    test_labels = torch.tensor([0, 1, 0, 1])  # CheXpert, MIMIC, CheXpert, MIMIC
    
    print(f"📊 Test input shape: {test_input.shape}")
    print(f"📊 Test labels: {test_labels}")
    
    # Test different architectures
    architectures = ['simple_cnn', 'resnet18', 'lightweight']
    
    for arch in architectures:
        print(f"\n🔬 Testing {arch} architecture...")
        
        try:
            if arch == 'lightweight':
                discriminator = LightweightDatasetDiscriminator(num_datasets=2, input_channels=1)
            else:
                discriminator = DatasetDiscriminator(
                    num_datasets=2,
                    input_channels=1,
                    architecture=arch,
                    dropout_rate=0.3,
                    use_spectral_norm=False
                )
            
            # Test forward pass
            with torch.no_grad():
                logits = discriminator(test_input)
                probs = discriminator.get_probabilities(test_input) if hasattr(discriminator, 'get_probabilities') else F.softmax(logits, dim=1)
                
                print(f"  ✅ Output logits shape: {logits.shape}")
                print(f"  ✅ Output probabilities shape: {probs.shape}")
                print(f"  📊 Sample probabilities: {probs[0].numpy()}")
                print(f"  📊 Predicted classes: {torch.argmax(probs, dim=1).numpy()}")
            
            # Test loss computation
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, test_labels)
            print(f"  📊 Test loss: {loss.item():.4f}")
            
            # Test backward pass
            loss.backward()
            print(f"  ✅ Backward pass successful")
            
            # Test feature extraction (if available)
            if hasattr(discriminator, 'get_features'):
                with torch.no_grad():
                    features = discriminator.get_features(test_input)
                    print(f"  ✅ Feature extraction shape: {features.shape}")
            
        except Exception as e:
            print(f"  ❌ Error testing {arch}: {e}")
    
    print("\n✅ Dataset Discriminator testing completed!")


if __name__ == "__main__":
    test_dataset_discriminators()
