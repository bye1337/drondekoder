"""
Siamese Network для сопоставления изображений карты и снимка дрона.

Использует архитектуру Siamese Network с CNN backbone для извлечения признаков
и сравнения изображений.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class FeatureExtractor(nn.Module):
    """
    Извлечение признаков из изображения с использованием предобученного ResNet.
    """
    
    def __init__(self, backbone='resnet50', feature_dim=128, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            # Удаляем последний слой для извлечения признаков
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection head для уменьшения размерности
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor [batch_size, 3, H, W]
            
        Returns:
            features: Feature vector [batch_size, feature_dim]
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        features = self.projector(x)
        # L2 normalization
        features = nn.functional.normalize(features, p=2, dim=1)
        return features


class SiameseNetwork(nn.Module):
    """
    Siamese Network для сопоставления двух изображений.
    
    Принимает пару изображений и возвращает оценку схожести.
    """
    
    def __init__(self, backbone='resnet50', feature_dim=128, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone, feature_dim, pretrained)
        
    def forward(self, img1, img2):
        """
        Args:
            img1: Первое изображение [batch_size, 3, H, W]
            img2: Второе изображение [batch_size, 3, H, W]
            
        Returns:
            similarity: Косинусное сходство [batch_size, 1]
        """
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        # Косинусное сходство
        similarity = torch.sum(features1 * features2, dim=1, keepdim=True)
        return similarity
    
    def extract_features(self, img):
        """
        Извлечение признаков из одного изображения.
        
        Args:
            img: Изображение [batch_size, 3, H, W]
            
        Returns:
            features: Feature vector [batch_size, feature_dim]
        """
        return self.feature_extractor(img)


class TripletNetwork(nn.Module):
    """
    Triplet Network для обучения с тройками (anchor, positive, negative).
    Более стабильное обучение для задач сопоставления.
    """
    
    def __init__(self, backbone='resnet50', feature_dim=128, pretrained=True):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone, feature_dim, pretrained)
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor изображение [batch_size, 3, H, W]
            positive: Положительное изображение (та же локация) [batch_size, 3, H, W]
            negative: Отрицательное изображение (другая локация) [batch_size, 3, H, W]
            
        Returns:
            anchor_feat, positive_feat, negative_feat: Векторы признаков
        """
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)
        
        return anchor_feat, positive_feat, negative_feat
    
    def extract_features(self, img):
        """Извлечение признаков из одного изображения."""
        return self.feature_extractor(img)

