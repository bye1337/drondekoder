"""
Loss функции для обучения моделей сопоставления изображений.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss для обучения Siamese Network.
    
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    где Y=1 для похожих изображений, Y=0 для разных
    D - евклидово расстояние между признаками
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, features1, features2, labels):
        """
        Args:
            features1: Признаки первого изображения [batch_size, feature_dim]
            features2: Признаки второго изображения [batch_size, feature_dim]
            labels: 1 для похожих, 0 для разных [batch_size]
            
        Returns:
            loss: Значение функции потерь
        """
        euclidean_distance = F.pairwise_distance(features1, features2)
        loss_contrastive = torch.mean(
            (1 - labels) * torch.pow(euclidean_distance, 2) +
            labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet Loss для обучения с тройками изображений.
    
    L = max(0, margin + D(a,p) - D(a,n))
    где a - anchor, p - positive, n - negative
    D - евклидово расстояние
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor features [batch_size, feature_dim]
            positive: Positive features [batch_size, feature_dim]
            negative: Negative features [batch_size, feature_dim]
            
        Returns:
            loss: Значение функции потерь
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(
            torch.clamp(
                self.margin + distance_positive - distance_negative,
                min=0.0
            )
        )
        return loss


class ContrastiveCosineLoss(nn.Module):
    """
    Contrastive Loss на основе косинусного сходства.
    Подходит для случаев когда уже есть нормализованные признаки.
    """
    
    def __init__(self, margin=0.5, temperature=0.07):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, features1, features2, labels):
        """
        Args:
            features1: Признаки первого изображения [batch_size, feature_dim]
            features2: Признаки второго изображения [batch_size, feature_dim]
            labels: 1 для похожих, 0 для разных [batch_size]
            
        Returns:
            loss: Значение функции потерь
        """
        # Косинусное сходство (признаки уже нормализованы)
        cosine_sim = F.cosine_similarity(features1, features2)
        
        # Для похожих изображений минимизируем расстояние
        # Для разных - максимизируем расстояние
        loss = torch.mean(
            labels * torch.pow(cosine_sim - 1, 2) +
            (1 - labels) * torch.pow(torch.clamp(cosine_sim + self.margin, max=1.0) - 1, 2)
        )
        return loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Mining для автоматического выбора triplets.
    """
    
    def __init__(self, margin=1.0, mining_strategy='hard'):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor features [batch_size, feature_dim]
            positive: Positive features [batch_size, feature_dim]
            negative: Negative features [batch_size, feature_dim]
            
        Returns:
            loss: Значение функции потерь
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        if self.mining_strategy == 'hard':
            # Hard mining: выбираем наиболее сложные triplets
            mask = torch.gt(distance_positive + self.margin, distance_negative)
        elif self.mining_strategy == 'all':
            # Все triplets
            mask = torch.ones_like(distance_positive, dtype=torch.bool)
        else:
            # Easy mining
            mask = torch.ones_like(distance_positive, dtype=torch.bool)
        
        loss = torch.mean(
            torch.clamp(
                self.margin + distance_positive - distance_negative,
                min=0.0
            )[mask]
        )
        
        return loss

