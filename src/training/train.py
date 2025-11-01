"""
Скрипт для обучения модели сопоставления изображений.
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import TripletNetwork, SiameseNetwork
from models.losses import TripletLoss, ContrastiveLoss
from preprocessing import TripletMapDataset, MapImagePairDataset


class Trainer:
    """
    Класс для обучения модели.
    """
    
    def __init__(self, config_path):
        """
        Args:
            config_path: Путь к конфигурационному файлу
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Создаем директории
        os.makedirs(self.config['paths']['model_save_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
        # Инициализация модели
        self.model = self._build_model()
        
        # Инициализация loss
        self.criterion = self._build_loss()
        
        # Инициализация оптимизатора
        self.optimizer = self._build_optimizer()
        
        # Инициализация scheduler
        self.scheduler = self._build_scheduler()
        
        # Загрузка данных
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Лучшая точность
        self.best_acc = 0.0
        
    def _build_model(self):
        """Построение модели."""
        model_config = self.config['model']
        
        if self.config['training']['use_triplet']:
            model = TripletNetwork(
                backbone=model_config['backbone'],
                feature_dim=model_config['feature_dim'],
                pretrained=model_config['pretrained']
            )
        else:
            model = SiameseNetwork(
                backbone=model_config['backbone'],
                feature_dim=model_config['feature_dim'],
                pretrained=model_config['pretrained']
            )
        
        return model.to(self.device)
    
    def _build_loss(self):
        """Построение функции потерь."""
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type']
        
        if loss_type == 'triplet':
            return TripletLoss(margin=loss_config.get('margin', 1.0))
        elif loss_type == 'contrastive':
            return ContrastiveLoss(margin=loss_config.get('margin', 1.0))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _build_optimizer(self):
        """Построение оптимизатора."""
        opt_config = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if opt_config['type'] == 'adam':
            return Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_config['type'] == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    def _build_scheduler(self):
        """Построение scheduler."""
        sched_config = self.config['training']['scheduler']
        num_epochs = self.config['training']['num_epochs']
        
        if sched_config['type'] == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=sched_config.get('eta_min', 0)
            )
        elif sched_config['type'] == 'step':
            return StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _build_dataloaders(self):
        """Построение загрузчиков данных."""
        data_dir = self.config['paths']['data_dir']
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        
        input_size = self.config['model']['input_size']
        
        if self.config['training']['use_triplet']:
            train_dataset = TripletMapDataset(
                data_dir,
                mode='train',
                input_size=input_size
            )
            val_dataset = TripletMapDataset(
                data_dir,
                mode='val',
                input_size=input_size
            )
        else:
            train_dataset = MapImagePairDataset(
                data_dir,
                mode='train',
                input_size=input_size
            )
            val_dataset = MapImagePairDataset(
                data_dir,
                mode='val',
                input_size=input_size
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        """Обучение одной эпохи."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Перемещаем данные на устройство
            if self.config['training']['use_triplet']:
                anchor, positive, negative = batch_data
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_feat, pos_feat, neg_feat = self.model(anchor, positive, negative)
                
                # Вычисление loss
                loss = self.criterion(anchor_feat, pos_feat, neg_feat)
            else:
                img1, img2 = batch_data
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # Forward pass
                similarity = self.model(img1, img2)
                
                # Для contrastive loss нужны labels (упрощенная версия)
                # В реальности нужны метки похожести
                labels = torch.ones(similarity.size(0), 1).to(self.device)
                loss = nn.MSELoss()(similarity, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Обновление progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate_epoch(self):
        """Валидация одной эпохи."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch_data in pbar:
            if self.config['training']['use_triplet']:
                anchor, positive, negative = batch_data
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_feat, pos_feat, neg_feat = self.model(anchor, positive, negative)
                loss = self.criterion(anchor_feat, pos_feat, neg_feat)
                
                # Простая метрика: расстояние до positive должно быть меньше чем до negative
                dist_pos = torch.norm(anchor_feat - pos_feat, dim=1)
                dist_neg = torch.norm(anchor_feat - neg_feat, dim=1)
                correct_predictions += (dist_pos < dist_neg).sum().item()
                total_predictions += anchor.size(0)
            else:
                img1, img2 = batch_data
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                similarity = self.model(img1, img2)
                labels = torch.ones(similarity.size(0), 1).to(self.device)
                loss = nn.MSELoss()(similarity, labels)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Сохранение checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        # Сохранение последнего checkpoint
        last_path = os.path.join(
            self.config['paths']['model_save_dir'],
            'last_checkpoint.pth'
        )
        torch.save(checkpoint, last_path)
        
        # Сохранение лучшего
        if is_best:
            best_path = os.path.join(
                self.config['paths']['model_save_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_acc:.4f}")
    
    def train(self):
        """Полный цикл обучения."""
        num_epochs = self.config['training']['num_epochs']
        
        print("Starting training...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Обучение
            train_loss = self.train_epoch()
            
            # Валидация
            val_loss, val_acc = self.validate_epoch()
            
            # Обновление learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Логирование
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Сохранение best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            
            # Сохранение checkpoint
            if epoch % self.config['training'].get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train visual positioning model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

