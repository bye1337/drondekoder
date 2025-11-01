"""
Dataset классы для загрузки и обработки данных для обучения модели.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class MapImagePairDataset(Dataset):
    """
    Dataset для обучения на парах изображений (карта и снимок дрона).
    
    Ожидаемая структура данных:
    data/
        pairs/
            image_001_map.jpg      # Большое изображение карты
            image_001_drone.jpg    # Маленькое изображение с дрона
            image_002_map.jpg
            image_002_drone.jpg
            ...
    """
    
    def __init__(self, data_dir, mode='train', transform=None, input_size=(512, 512)):
        """
        Args:
            data_dir: Путь к директории с данными
            mode: 'train', 'val' или 'test'
            transform: Albumentations трансформации
            input_size: Размер входного изображения
        """
        self.data_dir = data_dir
        self.mode = mode
        self.input_size = input_size
        
        # Загружаем список пар
        self.pairs = self._load_pairs()
        
        # Определяем трансформации
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _load_pairs(self):
        """Загрузка списка пар изображений."""
        pairs_dir = os.path.join(self.data_dir, 'pairs')
        map_files = sorted([f for f in os.listdir(pairs_dir) if f.endswith('_map.jpg')])
        
        pairs = []
        for map_file in map_files:
            drone_file = map_file.replace('_map.jpg', '_drone.jpg')
            map_path = os.path.join(pairs_dir, map_file)
            drone_path = os.path.join(pairs_dir, drone_file)
            
            if os.path.exists(drone_path):
                pairs.append({
                    'map': map_path,
                    'drone': drone_path
                })
        
        # Разделение на train/val/test
        n_total = len(pairs)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        if self.mode == 'train':
            pairs = pairs[:n_train]
        elif self.mode == 'val':
            pairs = pairs[n_train:n_train+n_val]
        else:  # test
            pairs = pairs[n_train+n_val:]
        
        return pairs
    
    def _get_default_transforms(self):
        """Получение стандартных трансформаций."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Возвращает пару изображений.
        
        Returns:
            map_img: Изображение карты [3, H, W]
            drone_img: Изображение с дрона [3, H, W]
        """
        pair = self.pairs[idx]
        
        # Загрузка изображений
        map_img = cv2.imread(pair['map'])
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        
        drone_img = cv2.imread(pair['drone'])
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        
        # Применение трансформаций
        if self.transform:
            # Применяем трансформации отдельно к каждому изображению
            map_transformed = self.transform(image=map_img)
            drone_transformed = self.transform(image=drone_img)
            map_img = map_transformed['image']
            drone_img = drone_transformed['image']
        else:
            map_img = torch.from_numpy(map_img).float().permute(2, 0, 1) / 255.0
            drone_img = torch.from_numpy(drone_img).float().permute(2, 0, 1) / 255.0
        
        return map_img, drone_img


class TripletMapDataset(Dataset):
    """
    Dataset для обучения на triplets (anchor, positive, negative).
    Более эффективно для обучения моделей сопоставления.
    """
    
    def __init__(self, data_dir, mode='train', transform=None, input_size=(512, 512)):
        """
        Args:
            data_dir: Путь к директории с данными
            mode: 'train', 'val' или 'test'
            transform: Albumentations трансформации
            input_size: Размер входного изображения
        """
        self.data_dir = data_dir
        self.mode = mode
        self.input_size = input_size
        
        # Загружаем пары
        self.pairs = self._load_pairs()
        
        # Формируем triplets
        self.triplets = self._create_triplets()
        
        # Определяем трансформации
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _load_pairs(self):
        """Загрузка списка пар изображений."""
        pairs_dir = os.path.join(self.data_dir, 'pairs')
        map_files = sorted([f for f in os.listdir(pairs_dir) if f.endswith('_map.jpg')])
        
        pairs = []
        for map_file in map_files:
            drone_file = map_file.replace('_map.jpg', '_drone.jpg')
            map_path = os.path.join(pairs_dir, map_file)
            drone_path = os.path.join(pairs_dir, drone_file)
            
            if os.path.exists(drone_path):
                pairs.append({
                    'map': map_path,
                    'drone': drone_path
                })
        
        # Разделение на train/val/test
        n_total = len(pairs)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        if self.mode == 'train':
            pairs = pairs[:n_train]
        elif self.mode == 'val':
            pairs = pairs[n_train:n_train+n_val]
        else:  # test
            pairs = pairs[n_train+n_val:]
        
        return pairs
    
    def _create_triplets(self):
        """
        Создание triplets для обучения.
        Anchor - снимок дрона
        Positive - соответствующая область карты
        Negative - случайная область карты из другой пары
        """
        triplets = []
        
        for i, pair in enumerate(self.pairs):
            # Получаем случайный индекс для negative
            negative_idx = np.random.randint(0, len(self.pairs))
            while negative_idx == i:
                negative_idx = np.random.randint(0, len(self.pairs))
            
            triplets.append({
                'anchor': pair['drone'],
                'positive': pair['map'],
                'negative': self.pairs[negative_idx]['map']
            })
        
        return triplets
    
    def _get_default_transforms(self):
        """Получение стандартных трансформаций."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Возвращает triplet изображений.
        
        Returns:
            anchor: Anchor изображение [3, H, W]
            positive: Positive изображение [3, H, W]
            negative: Negative изображение [3, H, W]
        """
        triplet = self.triplets[idx]
        
        # Загрузка изображений
        anchor = cv2.imread(triplet['anchor'])
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        
        positive = cv2.imread(triplet['positive'])
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
        
        negative = cv2.imread(triplet['negative'])
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
        
        # Применение трансформаций
        if self.transform:
            transformed_anchor = self.transform(image=anchor)
            transformed_positive = self.transform(image=positive)
            transformed_negative = self.transform(image=negative)
            
            anchor = transformed_anchor['image']
            positive = transformed_positive['image']
            negative = transformed_negative['image']
        
        return anchor, positive, negative

