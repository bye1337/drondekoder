"""
Модуль для поиска соответствий между изображением дрона и картой.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2
from pathlib import Path


class ImageMatcher:
    """
    Класс для поиска соответствий между снимком дрона и спутниковой картой.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Обученная модель SiameseNetwork
            device: CUDA или CPU
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def extract_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Извлечение признаков из изображений.
        
        Args:
            image_batch: Batch изображений [N, 3, H, W]
            
        Returns:
            features: Векторы признаков [N, feature_dim]
        """
        with torch.no_grad():
            features = self.model.extract_features(image_batch.to(self.device))
        return features
    
    def match_using_sliding_window(
        self,
        drone_image: np.ndarray,
        map_image: np.ndarray,
        window_size: Tuple[int, int] = (512, 512),
        stride: int = 128,
        top_k: int = 5
    ) -> list:
        """
        Поиск соответствий методом скользящего окна.
        
        Args:
            drone_image: Изображение с дрона [H, W, 3]
            map_image: Большое изображение карты [H, W, 3]
            window_size: Размер окна для поиска
            stride: Шаг скользящего окна
            top_k: Количество лучших совпадений
            
        Returns:
            matches: Список найденных соответствий (x, y, similarity)
        """
        # Препроцессинг изображения дрона
        drone_tensor = self._preprocess_image(drone_image)
        drone_features = self.extract_features(drone_tensor.unsqueeze(0))
        
        # Скользящее окно по карте
        h, w = window_size
        map_h, map_w = map_image.shape[:2]
        
        matches = []
        batch = []
        positions = []
        
        for y in range(0, map_h - h + 1, stride):
            for x in range(0, map_w - w + 1, stride):
                window = map_image[y:y+h, x:x+w]
                window_tensor = self._preprocess_image(window)
                batch.append(window_tensor)
                positions.append((x, y))
        
        # Обработка batch'ами для эффективности
        batch_size = 32
        similarities = []
        
        for i in range(0, len(batch), batch_size):
            batch_tensors = torch.stack(batch[i:i+batch_size])
            window_features = self.extract_features(batch_tensors)
            
            # Косинусное сходство
            sim = F.cosine_similarity(
                drone_features.expand(len(window_features), -1),
                window_features,
                dim=1
            )
            similarities.extend(sim.cpu().numpy())
        
        # Сортируем по убыванию сходства
        matches = sorted(
            zip(positions, similarities),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return matches
    
    def match_using_pyramid(
        self,
        drone_image: np.ndarray,
        map_image: np.ndarray,
        scales: list = [0.5, 0.75, 1.0],
        top_k: int = 5
    ) -> list:
        """
        Поиск соответствий с использованием image pyramid для учета масштаба.
        
        Args:
            drone_image: Изображение с дрона [H, W, 3]
            map_image: Большое изображение карты [H, W, 3]
            scales: Список масштабов для поиска
            top_k: Количество лучших совпадений
            
        Returns:
            matches: Список найденных соответствий (x, y, scale, similarity)
        """
        all_matches = []
        
        for scale in scales:
            # Масштабируем карту
            scaled_map = cv2.resize(
                map_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            
            # Поиск на масштабированной карте
            matches = self.match_using_sliding_window(
                drone_image,
                scaled_map,
                stride=128,
                top_k=top_k
            )
            
            # Добавляем информацию о масштабе
            for (x, y), similarity in matches:
                all_matches.append(((int(x/scale), int(y/scale)), scale, similarity))
        
        # Возвращаем топ-k соответствий
        all_matches.sort(key=lambda x: x[2], reverse=True)
        return all_matches[:top_k]
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Препроцессинг изображения для модели.
        
        Args:
            image: Изображение [H, W, 3] в RGB
            
        Returns:
            tensor: Подготовленный тензор [3, 512, 512]
        """
        # Resize до 512x512
        image = cv2.resize(image, (512, 512))
        
        # Нормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # Конвертация в tensor и перестановка каналов
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return tensor


class CoordinateEstimator:
    """
    Класс для оценки координат на основе найденных соответствий.
    """
    
    def __init__(self, map_metadata: dict):
        """
        Args:
            map_metadata: Метаданные карты {
                'center_lat': float,
                'center_lon': float,
                'pixels_per_meter': float,
                'image_size': (width, height)
            }
        """
        self.map_metadata = map_metadata
    
    def pixels_to_coordinates(
        self,
        pixel_x: int,
        pixel_y: int
    ) -> Tuple[float, float]:
        """
        Преобразование координат пикселей в географические координаты.
        
        Args:
            pixel_x: Координата X в пикселях
            pixel_y: Координата Y в пикселях
            
        Returns:
            lat, lon: Географические координаты
        """
        # Простое преобразование (может быть более сложным в зависимости от проекции)
        center_lat = self.map_metadata['center_lat']
        center_lon = self.map_metadata['center_lon']
        pixels_per_meter = self.map_metadata['pixels_per_meter']
        
        img_w, img_h = self.map_metadata['image_size']
        
        # Смещение от центра в метрах
        offset_x_m = (pixel_x - img_w / 2) / pixels_per_meter
        offset_y_m = (pixel_y - img_h / 2) / pixels_per_meter
        
        # Приблизительное преобразование в градусы
        # 1 градус широты ≈ 111 км
        # 1 градус долготы ≈ 111 км * cos(latitude)
        lat = center_lat + offset_y_m / 111000
        lon = center_lon + offset_x_m / (111000 * np.cos(np.radians(center_lat)))
        
        return lat, lon
    
    def estimate_position(
        self,
        matches: list
    ) -> Tuple[float, float, float]:
        """
        Оценка позиции на основе нескольких соответствий (взвешенное усреднение).
        
        Args:
            matches: Список соответствий [(x, y, similarity), ...]
            
        Returns:
            lat, lon, confidence: Оцененные координаты и уверенность
        """
        if not matches:
            return None, None, 0.0
        
        # Взвешенное усреднение по уверенности
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for (x, y), similarity in matches:
            # Используем similarity как весовой коэффициент
            weight = max(0, similarity)  # Обрезаем отрицательные
            total_weight += weight
            weighted_x += x * weight
            weighted_y += y * weight
        
        if total_weight > 0:
            avg_x = weighted_x / total_weight
            avg_y = weighted_y / total_weight
            
            # Конвертация в координаты
            lat, lon = self.pixels_to_coordinates(int(avg_x), int(avg_y))
            
            # Уверенность = средняя similarity
            confidence = total_weight / len(matches)
            
            return lat, lon, confidence
        
        return None, None, 0.0

