"""
Модуль для стабилизации позиции БАС по видеопотоку с нижней камеры
Оптимизирован для реального времени и работы с двумя камерами
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque
import time


class PositionStabilizer:
    """
    Класс для стабилизации позиции БАС по видеопотоку.
    
    Использует оптический поток (Optical Flow) для отслеживания смещения
    между последовательными кадрами в реальном времени.
    """
    
    def __init__(
        self,
        method: str = 'lucas_kanade',  # 'lucas_kanade' или 'farneback'
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        window_size: Tuple[int, int] = (15, 15),
        max_level: int = 2,
        criteria: Tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    ):
        """
        Args:
            method: Метод отслеживания ('lucas_kanade' или 'farneback')
            max_corners: Максимальное количество точек для отслеживания
            quality_level: Порог качества для детекции углов (Shi-Tomasi)
            min_distance: Минимальное расстояние между точками
            window_size: Размер окна для Lucas-Kanade
            max_level: Максимальный уровень пирамиды
            criteria: Критерии остановки для оптического потока
        """
        self.method = method
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.window_size = window_size
        self.max_level = max_level
        self.criteria = criteria
        
        # Состояние отслеживания
        self.previous_frame = None
        self.previous_gray = None
        self.track_points = None
        self.initial_position = None
        self.current_offset = np.array([0.0, 0.0], dtype=np.float32)  # Накопленное смещение
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # Текущая скорость
        
        # История для фильтрации
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        # Для Farneback метода
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
    
    def reset(self):
        """Сброс состояния стабилизатора"""
        self.previous_frame = None
        self.previous_gray = None
        self.track_points = None
        self.initial_position = None
        self.current_offset = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.position_history.clear()
        self.velocity_history.clear()
    
    def initialize(self, frame: np.ndarray) -> bool:
        """
        Инициализация стабилизатора на первом кадре
        
        Args:
            frame: Первый кадр видеопотока
            
        Returns:
            True если инициализация успешна
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Детектируем углы для отслеживания (Shi-Tomasi)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )
        
        if corners is not None and len(corners) > 10:
            self.track_points = corners
            self.previous_gray = gray.copy()
            self.initial_position = np.array([frame.shape[1] // 2, frame.shape[0] // 2])
            self.current_offset = np.array([0.0, 0.0])
            return True
        
        return False
    
    def update_lucas_kanade(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обновление позиции методом Lucas-Kanade оптического потока
        
        Args:
            frame: Текущий кадр
            
        Returns:
            Словарь с информацией о смещении и позиции
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.previous_gray is None or self.track_points is None:
            return None
        
        if len(self.track_points) == 0:
            return None
        
        # Вычисляем оптический поток
        new_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.track_points,
            None,
            winSize=self.window_size,
            maxLevel=self.max_level,
            criteria=self.criteria
        )
        
        # Фильтруем хорошие точки
        good_points = new_points[status == 1]
        old_points = self.track_points[status == 1]
        
        if len(good_points) < 4:
            # Недостаточно точек - переинициализация
            return None
        
        # Вычисляем смещение (медианное для устойчивости к выбросам)
        displacement = good_points - old_points
        median_displacement = np.median(displacement, axis=0)
        
        # Обновляем накопленное смещение
        self.current_offset += median_displacement
        
        # Обновляем скорость (экспоненциальное сглаживание)
        alpha = 0.7
        self.velocity = alpha * self.velocity + (1 - alpha) * median_displacement
        
        # Обновляем состояние
        self.track_points = good_points.reshape(-1, 1, 2)
        self.previous_gray = gray.copy()
        
        # Добавляем выборочные точки, если их стало мало
        if len(self.track_points) < self.max_corners // 2:
            new_corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners - len(self.track_points),
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            if new_corners is not None:
                self.track_points = np.vstack([self.track_points, new_corners])
        
        # Фильтруем позицию (медианный фильтр)
        self.position_history.append(self.current_offset.copy())
        filtered_offset = np.median(list(self.position_history), axis=0)
        
        # Вычисляем текущую позицию относительно начальной
        current_position = self.initial_position + filtered_offset
        
        return {
            'position': current_position.astype(int).tolist(),
            'offset': filtered_offset.tolist(),
            'velocity': self.velocity.tolist(),
            'tracked_points': len(self.track_points),
            'method': 'lucas_kanade',
            'confidence': min(1.0, len(self.track_points) / self.max_corners)
        }
    
    def update_farneback(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обновление позиции методом Farneback оптического потока
        
        Args:
            frame: Текущий кадр
            
        Returns:
            Словарь с информацией о смещении и позиции
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.previous_gray is None:
            return None
        
        # Вычисляем плотный оптический поток
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            **self.farneback_params
        )
        
        # Вычисляем среднее смещение в центре кадра (область интереса)
        h, w = gray.shape
        center_region = flow[h//4:3*h//4, w//4:3*w//4]
        
        # Медианное смещение для устойчивости
        median_flow = np.median(center_region.reshape(-1, 2), axis=0)
        
        # Обновляем накопленное смещение
        self.current_offset += median_flow
        
        # Обновляем скорость
        alpha = 0.7
        self.velocity = alpha * self.velocity + (1 - alpha) * median_flow
        
        # Обновляем состояние
        self.previous_gray = gray.copy()
        
        # Фильтруем позицию
        self.position_history.append(self.current_offset.copy())
        filtered_offset = np.median(list(self.position_history), axis=0)
        
        # Вычисляем текущую позицию
        if self.initial_position is None:
            self.initial_position = np.array([w // 2, h // 2])
        
        current_position = self.initial_position + filtered_offset
        
        # Вычисляем уверенность на основе вариации потока
        flow_variance = np.var(center_region.reshape(-1, 2), axis=0)
        confidence = 1.0 / (1.0 + np.mean(flow_variance) / 100.0)
        
        return {
            'position': current_position.astype(int).tolist(),
            'offset': filtered_offset.tolist(),
            'velocity': self.velocity.tolist(),
            'method': 'farneback',
            'confidence': float(confidence),
            'flow_magnitude': float(np.linalg.norm(median_flow))
        }
    
    def update(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обновление позиции на основе нового кадра
        
        Args:
            frame: Текущий кадр видеопотока
            
        Returns:
            Словарь с информацией о позиции и смещении
        """
        # Инициализация при первом кадре
        if self.previous_gray is None:
            if not self.initialize(frame):
                return None
        
        # Выбираем метод обновления
        if self.method == 'lucas_kanade':
            result = self.update_lucas_kanade(frame)
        elif self.method == 'farneback':
            result = self.update_farneback(frame)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        return result
    
    def get_stability_metrics(self) -> Dict:
        """
        Возвращает метрики стабильности позиции
        
        Returns:
            Словарь с метриками (вариация позиции, скорость дрифта и т.д.)
        """
        if len(self.position_history) < 2:
            return {
                'position_variance': 0.0,
                'drift_rate': 0.0,
                'is_stable': True
            }
        
        positions = np.array(list(self.position_history))
        position_variance = np.var(positions, axis=0)
        
        # Скорость дрифта (изменение смещения за последние кадры)
        if len(self.position_history) >= 5:
            recent_positions = list(self.position_history)[-5:]
            drift = np.mean(np.diff(recent_positions, axis=0), axis=0)
            drift_rate = np.linalg.norm(drift)
        else:
            drift_rate = 0.0
        
        # Позиция считается стабильной, если вариация < 5 пикселей
        is_stable = np.mean(position_variance) < 25.0  # 5 пикселей в квадрате
        
        return {
            'position_variance': position_variance.tolist(),
            'drift_rate': float(drift_rate),
            'is_stable': bool(is_stable),
            'avg_position': np.mean(positions, axis=0).tolist()
        }


class DualCameraStabilizer:
    """
    Класс для стабилизации позиции с использованием двух камер
    (например, нижняя камера для позиционирования, вторая для валидации)
    """
    
    def __init__(
        self,
        primary_method: str = 'lucas_kanade',
        secondary_method: str = 'farneback'
    ):
        """
        Args:
            primary_method: Метод для основной камеры (нижняя)
            secondary_method: Метод для вторичной камеры (опционально)
        """
        self.primary_stabilizer = PositionStabilizer(method=primary_method)
        self.secondary_stabilizer = PositionStabilizer(method=secondary_method) if secondary_method else None
        self.fusion_weight = 0.8  # Вес основной камеры
    
    def update(
        self,
        primary_frame: np.ndarray,
        secondary_frame: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Обновление позиции с учетом двух камер
        
        Args:
            primary_frame: Кадр с основной (нижней) камеры
            secondary_frame: Кадр со второй камеры (опционально)
            
        Returns:
            Объединенный результат стабилизации
        """
        # Обновление основной камеры
        primary_result = self.primary_stabilizer.update(primary_frame)
        
        if primary_result is None:
            return None
        
        # Если вторая камера доступна, используем ее для валидации
        if secondary_frame is not None and self.secondary_stabilizer:
            secondary_result = self.secondary_stabilizer.update(secondary_frame)
            
            if secondary_result:
                # Взвешенное объединение результатов
                w1, w2 = self.fusion_weight, 1.0 - self.fusion_weight
                
                combined_position = (
                    np.array(primary_result['position']) * w1 +
                    np.array(secondary_result['position']) * w2
                )
                
                combined_velocity = (
                    np.array(primary_result['velocity']) * w1 +
                    np.array(secondary_result['velocity']) * w2
                )
                
                combined_confidence = (
                    primary_result['confidence'] * w1 +
                    secondary_result['confidence'] * w2
                )
                
                return {
                    'position': combined_position.astype(int).tolist(),
                    'offset': primary_result['offset'],
                    'velocity': combined_velocity.tolist(),
                    'confidence': combined_confidence,
                    'primary_confidence': primary_result['confidence'],
                    'secondary_confidence': secondary_result['confidence'],
                    'method': 'dual_camera_fusion'
                }
        
        return primary_result
    
    def reset(self):
        """Сброс обоих стабилизаторов"""
        self.primary_stabilizer.reset()
        if self.secondary_stabilizer:
            self.secondary_stabilizer.reset()


def visualize_stabilization(
    frame: np.ndarray,
    result: Dict,
    draw_tracks: bool = True,
    draw_grid: bool = False
) -> np.ndarray:
    """
    Визуализация результатов стабилизации на кадре
    
    Args:
        frame: Кадр для визуализации
        result: Результат стабилизации
        draw_tracks: Рисовать ли треки оптического потока
        draw_grid: Рисовать ли сетку для оценки стабильности
        
    Returns:
        Кадр с визуализацией
    """
    vis_frame = frame.copy()
    
    if result is None:
        return vis_frame
    
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    position = result.get('position', center)
    
    # Рисуем центр кадра (опорная точка)
    cv2.circle(vis_frame, center, 10, (0, 255, 0), 2)
    cv2.circle(vis_frame, center, 5, (0, 255, 0), -1)
    
    # Рисуем текущую позицию
    pos = tuple(position)
    offset = np.array(result.get('offset', [0, 0]))
    offset_pixels = offset.astype(int)
    
    # Вектор смещения
    end_point = (center[0] + offset_pixels[0], center[1] + offset_pixels[1])
    cv2.arrowedLine(vis_frame, center, end_point, (0, 0, 255), 3, tipLength=0.3)
    cv2.circle(vis_frame, pos, 8, (255, 0, 0), -1)
    
    # Информация на кадре
    confidence = result.get('confidence', 0.0)
    tracked_points = result.get('tracked_points', 0)
    velocity = np.array(result.get('velocity', [0, 0]))
    velocity_magnitude = np.linalg.norm(velocity)
    
    info_text = [
        f"Confidence: {confidence:.2f}",
        f"Tracked: {tracked_points}",
        f"Velocity: {velocity_magnitude:.2f} px/frame",
        f"Offset: ({offset_pixels[0]}, {offset_pixels[1]})"
    ]
    
    y_offset = 30
    for text in info_text:
        cv2.putText(vis_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 25
    
    # Сетка для оценки стабильности
    if draw_grid:
        grid_size = 50
        for x in range(0, w, grid_size):
            cv2.line(vis_frame, (x, 0), (x, h), (100, 100, 100), 1)
        for y in range(0, h, grid_size):
            cv2.line(vis_frame, (0, y), (w, y), (100, 100, 100), 1)
    
    return vis_frame

