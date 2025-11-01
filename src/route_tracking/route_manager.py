"""
Модуль для управления маршрутом дрона и отслеживания отклонений.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DeviationStatus(Enum):
    """Статусы отклонения от маршрута."""
    ON_ROUTE = "on_route"
    MINOR_DEVIATION = "minor_deviation"
    MAJOR_DEVIATION = "major_deviation"
    OFF_ROUTE = "off_route"


@dataclass
class RoutePoint:
    """
    Точка маршрута.
    
    Attributes:
        lat: Широта
        lon: Долгота
        altitude: Высота (опционально)
        tolerance: Допустимое отклонение в метрах
    """
    lat: float
    lon: float
    altitude: Optional[float] = None
    tolerance: float = 10.0  # метров


@dataclass
class Position:
    """
    Текущая позиция дрона.
    
    Attributes:
        lat: Широта
        lon: Долгота
        altitude: Высота (опционально)
        confidence: Уверенность в позиции (0-1)
        timestamp: Временная метка
    """
    lat: float
    lon: float
    altitude: Optional[float] = None
    confidence: float = 1.0
    timestamp: Optional[float] = None


@dataclass
class DeviationAlert:
    """
    Оповещение об отклонении от маршрута.
    
    Attributes:
        status: Статус отклонения
        deviation_distance: Расстояние отклонения в метрах
        nearest_point: Ближайшая точка маршрута
        message: Текстовое сообщение
    """
    status: DeviationStatus
    deviation_distance: float
    nearest_point: Tuple[int, float, float]  # (index, lat, lon)
    message: str


class RouteManager:
    """
    Менеджер для управления маршрутом дрона и отслеживания отклонений.
    """
    
    def __init__(
        self,
        route_points: List[RoutePoint],
        max_deviation: float = 50.0,
        minor_threshold: float = 25.0
    ):
        """
        Args:
            route_points: Список точек маршрута
            max_deviation: Максимальное допустимое отклонение в метрах
            minor_threshold: Порог для "незначительного" отклонения
        """
        self.route_points = route_points
        self.max_deviation = max_deviation
        self.minor_threshold = minor_threshold
        self.current_segment_index = 0
    
    def update_position(self, position: Position) -> Optional[DeviationAlert]:
        """
        Обновление позиции дрона и проверка отклонения.
        
        Args:
            position: Текущая позиция дрона
            
        Returns:
            DeviationAlert если есть отклонение, иначе None
        """
        # Находим ближайший сегмент маршрута
        nearest_idx, distance = self._find_nearest_point(position)
        
        # Определяем статус
        status = self._classify_deviation(distance)
        
        # Если отклонение значительное, создаем оповещение
        if status != DeviationStatus.ON_ROUTE:
            return DeviationAlert(
                status=status,
                deviation_distance=distance,
                nearest_point=(
                    nearest_idx,
                    self.route_points[nearest_idx].lat,
                    self.route_points[nearest_idx].lon
                ),
                message=self._generate_message(status, distance, nearest_idx)
            )
        
        return None
    
    def _find_nearest_point(self, position: Position) -> Tuple[int, float]:
        """
        Поиск ближайшей точки маршрута.
        
        Returns:
            (index, distance): Индекс точки и расстояние в метрах
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, point in enumerate(self.route_points):
            distance = self._haversine_distance(
                position.lat, position.lon,
                point.lat, point.lon
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx, min_distance
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Вычисление расстояния между двумя точками на сфере (Haversine формула).
        
        Args:
            lat1, lon1: Координаты первой точки
            lat2, lon2: Координаты второй точки
            
        Returns:
            distance: Расстояние в метрах
        """
        R = 6371000  # Радиус Земли в метрах
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi / 2)**2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def _classify_deviation(self, distance: float) -> DeviationStatus:
        """
        Классификация отклонения по расстоянию.
        
        Args:
            distance: Расстояние отклонения в метрах
            
        Returns:
            status: Статус отклонения
        """
        if distance <= self.minor_threshold:
            return DeviationStatus.ON_ROUTE
        elif distance <= self.minor_threshold * 2:
            return DeviationStatus.MINOR_DEVIATION
        elif distance <= self.max_deviation:
            return DeviationStatus.MAJOR_DEVIATION
        else:
            return DeviationStatus.OFF_ROUTE
    
    def _generate_message(
        self,
        status: DeviationStatus,
        distance: float,
        point_idx: int
    ) -> str:
        """
        Генерация текстового сообщения об отклонении.
        
        Args:
            status: Статус отклонения
            distance: Расстояние в метрах
            point_idx: Индекс ближайшей точки
            
        Returns:
            message: Текстовое сообщение
        """
        if status == DeviationStatus.MINOR_DEVIATION:
            return f"[WARNING] Minor deviation: {distance:.1f}m from point #{point_idx}"
        elif status == DeviationStatus.MAJOR_DEVIATION:
            return f"[ALERT] Major deviation: {distance:.1f}m from point #{point_idx}"
        elif status == DeviationStatus.OFF_ROUTE:
            return f"[CRITICAL] OFF ROUTE: {distance:.1f}m from point #{point_idx}"
        
        return "On route"
    
    def get_current_target(self) -> Optional[RoutePoint]:
        """
        Получение текущей целевой точки маршрута.
        
        Returns:
            RoutePoint если есть, иначе None
        """
        if self.current_segment_index < len(self.route_points):
            return self.route_points[self.current_segment_index]
        return None
    
    def advance_to_next_point(self):
        """Переход к следующей точке маршрута."""
        if self.current_segment_index < len(self.route_points) - 1:
            self.current_segment_index += 1
    
    def get_route_progress(self) -> float:
        """
        Получение прогресса по маршруту.
        
        Returns:
            progress: Прогресс от 0 до 1
        """
        return self.current_segment_index / len(self.route_points) if self.route_points else 0.0

