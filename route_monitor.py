"""
Модуль для мониторинга маршрута и определения отклонений
"""
import numpy as np
from typing import List, Tuple, Dict, Optional


class RouteMonitor:
    """Класс для отслеживания маршрута и определения отклонений"""
    
    def __init__(self):
        self.waypoints: List[Tuple[float, float]] = []
        self.current_segment = 0
        self.allowed_deviation = 50.0  # Разрешенное отклонение в пикселях
        
    def set_route(self, waypoints: List[Tuple[float, float]]):
        """
        Устанавливает маршрут
        
        Args:
            waypoints: Список точек маршрута [(x1, y1), (x2, y2), ...]
        """
        if len(waypoints) < 2:
            raise ValueError("Маршрут должен содержать минимум 2 точки")
        self.waypoints = waypoints
        self.current_segment = 0
        
    def get_current_segment(self, position: Tuple[float, float]) -> int:
        """
        Определяет текущий сегмент маршрута на основе позиции
        
        Args:
            position: Текущая позиция (x, y)
            
        Returns:
            Индекс текущего сегмента
        """
        if len(self.waypoints) < 2:
            return 0
        
        min_dist = float('inf')
        best_segment = 0
        
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i + 1]
            
            # Расстояние от точки до отрезка
            dist = self._distance_to_segment(position, start, end)
            
            if dist < min_dist:
                min_dist = dist
                best_segment = i
        
        self.current_segment = best_segment
        return best_segment
    
    def check_deviation(self, position: Tuple[float, float]) -> Dict:
        """
        Проверяет отклонение от маршрута
        
        Args:
            position: Текущая позиция (x, y)
            
        Returns:
            Словарь с информацией об отклонении
        """
        if len(self.waypoints) < 2:
            return {
                'is_on_route': True,
                'deviation': 0.0,
                'segment': 0,
                'message': 'Маршрут не задан'
            }
        
        segment = self.get_current_segment(position)
        
        if segment >= len(self.waypoints) - 1:
            return {
                'is_on_route': True,
                'deviation': 0.0,
                'segment': segment,
                'message': 'Маршрут завершен'
            }
        
        start = self.waypoints[segment]
        end = self.waypoints[segment + 1]
        
        # Расстояние от текущей позиции до линии маршрута
        deviation = self._distance_to_segment(position, start, end)
        
        # Направление отклонения
        deviation_vector = self._get_deviation_vector(position, start, end)
        
        is_on_route = deviation <= self.allowed_deviation
        
        message = 'На маршруте'
        if not is_on_route:
            message = f'Отклонение от маршрута: {deviation:.1f} пикселей'
        
        return {
            'is_on_route': is_on_route,
            'deviation': float(deviation),
            'segment': segment,
            'segment_start': start,
            'segment_end': end,
            'deviation_vector': deviation_vector,
            'message': message,
            'current_position': position
        }
    
    def _distance_to_segment(self, point: Tuple[float, float], 
                             start: Tuple[float, float], 
                             end: Tuple[float, float]) -> float:
        """
        Вычисляет расстояние от точки до отрезка
        
        Args:
            point: Точка
            start: Начало отрезка
            end: Конец отрезка
            
        Returns:
            Расстояние в пикселях
        """
        px, py = point
        sx, sy = start
        ex, ey = end
        
        # Вектор отрезка
        dx = ex - sx
        dy = ey - sy
        
        if dx == 0 and dy == 0:
            # Отрезок - это точка
            return np.sqrt((px - sx)**2 + (py - sy)**2)
        
        # Параметр t для проекции точки на отрезок
        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
        
        # Ближайшая точка на отрезке
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        
        # Расстояние от точки до ближайшей точки на отрезке
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def _get_deviation_vector(self, point: Tuple[float, float],
                             start: Tuple[float, float],
                             end: Tuple[float, float]) -> Tuple[float, float]:
        """
        Получает вектор отклонения от маршрута
        
        Returns:
            Кортеж (dx, dy) - вектор отклонения
        """
        px, py = point
        sx, sy = start
        ex, ey = end
        
        dx_seg = ex - sx
        dy_seg = ey - sy
        
        if dx_seg == 0 and dy_seg == 0:
            return (px - sx, py - sy)
        
        t = max(0, min(1, ((px - sx) * dx_seg + (py - sy) * dy_seg) / (dx_seg * dx_seg + dy_seg * dy_seg)))
        
        closest_x = sx + t * dx_seg
        closest_y = sy + t * dy_seg
        
        return (px - closest_x, py - closest_y)
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """Возвращает следующую точку маршрута"""
        if self.current_segment + 1 < len(self.waypoints):
            return self.waypoints[self.current_segment + 1]
        return None
    
    def set_allowed_deviation(self, pixels: float):
        """Устанавливает разрешенное отклонение в пикселях"""
        self.allowed_deviation = pixels

