"""
Модуль возврата домой (RTH - Return to Home)
Использует накопленные смещения для возврата в точку взлета
"""
import math
from typing import Optional, Tuple
from gps_integration import GPSInterface


class HomeReturn:
    """
    Модуль возврата домой
    Вычисляет необходимые команды для возврата в точку взлета
    """
    
    def __init__(self, gps: Optional[GPSInterface] = None):
        """
        Args:
            gps: GPS интерфейс (опционально, для улучшенной навигации)
        """
        self.gps = gps
        self.home_position: Optional[Tuple[float, float]] = None  # (x, y) в пикселях
        self.current_position: Optional[Tuple[float, float]] = None
        
        # Максимальное расстояние до дома для успешного возврата
        self.arrival_threshold = 50.0  # пикселей (примерно 5м при масштабе 0.1м/пикс)
        
        # Масштаб: пиксели -> метры (зависит от высоты)
        self.pixels_per_meter = 10.0  # По умолчанию: 10 пикселей = 1 метр
    
    def set_home(self, x: float, y: float):
        """Устанавливает домашнюю точку"""
        self.home_position = (x, y)
        print(f"Домашняя точка установлена: ({x}, {y})")
    
    def set_pixels_per_meter(self, scale: float):
        """
        Устанавливает масштаб преобразования пикселей в метры
        
        Args:
            scale: Количество пикселей на метр
        """
        self.pixels_per_meter = scale
    
    def update_position(self, x: float, y: float):
        """Обновляет текущую позицию"""
        self.current_position = (x, y)
    
    def get_distance_to_home(self) -> float:
        """Вычисляет расстояние до дома (в пикселях)"""
        if not self.home_position or not self.current_position:
            return float('inf')
        
        dx = self.current_position[0] - self.home_position[0]
        dy = self.current_position[1] - self.home_position[1]
        
        return math.sqrt(dx*dx + dy*dy)
    
    def get_distance_to_home_meters(self) -> float:
        """Вычисляет расстояние до дома (в метрах)"""
        pixels = self.get_distance_to_home()
        return pixels / self.pixels_per_meter
    
    def get_bearing_to_home(self) -> float:
        """
        Вычисляет направление до дома (в градусах, 0-360)
        0° = север, 90° = восток, 180° = юг, 270° = запад
        """
        if not self.home_position or not self.current_position:
            return 0.0
        
        dx = self.home_position[0] - self.current_position[0]
        dy = self.home_position[1] - self.current_position[1]
        
        # Вычисляем угол в радианах
        angle_rad = math.atan2(dx, -dy)  # -dy потому что ось Y направлена вниз
        
        # Преобразуем в градусы и нормализуем
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360
        
        return angle_deg
    
    def is_home_reached(self) -> bool:
        """Проверяет, достигнута ли домашняя точка"""
        distance = self.get_distance_to_home()
        return distance <= self.arrival_threshold
    
    def get_control_command(self, current_yaw: float = 0.0) -> dict:
        """
        Вычисляет команды управления для возврата домой
        
        Args:
            current_yaw: Текущий курс дрона (в градусах, 0-360)
            
        Returns:
            Словарь с командами управления:
            {
                'heading': целевой курс,
                'heading_error': ошибка курса,
                'distance': расстояние до дома,
                'speed': рекомендуемая скорость,
                'action': 'move' | 'hover' | 'arrived'
            }
        """
        if not self.home_position or not self.current_position:
            return {
                'action': 'error',
                'message': 'Home position or current position not set'
            }
        
        distance = self.get_distance_to_home()
        bearing = self.get_bearing_to_home()
        
        if self.is_home_reached():
            return {
                'action': 'arrived',
                'heading': current_yaw,
                'heading_error': 0.0,
                'distance': distance,
                'distance_meters': self.get_distance_to_home_meters(),
                'speed': 0.0
            }
        
        # Вычисляем ошибку курса
        heading_error = bearing - current_yaw
        
        # Нормализуем ошибку до диапазона -180..180
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360
        
        # Рекомендуемая скорость (зависит от расстояния)
        # Ближе к дому - медленнее
        max_speed = 2.0  # м/с
        min_speed = 0.5  # м/с
        
        distance_m = distance / self.pixels_per_meter
        
        if distance_m > 100:
            speed = max_speed
        elif distance_m > 50:
            speed = max_speed * 0.7
        elif distance_m > 20:
            speed = max_speed * 0.5
        else:
            speed = min_speed
        
        # Если ошибка курса большая, сначала выравниваем курс
        if abs(heading_error) > 30:
            speed = 0.0  # Останавливаемся для разворота
        
        return {
            'action': 'move',
            'heading': bearing,
            'heading_error': heading_error,
            'distance': distance,
            'distance_meters': distance_m,
            'speed': speed,
            'pixels_per_meter': self.pixels_per_meter
        }
    
    def get_status(self) -> dict:
        """Возвращает статус модуля возврата домой"""
        return {
            'home_set': self.home_position is not None,
            'position_set': self.current_position is not None,
            'distance': self.get_distance_to_home() if self.home_position and self.current_position else None,
            'distance_meters': self.get_distance_to_home_meters() if self.home_position and self.current_position else None,
            'bearing': self.get_bearing_to_home() if self.home_position and self.current_position else None,
            'home_reached': self.is_home_reached() if self.home_position and self.current_position else False,
            'arrival_threshold': self.arrival_threshold
        }

