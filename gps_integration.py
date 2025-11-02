"""
Модуль интеграции с GPS/ГЛОНАСС для резервирования системы стабилизации
"""
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GPSFix:
    """Данные GPS фиксации"""
    latitude: float
    longitude: float
    altitude: float
    timestamp: float
    hdop: float  # Horizontal Dilution of Precision
    fix_quality: int  # 0=invalid, 1=GPS, 2=DGPS, etc.
    satellites: int
    valid: bool = False


class GPSInterface:
    """
    Интерфейс для работы с GPS/ГЛОНАСС приемником
    Предполагается получение данных через TCP/UDP или последовательный порт
    """
    
    def __init__(self, gps_source=None):
        """
        Args:
            gps_source: Источник GPS данных (TCP socket, serial port, etc.)
        """
        self.gps_source = gps_source
        self.last_fix: Optional[GPSFix] = None
        self.home_fix: Optional[GPSFix] = None
        self.calibrated = False
        
    def parse_nmea(self, nmea_sentence: str) -> Optional[GPSFix]:
        """
        Парсинг NMEA предложения (например, $GPGGA)
        
        Формат GPGGA:
        $GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
        """
        if not nmea_sentence.startswith('$GP'):
            return None
        
        try:
            parts = nmea_sentence.split(',')
            
            if parts[0] == '$GPGGA':
                # Parse GPGGA sentence
                if len(parts) < 15:
                    return None
                
                # Время (не используем напрямую)
                time_str = parts[1] if parts[1] else None
                
                # Широта
                lat_deg = float(parts[2][:2]) if len(parts[2]) >= 2 else 0.0
                lat_min = float(parts[2][2:]) if len(parts[2]) > 2 else 0.0
                lat_hem = parts[3]
                latitude = lat_deg + lat_min / 60.0
                if lat_hem == 'S':
                    latitude = -latitude
                
                # Долгота
                lon_deg = float(parts[4][:3]) if len(parts[4]) >= 3 else 0.0
                lon_min = float(parts[4][3:]) if len(parts[4]) > 3 else 0.0
                lon_hem = parts[5]
                longitude = lon_deg + lon_min / 60.0
                if lon_hem == 'W':
                    longitude = -longitude
                
                # Качество фикса
                fix_quality = int(parts[6]) if parts[6] else 0
                
                # Количество спутников
                satellites = int(parts[7]) if parts[7] else 0
                
                # HDOP
                hdop = float(parts[8]) if parts[8] else 99.9
                
                # Высота
                altitude = float(parts[9]) if parts[9] else 0.0
                
                fix = GPSFix(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    timestamp=time.time(),
                    hdop=hdop,
                    fix_quality=fix_quality,
                    satellites=satellites,
                    valid=(fix_quality > 0 and satellites >= 4)
                )
                
                return fix
                
        except (ValueError, IndexError) as e:
            print(f"Ошибка парсинга NMEA: {e}")
            return None
        
        return None
    
    def get_current_fix(self) -> Optional[GPSFix]:
        """Получает текущий GPS фикс"""
        # Здесь должен быть код получения данных от GPS приемника
        # Для примера возвращаем последний известный фикс
        return self.last_fix
    
    def set_home(self, fix: Optional[GPSFix] = None):
        """
        Устанавливает домашнюю точку (точка взлета)
        
        Args:
            fix: GPS фикс домашней точки. Если None, использует текущий фикс
        """
        if fix is None:
            fix = self.get_current_fix()
        
        if fix and fix.valid:
            self.home_fix = fix
            self.calibrated = True
            print(f"Домашняя точка установлена по GPS: "
                  f"lat={fix.latitude:.6f}, lon={fix.longitude:.6f}, "
                  f"alt={fix.altitude:.1f}m")
            return True
        else:
            print("Предупреждение: GPS фикс невалиден для установки дома")
            return False
    
    def get_distance_to_home(self, current_fix: Optional[GPSFix] = None) -> float:
        """
        Вычисляет расстояние до домашней точки (в метрах)
        
        Args:
            current_fix: Текущий GPS фикс. Если None, использует get_current_fix()
        """
        if not self.home_fix:
            return 0.0
        
        if current_fix is None:
            current_fix = self.get_current_fix()
        
        if not current_fix or not current_fix.valid:
            return 0.0
        
        # Формула гаверсинуса для вычисления расстояния на сфере
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Радиус Земли в метрах
        
        lat1 = radians(self.home_fix.latitude)
        lat2 = radians(current_fix.latitude)
        dlat = lat2 - lat1
        dlon = radians(current_fix.longitude - self.home_fix.longitude)
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c
        
        return distance
    
    def get_bearing_to_home(self, current_fix: Optional[GPSFix] = None) -> float:
        """
        Вычисляет азимут до домашней точки (в градусах, 0-360)
        
        Args:
            current_fix: Текущий GPS фикс
        """
        if not self.home_fix:
            return 0.0
        
        if current_fix is None:
            current_fix = self.get_current_fix()
        
        if not current_fix or not current_fix.valid:
            return 0.0
        
        from math import radians, degrees, sin, cos, atan2
        
        lat1 = radians(self.home_fix.latitude)
        lat2 = radians(current_fix.latitude)
        dlon = radians(current_fix.longitude - self.home_fix.longitude)
        
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = degrees(atan2(y, x))
        bearing = (bearing + 360) % 360  # Нормализация до 0-360
        
        return bearing
    
    def is_gps_reliable(self, fix: Optional[GPSFix] = None) -> bool:
        """
        Проверяет надежность GPS фикса
        
        Args:
            fix: GPS фикс для проверки
        """
        if fix is None:
            fix = self.get_current_fix()
        
        if not fix:
            return False
        
        # Критерии надежности:
        # - Валидный фикс
        # - Достаточно спутников (>= 6 для хорошего фикса)
        # - Низкий HDOP (< 2.0 для хорошего фикса)
        # - Недавние данные (< 1 секунды)
        
        if not fix.valid:
            return False
        
        if fix.satellites < 4:
            return False
        
        if fix.hdop > 3.0:
            return False
        
        if time.time() - fix.timestamp > 1.0:
            return False
        
        return True
    
    def fuse_with_visual(self, gps_position: Tuple[float, float], 
                        visual_position: Tuple[float, float],
                        gps_confidence: float = 1.0,
                        visual_confidence: float = 1.0) -> Tuple[float, float]:
        """
        Объединение GPS и визуальных данных (простой фильтр Калмана)
        
        Args:
            gps_position: Позиция по GPS (lat, lon)
            visual_position: Позиция по визуальной системе (x, y в пикселях)
            gps_confidence: Уверенность GPS (0-1)
            visual_confidence: Уверенность визуальной системы (0-1)
        """
        # Простое взвешенное усреднение
        # В реальности нужно преобразование координат и более сложный фильтр
        
        total_confidence = gps_confidence + visual_confidence
        if total_confidence == 0:
            return visual_position
        
        # Для простоты возвращаем визуальную позицию, если она надежна
        # Или GPS, если визуальная система ненадежна
        if visual_confidence > 0.7:
            return visual_position
        elif gps_confidence > 0.7:
            # Здесь нужно преобразование GPS -> пиксели
            return visual_position  # Заглушка
        else:
            # Взвешенное среднее
            w1 = visual_confidence / total_confidence
            w2 = gps_confidence / total_confidence
            return (
                visual_position[0] * w1 + gps_position[0] * w2,
                visual_position[1] * w1 + gps_position[1] * w2
            )

