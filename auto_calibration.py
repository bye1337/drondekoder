"""
Модуль автокалибровки системы стабилизации при взлете
Использует GPS данные в первые 5 минут после взлета
"""
import time
from typing import Optional, Tuple, Callable
from stabilization_processor import PositionStabilizer
from gps_integration import GPSInterface, GPSFix


class AutoCalibration:
    """
    Автоматическая калибровка системы стабилизации
    Работает в первые 5 минут после взлета
    """
    
    def __init__(
        self,
        stabilizer: PositionStabilizer,
        gps: GPSInterface,
        calibration_window: float = 300.0  # 5 минут
    ):
        """
        Args:
            stabilizer: Стабилизатор позиции
            gps: GPS интерфейс
            calibration_window: Окно калибровки в секундах
        """
        self.stabilizer = stabilizer
        self.gps = gps
        self.calibration_window = calibration_window
        
        self.takeoff_time: Optional[float] = None
        self.calibrated = False
        self.calibration_complete = False
        
        # Точка взлета (визуальная и GPS)
        self.visual_home: Optional[Tuple[float, float]] = None
        self.gps_home: Optional[GPSFix] = None
        
    def start_calibration(self):
        """Запускает процесс калибровки (вызывается при взлете)"""
        self.takeoff_time = time.time()
        self.calibrated = False
        self.calibration_complete = False
        print("Автокалибровка запущена")
    
    def update(self, frame, visual_result: Optional[dict] = None) -> bool:
        """
        Обновляет процесс калибровки
        
        Args:
            frame: Текущий кадр (для инициализации стабилизатора)
            visual_result: Результат визуальной стабилизации
            
        Returns:
            True если калибровка завершена
        """
        if self.calibration_complete:
            return True
        
        if self.takeoff_time is None:
            return False
        
        elapsed = time.time() - self.takeoff_time
        
        # Проверка окна калибровки
        if elapsed > self.calibration_window:
            if not self.calibrated:
                print("Предупреждение: калибровка не завершена за отведенное время")
            self.calibration_complete = True
            return self.calibrated
        
        # Инициализация визуальной системы при первом кадре
        if visual_result is None and frame is not None:
            self.stabilizer.initialize(frame)
            visual_result = self.stabilizer.update(frame)
        
        # Получение GPS фикса
        gps_fix = self.gps.get_current_fix()
        
        # Условия завершения калибровки:
        # 1. Визуальная система инициализирована
        # 2. GPS фикс валиден и надежен
        # 3. Прошло минимум 5 секунд для стабилизации
        
        if elapsed >= 5.0:  # Минимум 5 секунд после взлета
            if visual_result and visual_result.get('confidence', 0) > 0.5:
                # Устанавливаем визуальную домашнюю точку
                position = visual_result.get('position', [0, 0])
                self.visual_home = (position[0], position[1])
                
                # Устанавливаем GPS домашнюю точку
                if gps_fix and self.gps.is_gps_reliable(gps_fix):
                    self.gps.set_home(gps_fix)
                    self.gps_home = gps_fix
                    
                    self.calibrated = True
                    self.calibration_complete = True
                    
                    print(f"Калибровка завершена:")
                    print(f"  Визуальная точка: {self.visual_home}")
                    print(f"  GPS точка: lat={gps_fix.latitude:.6f}, "
                          f"lon={gps_fix.longitude:.6f}, alt={gps_fix.altitude:.1f}m")
                    return True
        
        return False
    
    def get_calibration_status(self) -> dict:
        """Возвращает статус калибровки"""
        elapsed = 0.0
        if self.takeoff_time:
            elapsed = time.time() - self.takeoff_time
        
        return {
            'active': self.takeoff_time is not None and not self.calibration_complete,
            'calibrated': self.calibrated,
            'elapsed_time': elapsed,
            'remaining_time': max(0, self.calibration_window - elapsed),
            'visual_home': self.visual_home,
            'gps_home': {
                'latitude': self.gps_home.latitude if self.gps_home else None,
                'longitude': self.gps_home.longitude if self.gps_home else None,
                'altitude': self.gps_home.altitude if self.gps_home else None
            } if self.gps_home else None
        }
    
    def reset(self):
        """Сброс калибровки"""
        self.takeoff_time = None
        self.calibrated = False
        self.calibration_complete = False
        self.visual_home = None
        self.gps_home = None

