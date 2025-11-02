"""
Главный модуль системы навигации БАС
Объединяет стабилизацию, GPS, автокалибровку и возврат домой
Оптимизирован для работы на встроенном вычислителе с задержками <0.1с
"""
import cv2
import numpy as np
import time
import threading
from typing import Optional, Callable, Dict
from video_processor import VideoProcessor
from stabilization_processor import PositionStabilizer
from navigation_server import NavigationServer
from gps_integration import GPSInterface
from auto_calibration import AutoCalibration
from home_return import HomeReturn


class DroneNavigationSystem:
    """
    Главная система навигации БАС
    
    Функции:
    - Стабилизация позиции по видеопотоку
    - Интеграция с GPS для резервирования
    - Автокалибровка при взлете
    - Возврат домой
    - TCP/UDP сервер для связи с системой управления
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        tcp_port: int = 5001,
        udp_port: int = 5002,
        target_fps: float = 10.0,  # Целевая частота обновления (10 Гц как у GNSS)
        optimize_for_latency: bool = True
    ):
        """
        Args:
            camera_id: ID камеры
            tcp_port: Порт TCP сервера
            udp_port: Порт UDP сервера
            target_fps: Целевая частота обновления (по умолчанию 10 Гц как у GNSS)
            optimize_for_latency: Оптимизация под низкие задержки
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.optimize_for_latency = optimize_for_latency
        
        # Инициализация компонентов
        self.video_processor = VideoProcessor(
            use_dual_camera=False,
            primary_method='lucas_kanade'  # Самый быстрый метод
        )
        
        # Настройка для низких задержек
        if optimize_for_latency:
            # Уменьшаем количество точек для ускорения
            self.video_processor.stabilizer.max_corners = 100
            self.video_processor.stabilizer.quality_level = 0.02
        
        # GPS интерфейс
        self.gps = GPSInterface()
        
        # Автокалибровка
        self.calibration = AutoCalibration(
            stabilizer=self.video_processor.stabilizer,
            gps=self.gps
        )
        
        # Возврат домой
        self.home_return = HomeReturn(gps=self.gps)
        
        # Навигационный сервер
        self.nav_server = NavigationServer(
            tcp_port=tcp_port,
            udp_port=udp_port,
            video_processor=self.video_processor
        )
        
        # Состояние
        self.is_running = False
        self.processing_thread = None
        self.cap = None
        
        # Статистика
        self.frame_count = 0
        self.processing_times = []
        self.last_latency = 0.0
        
    def start(self):
        """Запускает систему навигации"""
        if self.is_running:
            return
        
        # Открываем камеру
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть камеру {self.camera_id}")
        
        # Настраиваем камеру для низких задержек
        if self.optimize_for_latency:
            # Уменьшаем разрешение для ускорения (если возможно)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Уменьшаем буфер кадров
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Запускаем навигационный сервер
        self.nav_server.start()
        
        # Запускаем обработку
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        print("Система навигации запущена")
    
    def stop(self):
        """Останавливает систему навигации"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.nav_server.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("Система навигации остановлена")
    
    def takeoff(self):
        """Сигнал взлета - запускает автокалибровку"""
        self.calibration.start_calibration()
        print("Сигнал взлета получен, автокалибровка запущена")
    
    def _processing_loop(self):
        """Основной цикл обработки (оптимизирован для низких задержек)"""
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        
        while self.is_running:
            start_time = time.time()
            
            # Читаем кадр
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Пропускаем кадры, если обрабатываем быстрее целевой частоты
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            
            last_frame_time = time.time()
            
            # Обработка кадра
            result = self.video_processor.process_frame(primary_frame=frame)
            
            if result:
                # Обновляем автокалибровку
                if not self.calibration.calibration_complete:
                    self.calibration.update(frame, result)
                
                # Если калибровка завершена, устанавливаем домашнюю точку
                if self.calibration.calibrated and self.home_return.home_position is None:
                    position = result.get('position', [0, 0])
                    self.home_return.set_home(position[0], position[1])
                    self.nav_server.set_home_position(position[0], position[1])
                
                # Обновляем текущую позицию для возврата домой
                position = result.get('position', [0, 0])
                self.home_return.update_position(position[0], position[1])
                
                # Добавляем информацию о возврате домой
                rth_status = self.home_return.get_status()
                rth_command = self.home_return.get_control_command()
                
                result['home_return'] = {
                    'status': rth_status,
                    'command': rth_command
                }
                
                # Отправляем в навигационный сервер
                self.nav_server.update_measurement(result)
                self.nav_server.broadcast_measurement(result)
            
            # Измеряем задержку
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            self.last_latency = processing_time
            self.frame_count += 1
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику работы системы"""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0.0
        max_time = np.max(self.processing_times) if self.processing_times else 0.0
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time_ms': avg_time * 1000,
            'max_processing_time_ms': max_time * 1000,
            'last_latency_ms': self.last_latency * 1000,
            'target_fps': self.target_fps,
            'actual_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'calibration': self.calibration.get_calibration_status(),
            'home_return': self.home_return.get_status()
        }
    
    def get_navigation_data(self) -> Optional[Dict]:
        """Возвращает текущие навигационные данные"""
        if not self.nav_server.last_measurement:
            return None
        
        measurement = self.nav_server.last_measurement.copy()
        
        # Добавляем информацию о GPS
        if self.gps:
            gps_fix = self.gps.get_current_fix()
            if gps_fix:
                measurement['gps'] = {
                    'latitude': gps_fix.latitude,
                    'longitude': gps_fix.longitude,
                    'altitude': gps_fix.altitude,
                    'reliable': self.gps.is_gps_reliable(gps_fix)
                }
        
        return measurement


def main():
    """Пример использования"""
    print("=" * 60)
    print("Система навигации БАС")
    print("=" * 60)
    
    # Создание системы
    nav_system = DroneNavigationSystem(
        camera_id=0,
        tcp_port=5001,
        udp_port=5002,
        target_fps=10.0,  # 10 Гц как у GNSS
        optimize_for_latency=True
    )
    
    try:
        # Запуск
        nav_system.start()
        
        # Симуляция взлета
        print("\nСигнал взлета...")
        nav_system.takeoff()
        
        print("\nСистема работает. Нажмите Ctrl+C для остановки\n")
        
        # Мониторинг
        while True:
            time.sleep(5)
            stats = nav_system.get_statistics()
            print(f"FPS: {stats['actual_fps']:.1f}, "
                  f"Latency: {stats['last_latency_ms']:.1f}ms, "
                  f"Frames: {stats['frames_processed']}")
            
            if stats['calibration']['calibrated']:
                print("✅ Калибровка завершена")
            
    except KeyboardInterrupt:
        print("\n\nОстановка системы...")
    finally:
        nav_system.stop()


if __name__ == '__main__':
    main()

