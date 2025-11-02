"""
Модуль для обработки видеопотока в реальном времени с использованием Optical Flow
для стабилизации позиции БАС
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
from stabilization_processor import (
    PositionStabilizer,
    DualCameraStabilizer,
    visualize_stabilization
)


class VideoProcessor:
    """Класс для обработки видеопотока с стабилизацией позиции"""
    
    def __init__(
        self,
        use_dual_camera: bool = False,
        primary_method: str = 'lucas_kanade',
        secondary_method: str = 'farneback',
        route_monitor=None
    ):
        """
        Args:
            use_dual_camera: Использовать ли две камеры
            primary_method: Метод для основной камеры ('lucas_kanade' или 'farneback')
            secondary_method: Метод для второй камеры
            route_monitor: Монитор маршрута (опционально)
        """
        self.use_dual_camera = use_dual_camera
        self.route_monitor = route_monitor
        
        # Инициализация стабилизатора
        if use_dual_camera:
            self.stabilizer = DualCameraStabilizer(
                primary_method=primary_method,
                secondary_method=secondary_method
            )
        else:
            self.stabilizer = PositionStabilizer(method=primary_method)
        
        self.is_processing = False
        self.processing_thread = None
        self.frame_callback = None
        self.frame_count = 0
        self.last_result = None
        
        # Для статистики
        self.fps_history = []
        self.processing_times = []
        
    def start_processing(
        self,
        primary_video_source,
        secondary_video_source: Optional = None,
        frame_callback: Optional[Callable] = None
    ):
        """
        Начинает обработку видеопотока
        
        Args:
            primary_video_source: Источник видео основной камеры
            secondary_video_source: Источник видео второй камеры (если use_dual_camera=True)
            frame_callback: Функция обратного вызова для обработки каждого кадра
        """
        if self.is_processing:
            self.stop_processing()
        
        self.is_processing = True
        self.frame_callback = frame_callback
        self.frame_count = 0
        
        # Открываем основную камеру
        if isinstance(primary_video_source, (str, int)):
            self.primary_cap = cv2.VideoCapture(primary_video_source)
        else:
            self.primary_cap = primary_video_source
            
        if not self.primary_cap.isOpened():
            raise ValueError("Не удалось открыть видеопоток основной камеры")
        
        # Открываем вторую камеру, если нужно
        if self.use_dual_camera and secondary_video_source is not None:
            if isinstance(secondary_video_source, (str, int)):
                self.secondary_cap = cv2.VideoCapture(secondary_video_source)
            else:
                self.secondary_cap = secondary_video_source
        else:
            self.secondary_cap = None
        
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Останавливает обработку видеопотока"""
        self.is_processing = False
        
        if hasattr(self, 'primary_cap'):
            self.primary_cap.release()
        if hasattr(self, 'secondary_cap') and self.secondary_cap is not None:
            self.secondary_cap.release()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def _process_loop(self):
        """Основной цикл обработки кадров"""
        while self.is_processing:
            ret1, primary_frame = self.primary_cap.read()
            if not ret1:
                break
            
            # Получаем кадр со второй камеры, если используется
            secondary_frame = None
            if self.secondary_cap is not None and self.use_dual_camera:
                ret2, secondary_frame = self.secondary_cap.read()
                if not ret2:
                    secondary_frame = None  # Продолжаем без второй камеры
            
            self.frame_count += 1
            start_time = time.time()
            
            # Обрабатываем кадр
            try:
                if self.use_dual_camera:
                    result = self.stabilizer.update(
                        primary_frame=primary_frame,
                        secondary_frame=secondary_frame
                    )
                else:
                    result = self.stabilizer.update(primary_frame)
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                if result:
                    result['processing_time_ms'] = processing_time * 1000
                    result['fps'] = 1.0 / processing_time if processing_time > 0 else 0
                    result['frame_number'] = self.frame_count
                    
                    # Добавляем метрики стабильности
                    if hasattr(self.stabilizer, 'get_stability_metrics'):
                        stability = self.stabilizer.get_stability_metrics()
                        result['stability'] = stability
                    elif hasattr(self.stabilizer, 'primary_stabilizer'):
                        stability = self.stabilizer.primary_stabilizer.get_stability_metrics()
                        result['stability'] = stability
                    
                    self.last_result = result
                    
                    # Вызываем callback, если установлен
                    if self.frame_callback:
                        self.frame_callback(result, primary_frame, secondary_frame)
                        
            except Exception as e:
                print(f"Ошибка обработки кадра {self.frame_count}: {e}")
            
            # Небольшая задержка для предотвращения перегрузки CPU
            time.sleep(0.001)
    
    def process_frame(
        self,
        primary_frame: np.ndarray,
        secondary_frame: Optional[np.ndarray] = None
    ) -> Optional[dict]:
        """
        Обрабатывает один кадр видео
        
        Args:
            primary_frame: Кадр с основной камеры
            secondary_frame: Кадр со второй камеры (опционально)
            
        Returns:
            Результат обработки с позицией, смещением, скоростью и уверенностью
        """
        try:
            if self.use_dual_camera:
                result = self.stabilizer.update(
                    primary_frame=primary_frame,
                    secondary_frame=secondary_frame
                )
            else:
                result = self.stabilizer.update(primary_frame)
            
            if result:
                self.last_result = result
                
                # Добавляем метрики стабильности
                if hasattr(self.stabilizer, 'get_stability_metrics'):
                    result['stability'] = self.stabilizer.get_stability_metrics()
                elif hasattr(self.stabilizer, 'primary_stabilizer'):
                    result['stability'] = self.stabilizer.primary_stabilizer.get_stability_metrics()
                
                # Проверяем отклонение от маршрута, если маршрут установлен
                if self.route_monitor and self.route_monitor.waypoints:
                    position = result.get('position', [0, 0])
                    if position:
                        deviation_info = self.route_monitor.check_deviation(
                            (position[0], position[1])
                        )
                        result['deviation'] = deviation_info
                else:
                    result['deviation'] = {
                        'is_on_route': True,
                        'deviation': 0.0,
                        'message': 'Маршрут не задан'
                    }
            
            return result
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return None
    
    def encode_frame_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """
        Кодирует кадр в JPEG формат
        
        Args:
            frame: Кадр видео
            quality: Качество JPEG (0-100)
            
        Returns:
            Закодированный кадр в формате bytes
        """
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if success:
            return buffer.tobytes()
        return None
    
    def draw_result_on_frame(
        self,
        frame: np.ndarray,
        result: dict,
        draw_info: bool = True,
        draw_grid: bool = False
    ) -> np.ndarray:
        """
        Рисует результат обработки на кадре
        
        Args:
            frame: Кадр видео
            result: Результат обработки стабилизатором
            draw_info: Рисовать ли текстовую информацию
            draw_grid: Рисовать ли сетку для оценки стабильности
            
        Returns:
            Кадр с нарисованными результатами
        """
        return visualize_stabilization(
            frame,
            result,
            draw_tracks=draw_info,
            draw_grid=draw_grid
        )
    
    def get_statistics(self) -> dict:
        """Возвращает статистику обработки"""
        if not self.processing_times:
            return {
                'avg_fps': 0.0,
                'avg_processing_time_ms': 0.0,
                'total_frames': self.frame_count
            }
        
        avg_time = np.mean(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_fps': avg_fps,
            'avg_processing_time_ms': avg_time * 1000,
            'min_processing_time_ms': np.min(self.processing_times) * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'total_frames': self.frame_count,
            'last_result': self.last_result is not None
        }
    
    def reset_stabilizer(self):
        """Сбрасывает стабилизатор"""
        if hasattr(self.stabilizer, 'reset'):
            self.stabilizer.reset()
        elif hasattr(self.stabilizer, 'primary_stabilizer'):
            self.stabilizer.primary_stabilizer.reset()
            if hasattr(self.stabilizer, 'secondary_stabilizer') and self.stabilizer.secondary_stabilizer:
                self.stabilizer.secondary_stabilizer.reset()
