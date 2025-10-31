"""
Модуль для обработки видеопотока в реальном времени
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
from image_matcher import ImageMatcher
from route_monitor import RouteMonitor


class VideoProcessor:
    """Класс для обработки видеопотока"""
    
    def __init__(self, matcher: ImageMatcher, route_monitor: RouteMonitor):
        self.matcher = matcher
        self.route_monitor = route_monitor
        self.is_processing = False
        self.processing_thread = None
        self.frame_callback = None
        self.process_every_n_frames = 3  # Обрабатываем каждый N-й кадр для производительности
        self.frame_count = 0
        self.last_result = None
        
    def start_processing(self, video_source, frame_callback: Optional[Callable] = None):
        """
        Начинает обработку видеопотока
        
        Args:
            video_source: Источник видео (cv2.VideoCapture, файл или веб-камера)
            frame_callback: Функция обратного вызова для обработки каждого кадра
        """
        if self.is_processing:
            self.stop_processing()
        
        self.is_processing = True
        self.frame_callback = frame_callback
        self.frame_count = 0
        
        if isinstance(video_source, (str, int)):
            self.cap = cv2.VideoCapture(video_source)
        else:
            self.cap = video_source
            
        if not self.cap.isOpened():
            raise ValueError("Не удалось открыть видеопоток")
        
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Останавливает обработку видеопотока"""
        self.is_processing = False
        if hasattr(self, 'cap'):
            self.cap.release()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def _process_loop(self):
        """Основной цикл обработки кадров"""
        while self.is_processing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Обрабатываем каждый N-й кадр для производительности
            if self.frame_count % self.process_every_n_frames == 0:
                if self.frame_callback:
                    try:
                        self.frame_callback(frame.copy(), self.frame_count)
                    except Exception as e:
                        print(f"Ошибка обработки кадра: {e}")
            
            # Небольшая задержка для предотвращения перегрузки CPU
            time.sleep(0.01)
    
    def process_frame(self, frame: np.ndarray, large_map: Optional[np.ndarray] = None) -> Optional[dict]:
        """
        Обрабатывает один кадр видео
        
        Args:
            frame: Кадр видео
            large_map: Большая карта (если None, используется из route_monitor)
            
        Returns:
            Результат обработки или None
        """
        if large_map is None:
            return None
        
        result = self.matcher.find_location(large_map, frame)
        
        if result:
            self.last_result = result
            position = (result['x'], result['y'])
            
            # Проверяем отклонение от маршрута
            if self.route_monitor.waypoints:
                deviation_info = self.route_monitor.check_deviation(position)
                result['deviation'] = deviation_info
            else:
                result['deviation'] = {
                    'is_on_route': True,
                    'deviation': 0.0,
                    'message': 'Маршрут не задан'
                }
        
        return result
    
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
    
    def draw_result_on_frame(self, frame: np.ndarray, result: dict, 
                           draw_info: bool = True) -> np.ndarray:
        """
        Рисует результат обработки на кадре
        
        Args:
            frame: Кадр видео
            result: Результат обработки
            draw_info: Рисовать ли текстовую информацию
            
        Returns:
            Кадр с нарисованными результатами
        """
        output_frame = frame.copy()
        
        if result and 'position' in result:
            pos = result['position']
            x, y = int(pos['x']), int(pos['y'])
            
            # Рисуем точку местоположения
            cv2.circle(output_frame, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(output_frame, (x, y), 20, (0, 255, 0), 2)
            
            if draw_info:
                # Добавляем текстовую информацию
                info_text = [
                    f"Confidence: {pos['confidence']:.1f}%",
                    f"Matches: {pos['matches_count']}",
                    f"Angle: {pos['angle']:.1f}°"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(output_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(output_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_offset += 25
        
        # Рисуем информацию об отклонении
        if result and 'deviation' in result:
            deviation = result['deviation']
            if not deviation.get('is_on_route', True):
                text = f"Deviation: {deviation['deviation']:.1f}px"
                color = (0, 0, 255)  # Красный
                cv2.putText(output_frame, text, (10, output_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        return output_frame

