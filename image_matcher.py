"""
Модуль для сопоставления изображений и определения местоположения
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class ImageMatcher:
    """Класс для сопоставления малого изображения с большой картой"""
    
    def __init__(self):
        # Используем ORB детектор для совместимости (не требует лицензирования)
        # Можно использовать SIFT/SURF если нужна лучшая точность
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def find_location(self, large_map: np.ndarray, small_image: np.ndarray) -> Optional[Dict]:
        """
        Находит местоположение малого изображения на большой карте
        
        Args:
            large_map: Большая карта (BGR изображение)
            small_image: Малое изображение с камеры дрона (BGR изображение)
            
        Returns:
            Словарь с координатами центра, углом поворота и уверенностью, или None
        """
        # Конвертируем в серый для детекции признаков
        gray_map = cv2.cvtColor(large_map, cv2.COLOR_BGR2GRAY) if len(large_map.shape) == 3 else large_map
        gray_small = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY) if len(small_image.shape) == 3 else small_image
        
        # Детектируем ключевые точки и дескрипторы
        kp_map, des_map = self.orb.detectAndCompute(gray_map, None)
        kp_small, des_small = self.orb.detectAndCompute(gray_small, None)
        
        if des_map is None or des_small is None or len(des_map) < 4 or len(des_small) < 4:
            return None
        
        # Сопоставляем дескрипторы
        matches = self.matcher.knnMatch(des_small, des_map, k=2)
        
        # Фильтруем хорошие совпадения по методу Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        # Извлекаем координаты совпавших точек
        src_pts = np.float32([kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Вычисляем гомографию
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                  cv2.RANSAC, 5.0)
        except:
            return None
        
        if homography is None:
            return None
        
        # Получаем углы малого изображения
        h, w = gray_small.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Трансформируем углы на большую карту
        try:
            transformed_corners = cv2.perspectiveTransform(corners, homography)
        except:
            return None
        
        # Вычисляем центр малого изображения на карте
        center_x = int(np.mean(transformed_corners[:, 0, 0]))
        center_y = int(np.mean(transformed_corners[:, 0, 1]))
        
        # Вычисляем угол поворота из гомографии
        angle = np.arctan2(homography[1, 0], homography[0, 0]) * 180 / np.pi
        
        # Уверенность вычисляется на основе количества хороших совпадений
        confidence = len(good_matches) / max(len(kp_small), len(kp_map)) * 100
        
        return {
            'x': center_x,
            'y': center_y,
            'angle': float(angle),
            'confidence': float(confidence),
            'matches_count': len(good_matches),
            'corners': transformed_corners.reshape(-1, 2).tolist()
        }
    
    def visualize_match(self, large_map: np.ndarray, small_image: np.ndarray, 
                       match_result: Dict, draw_corners: bool = True) -> np.ndarray:
        """
        Визуализирует результат сопоставления
        
        Args:
            large_map: Большая карта
            small_image: Малое изображение
            match_result: Результат find_location
            draw_corners: Рисовать ли углы области
            
        Returns:
            Изображение с визуализацией
        """
        result = large_map.copy()
        
        if draw_corners and 'corners' in match_result:
            corners = np.array(match_result['corners'], dtype=np.int32)
            cv2.polylines(result, [corners], True, (0, 255, 0), 3)
            
            # Рисуем центр
            cv2.circle(result, (match_result['x'], match_result['y']), 10, (0, 0, 255), -1)
            cv2.circle(result, (match_result['x'], match_result['y']), 20, (0, 0, 255), 2)
        
        return result

