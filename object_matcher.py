"""
ML система для детекции объектов и их сравнения на картах
Использует YOLOv8 для детекции объектов и векторное представление для поиска
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO/PyTorch не установлены. Установите: pip install ultralytics torch torchvision")


class ObjectDescriptor:
    """Класс для описания объекта на карте"""
    
    def __init__(self, bbox: Tuple[int, int, int, int], 
                 class_id: int, confidence: float, 
                 embedding: np.ndarray, image_patch: np.ndarray):
        """
        Args:
            bbox: (x1, y1, x2, y2) координаты bounding box
            class_id: ID класса объекта
            confidence: Уверенность детекции
            embedding: Векторное представление объекта
            image_patch: Патч изображения с объектом
        """
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.embedding = embedding
        self.image_patch = image_patch
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def distance(self, other: 'ObjectDescriptor') -> float:
        """Вычисляет косинусное расстояние между объектами"""
        return 1 - np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding) + 1e-8
        )


class ObjectMatcher:
    """
    ML система для поиска местоположения на карте по объектам
    Использует YOLOv8 для детекции объектов и векторное сравнение
    """
    
    def __init__(self, model_size: str = 'n', confidence_threshold: float = 0.25):
        """
        Args:
            model_size: Размер модели YOLO ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Порог уверенности детекции
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.yolo_available = YOLO_AVAILABLE
        self.yolo_model = None
        self.feature_extractor = None
        self.device = None
        self.transform = None
        
        if self.yolo_available:
            self._init_models()
        else:
            print("Предупреждение: YOLO не установлен. Используйте традиционный метод.")
    
    def _init_models(self):
        """Инициализирует модели YOLO и SentenceTransformers"""
        try:
            print("Инициализация ML моделей...")
            
            # Определяем устройство
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Устройство: {self.device}")
            
            # Загружаем YOLOv8
            model_path = f'yolov8{self.model_size}.pt'
            self.yolo_model = YOLO(model_path)
            print(f"✓ YOLOv8-{self.model_size} загружен")
            
            # Загружаем модель для извлечения признаков из патчей
            # Используем MobileNetV2 как легкий feature extractor
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            mobilenet = mobilenet_v2(weights=weights)
            # Удаляем последний классификационный слой
            self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
            print("✓ MobileNetV2 загружен для извлечения признаков")
            
            # Трансформации для предобработки
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✓ ML модели готовы к работе")
            
        except Exception as e:
            print(f"Ошибка инициализации моделей: {e}")
            self.yolo_available = False
    
    def detect_objects(self, image: np.ndarray) -> List[ObjectDescriptor]:
        """
        Детектирует объекты на изображении
        
        Args:
            image: Изображение (BGR)
            
        Returns:
            Список дескрипторов объектов
        """
        if not self.yolo_available or self.yolo_model is None:
            return []
        
        # Конвертируем BGR в RGB для YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детектируем объекты
        results = self.yolo_model(image_rgb, 
                                 conf=self.confidence_threshold,
                                 verbose=False)
        
        objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # Получаем координаты
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Получаем класс и уверенность
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Извлекаем патч изображения
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                # Извлекаем признаки патча
                patch_features = self._extract_visual_features(patch)
                
                objects.append(ObjectDescriptor(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    confidence=confidence,
                    embedding=patch_features,
                    image_patch=patch
                ))
        
        return objects
    
    def _extract_visual_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Извлекает визуальные признаки из патча
        
        Args:
            patch: Патч изображения
            
        Returns:
            Вектор признаков
        """
        # Используем комбинацию различных признаков
        features = []
        
        # 1. Гистограмма цветов в HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # 2. Край (границы объектов)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (patch.shape[0] * patch.shape[1])
        
        # 3. Текстура (вариация яркости)
        texture_variance = np.var(gray)
        
        # 4. Сверточные признаки через MobileNet (если доступны)
        if self.feature_extractor is not None and self.transform is not None:
            try:
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_tensor = self.transform(patch_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    features = self.feature_extractor(patch_tensor)
                    features = features.squeeze().cpu().numpy()
                    # Берем первые 256 признаков
                    conv_features = features.flatten()[:256]
            except Exception as e:
                conv_features = np.zeros(256)
        else:
            conv_features = np.zeros(256)
        
        # Объединяем все признаки
        features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten(),
            [edge_density, texture_variance],
            conv_features
        ])
        
        # Нормализуем
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def find_location(self, large_map: np.ndarray, small_image: np.ndarray,
                     search_step: int = 500, top_k: int = 5) -> Optional[Dict]:
        """
        Находит местоположение малого изображения на большой карте
        
        Args:
            large_map: Большая карта
            small_image: Малое изображение с камеры дрона
            search_step: Шаг сканирования карты (для больших карт)
            top_k: Количество лучших кандидатов для анализа
            
        Returns:
            Словарь с координатами и уверенностью
        """
        if not self.yolo_available:
            return None
        
        print(f"Детекция объектов на малом изображении...")
        small_objects = self.detect_objects(small_image)
        
        if len(small_objects) == 0:
            print("⚠ Объекты на малом изображении не найдены")
            print("🔄 Используем резервный ORB метод...")
            return self._find_location_fallback(large_map, small_image)
        
        print(f"✓ Найдено {len(small_objects)} объектов на малом изображении")
        
        # Для больших карт сканируем по частям
        map_h, map_w = large_map.shape[:2]
        small_h, small_w = small_image.shape[:2]
        
        # Определяем стратегию поиска в зависимости от размера карты
        if map_w > 5000 or map_h > 5000:
            result = self._search_in_large_map(large_map, small_objects, small_h, small_w, search_step)
        else:
            result = self._search_small_map(large_map, small_objects, small_h, small_w, top_k)
        
        # Если YOLO не сработал, пробуем ORB
        if result is None:
            print("⚠ YOLO не нашел совпадений")
            print("🔄 Используем резервный ORB метод...")
            return self._find_location_fallback(large_map, small_image)
        
        return result
    
    def _find_location_fallback(self, large_map: np.ndarray, small_image: np.ndarray) -> Optional[Dict]:
        """
        Резервный метод поиска с использованием ORB
        Используется когда YOLO не находит объекты
        """
        try:
            from image_matcher import ImageMatcher
            orb_matcher = ImageMatcher()
            return orb_matcher.find_location(large_map, small_image)
        except Exception as e:
            print(f"❌ Ошибка ORB метода: {e}")
            return None
    
    def _search_small_map(self, large_map: np.ndarray, small_objects: List[ObjectDescriptor],
                         small_h: int, small_w: int, top_k: int) -> Optional[Dict]:
        """Поиск на маленькой карте (< 5000x5000)"""
        print("Детекция объектов на большой карте...")
        large_objects = self.detect_objects(large_map)
        
        if len(large_objects) == 0:
            print("Объекты на большой карте не найдены")
            return None
        
        print(f"Найдено {len(large_objects)} объектов на большой карте")
        
        # Ищем совпадения объектов
        matches = self._find_object_matches(small_objects, large_objects, top_k)
        
        if len(matches) == 0:
            print("Совпадений объектов не найдено")
            return None
        
        # Вычисляем позицию на основе совпадений
        return self._estimate_position(matches, large_map.shape, small_h, small_w)
    
    def _search_in_large_map(self, large_map: np.ndarray, small_objects: List[ObjectDescriptor],
                            small_h: int, small_w: int, search_step: int) -> Optional[Dict]:
        """Поиск на большой карте (>= 5000x5000) с многоуровневым сканированием"""
        print(f"Большая карта: {large_map.shape[1]}x{large_map.shape[0]}")
        print("Многоуровневый поиск...")
        
        map_h, map_w = large_map.shape[:2]
        
        # Уровень 1: Грубое сканирование с большим шагом
        print("Уровень 1: Грубое сканирование...")
        best_candidates = []
        
        for y in range(0, map_h - small_h, search_step * 3):
            for x in range(0, map_w - small_w, search_step * 3):
                region = large_map[y:min(y + small_h, map_h), 
                                 x:min(x + small_w, map_w)]
                
                region_objects = self.detect_objects(region)
                if len(region_objects) > 0:
                    # Корректируем координаты обратно на большую карту
                    for obj in region_objects:
                        obj.bbox = (obj.bbox[0] + x, obj.bbox[1] + y,
                                   obj.bbox[2] + x, obj.bbox[3] + y)
                        obj.center = (obj.center[0] + x, obj.center[1] + y)
                    
                    matches = self._find_object_matches(small_objects, region_objects, top_k=3)
                    if len(matches) > 0:
                        confidence = sum(m.distance for m in matches) / len(matches)
                        best_candidates.append((x, y, confidence, matches))
        
        if len(best_candidates) == 0:
            print("Кандидатов не найдено")
            return None
        
        # Сортируем по уверенности
        best_candidates.sort(key=lambda x: x[2])
        best_candidates = best_candidates[:3]  # Топ-3
        
        # Уровень 2: Уточнение в окрестностях лучших кандидатов
        print(f"Уровень 2: Уточнение {len(best_candidates)} кандидатов...")
        refined_candidates = []
        
        for candidate_x, candidate_y, _, matches in best_candidates:
            # Сканируем окрестность с меньшим шагом
            refine_step = search_step // 2
            
            for y in range(max(0, candidate_y - refine_step), 
                          min(map_h - small_h, candidate_y + refine_step), 
                          refine_step // 2):
                for x in range(max(0, candidate_x - refine_step),
                              min(map_w - small_w, candidate_x + refine_step),
                              refine_step // 2):
                    
                    region = large_map[y:min(y + small_h, map_h),
                                     x:min(x + small_w, map_w)]
                    
                    region_objects = self.detect_objects(region)
                    if len(region_objects) > 0:
                        for obj in region_objects:
                            obj.bbox = (obj.bbox[0] + x, obj.bbox[1] + y,
                                       obj.bbox[2] + x, obj.bbox[3] + y)
                            obj.center = (obj.center[0] + x, obj.center[1] + y)
                        
                        new_matches = self._find_object_matches(small_objects, region_objects, top_k=5)
                        if len(new_matches) > 0:
                            confidence = sum(m.distance for m in new_matches) / len(new_matches)
                            refined_candidates.append((x, y, confidence, new_matches))
        
        if len(refined_candidates) == 0:
            # Используем результат грубого сканирования
            best_x, best_y, _, matches = best_candidates[0]
        else:
            refined_candidates.sort(key=lambda x: x[2])
            best_x, best_y, _, matches = refined_candidates[0]
        
        # Вычисляем финальную позицию
        result = self._estimate_position(matches, large_map.shape, small_h, small_w)
        if result:
            result['x'] = best_x + small_w // 2
            result['y'] = best_y + small_h // 2
        
        return result
    
    def _find_object_matches(self, small_objects: List[ObjectDescriptor],
                           large_objects: List[ObjectDescriptor],
                           top_k: int) -> List:
        """
        Находит совпадения объектов между малым и большим изображением
        
        Returns:
            Список кортежей (small_obj, large_obj, distance)
        """
        matches = []
        
        for small_obj in small_objects:
            for large_obj in large_objects:
                # Проверяем совпадение класса
                if small_obj.class_id != large_obj.class_id:
                    continue
                
                distance = small_obj.distance(large_obj)
                matches.append((small_obj, large_obj, distance))
        
        # Сортируем по расстоянию и берем топ-k
        matches.sort(key=lambda x: x[2])
        
        return matches[:top_k] if top_k > 0 else matches[:10]
    
    def _estimate_position(self, matches: List, map_shape: Tuple[int, int],
                          small_h: int, small_w: int) -> Optional[Dict]:
        """
        Оценивает позицию на основе совпадений объектов
        
        Args:
            matches: Список совпадений
            map_shape: Размеры карты
            small_h, small_w: Размеры малого изображения
            
        Returns:
            Результат с координатами
        """
        if len(matches) == 0:
            return None
        
        # Вычисляем среднее местоположение по совпадениям
        distances = [m[2] for m in matches]
        weights = [1 / (d + 0.1) for d in distances]  # Обратные веса
        total_weight = sum(weights)
        
        # Взвешенное среднее центров совпадений
        center_x = sum(large_obj.center[0] * w for _, large_obj, w in zip(matches, [m[1] for m in matches], weights)) / total_weight
        center_y = sum(large_obj.center[1] * w for _, large_obj, w in zip(matches, [m[1] for m in matches], weights)) / total_weight
        
        # Смещаем с учетом размера малого изображения
        center_x = center_x - small_w // 2
        center_y = center_y - small_h // 2
        
        # Ограничиваем границами карты
        center_x = max(small_w // 2, min(map_shape[1] - small_w // 2, center_x))
        center_y = max(small_h // 2, min(map_shape[0] - small_h // 2, center_y))
        
        # Вычисляем уверенность
        avg_distance = sum(distances) / len(distances)
        confidence = max(0, (1 - avg_distance) * 100)
        
        # Создаем углы области
        corners = [
            [int(center_x - small_w // 2), int(center_y - small_h // 2)],
            [int(center_x + small_w // 2), int(center_y - small_h // 2)],
            [int(center_x + small_w // 2), int(center_y + small_h // 2)],
            [int(center_x - small_w // 2), int(center_y + small_h // 2)]
        ]
        
        return {
            'x': int(center_x),
            'y': int(center_y),
            'angle': 0.0,  # Угол не определяется в данном методе
            'confidence': float(confidence),
            'matches_count': len(matches),
            'corners': corners,
            'detected_objects': len(matches)
        }

