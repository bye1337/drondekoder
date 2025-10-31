#!/usr/bin/env python3
"""
Реальный тест ML системы с загрузкой карты и поиском
"""
import cv2
import numpy as np
from object_matcher import ObjectMatcher

def test_with_uploaded_maps():
    """Тест с реальными картами из проекта"""
    print("=" * 60)
    print("РЕАЛЬНЫЙ ТЕСТ ML СИСТЕМЫ")
    print("=" * 60)
    
    import glob
    
    # Ищем карты
    map_files = glob.glob("maps_storage/*.jpg") + glob.glob("uploads/*.jpg")
    
    if not map_files:
        print("❌ Карты не найдены!")
        return False
    
    print(f"\nНайдено карт: {len(map_files)}")
    
    for map_path in map_files[:2]:  # Тестируем первые 2 карты
        print(f"\n{'='*60}")
        print(f"Карта: {map_path}")
        print('='*60)
        
        # Загружаем карту
        map_image = cv2.imread(map_path)
        if map_image is None:
            print("❌ Не удалось загрузить карту")
            continue
        
        print(f"Размер: {map_image.shape[1]}x{map_image.shape[0]}")
        
        # Инициализируем ML
        matcher = ObjectMatcher(model_size='n', confidence_threshold=0.2)
        
        # Детектируем объекты на всей карте
        print("\nДетекция объектов на карте...")
        objects = matcher.detect_objects(map_image)
        print(f"✓ Найдено объектов: {len(objects)}")
        
        if len(objects) > 0:
            print("\nОбнаруженные объекты:")
            class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 15: 'cat', 16: 'dog', 24: 'backpack', 25: 'umbrella'
            }
            
            for i, obj in enumerate(objects[:10]):  # Показываем первые 10
                class_name = class_names.get(obj.class_id, f'class_{obj.class_id}')
                print(f"  {i+1}. {class_name} - уверенность: {obj.confidence:.2f}, "
                      f"центр: ({obj.center[0]}, {obj.center[1]})")
        
        # Создаем маленькое изображение из центра карты
        h, w = map_image.shape[:2]
        center_y, center_x = h // 2, w // 2
        crop_size = min(h, w) // 3
        
        small_image = map_image[
            max(0, center_y - crop_size//2):min(h, center_y + crop_size//2),
            max(0, center_x - crop_size//2):min(w, center_x + crop_size//2)
        ]
        
        print(f"\nСоздано маленькое изображение: {small_image.shape[1]}x{small_image.shape[0]}")
        
        # Детектируем объекты на маленьком изображении
        small_objects = matcher.detect_objects(small_image)
        print(f"✓ Объектов на маленьком изображении: {len(small_objects)}")
        
        # Пробуем найти местоположение
        print("\nПоиск местоположения...")
        result = matcher.find_location(map_image, small_image, search_step=100)
        
        if result:
            print(f"✅ МЕСТОПОЛОЖЕНИЕ НАЙДЕНО!")
            print(f"   Центр: ({result['x']}, {result['y']})")
            print(f"   Ожидалось: ({center_x}, {center_y})")
            print(f"   Уверенность: {result['confidence']:.2f}%")
            print(f"   Совпадений объектов: {result['matches_count']}")
            
            error = np.sqrt((center_x - result['x'])**2 + (center_y - result['y'])**2)
            print(f"   Ошибка: {error:.1f} пикселей")
            
            if error < 100:
                print("   🎯 Отличная точность!")
            elif error < 300:
                print("   ✓ Хорошая точность")
            else:
                print("   ⚠ Точность можно улучшить")
        else:
            print("⚠ Местоположение не найдено")
            if len(objects) == 0:
                print("   Причина: на карте не найдено объектов YOLO")
            elif len(small_objects) == 0:
                print("   Причина: на маленьком изображении не найдено объектов")
            else:
                print("   Причина: объекты не совпадают")
    
    return True


if __name__ == "__main__":
    success = test_with_uploaded_maps()
    if success:
        print("\n" + "="*60)
        print("✓ Тест завершен")
        print("="*60)

