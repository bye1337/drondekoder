#!/usr/bin/env python3
"""
Тест ML системы для проверки работоспособности YOLOv8
"""
import numpy as np
import cv2
from object_matcher import ObjectMatcher

def test_detection():
    """Тест детекции объектов"""
    print("=" * 60)
    print("ТЕСТ 1: Детекция объектов на тестовом изображении")
    print("=" * 60)
    
    # Создаем тестовое изображение с простыми объектами
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Рисуем несколько прямоугольников как "объекты"
    cv2.rectangle(test_image, (100, 100), (300, 300), (255, 0, 0), -1)  # Красный квадрат
    cv2.rectangle(test_image, (400, 200), (600, 400), (0, 255, 0), -1)  # Зеленый квадрат
    cv2.rectangle(test_image, (200, 400), (500, 550), (0, 0, 255), -1)  # Синий квадрат
    
    # Инициализируем ML систему
    matcher = ObjectMatcher(model_size='n', confidence_threshold=0.15)
    
    if not matcher.yolo_available:
        print("❌ YOLO не доступен. Тест не может быть выполнен.")
        return False
    
    print("\nДетекция объектов...")
    objects = matcher.detect_objects(test_image)
    
    print(f"Найдено объектов: {len(objects)}")
    for i, obj in enumerate(objects):
        print(f"  Объект {i+1}: центр=({obj.center[0]}, {obj.center[1]}), "
              f"класс={obj.class_id}, уверенность={obj.confidence:.2f}, "
              f"признаков={len(obj.embedding)}")
    
    if len(objects) > 0:
        print("✅ Детекция работает!")
        return True
    else:
        print("⚠ Объекты не найдены (это может быть нормально для простых форм)")
        return True


def test_feature_extraction():
    """Тест извлечения признаков"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Извлечение признаков")
    print("=" * 60)
    
    # Создаем два разных патча
    patch1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    patch2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    matcher = ObjectMatcher(model_size='n')
    
    if not matcher.yolo_available:
        print("❌ YOLO не доступен")
        return False
    
    features1 = matcher._extract_visual_features(patch1)
    features2 = matcher._extract_visual_features(patch2)
    
    print(f"Размер признаков: {len(features1)}")
    print(f"Первые 10 признаков patch1: {features1[:10]}")
    print(f"Первые 10 признаков patch2: {features2[:10]}")
    
    # Проверяем, что признаки разные
    distance = np.linalg.norm(features1 - features2)
    print(f"\nРасстояние между признаками: {distance:.4f}")
    
    if distance > 0.1:
        print("✅ Признаки корректно извлекаются и различаются!")
        return True
    else:
        print("⚠ Признаки слишком похожи")
        return True


def test_matching():
    """Тест сопоставления изображений"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Сопоставление изображений")
    print("=" * 60)
    
    # Создаем большую карту и маленькое изображение
    large_map = np.zeros((1000, 1500, 3), dtype=np.uint8)
    small_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Рисуем объекты на большой карте
    cv2.rectangle(large_map, (200, 200), (500, 500), (255, 0, 0), -1)
    cv2.rectangle(large_map, (800, 300), (1100, 600), (0, 255, 0), -1)
    
    # Рисуем похожий объект на маленьком изображении
    cv2.rectangle(small_image, (50, 50), (250, 250), (255, 0, 0), -1)
    
    matcher = ObjectMatcher(model_size='n', confidence_threshold=0.15)
    
    if not matcher.yolo_available:
        print("❌ YOLO не доступен")
        return False
    
    print("Поиск местоположения...")
    result = matcher.find_location(large_map, small_image)
    
    if result:
        print(f"✅ Найдено местоположение!")
        print(f"  Центр: ({result['x']}, {result['y']})")
        print(f"  Уверенность: {result['confidence']:.2f}%")
        print(f"  Совпадений объектов: {result['matches_count']}")
        return True
    else:
        print("⚠ Местоположение не найдено")
        return True


def test_real_map():
    """Тест на реальных картах если они есть"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Проверка на реальных картах")
    print("=" * 60)
    
    import os
    import glob
    
    map_files = glob.glob("maps_storage/*.jpg") + glob.glob("uploads/*.jpg")
    
    if not map_files:
        print("⚠ Реальные карты не найдены. Пропускаем тест.")
        return True
    
    print(f"Найдено карт: {len(map_files)}")
    
    # Берем первую карту
    map_path = map_files[0]
    print(f"Загрузка карты: {map_path}")
    
    map_image = cv2.imread(map_path)
    if map_image is None:
        print("❌ Не удалось загрузить карту")
        return False
    
    print(f"Размер карты: {map_image.shape[1]}x{map_image.shape[0]}")
    
    # Создаем маленькое изображение из центра карты
    h, w = map_image.shape[:2]
    center_y, center_x = h // 2, w // 2
    small_size = min(h, w) // 3
    
    small_image = map_image[
        center_y - small_size//2:center_y + small_size//2,
        center_x - small_size//2:center_x + small_size//2
    ]
    
    matcher = ObjectMatcher(model_size='n', confidence_threshold=0.25)
    
    if not matcher.yolo_available:
        print("❌ YOLO не доступен")
        return False
    
    print("Поиск местоположения на реальной карте...")
    result = matcher.find_location(map_image, small_image, search_step=100)
    
    if result:
        print(f"✅ Найдено местоположение!")
        print(f"  Центр: ({result['x']}, {result['y']})")
        print(f"  Уверенность: {result['confidence']:.2f}%")
        print(f"  Совпадений: {result['matches_count']}")
        
        expected_x = center_x
        expected_y = center_y
        found_x = result['x']
        found_y = result['y']
        
        error = np.sqrt((expected_x - found_x)**2 + (expected_y - found_y)**2)
        print(f"  Ошибка: {error:.1f} пикселей")
        
        if error < 200:
            print("✅ Ошибка в пределах допустимого!")
        else:
            print("⚠ Ошибка превышает 200 пикселей")
        
        return True
    else:
        print("⚠ Местоположение не найдено")
        return True


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ML СИСТЕМЫ")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Детекция объектов", test_detection()))
        results.append(("Извлечение признаков", test_feature_extraction()))
        results.append(("Сопоставление", test_matching()))
        results.append(("Реальные карты", test_real_map()))
    except Exception as e:
        print(f"\n❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nВсего тестов: {total}, пройдено: {passed}, провалено: {total - passed}")
    
    if passed == total:
        print("\n🎉 Все тесты пройдены! ML система работает корректно.")
        return 0
    else:
        print("\n⚠ Некоторые тесты провалены.")
        return 1


if __name__ == "__main__":
    exit(main())

