"""
Скрипт для создания синтетического датасета из большого изображения карты.
"""

import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def augment_drone_image(image: np.ndarray) -> np.ndarray:
    """
    Аугментация изображения для имитации снимка дрона.
    
    Args:
        image: Входное изображение [H, W, 3]
        
    Returns:
        Аугментированное изображение
    """
    # Случайный поворот
    angle = np.random.randint(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Случайная яркость/контраст
    brightness = np.random.randint(-20, 20)
    contrast = np.random.uniform(0.9, 1.1)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    # Случайный цветовой сдвиг
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + np.random.randint(-10, 10), 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] + np.random.randint(-20, 20), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + np.random.randint(-20, 20), 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Добавление шума
    noise = np.random.randn(*image.shape) * 5
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    return image


def create_synthetic_pairs(
    map_image_path: str,
    output_dir: str,
    num_pairs: int = 100,
    patch_size: int = 512,
    map_keep_original: bool = False
):
    """
    Создание синтетических пар изображений из карты.
    
    Args:
        map_image_path: Путь к большой карте
        output_dir: Директория для сохранения пар
        num_pairs: Количество пар для создания
        patch_size: Размер вырезаемого фрагмента
        map_keep_original: Сохранять ли исходную карту без изменений
    """
    # Загрузка карты
    print(f"Loading map image from {map_image_path}...")
    map_img = cv2.imread(map_image_path)
    if map_img is None:
        raise ValueError(f"Could not load image from {map_image_path}")
    
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    h, w = map_img.shape[:2]
    
    print(f"Map image size: {w}x{h}")
    
    # Проверка размера
    if h < patch_size or w < patch_size:
        raise ValueError(f"Map image too small: {w}x{h}, need at least {patch_size}x{patch_size}")
    
    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_pairs} image pairs...")
    
    for i in tqdm(range(num_pairs)):
        # Случайная позиция для вырезки
        max_y = h - patch_size
        max_x = w - patch_size
        
        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)
        
        # Вырезка фрагмента
        patch = map_img[y:y+patch_size, x:x+patch_size]
        
        # Применение аугментаций для drone изображения
        drone_patch = augment_drone_image(patch.copy())
        
        # Сохранение map изображения (полная карта)
        if not map_keep_original:
            # Обрезаем map до размера, чтобы показать локальную область
            # Можно просто сохранить весь map_img или обрезать его
            map_to_save = map_img
        else:
            map_to_save = map_img
        
        # Сохранение
        map_filename = os.path.join(output_dir, f"image_{i:04d}_map.jpg")
        drone_filename = os.path.join(output_dir, f"image_{i:04d}_drone.jpg")
        
        cv2.imwrite(map_filename, cv2.cvtColor(map_to_save, cv2.COLOR_RGB2BGR))
        cv2.imwrite(drone_filename, cv2.cvtColor(drone_patch, cv2.COLOR_RGB2BGR))
    
    print(f"\nDataset created successfully in {output_dir}")
    print(f"Created {num_pairs} pairs of images")


def main():
    parser = argparse.ArgumentParser(
        description='Create synthetic dataset from map images'
    )
    parser.add_argument(
        '--map-image',
        type=str,
        required=True,
        help='Path to large map image'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/pairs',
        help='Output directory for pairs'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=100,
        help='Number of image pairs to create'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=512,
        help='Size of drone image patches'
    )
    parser.add_argument(
        '--keep-original-map',
        action='store_true',
        help='Keep original full map for each pair'
    )
    
    args = parser.parse_args()
    
    create_synthetic_pairs(
        map_image_path=args.map_image,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        patch_size=args.patch_size,
        map_keep_original=args.keep_original_map
    )


if __name__ == '__main__':
    main()

