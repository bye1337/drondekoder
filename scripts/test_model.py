"""
Скрипт для тестирования обученной модели.
"""

import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models import SiameseNetwork, TripletNetwork
from src.inference import ImageMatcher, CoordinateEstimator


def load_test_model(model_path: str, device='cuda'):
    """Загрузка обученной модели."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    model = SiameseNetwork(
        backbone=model_config.get('backbone', 'resnet50'),
        feature_dim=model_config.get('feature_dim', 128),
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def test_single_pair(model, map_image_path: str, drone_image_path: str):
    """Тестирование на одной паре изображений."""
    # Загрузка изображений
    map_img = np.array(Image.open(map_image_path).convert('RGB'))
    drone_img = np.array(Image.open(drone_image_path).convert('RGB'))
    
    # Создание matcher
    matcher = ImageMatcher(model, device='cpu')
    
    # Поиск соответствий
    matches = matcher.match_using_sliding_window(
        drone_img,
        map_img,
        window_size=(512, 512),
        stride=128,
        top_k=5
    )
    
    print(f"\nFound {len(matches)} matches:")
    for i, ((x, y), similarity) in enumerate(matches):
        print(f"  {i+1}. Position: ({x}, {y}), Similarity: {similarity:.4f}")
    
    return matches


def test_batch_inference(model, test_dir: str, num_tests: int = 10):
    """Batch тестирование."""
    test_dir = Path(test_dir)
    
    map_files = sorted(test_dir.glob("*_map.jpg"))
    
    if not map_files:
        print("No test images found!")
        return
    
    print(f"Testing on {min(num_tests, len(map_files))} pairs...")
    
    for i, map_file in enumerate(map_files[:num_tests]):
        drone_file = test_dir / map_file.name.replace("_map.jpg", "_drone.jpg")
        
        if drone_file.exists():
            print(f"\nTest {i+1}: {map_file.name}")
            try:
                matches = test_single_pair(model, str(map_file), str(drone_file))
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--map-image',
        type=str,
        help='Path to map image for single test'
    )
    parser.add_argument(
        '--drone-image',
        type=str,
        help='Path to drone image for single test'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Directory with test pairs for batch testing'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=10,
        help='Number of tests for batch mode'
    )
    
    args = parser.parse_args()
    
    # Загрузка модели
    print("Loading model...")
    model, checkpoint = load_test_model(args.model)
    print(f"Model loaded! Epoch: {checkpoint['epoch']}, Best Acc: {checkpoint['best_acc']:.4f}")
    
    # Тестирование
    if args.map_image and args.drone_image:
        # Одиночное тестирование
        test_single_pair(model, args.map_image, args.drone_image)
    elif args.test_dir:
        # Batch тестирование
        test_batch_inference(model, args.test_dir, args.num_tests)
    else:
        print("Please provide either --map-image/--drone-image or --test-dir")
        parser.print_help()


if __name__ == '__main__':
    main()

