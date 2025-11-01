"""
Example usage of the visual positioning system.
"""

import numpy as np
import torch
from src.models import SiameseNetwork, TripletNetwork
from src.inference import ImageMatcher, CoordinateEstimator
from src.route_tracking import RouteManager, RoutePoint, Position, DeviationStatus


def example_model_usage():
    """Example of model usage."""
    print("=== Model Usage Example ===\n")
    
    # Creating model
    print("1. Creating SiameseNetwork model...")
    try:
        model = SiameseNetwork(
            backbone='resnet50',
            feature_dim=128,
            pretrained=True
        )
        print(f"   Model created: {model.__class__.__name__}")
        
        # Проверка forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        img1 = torch.randn(batch_size, 3, 512, 512)
        img2 = torch.randn(batch_size, 3, 512, 512)
        
        with torch.no_grad():
            similarity = model(img1, img2)
            print(f"   Input: {batch_size} image pairs 512x512")
            print(f"   Output similarity: {similarity.shape}")
        
        # Извлечение признаков
        print("\n3. Extracting features...")
        with torch.no_grad():
            features = model.extract_features(img1)
            print(f"   Input: {batch_size} images 512x512")
            print(f"   Output features: {features.shape}")
            print(f"   Feature dim: {features.shape[1]}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Note: This requires pre-trained weights and may not work without Internet")


def example_route_tracking():
    """Example of route tracking."""
    print("\n\n=== Route Tracking Example ===\n")
    
    # Creating route
    print("1. Creating route with 3 points...")
    route_points = [
        RoutePoint(lat=55.7558, lon=37.6173, tolerance=10.0),
        RoutePoint(lat=55.7658, lon=37.6273, tolerance=15.0),
        RoutePoint(lat=55.7758, lon=37.6373, tolerance=10.0),
    ]
    
    route_manager = RouteManager(
        route_points=route_points,
        max_deviation=50.0,
        minor_threshold=25.0
    )
    print(f"   Route created with {len(route_points)} points")
    
    # Testing positions
    print("\n2. Testing different positions...")
    
    test_positions = [
        (55.7558, 37.6173, "On route"),
        (55.7650, 37.6280, "Minor deviation"),
        (55.7700, 37.6300, "Major deviation"),
        (55.8000, 37.7000, "Off route"),
    ]
    
    for lat, lon, description in test_positions:
        position = Position(lat=lat, lon=lon, confidence=0.9)
        alert = route_manager.update_position(position)
        
        if alert:
            status_symbol = {
                DeviationStatus.ON_ROUTE: "[OK]",
                DeviationStatus.MINOR_DEVIATION: "[WARNING]",
                DeviationStatus.MAJOR_DEVIATION: "[ALERT]",
                DeviationStatus.OFF_ROUTE: "[CRITICAL]"
            }
            symbol = status_symbol.get(alert.status, "[?]")
            print(f"   {description}: {symbol} {alert.deviation_distance:.1f}m - {alert.message}")
        else:
            print(f"   {description}: [OK] On route")


def example_coordinate_estimation():
    """Example of coordinate estimation."""
    print("\n\n=== Coordinate Estimation Example ===\n")
    
    # Map metadata
    map_metadata = {
        'center_lat': 55.7558,
        'center_lon': 37.6173,
        'pixels_per_meter': 1.0,
        'image_size': (2048, 2048)
    }
    
    # Creating estimator
    estimator = CoordinateEstimator(map_metadata)
    
    # Test matches
    print("1. Test matches...")
    matches = [
        ((1024, 1024), 0.95),  # Center of map
        ((1000, 1000), 0.90),  # Near center
        ((1050, 1050), 0.85),  # Near center
    ]
    
    # Estimating position
    lat, lon, confidence = estimator.estimate_position(matches)
    
    print(f"   Average position: ({lat:.6f}, {lon:.6f})")
    print(f"   Confidence: {confidence:.2%}")


def main():
    """Main function to run examples."""
    print("Drone Visual Positioning System - Examples\n")
    print("=" * 70)
    
    try:
        # Example 1: Model usage
        example_model_usage()
        
        # Example 2: Route tracking
        example_route_tracking()
        
        # Example 3: Coordinate estimation
        example_coordinate_estimation()
        
        print("\n" + "=" * 70)
        print("\n[SUCCESS] All examples completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

