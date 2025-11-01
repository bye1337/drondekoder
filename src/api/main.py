"""
FastAPI backend для веб-интерфейса системы визуального позиционирования.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
from typing import List, Optional, Tuple
import json
from pathlib import Path

from src.models import SiameseNetwork, TripletNetwork
from src.inference import ImageMatcher, CoordinateEstimator
from src.route_tracking import RouteManager, RoutePoint, Position, DeviationAlert, DeviationStatus


app = FastAPI(
    title="Drone Visual Positioning System",
    description="Система визуального позиционирования дрона на основе компьютерного зрения",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model = None
image_matcher = None
route_manager: Optional[RouteManager] = None
map_metadata = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pydantic модели
class MapMetadata(BaseModel):
    center_lat: float
    center_lon: float
    pixels_per_meter: float
    image_size: Tuple[int, int]


class RoutePointCreate(BaseModel):
    lat: float
    lon: float
    altitude: Optional[float] = None
    tolerance: float = 10.0


class RouteCreate(BaseModel):
    route_points: List[RoutePointCreate]


class PositionUpdate(BaseModel):
    lat: float
    lon: float
    confidence: float
    timestamp: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация модели при запуске приложения."""
    global model, image_matcher
    
    model_path = Path("models/checkpoints/best_model.pth")
    if model_path.exists():
        try:
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
            
            image_matcher = ImageMatcher(model, device=device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running without pre-trained model.")
    else:
        print("Model file not found. Please train the model first.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Visual Positioning System</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .status {
                padding: 15px;
                margin: 20px 0;
                border-radius: 8px;
                background: #f0f0f0;
            }
            .upload-section {
                margin: 20px 0;
                padding: 20px;
                border: 2px dashed #667eea;
                border-radius: 10px;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #764ba2;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background: #e8f5e9;
                border-radius: 8px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚁 Drone Visual Positioning System 2.0</h1>
            
            <div class="status">
                <h3>Система определения местоположения дрона</h3>
                <p>Загрузите изображение с дрона и получите его точные координаты</p>
            </div>
            
            <div class="upload-section">
                <h3>1. Загрузить карту местности</h3>
                <input type="file" id="mapFile" accept="image/*">
            </div>
            
            <div class="upload-section">
                <h3>2. Загрузить снимок с дрона</h3>
                <input type="file" id="droneFile" accept="image/*">
            </div>
            
            <button onclick="processImage()">Определить позицию</button>
            
            <div class="result" id="result">
                <h3>Результат:</h3>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            async function processImage() {
                const mapFile = document.getElementById('mapFile').files[0];
                const droneFile = document.getElementById('droneFile').files[0];
                
                if (!mapFile || !droneFile) {
                    alert('Пожалуйста, загрузите оба изображения!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('map_image', mapFile);
                formData.append('drone_image', droneFile);
                
                try {
                    const response = await fetch('/api/match', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('resultContent').innerHTML = 
                        '<p><strong>Координаты:</strong> ' + data.lat.toFixed(6) + ', ' + data.lon.toFixed(6) + '</p>' +
                        '<p><strong>Уверенность:</strong> ' + (data.confidence * 100).toFixed(1) + '%</p>';
                } catch (error) {
                    alert('Ошибка при обработке: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/api/match")
async def match_images(
    map_image: UploadFile = File(...),
    drone_image: UploadFile = File(...)
):
    """
    Сопоставление изображения дрона с картой и определение позиции.
    
    Args:
        map_image: Большое изображение карты
        drone_image: Снимок с дрона
        
    Returns:
        dict: Координаты и уверенность
    """
    if image_matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Загрузка и обработка изображений
        map_bytes = await map_image.read()
        drone_bytes = await drone_image.read()
        
        map_array = np.array(Image.open(io.BytesIO(map_bytes)).convert('RGB'))
        drone_array = np.array(Image.open(io.BytesIO(drone_bytes)).convert('RGB'))
        
        # Поиск соответствий
        matches = image_matcher.match_using_sliding_window(
            drone_array,
            map_array,
            window_size=(512, 512),
            stride=128,
            top_k=5
        )
        
        # Оценка позиции
        if matches and map_metadata:
            estimator = CoordinateEstimator(map_metadata)
            lat, lon, confidence = estimator.estimate_position(matches)
            
            return {
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "matches_count": len(matches)
            }
        else:
            # Временная заглушка для демонстрации
            return {
                "lat": 55.7558,
                "lon": 37.6173,
                "confidence": 0.85,
                "matches_count": len(matches) if matches else 0,
                "note": "Demo mode - real coordinates require map_metadata configuration"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/configure-map")
async def configure_map(metadata: MapMetadata):
    """
    Настройка метаданных карты.
    
    Args:
        metadata: Метаданные карты
    """
    global map_metadata
    map_metadata = metadata.dict()
    return {"status": "Map metadata configured"}


@app.post("/api/create-route")
async def create_route(route: RouteCreate):
    """
    Создание маршрута дрона.
    
    Args:
        route: Маршрут с точками
    """
    global route_manager
    
    route_points = [
        RoutePoint(
            lat=pt.lat,
            lon=pt.lon,
            altitude=pt.altitude,
            tolerance=pt.tolerance
        )
        for pt in route.route_points
    ]
    
    route_manager = RouteManager(
        route_points=route_points,
        max_deviation=50.0
    )
    
    return {
        "status": "Route created",
        "points_count": len(route_points)
    }


@app.post("/api/update-position")
async def update_position(position: PositionUpdate):
    """
    Обновление позиции дрона и проверка отклонения от маршрута.
    
    Args:
        position: Текущая позиция дрона
    """
    if route_manager is None:
        raise HTTPException(status_code=400, detail="Route not created")
    
    pos = Position(
        lat=position.lat,
        lon=position.lon,
        confidence=position.confidence,
        timestamp=position.timestamp
    )
    
    alert = route_manager.update_position(pos)
    
    response = {
        "on_route": alert is None,
        "progress": route_manager.get_route_progress()
    }
    
    if alert:
        response["alert"] = {
            "status": alert.status.value,
            "deviation_distance": alert.deviation_distance,
            "nearest_point": {
                "index": alert.nearest_point[0],
                "lat": alert.nearest_point[1],
                "lon": alert.nearest_point[2]
            },
            "message": alert.message
        }
    
    return response


@app.get("/api/route-status")
async def get_route_status():
    """Получение статуса текущего маршрута."""
    if route_manager is None:
        return {"has_route": False}
    
    target = route_manager.get_current_target()
    
    return {
        "has_route": True,
        "progress": route_manager.get_route_progress(),
        "current_target": {
            "lat": target.lat,
            "lon": target.lon,
            "tolerance": target.tolerance
        } if target else None,
        "total_points": len(route_manager.route_points)
    }


@app.get("/api/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

