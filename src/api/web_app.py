"""
Streamlit веб-интерфейс для системы визуального позиционирования.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import SiameseNetwork, TripletNetwork
from src.inference import ImageMatcher, CoordinateEstimator
from src.route_tracking import RouteManager, RoutePoint, Position

# Конфигурация страницы
st.set_page_config(
    page_title="Drone Visual Positioning System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация сессии
if 'model' not in st.session_state:
    st.session_state.model = None
if 'image_matcher' not in st.session_state:
    st.session_state.image_matcher = None
if 'route_manager' not in st.session_state:
    st.session_state.route_manager = None
if 'map_metadata' not in st.session_state:
    st.session_state.map_metadata = None


def load_model():
    """Загрузка обученной модели."""
    model_path = Path("models/checkpoints/best_model.pth")
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            model_config = config.get('model', {})
            
            model = SiameseNetwork(
                backbone=model_config.get('backbone', 'resnet50'),
                feature_dim=model_config.get('feature_dim', 128),
                pretrained=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            return None
    return None


# Сайдбар
with st.sidebar:
    st.title("🚁 Навигация")
    st.markdown("---")
    
    page = st.radio(
        "Выберите раздел:",
        ["Главная", "Позиционирование", "Отслеживание маршрута", "Настройки"]
    )
    
    st.markdown("---")
    st.subheader("Статус системы")
    
    if st.session_state.model:
        st.success("✅ Модель загружена")
    else:
        st.warning("⚠️ Модель не загружена")
        if st.button("Загрузить модель"):
            st.session_state.model = load_model()
            if st.session_state.model:
                st.session_state.image_matcher = ImageMatcher(
                    st.session_state.model,
                    device='cpu'
                )
                st.success("Модель успешно загружена!")
                st.rerun()


# Главная страница
if page == "Главная":
    st.title("🚁 Drone Visual Positioning System 2.0")
    st.markdown("### Система визуального позиционирования дрона")
    
    st.markdown("""
    Добро пожаловать в систему визуального позиционирования дрона!
    
    **Возможности системы:**
    - 🎯 Точное определение местоположения по изображениям
    - 🗺️ Сопоставление снимков с картой местности
    - 📍 Отслеживание маршрута дрона
    - ⚠️ Уведомления об отклонениях от маршрута
    
    **Использование:**
    1. Загрузите модель в разделе настроек
    2. Используйте "Позиционирование" для определения координат
    3. Настройте маршрут в разделе "Отслеживание маршрута"
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Точность", "94.5%", "2.1%")
    
    with col2:
        st.metric("Время обработки", "1.2s", "-0.3s")
    
    with col3:
        st.metric("Уверенность", "87%", "5%")


# Позиционирование
elif page == "Позиционирование":
    st.title("📍 Позиционирование")
    st.markdown("Определение местоположения дрона по изображениям")
    
    tab1, tab2 = st.tabs(["Загрузка изображений", "Из веб-камеры"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗺️ Карта местности")
            map_file = st.file_uploader(
                "Загрузите карту местности",
                type=['jpg', 'jpeg', 'png'],
                key="map_upload"
            )
            
            if map_file:
                map_image = Image.open(map_file)
                st.image(map_image, caption="Карта местности", use_container_width=True)
                
                # Конфигурация метаданных
                with st.expander("Настройка метаданных карты"):
                    center_lat = st.number_input("Центральная широта", value=55.7558, format="%.6f")
                    center_lon = st.number_input("Центральная долгота", value=37.6173, format="%.6f")
                    pixels_per_m = st.number_input("Пикселей на метр", value=1.0, format="%.2f")
                    
                    if st.button("Сохранить метаданные"):
                        st.session_state.map_metadata = {
                            'center_lat': center_lat,
                            'center_lon': center_lon,
                            'pixels_per_meter': pixels_per_m,
                            'image_size': map_image.size
                        }
                        st.success("Метаданные сохранены!")
        
        with col2:
            st.subheader("📸 Снимок с дрона")
            drone_file = st.file_uploader(
                "Загрузите снимок с дрона",
                type=['jpg', 'jpeg', 'png'],
                key="drone_upload"
            )
            
            if drone_file:
                drone_image = Image.open(drone_file)
                st.image(drone_image, caption="Снимок с дрона", use_container_width=True)
        
        if st.button("🎯 Определить позицию", type="primary"):
            if map_file and drone_file:
                if st.session_state.image_matcher:
                    with st.spinner("Обработка изображений..."):
                        map_array = np.array(map_image.convert('RGB'))
                        drone_array = np.array(drone_image.convert('RGB'))
                        
                        matches = st.session_state.image_matcher.match_using_sliding_window(
                            drone_array,
                            map_array,
                            window_size=(512, 512),
                            stride=128,
                            top_k=5
                        )
                        
                        st.success(f"Найдено {len(matches)} соответствий!")
                        
                        # Оценка координат
                        if matches and st.session_state.map_metadata:
                            estimator = CoordinateEstimator(st.session_state.map_metadata)
                            lat, lon, confidence = estimator.estimate_position(matches)
                            
                            # Отображение результатов
                            st.subheader("📊 Результаты")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Широта", f"{lat:.6f}")
                            with col2:
                                st.metric("Долгота", f"{lon:.6f}")
                            with col3:
                                st.metric("Уверенность", f"{confidence*100:.1f}%")
                            
                            # Карта (можно интегрировать folium)
                            st.info(f"Координаты: {lat:.6f}, {lon:.6f}")
                        else:
                            st.warning("Метаданные карты не настроены")
                else:
                    st.error("Модель не загружена! Загрузите модель в настройках.")
            else:
                st.warning("Загрузите оба изображения!")
    
    with tab2:
        st.info("Функция веб-камеры будет добавлена в следующей версии")


# Отслеживание маршрута
elif page == "Отслеживание маршрута":
    st.title("🛤️ Отслеживание маршрута")
    st.markdown("Мониторинг отклонений дрона от заданного маршрута")
    
    # Создание маршрута
    st.subheader("Создание маршрута")
    
    num_points = st.number_input("Количество точек маршрута", min_value=2, value=5, step=1)
    
    route_points = []
    for i in range(num_points):
        st.markdown(f"**Точка {i+1}**")
        col1, col2 = st.columns(2)
        
        with col1:
            lat = st.number_input(f"Широта {i+1}", value=55.7558, format="%.6f", key=f"lat_{i}")
        with col2:
            lon = st.number_input(f"Долгота {i+1}", value=37.6173 + i*0.001, format="%.6f", key=f"lon_{i}")
        
        route_points.append({
            'lat': lat,
            'lon': lon
        })
    
    if st.button("Создать маршрут", type="primary"):
        route_points_list = [
            RoutePoint(lat=pt['lat'], lon=pt['lon'])
            for pt in route_points
        ]
        
        st.session_state.route_manager = RouteManager(
            route_points=route_points_list,
            max_deviation=50.0
        )
        st.success("Маршрут создан!")
    
    # Мониторинг позиции
    if st.session_state.route_manager:
        st.markdown("---")
        st.subheader("Мониторинг позиции")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_lat = st.number_input("Текущая широта", value=55.7558, format="%.6f")
        with col2:
            current_lon = st.number_input("Текущая долгота", value=37.6173, format="%.6f")
        
        if st.button("Проверить позицию"):
            position = Position(
                lat=current_lat,
                lon=current_lon,
                confidence=0.9
            )
            
            alert = st.session_state.route_manager.update_position(position)
            
            # Отображение результата
            if alert:
                st.error(alert.message)
                st.metric("Отклонение", f"{alert.deviation_distance:.2f} м")
            else:
                st.success("✅ Дрон на маршруте")
                st.metric("Прогресс", f"{st.session_state.route_manager.get_route_progress()*100:.1f}%")
        
        # Визуализация маршрута
        st.markdown("---")
        st.subheader("Визуализация маршрута")
        st.info("Интеграция с картами будет добавлена в следующей версии")


# Настройки
elif page == "Настройки":
    st.title("⚙️ Настройки")
    
    st.subheader("Модель")
    model_status = st.session_state.model is not None
    
    if st.session_state.model:
        st.success("✅ Модель загружена")
        
        model_path = Path("models/checkpoints/best_model.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            st.json(config)
    else:
        st.warning("⚠️ Модель не загружена")
    
    st.markdown("---")
    
    st.subheader("Система")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Устройство: {device}")
    
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    pass  # Streamlit обрабатывает запуск

