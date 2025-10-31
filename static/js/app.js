// Глобальное состояние приложения
let state = {
    mapImage: null,
    mapCanvas: null,
    mapContext: null,
    waypoints: [],
    currentPosition: null,
    routeSet: false,
    videoStream: null,
    videoProcessing: false,
    processingInterval: null
};

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Получаем элементы
    const mapFileInput = document.getElementById('map-file');
    const droneFileInput = document.getElementById('drone-file');
    const findLocationBtn = document.getElementById('find-location-btn');
    const clearRouteBtn = document.getElementById('clear-route-btn');
    const startVideoBtn = document.getElementById('start-video-btn');
    const stopVideoBtn = document.getElementById('stop-video-btn');
    const videoStream = document.getElementById('video-stream');
    const mapCanvas = document.getElementById('map-canvas');
    
    state.mapCanvas = mapCanvas;
    state.mapContext = mapCanvas.getContext('2d');
    state.videoStream = videoStream;
    
    // Обработчики событий
    mapFileInput.addEventListener('change', handleMapUpload);
    droneFileInput.addEventListener('change', () => {
        updateStatus('drone-status', 'Изображение выбрано');
        findLocationBtn.disabled = false;
    });
    findLocationBtn.addEventListener('click', handleFindLocation);
    clearRouteBtn.addEventListener('click', clearRoute);
    startVideoBtn.addEventListener('click', startVideoProcessing);
    stopVideoBtn.addEventListener('click', stopVideoProcessing);
    
    // Обработчик клика на карте для добавления точек маршрута
    mapCanvas.addEventListener('click', handleMapClick);
    
    // Обновляем статус при загрузке
    updateSystemStatus();
}

async function handleMapUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    updateStatus('map-status', 'Загрузка карты...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload_map', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            updateStatus('map-status', `Карта загружена: ${data.filename} (${data.width}x${data.height})`);
            await loadMapImage(file);
        } else {
            updateStatus('map-status', `Ошибка: ${data.error}`, 'error');
        }
    } catch (error) {
        updateStatus('map-status', `Ошибка: ${error.message}`, 'error');
    }
    
    updateSystemStatus();
}

async function loadMapImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                state.mapImage = img;
                state.mapCanvas.width = img.width;
                state.mapCanvas.height = img.height;
                state.mapContext.drawImage(img, 0, 0);
                document.getElementById('map-placeholder').style.display = 'none';
                resolve();
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function handleFindLocation() {
    const fileInput = document.getElementById('drone-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Выберите изображение с камеры дрона');
        return;
    }
    
    if (!state.mapImage) {
        alert('Сначала загрузите карту');
        return;
    }
    
    updateStatus('drone-status', 'Анализ изображения...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/find_location', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            const pos = data.position;
            state.currentPosition = { x: pos.x, y: pos.y };
            
            // Рисуем позицию на карте
            drawMap();
            drawCurrentPosition(pos.x, pos.y);
            
            // Показываем информацию
            showLocationInfo(data);
            
            // Проверяем отклонение
            if (data.deviation) {
                handleDeviation(data.deviation);
            }
            
            updateStatus('drone-status', 
                `Местоположение найдено! (${Math.round(pos.x)}, ${Math.round(pos.y)})`);
        } else {
            updateStatus('drone-status', 
                data.error || 'Не удалось найти местоположение', 'error');
        }
    } catch (error) {
        updateStatus('drone-status', `Ошибка: ${error.message}`, 'error');
    }
    
    updateSystemStatus();
}

function showLocationInfo(data) {
    const locationInfo = document.getElementById('location-info');
    const locationDetails = document.getElementById('location-details');
    
    const pos = data.position;
    let html = `
        <p><strong>Координаты:</strong> (${Math.round(pos.x)}, ${Math.round(pos.y)})</p>
        <p><strong>Угол поворота:</strong> ${pos.angle.toFixed(2)}°</p>
        <p><strong>Уверенность:</strong> ${pos.confidence.toFixed(2)}%</p>
        <p><strong>Совпадений:</strong> ${pos.matches_count}</p>
    `;
    
    locationDetails.innerHTML = html;
    locationInfo.style.display = 'block';
}

function handleDeviation(deviationInfo) {
    const deviationStatus = document.getElementById('deviation-status');
    
    if (!deviationInfo.is_on_route) {
        deviationStatus.textContent = `${deviationInfo.deviation.toFixed(1)} пикселей`;
        deviationStatus.className = 'value error';
        
        // Показываем предупреждение
        alert(`⚠️ ВНИМАНИЕ: ${deviationInfo.message}\nОтклонение: ${deviationInfo.deviation.toFixed(1)} пикселей`);
    } else {
        deviationStatus.textContent = 'На маршруте';
        deviationStatus.className = 'value success';
    }
}

function handleMapClick(event) {
    if (!state.mapImage) {
        alert('Сначала загрузите карту');
        return;
    }
    
    const rect = state.mapCanvas.getBoundingClientRect();
    const scaleX = state.mapCanvas.width / rect.width;
    const scaleY = state.mapCanvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    // Добавляем точку маршрута
    state.waypoints.push({ x, y });
    drawMap();
    drawRoute();
    
    // Если точек достаточно, отправляем маршрут на сервер
    if (state.waypoints.length >= 2) {
        setRoute();
    }
}

function drawMap() {
    if (!state.mapImage) return;
    
    state.mapContext.clearRect(0, 0, state.mapCanvas.width, state.mapCanvas.height);
    state.mapContext.drawImage(state.mapImage, 0, 0);
}

function drawRoute() {
    if (state.waypoints.length < 2) return;
    
    const ctx = state.mapContext;
    
    // Рисуем линию маршрута
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(state.waypoints[0].x, state.waypoints[0].y);
    
    for (let i = 1; i < state.waypoints.length; i++) {
        ctx.lineTo(state.waypoints[i].x, state.waypoints[i].y);
    }
    ctx.stroke();
    
    // Рисуем точки маршрута
    state.waypoints.forEach((wp, index) => {
        ctx.fillStyle = index === 0 ? '#4CAF50' : '#2196F3';
        ctx.beginPath();
        ctx.arc(wp.x, wp.y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Номер точки
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText((index + 1).toString(), wp.x, wp.y);
    });
}

function drawCurrentPosition(x, y) {
    const ctx = state.mapContext;
    
    // Рисуем большой круг для текущей позиции
    ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Центральная точка
    ctx.fillStyle = '#FF0000';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
}

async function setRoute() {
    const deviationThreshold = parseFloat(document.getElementById('deviation-threshold').value);
    
    try {
        const response = await fetch('/api/set_route', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                waypoints: state.waypoints,
                allowed_deviation: deviationThreshold
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            state.routeSet = true;
            updateSystemStatus();
            
            // Если есть текущая позиция, проверяем отклонение
            if (state.currentPosition) {
                checkDeviation(state.currentPosition.x, state.currentPosition.y);
            }
        } else {
            alert(`Ошибка установки маршрута: ${data.error}`);
        }
    } catch (error) {
        console.error('Ошибка установки маршрута:', error);
    }
}

function clearRoute() {
    state.waypoints = [];
    state.routeSet = false;
    drawMap();
    if (state.currentPosition) {
        drawCurrentPosition(state.currentPosition.x, state.currentPosition.y);
    }
    updateSystemStatus();
    
    // Очищаем маршрут на сервере
    fetch('/api/set_route', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            waypoints: []
        })
    });
}

async function checkDeviation(x, y) {
    try {
        const response = await fetch('/api/check_deviation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ x, y })
        });
        
        const data = await response.json();
        handleDeviation(data);
    } catch (error) {
        console.error('Ошибка проверки отклонения:', error);
    }
}

async function updateSystemStatus() {
    try {
        const response = await fetch('/api/get_status');
        const data = await response.json();
        
        // Обновляем статус карты
        const mapStatus = document.getElementById('map-loaded-status');
        if (data.map_loaded) {
            mapStatus.textContent = 'Загружена';
            mapStatus.className = 'value success';
        } else {
            mapStatus.textContent = 'Не загружена';
            mapStatus.className = 'value';
        }
        
        // Обновляем статус позиции
        const positionStatus = document.getElementById('position-status');
        if (data.current_position) {
            const pos = data.current_position;
            positionStatus.textContent = `(${Math.round(pos[0])}, ${Math.round(pos[1])})`;
            positionStatus.className = 'value success';
        } else {
            positionStatus.textContent = 'Не определена';
            positionStatus.className = 'value';
        }
        
        // Обновляем статус маршрута
        const routeStatus = document.getElementById('route-status');
        if (data.route_set && data.waypoints_count > 0) {
            routeStatus.textContent = `${data.waypoints_count} точек`;
            routeStatus.className = 'value success';
        } else {
            routeStatus.textContent = 'Не задан';
            routeStatus.className = 'value';
        }
        
        // Обновляем отклонение
        if (data.deviation) {
            handleDeviation(data.deviation);
        }
    } catch (error) {
        console.error('Ошибка обновления статуса:', error);
    }
}

function updateStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = 'status-text';
    if (type === 'error') {
        element.style.color = '#dc3545';
    } else if (type === 'success') {
        element.style.color = '#28a745';
    } else {
        element.style.color = '#666';
    }
}

// Функции для обработки видео
async function startVideoProcessing() {
    if (!state.mapImage) {
        alert('Сначала загрузите карту');
        return;
    }
    
    try {
        // Запрашиваем доступ к веб-камере
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        
        state.videoStream.srcObject = stream;
        state.videoStream.style.display = 'block';
        state.videoProcessing = true;
        
        // Обновляем кнопки
        document.getElementById('start-video-btn').disabled = true;
        document.getElementById('stop-video-btn').disabled = false;
        
        // Начинаем обработку кадров
        processVideoFrames();
        
        updateStatus('drone-status', 'Видеопоток запущен');
    } catch (error) {
        console.error('Ошибка доступа к камере:', error);
        alert('Не удалось получить доступ к веб-камере. Проверьте разрешения.');
    }
}

function stopVideoProcessing() {
    if (state.videoStream && state.videoStream.srcObject) {
        const tracks = state.videoStream.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        state.videoStream.srcObject = null;
        state.videoStream.style.display = 'none';
    }
    
    if (state.processingInterval) {
        clearInterval(state.processingInterval);
        state.processingInterval = null;
    }
    
    state.videoProcessing = false;
    
    // Обновляем кнопки
    document.getElementById('start-video-btn').disabled = false;
    document.getElementById('stop-video-btn').disabled = true;
    
    updateStatus('drone-status', 'Видеопоток остановлен');
}

function processVideoFrames() {
    if (!state.videoProcessing) return;
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = state.videoStream.videoWidth || 640;
    canvas.height = state.videoStream.videoHeight || 480;
    
    function captureAndProcess() {
        if (!state.videoProcessing || !state.videoStream.videoWidth) {
            return;
        }
        
        try {
            ctx.drawImage(state.videoStream, 0, 0, canvas.width, canvas.height);
            
            // Конвертируем кадр в base64
            canvas.toBlob(async (blob) => {
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const base64data = reader.result;
                    
                    try {
                        const response = await fetch('/api/process_video_frame', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                frame: base64data
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok && data.success && data.position) {
                            const pos = data.position;
                            state.currentPosition = { x: pos.x, y: pos.y };
                            
                            // Обновляем карту
                            drawMap();
                            drawRoute();
                            drawCurrentPosition(pos.x, pos.y);
                            
                            // Обновляем информацию
                            if (data.deviation) {
                                handleDeviation(data.deviation);
                            }
                            
                            updateSystemStatus();
                        }
                    } catch (error) {
                        console.error('Ошибка обработки кадра:', error);
                    }
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Ошибка захвата кадра:', error);
        }
    }
    
    // Обрабатываем кадры каждые 500мс (2 FPS для обработки)
    state.processingInterval = setInterval(captureAndProcess, 500);
}

// Обновляем доступность кнопки видео при загрузке карты
const originalHandleMapUpload = handleMapUpload;
handleMapUpload = async function(event) {
    await originalHandleMapUpload(event);
    const startVideoBtn = document.getElementById('start-video-btn');
    if (state.mapImage) {
        startVideoBtn.disabled = false;
    }
};

// Периодическое обновление статуса
setInterval(updateSystemStatus, 2000);

