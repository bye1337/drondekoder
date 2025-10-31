"""
Главный файл приложения Flask
"""
import os
import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
from route_monitor import RouteMonitor
from video_processor import VideoProcessor

# Импортируем ML систему на базе YOLOv8
from object_matcher import ObjectMatcher

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max для больших карт

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Глобальные объекты - используем только YOLOv8 для детекции объектов
try:
    matcher = ObjectMatcher(model_size='n', confidence_threshold=0.25)
    print("✓ ML система загружена (ObjectMatcher на YOLOv8)")
except Exception as e:
    print(f"❌ Ошибка инициализации ML системы: {e}")
    print("Установите зависимости: pip install ultralytics torch torchvision")
    raise

route_monitor = RouteMonitor()
video_processor = VideoProcessor(matcher, route_monitor)

# Глобальное состояние
large_map = None
map_filename = None
current_position = None
video_processing_active = False


def allowed_file(filename):
    """Проверяет разрешенное расширение файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image(file_path):
    """Загружает изображение"""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {file_path}")
    return img


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/upload_map', methods=['POST'])
def upload_map():
    """Загружает большую карту"""
    global large_map, map_filename
    
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            large_map = load_image(filepath)
            map_filename = filename
            height, width = large_map.shape[:2]
            
            return jsonify({
                'success': True,
                'filename': filename,
                'width': width,
                'height': height,
                'message': 'Карта успешно загружена'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Недопустимый тип файла'}), 400


@app.route('/api/find_location', methods=['POST'])
def find_location():
    """Определяет местоположение на карте"""
    global large_map, current_position
    
    if large_map is None:
        return jsonify({'error': 'Сначала загрузите карту'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            small_image = load_image(filepath)
            result = matcher.find_location(large_map, small_image)
            
            if result is None:
                return jsonify({
                    'success': False,
                    'error': 'Не удалось найти совпадение. Попробуйте другое изображение.'
                }), 400
            
            current_position = (result['x'], result['y'])
            
            # Проверяем отклонение от маршрута
            deviation_info = route_monitor.check_deviation(current_position)
            
            response = {
                'success': True,
                'position': {
                    'x': result['x'],
                    'y': result['y'],
                    'angle': result['angle'],
                    'confidence': result['confidence'],
                    'matches_count': result['matches_count']
                },
                'deviation': deviation_info
            }
            
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Недопустимый тип файла'}), 400


@app.route('/api/set_route', methods=['POST'])
def set_route():
    """Устанавливает маршрут"""
    data = request.get_json()
    
    if 'waypoints' not in data:
        return jsonify({'error': 'Точки маршрута не указаны'}), 400
    
    waypoints = data['waypoints']
    
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        return jsonify({'error': 'Маршрут должен содержать минимум 2 точки'}), 400
    
    try:
        route_points = [(float(wp['x']), float(wp['y'])) for wp in waypoints]
        route_monitor.set_route(route_points)
        
        if 'allowed_deviation' in data:
            route_monitor.set_allowed_deviation(float(data['allowed_deviation']))
        
        return jsonify({
            'success': True,
            'message': f'Маршрут установлен с {len(route_points)} точками',
            'waypoints_count': len(route_points)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_deviation', methods=['POST'])
def check_deviation():
    """Проверяет отклонение от маршрута для заданной позиции"""
    data = request.get_json()
    
    if 'x' not in data or 'y' not in data:
        return jsonify({'error': 'Координаты не указаны'}), 400
    
    try:
        position = (float(data['x']), float(data['y']))
        deviation_info = route_monitor.check_deviation(position)
        return jsonify(deviation_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_status', methods=['GET'])
def get_status():
    """Возвращает текущий статус системы"""
    status = {
        'map_loaded': large_map is not None,
        'current_position': current_position,
        'route_set': len(route_monitor.waypoints) > 0,
        'waypoints_count': len(route_monitor.waypoints),
        'video_processing': video_processing_active
    }
    
    if current_position and route_monitor.waypoints:
        deviation_info = route_monitor.check_deviation(current_position)
        status['deviation'] = deviation_info
    
    return jsonify(status)


@app.route('/api/process_video_frame', methods=['POST'])
def process_video_frame():
    """Обрабатывает один кадр из видеопотока"""
    global large_map, current_position
    
    if large_map is None:
        return jsonify({'error': 'Сначала загрузите карту'}), 400
    
    if 'frame' not in request.files:
        # Попробуем получить base64 изображение
        data = request.get_json()
        if data and 'frame' in data:
            try:
                # Декодируем base64 изображение
                frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return jsonify({'error': 'Не удалось декодировать изображение'}), 400
                
                # Обрабатываем кадр
                result = video_processor.process_frame(frame, large_map)
                
                if result:
                    current_position = (result['x'], result['y'])
                    return jsonify({
                        'success': True,
                        'position': result,
                        'deviation': result.get('deviation')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Не удалось найти совпадение'
                    }), 400
                    
            except Exception as e:
                return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500
        
        return jsonify({'error': 'Кадр не найден'}), 400
    
    # Обработка файла
    file = request.files['frame']
    if file and allowed_file(file.filename):
        try:
            file_bytes = file.read()
            frame_array = np.frombuffer(file_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Не удалось декодировать изображение'}), 400
            
            # Обрабатываем кадр
            result = video_processor.process_frame(frame, large_map)
            
            if result:
                current_position = (result['x'], result['y'])
                return jsonify({
                    'success': True,
                    'position': result,
                    'deviation': result.get('deviation')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Не удалось найти совпадение'
                }), 400
                
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500
    
    return jsonify({'error': 'Недопустимый тип файла'}), 400


@app.route('/api/start_video_stream', methods=['POST'])
def start_video_stream():
    """Запускает обработку видеопотока"""
    global video_processing_active
    
    data = request.get_json()
    source = data.get('source', 0)  # 0 для веб-камеры
    
    try:
        video_processor.start_processing(source)
        video_processing_active = True
        
        return jsonify({
            'success': True,
            'message': 'Обработка видеопотока запущена'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop_video_stream', methods=['POST'])
def stop_video_stream():
    """Останавливает обработку видеопотока"""
    global video_processing_active
    
    try:
        video_processor.stop_processing()
        video_processing_active = False
        
        return jsonify({
            'success': True,
            'message': 'Обработка видеопотока остановлена'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/video_stream_feed')
def video_stream_feed():
    """Потоковая передача обработанного видео"""
    def generate():
        global large_map
        
        if large_map is None:
            return
        
        cap = cv2.VideoCapture(0)  # Веб-камера
        
        if not cap.isOpened():
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обрабатываем кадр
                result = video_processor.process_frame(frame, large_map)
                
                # Рисуем результат на кадре
                if result:
                    output_frame = video_processor.draw_result_on_frame(frame, result)
                else:
                    output_frame = frame
                
                # Кодируем в JPEG
                frame_bytes = video_processor.encode_frame_jpeg(output_frame)
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        finally:
            cap.release()
    
    return Response(stream_with_context(generate()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

