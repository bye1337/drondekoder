"""
Простой веб-сервер для системы стабилизации позиции БАС
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from video_processor import VideoProcessor
from route_monitor import RouteMonitor
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

# Инициализация
route_monitor = RouteMonitor()
video_processor = VideoProcessor(
    use_dual_camera=False,
    primary_method='lucas_kanade',
    route_monitor=route_monitor
)

current_position = None


@app.route('/')
def index():
    """Главная страница"""
    # Проверяем наличие HTML шаблона
    if os.path.exists('templates/index.html'):
        return render_template('index.html')
    else:
        # Если шаблона нет, возвращаем JSON
        return jsonify({
            'name': 'Система стабилизации позиции БАС',
            'version': '1.0',
            'endpoints': {
                'GET /api/status': 'Статус системы',
                'POST /api/process_frame': 'Обработка кадра',
                'POST /api/reset': 'Сброс стабилизатора',
                'POST /api/set_route': 'Установка маршрута'
            }
        })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Статус системы"""
    stats = video_processor.get_statistics()
    
    status = {
        'stabilization': {
            'active': True,
            'last_result': video_processor.last_result is not None,
            'stats': stats
        },
        'position': current_position,
        'route': {
            'set': len(route_monitor.waypoints) > 0,
            'waypoints_count': len(route_monitor.waypoints)
        }
    }
    
    if video_processor.last_result:
        status['stabilization']['last_confidence'] = video_processor.last_result.get('confidence', 0.0)
        status['stabilization']['last_position'] = video_processor.last_result.get('position')
        status['stabilization']['last_offset'] = video_processor.last_result.get('offset')
    
    return jsonify(status)


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Обработка кадра для стабилизации"""
    global current_position
    
    try:
        # Получаем кадр
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'Кадр не найден'}), 400
        
        # Декодируем base64
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Не удалось декодировать изображение'}), 400
        
        # Обработка
        result = video_processor.process_frame(primary_frame=frame)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Стабилизатор не инициализирован'
            }), 400
        
        # Обновляем позицию
        position = result.get('position', [0, 0])
        if position:
            current_position = {'x': position[0], 'y': position[1]}
        
        # Ответ
        response = {
            'success': True,
            'position': current_position,
            'offset': result.get('offset', [0, 0]),
            'velocity': result.get('velocity', [0, 0]),
            'confidence': result.get('confidence', 0.0),
            'method': result.get('method', 'unknown'),
            'fps': result.get('fps', 0.0)
        }
        
        # Стабильность
        if 'stability' in result:
            response['stability'] = result['stability']
        
        # Отклонение от маршрута
        if 'deviation' in result:
            response['deviation'] = result['deviation']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Сброс стабилизатора"""
    try:
        video_processor.reset_stabilizer()
        global current_position
        current_position = None
        return jsonify({'success': True, 'message': 'Стабилизатор сброшен'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set_route', methods=['POST'])
def set_route():
    """Установка маршрута"""
    try:
        data = request.get_json()
        if not data or 'waypoints' not in data:
            return jsonify({'error': 'Точки маршрута не указаны'}), 400
        
        waypoints = [(float(wp['x']), float(wp['y'])) for wp in data['waypoints']]
        route_monitor.set_route(waypoints)
        
        if 'allowed_deviation' in data:
            route_monitor.set_allowed_deviation(float(data['allowed_deviation']))
        
        return jsonify({
            'success': True,
            'waypoints_count': len(waypoints)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Система стабилизации позиции БАС")
    print("=" * 60)
    print("\nСервер запущен на http://localhost:5000")
    print("\nДоступные endpoints:")
    print("  GET  /api/status - статус системы")
    print("  POST /api/process_frame - обработка кадра")
    print("  POST /api/reset - сброс стабилизатора")
    print("  POST /api/set_route - установка маршрута")
    print("\nНажмите Ctrl+C для остановки\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

