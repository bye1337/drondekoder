"""
TCP/UDP сервер для связи с системой управления дроном
Выдает навигационные измерения для стабилизации позиции БАС
"""
import socket
import json
import struct
import threading
import time
from typing import Optional, Dict, Tuple
from video_processor import VideoProcessor
from stabilization_processor import PositionStabilizer


class NavigationServer:
    """
    Сервер для выдачи навигационных измерений по TCP/UDP
    Интегрируется с системой стабилизации позиции
    """
    
    # Формат пакета: [timestamp(float), position_x(float), position_y(float), 
    #                 offset_x(float), offset_y(float), velocity_x(float), 
    #                 velocity_y(float), confidence(float), status(uint8)]
    PACKET_FORMAT = 'dfffffffB'  # 8 floats + 1 byte
    PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        tcp_port: int = 5001,
        udp_port: int = 5002,
        video_processor: Optional[VideoProcessor] = None
    ):
        """
        Args:
            host: Адрес для прослушивания
            tcp_port: Порт TCP сервера
            udp_port: Порт UDP сервера
            video_processor: Процессор видеопотока
        """
        self.host = host
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.video_processor = video_processor
        
        self.tcp_socket = None
        self.udp_socket = None
        self.is_running = False
        self.clients = []
        
        # Домашняя точка (точка взлета)
        self.home_position: Optional[Tuple[float, float]] = None
        self.calibrated = False
        
        # Последние измерения
        self.last_measurement: Optional[Dict] = None
        self.measurement_lock = threading.Lock()
        
    def set_home_position(self, x: float, y: float):
        """Устанавливает домашнюю точку (точка взлета)"""
        self.home_position = (x, y)
        self.calibrated = True
        print(f"Домашняя точка установлена: ({x}, {y})")
    
    def get_home_distance(self, position: Tuple[float, float]) -> float:
        """Вычисляет расстояние до домашней точки"""
        if not self.home_position:
            return 0.0
        dx = position[0] - self.home_position[0]
        dy = position[1] - self.home_position[1]
        return (dx**2 + dy**2)**0.5
    
    def create_measurement_packet(self, measurement: Dict) -> bytes:
        """
        Создает бинарный пакет с навигационными измерениями
        
        Формат: timestamp, pos_x, pos_y, offset_x, offset_y, 
                vel_x, vel_y, confidence, status
        """
        position = measurement.get('position', [0.0, 0.0])
        offset = measurement.get('offset', [0.0, 0.0])
        velocity = measurement.get('velocity', [0.0, 0.0])
        confidence = measurement.get('confidence', 0.0)
        
        # Статус: bit 0 = calibrated, bit 1 = stable, bit 2 = home_reachable
        status = 0
        if self.calibrated:
            status |= 1
        if measurement.get('stability', {}).get('is_stable', False):
            status |= 2
        
        # Проверка досягаемости дома (в пределах 100м = ~1000 пикселей при масштабе 0.1м/пикс)
        if self.home_position:
            distance = self.get_home_distance((position[0], position[1]))
            if distance < 1000.0:  # Условно 100м
                status |= 4
        
        timestamp = time.time()
        
        # Упаковка данных
        packet = struct.pack(
            self.PACKET_FORMAT,
            timestamp,
            float(position[0]),
            float(position[1]),
            float(offset[0]),
            float(offset[1]),
            float(velocity[0]),
            float(velocity[1]),
            float(confidence),
            status
        )
        
        return packet
    
    def create_json_packet(self, measurement: Dict) -> bytes:
        """Создает JSON пакет с навигационными измерениями"""
        data = {
            'timestamp': time.time(),
            'position': measurement.get('position', [0.0, 0.0]),
            'offset': measurement.get('offset', [0.0, 0.0]),
            'velocity': measurement.get('velocity', [0.0, 0.0]),
            'confidence': measurement.get('confidence', 0.0),
            'stability': measurement.get('stability', {}),
            'calibrated': self.calibrated,
            'home_position': self.home_position,
            'home_distance': self.get_home_distance(
                tuple(measurement.get('position', [0.0, 0.0]))
            ) if self.calibrated else None
        }
        
        json_str = json.dumps(data)
        return json_str.encode('utf-8')
    
    def update_measurement(self, measurement: Dict):
        """Обновляет последнее измерение (вызывается из video_processor)"""
        with self.measurement_lock:
            self.last_measurement = measurement
    
    def handle_tcp_client(self, client_socket, address):
        """Обработка TCP клиента"""
        print(f"TCP клиент подключен: {address}")
        self.clients.append(client_socket)
        
        try:
            while self.is_running:
                with self.measurement_lock:
                    measurement = self.last_measurement
                
                if measurement:
                    # Отправляем бинарный пакет
                    packet = self.create_measurement_packet(measurement)
                    client_socket.send(packet)
                else:
                    # Отправляем пустой пакет при отсутствии данных
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Ошибка TCP клиента {address}: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
            print(f"TCP клиент отключен: {address}")
    
    def tcp_server_loop(self):
        """Основной цикл TCP сервера"""
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_socket.bind((self.host, self.tcp_port))
        self.tcp_socket.listen(5)
        
        print(f"TCP сервер запущен на {self.host}:{self.tcp_port}")
        
        while self.is_running:
            try:
                self.tcp_socket.settimeout(1.0)
                client_socket, address = self.tcp_socket.accept()
                
                # Запускаем обработку клиента в отдельном потоке
                client_thread = threading.Thread(
                    target=self.handle_tcp_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"Ошибка TCP сервера: {e}")
    
    def udp_server_loop(self):
        """Основной цикл UDP сервера"""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.host, self.udp_port))
        self.udp_socket.settimeout(1.0)
        
        print(f"UDP сервер запущен на {self.host}:{self.udp_port}")
        
        # Для UDP отправляем широковещательные пакеты
        last_client = None
        
        while self.is_running:
            try:
                # Читаем запрос от клиента
                data, addr = self.udp_socket.recvfrom(1024)
                last_client = addr
                
                # Отправляем измерение обратно
                with self.measurement_lock:
                    measurement = self.last_measurement
                
                if measurement:
                    # Отправляем JSON (проще для UDP)
                    packet = self.create_json_packet(measurement)
                    self.udp_socket.sendto(packet, addr)
                    
            except socket.timeout:
                # Периодически отправляем пакеты даже без запросов
                if last_client and self.last_measurement:
                    try:
                        packet = self.create_json_packet(self.last_measurement)
                        self.udp_socket.sendto(packet, last_client)
                    except:
                        pass
                continue
            except Exception as e:
                if self.is_running:
                    print(f"Ошибка UDP сервера: {e}")
    
    def start(self):
        """Запускает серверы"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Запускаем TCP сервер
        tcp_thread = threading.Thread(target=self.tcp_server_loop, daemon=True)
        tcp_thread.start()
        
        # Запускаем UDP сервер
        udp_thread = threading.Thread(target=self.udp_server_loop, daemon=True)
        udp_thread.start()
        
        print("Навигационный сервер запущен")
    
    def stop(self):
        """Останавливает серверы"""
        self.is_running = False
        
        # Закрываем соединения
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()
        
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except:
                pass
        
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
        
        print("Навигационный сервер остановлен")
    
    def broadcast_measurement(self, measurement: Dict):
        """Широковещательная рассылка измерения всем TCP клиентам"""
        packet = self.create_measurement_packet(measurement)
        disconnected = []
        
        for client in self.clients:
            try:
                client.send(packet)
            except:
                disconnected.append(client)
        
        # Удаляем отключенных клиентов
        for client in disconnected:
            if client in self.clients:
                self.clients.remove(client)


def create_callback(video_processor, nav_server):
    """Создает callback для video_processor для отправки данных в nav_server"""
    def callback(result, primary_frame, secondary_frame):
        nav_server.update_measurement(result)
        nav_server.broadcast_measurement(result)
    return callback

