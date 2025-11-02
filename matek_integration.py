"""
Интеграция системы стабилизации с полетным контроллером Matek F405
Использует MAVLink протокол для связи
"""
import time
import serial
import struct
from typing import Optional, Tuple, Dict
from enum import IntEnum
from drone_navigation import DroneNavigationSystem


# MAVLink версия 2.0
MAVLINK_VERSION = 2
MAVLINK_STX = 0xFD


class MAVLinkMessageID(IntEnum):
    """ID сообщений MAVLink"""
    HEARTBEAT = 0
    SYS_STATUS = 1
    SYSTEM_TIME = 2
    ATTITUDE = 30
    GLOBAL_POSITION_INT = 33
    LOCAL_POSITION_NED = 32
    ATTITUDE_TARGET = 83
    POSITION_TARGET_LOCAL_NED = 85
    HIL_STATE_QUATERNION = 115
    VISION_POSITION_ESTIMATE = 102


class MAVLink:
    """Простой MAVLink парсер/генератор для связи с Matek F405"""
    
    def __init__(self, serial_port: str, baudrate: int = 57600):
        """
        Args:
            serial_port: Путь к последовательному порту (например, '/dev/ttyUSB0')
            baudrate: Скорость передачи (обычно 57600 или 115200)
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.connection = None
        self.system_id = 255  # ID нашего компонента
        self.component_id = 190  # ID для visual navigation
        self.seq = 0
        
    def connect(self):
        """Подключается к полетному контроллеру"""
        try:
            self.connection = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=1.0
            )
            print(f"✅ Подключено к {self.serial_port} на скорости {self.baudrate}")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def disconnect(self):
        """Отключается от полетного контроллера"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("Отключено от полетного контроллера")
    
    def _calculate_checksum(self, message: bytes) -> int:
        """Вычисляет checksum для MAVLink сообщения"""
        crc = 0xFFFF
        for byte in message:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc
    
    def _create_mavlink_message(
        self,
        msg_id: int,
        payload: bytes
    ) -> bytes:
        """Создает MAVLink сообщение"""
        # Заголовок
        header = struct.pack(
            '<BBBBBBB',
            MAVLINK_STX,           # Стартовый байт
            len(payload),          # Длина payload
            self.seq,              # Sequence number
            self.system_id,        # System ID
            self.component_id,     # Component ID
            msg_id,                # Message ID
            msg_id >> 8            # Message ID (старший байт для v2)
        )
        
        # Собираем сообщение
        message = header + payload
        
        # Вычисляем checksum
        crc_extra = self._get_crc_extra(msg_id)
        checksum = self._calculate_checksum(message[1:] + bytes([crc_extra]))
        
        # Добавляем checksum
        message += struct.pack('<H', checksum)
        
        self.seq = (self.seq + 1) % 256
        
        return message
    
    def _get_crc_extra(self, msg_id: int) -> int:
        """CRC extra для различных сообщений MAVLink"""
        crc_table = {
            MAVLinkMessageID.VISION_POSITION_ESTIMATE: 102,
            MAVLinkMessageID.LOCAL_POSITION_NED: 185,
            MAVLinkMessageID.ATTITUDE: 39,
        }
        return crc_table.get(msg_id, 0)
    
    def send_vision_position_estimate(
        self,
        timestamp_us: int,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        covariance: Optional[list] = None
    ):
        """
        Отправляет VISION_POSITION_ESTIMATE сообщение
        Это основной способ передачи визуальной навигации в ArduPilot
        """
        if not self.connection or not self.connection.is_open:
            return False
        
        # Заполняем ковариацию (21 элементов)
        if covariance is None:
            covariance = [0.0] * 21
            # Устанавливаем уверенность по X, Y
            covariance[0] = 0.1  # X variance
            covariance[6] = 0.1  # Y variance
        
        payload = struct.pack(
            '<Qffffffffffffffffffffffff',
            timestamp_us,
            x, y, z,
            roll, pitch, yaw,
            *covariance[:18]  # Только 18 элементов в payload
        )
        
        message = self._create_mavlink_message(
            MAVLinkMessageID.VISION_POSITION_ESTIMATE,
            payload
        )
        
        try:
            self.connection.write(message)
            return True
        except Exception as e:
            print(f"Ошибка отправки VISION_POSITION_ESTIMATE: {e}")
            return False
    
    def send_local_position_ned(
        self,
        x: float,
        y: float,
        z: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0
    ):
        """Отправляет LOCAL_POSITION_NED сообщение"""
        if not self.connection or not self.connection.is_open:
            return False
        
        timestamp_ms = int(time.time() * 1000)
        
        payload = struct.pack(
            '<Qffffff',
            timestamp_ms,
            x, y, z,
            vx, vy, vz
        )
        
        message = self._create_mavlink_message(
            MAVLinkMessageID.LOCAL_POSITION_NED,
            payload
        )
        
        try:
            self.connection.write(message)
            return True
        except Exception as e:
            print(f"Ошибка отправки LOCAL_POSITION_NED: {e}")
            return False
    
    def send_heartbeat(self):
        """Отправляет heartbeat для поддержания связи"""
        if not self.connection or not self.connection.is_open:
            return False
        
        # Простой heartbeat (минимальная реализация)
        # В реальности нужно полноценное heartbeat сообщение
        pass
    
    def read_message(self) -> Optional[Dict]:
        """Читает входящее MAVLink сообщение"""
        if not self.connection or not self.connection.is_open:
            return None
        
        if self.connection.in_waiting == 0:
            return None
        
        try:
            # Простой парсинг (для полной реализации нужен полноценный парсер)
            data = self.connection.read(self.connection.in_waiting)
            # Здесь должен быть полноценный парсинг MAVLink
            return None
        except Exception as e:
            print(f"Ошибка чтения: {e}")
            return None


class MatekF405Bridge:
    """
    Мост между системой стабилизации и Matek F405
    Преобразует данные стабилизации в MAVLink команды
    """
    
    def __init__(
        self,
        navigation_system: DroneNavigationSystem,
        serial_port: str = None,  # Автопоиск или указать вручную
        baudrate: int = 57600,
        pixels_to_meters: float = 0.1  # Масштаб: 10 пикселей = 1 метр
    ):
        """
        Args:
            navigation_system: Система навигации
            serial_port: Последовательный порт (None = автопоиск USB-UART адаптера)
            baudrate: Скорость передачи (57600 или 115200)
            pixels_to_meters: Коэффициент преобразования пикселей в метры
        """
        self.nav_system = navigation_system
        self.pixels_to_meters = pixels_to_meters
        
        # Автопоиск порта, если не указан
        if serial_port is None:
            serial_port = self._find_uart_port()
            if serial_port:
                print(f"Автоматически найден порт: {serial_port}")
            else:
                print("⚠️  Порт не найден автоматически, укажите вручную")
        
        self.mavlink = MAVLink(serial_port or '/dev/ttyUSB0', baudrate)
        self.is_running = False
        self.last_update_time = 0.0
        self.update_rate = 10.0  # 10 Гц как у GNSS
        
        # Начальная точка (дом)
        self.home_position_ned = None
        
        # Накопленные смещения (в метрах NED)
        self.position_ned = [0.0, 0.0, 0.0]  # North, East, Down
        
    def connect(self) -> bool:
        """Подключается к Matek F405"""
        return self.mavlink.connect()
    
    def disconnect(self):
        """Отключается от Matek F405"""
        self.is_running = False
        self.mavlink.disconnect()
    
    def set_home(self, lat: float = None, lon: float = None, alt: float = None):
        """
        Устанавливает домашнюю точку
        
        Args:
            lat, lon, alt: GPS координаты (если доступны)
        """
        # Используем текущую позицию как (0, 0, 0) в NED координатах
        self.home_position_ned = [0.0, 0.0, 0.0]
        self.position_ned = [0.0, 0.0, 0.0]
        print("Домашняя точка установлена в системе координат NED")
    
    def update(self):
        """
        Обновляет данные в полетном контроллере
        Вызывается в цикле обработки
        """
        if not self.is_running:
            return
        
        current_time = time.time()
        
        # Ограничиваем частоту обновления
        if current_time - self.last_update_time < 1.0 / self.update_rate:
            return
        
        self.last_update_time = current_time
        
        # Получаем данные навигации
        nav_data = self.nav_system.get_navigation_data()
        
        if not nav_data:
            return
        
        # Извлекаем данные
        position = nav_data.get('position', [0, 0])
        offset = nav_data.get('offset', [0, 0])
        velocity = nav_data.get('velocity', [0, 0])
        confidence = nav_data.get('confidence', 0.0)
        
        # Преобразуем пиксели в метры
        # Предполагаем: ось Y камеры = North, ось X камеры = East
        offset_north = -offset[1] * self.pixels_to_meters  # Y -> North (инвертировано)
        offset_east = offset[0] * self.pixels_to_meters    # X -> East
        
        # Накопление позиции (относительно стартовой точки)
        if self.home_position_ned is not None:
            self.position_ned[0] = offset_north  # North
            self.position_ned[1] = offset_east   # East
            # Down (высота) - используем барометр или оставляем 0
        
        # Преобразуем скорость
        velocity_north = -velocity[1] * self.pixels_to_meters
        velocity_east = velocity[0] * self.pixels_to_meters
        
        # Timestamp в микросекундах
        timestamp_us = int(current_time * 1e6)
        
        # Отправляем VISION_POSITION_ESTIMATE
        # Это основной способ передачи визуальной навигации в ArduPilot
        self.mavlink.send_vision_position_estimate(
            timestamp_us=timestamp_us,
            x=self.position_ned[0],  # North
            y=self.position_ned[1],   # East
            z=self.position_ned[2],   # Down (высота)
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            covariance=[confidence * 0.1] * 3 + [0.0] * 18  # Уверенность в позиции
        )
        
        # Также отправляем LOCAL_POSITION_NED
        self.mavlink.send_local_position_ned(
            x=self.position_ned[0],
            y=self.position_ned[1],
            z=self.position_ned[2],
            vx=velocity_north,
            vy=velocity_east,
            vz=0.0
        )
    
    def start(self):
        """Запускает мост"""
        if self.connect():
            self.is_running = True
            self.set_home()
            print("✅ Мост Matek F405 запущен")
            return True
        return False
    
    def stop(self):
        """Останавливает мост"""
        self.disconnect()
    
    @staticmethod
    def _find_uart_port() -> Optional[str]:
        """
        Автоматический поиск USB-UART адаптера
        Возвращает путь к порту или None
        """
        try:
            import serial.tools.list_ports
            
            # Ищем USB-UART адаптеры
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                # Обычно USB-UART адаптеры имеют такие описания
                desc_lower = port.description.lower()
                if any(keyword in desc_lower for keyword in [
                    'uart', 'serial', 'ch340', 'ch341', 'cp210', 
                    'ftdi', 'pl2303', 'usb serial', 'usb-to-serial'
                ]):
                    return port.device
            
            # Если не нашли по описанию, берем первый USB порт
            for port in ports:
                if 'USB' in port.device or 'ttyUSB' in port.device:
                    return port.device
            
            # Linux: /dev/ttyUSB*, /dev/ttyACM*
            import glob
            for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
                ports = glob.glob(pattern)
                if ports:
                    return sorted(ports)[0]  # Возвращаем первый найденный
            
            return None
        except ImportError:
            # Если serial.tools недоступен, используем простой поиск
            import glob
            for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
                ports = glob.glob(pattern)
                if ports:
                    return sorted(ports)[0]
            return None


def main():
    """Пример использования интеграции с Matek F405"""
    print("=" * 60)
    print("Интеграция с Matek F405")
    print("=" * 60)
    
    # Создание системы навигации
    nav_system = DroneNavigationSystem(
        camera_id=0,
        target_fps=10.0,
        optimize_for_latency=True
    )
    
    # Создание моста с Matek F405
    # Для мини ПК: система автоматически найдет USB-UART адаптер
    # Можно указать вручную: '/dev/ttyUSB0', '/dev/ttyACM0' и т.д.
    bridge = MatekF405Bridge(
        navigation_system=nav_system,
        serial_port=None,            # Автопоиск USB-UART адаптера (или укажите вручную)
        baudrate=57600,              # Обычно 57600 или 115200 для ArduPilot
        pixels_to_meters=0.1         # Настройте под высоту полета
    )
    
    try:
        # Запуск системы навигации
        nav_system.start()
        
        # Запуск моста
        if bridge.start():
            print("\nСистема работает. Нажмите Ctrl+C для остановки\n")
            
            # Основной цикл
            while True:
                bridge.update()
                time.sleep(0.01)  # 10 Гц обновление
        else:
            print("❌ Не удалось подключиться к Matek F405")
            
    except KeyboardInterrupt:
        print("\n\nОстановка системы...")
    finally:
        bridge.stop()
        nav_system.stop()


if __name__ == '__main__':
    main()

