"""
Быстрый запуск интеграции с Matek F405 для мини ПК
Автоматически находит USB-UART адаптер и подключается
"""
from drone_navigation import DroneNavigationSystem
from matek_integration import MatekF405Bridge
import time
import sys


def main():
    print("=" * 60)
    print("Интеграция системы стабилизации с Matek F405")
    print("=" * 60)
    print("\nДля мини ПК на Ryzen 7:")
    print("  - Используется USB-UART адаптер")
    print("  - Автоматический поиск порта")
    print("  - Связь через MAVLink протокол\n")
    
    # 1. Создание системы навигации
    print("Инициализация системы навигации...")
    nav_system = DroneNavigationSystem(
        camera_id=0,  # ID камеры
        target_fps=10.0,  # 10 Гц как у GNSS
        optimize_for_latency=True
    )
    
    # 2. Создание моста с Matek F405
    print("Поиск USB-UART адаптера...")
    bridge = MatekF405Bridge(
        navigation_system=nav_system,
        serial_port=None,  # Автопоиск (или укажите вручную: '/dev/ttyUSB0')
        baudrate=57600,     # Скорость передачи (может быть 115200)
        pixels_to_meters=0.05  # 20 пикселей = 1 метр (настройте под высоту)
    )
    
    print("\nНастройка:")
    print(f"  Порт: {bridge.mavlink.serial_port}")
    print(f"  Скорость: {bridge.mavlink.baudrate} бод")
    print(f"  Масштаб: {bridge.pixels_to_meters} (1 пиксель = {1/bridge.pixels_to_meters:.1f} метр)")
    print()
    
    try:
        # 3. Запуск системы навигации
        print("Запуск системы навигации...")
        nav_system.start()
        
        # 4. Подключение к Matek F405
        print("Подключение к Matek F405...")
        if not bridge.start():
            print("❌ Не удалось подключиться к Matek F405")
            print("\nПроверьте:")
            print("  1. USB-UART адаптер подключен")
            print("  2. Провода подключены к UART2 на Matek F405")
            print("  3. Скорость передачи совпадает (57600 или 115200)")
            print("  4. Порт доступен: sudo chmod 666 /dev/ttyUSB0")
            return 1
        
        print("✅ Система запущена!")
        print("\nОтправка данных в Matek F405...")
        print("Нажмите Ctrl+C для остановки\n")
        
        # 5. Сигнал взлета (запускает автокалибровку)
        nav_system.takeoff()
        
        # 6. Основной цикл
        frame_count = 0
        last_stats_time = time.time()
        
        while True:
            # Обновление моста (отправка данных в Matek F405)
            bridge.update()
            
            # Статистика каждые 5 секунд
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                stats = nav_system.get_statistics()
                print(f"FPS: {stats['actual_fps']:.1f}, "
                      f"Latency: {stats['last_latency_ms']:.1f}ms, "
                      f"Frames: {stats['frames_processed']}")
                
                if stats['calibration']['calibrated']:
                    print("  ✅ Калибровка завершена")
                
                nav_data = nav_system.get_navigation_data()
                if nav_data:
                    position = nav_data.get('position', [0, 0])
                    confidence = nav_data.get('confidence', 0.0)
                    print(f"  Позиция: ({position[0]:.1f}, {position[1]:.1f}), "
                          f"Confidence: {confidence:.2f}")
                
                last_stats_time = current_time
            
            time.sleep(0.1)  # 10 Гц обновление
            
    except KeyboardInterrupt:
        print("\n\nОстановка системы...")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        bridge.stop()
        nav_system.stop()
        print("Система остановлена")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

