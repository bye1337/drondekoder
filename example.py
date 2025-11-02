"""
Простой пример использования системы стабилизации позиции БАС
"""
import cv2
import numpy as np
from video_processor import VideoProcessor

def main():
    print("=" * 60)
    print("Пример: Стабилизация позиции БАС")
    print("=" * 60)
    
    # Создание процессора
    processor = VideoProcessor(
        use_dual_camera=False,
        primary_method='lucas_kanade'
    )
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    print("\nОбработка видеопотока...")
    print("Нажмите 'q' для выхода, 'r' для сброса\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обработка кадра
        result = processor.process_frame(primary_frame=frame)
        
        if result:
            # Визуализация
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            position = result.get('position', center)
            offset = np.array(result.get('offset', [0, 0]), dtype=int)
            
            # Центр
            cv2.circle(frame, center, 10, (0, 255, 0), 2)
            
            # Смещение
            end_point = (center[0] + offset[0], center[1] + offset[1])
            cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 3, tipLength=0.3)
            cv2.circle(frame, tuple(position), 8, (255, 0, 0), -1)
            
            # Информация
            info = [
                f"Position: ({position[0]}, {position[1]})",
                f"Offset: ({offset[0]}, {offset[1]})",
                f"Confidence: {result.get('confidence', 0):.2f}",
                f"FPS: {result.get('fps', 0):.1f}"
            ]
            
            y = 30
            for text in info:
                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25
            
            # Стабильность
            stability = result.get('stability', {})
            if stability.get('is_stable', False):
                cv2.putText(frame, "STABLE", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "DRIFTING", (w - 180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if frame_count % 30 == 0:
                print(f"Кадр {frame_count}: pos={position}, conf={result.get('confidence', 0):.2f}")
        
        cv2.imshow('Stabilization', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            processor.reset_stabilizer()
            print("Стабилизатор сброшен")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Статистика
    stats = processor.get_statistics()
    print(f"\nСтатистика:")
    print(f"  Кадров обработано: {stats['total_frames']}")
    print(f"  Средний FPS: {stats['avg_fps']:.1f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        print(f"\n\nОшибка: {e}")
        import traceback
        traceback.print_exc()

