#!/bin/bash
# Скрипт для создания пакета развертывания на мини ПК

DEPLOY_DIR="deploy_package"
mkdir -p "$DEPLOY_DIR"

echo "Создание пакета развертывания..."

# Основные модули
echo "Копирование основных модулей..."
cp stabilization_processor.py "$DEPLOY_DIR/"
cp video_processor.py "$DEPLOY_DIR/"
cp navigation_server.py "$DEPLOY_DIR/"
cp gps_integration.py "$DEPLOY_DIR/"
cp auto_calibration.py "$DEPLOY_DIR/"
cp home_return.py "$DEPLOY_DIR/"
cp route_monitor.py "$DEPLOY_DIR/"
cp matek_integration.py "$DEPLOY_DIR/"
cp drone_navigation.py "$DEPLOY_DIR/"

# Скрипты запуска
cp quick_start_matek.py "$DEPLOY_DIR/"

# Конфигурация
cp requirements.txt "$DEPLOY_DIR/"

# Документация (только нужная)
mkdir -p "$DEPLOY_DIR/docs"
cp MINI_PC_SETUP.md "$DEPLOY_DIR/docs/"
cp ARDUPILOT_SETUP.md "$DEPLOY_DIR/docs/"
cp ardupilot_params.txt "$DEPLOY_DIR/docs/"

# README для развертывания
cat > "$DEPLOY_DIR/README_DEPLOY.md" << 'EOF'
# Развертывание на мини ПК

## Быстрая установка

1. Установить зависимости:
```bash
pip install -r requirements.txt
```

2. Запустить систему:
```bash
python quick_start_matek.py
```

## Подключение

1. Подключить USB-UART адаптер к мини ПК
2. Подключить адаптер к UART2 на Matek F405
3. Настроить ArduPilot (см. docs/ARDUPILOT_SETUP.md)
4. Запустить систему

## Файлы

- `drone_navigation.py` - главный модуль
- `quick_start_matek.py` - быстрый запуск
- `matek_integration.py` - интеграция с Matek F405

Подробнее: docs/MINI_PC_SETUP.md
EOF

echo ""
echo "✅ Пакет создан в папке: $DEPLOY_DIR"
echo ""
echo "Файлы для отправки на мини ПК:"
echo "  cd $DEPLOY_DIR"
echo "  tar -czf ../deploy.tar.gz *"
echo ""
echo "Или отправьте всю папку $DEPLOY_DIR на мини ПК"

