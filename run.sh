#!/bin/bash
# Скрипт запуска приложения

echo "Запуск системы навигации дрона..."
echo ""

# Активируем виртуальное окружение если оно есть
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Виртуальное окружение активировано"
else
    echo "⚠ Виртуальное окружение не найдено"
fi

# Проверяем Flask
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask не установлен!"
    echo "Установите зависимости: pip install -r requirements.txt"
    exit 1
fi

echo "✓ Запуск приложения..."
python app.py

