#!/bin/bash
# Скрипт остановки приложения

echo "Остановка системы навигации дрона..."

if [ -f "app.pid" ]; then
    PID=$(cat app.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "✓ Процесс $PID остановлен"
    else
        echo "⚠ Процесс $PID не найден"
    fi
    rm -f app.pid
else
    echo "⚠ PID файл не найден"
    # Пробуем найти и остановить по имени
    pkill -f "python app.py"
    echo "✓ Все процессы app.py остановлены"
fi

echo "Готово"

