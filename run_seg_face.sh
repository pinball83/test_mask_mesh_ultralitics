#!/bin/bash

# Устанавливаем права на исполнение
chmod +x "$0"

# Директория, где лежит скрипт и куда будут скачиваться модели
SCRIPT_DIR="python_video_processing"
cd "$SCRIPT_DIR" || exit

# --- 1. Установка зависимостей ---
echo "Checking Python dependencies..."
pip install -r requirements.txt --quiet

# --- 2. Загрузка моделей, если их нет ---
SEG_MODEL_ARG=$(basename "$2")
POSE_MODEL_ARG=$(basename "$3")

# Скачиваем модель сегментации, если она не существует
if [ ! -f "$SEG_MODEL_ARG" ]; then
    echo "Downloading segmentation model: $SEG_MODEL_ARG..."
    # Используем yolov8n-seg.pt как стандартную
    wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt" -O "$SEG_MODEL_ARG"
fi

# Скачиваем модель поз, если она не существует
if [ ! -f "$POSE_MODEL_ARG" ]; then
    echo "Downloading pose model: $POSE_MODEL_ARG..."
    # Используем yolov8n-pose.pt как стандартную
    wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt" -O "$POSE_MODEL_ARG"
fi

echo "Setup complete. Starting video processing..."
echo "--------------------------------------------"

# --- 3. Запуск Python-скрипта ---
# Передаем все аргументы командной строки (input.mp4, модели, маска, режим)
# в наш основной скрипт.
python3 run_video_processing.py "$@"
