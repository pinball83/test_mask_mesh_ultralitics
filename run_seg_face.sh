#!/bin/bash

# --- Docker-совместимый скрипт-обертка ---
# Этот скрипт предназначен для работы в качестве ENTRYPOINT в Docker.
# Он предполагает, что скрипты приложения лежат в /usr/src/app,
# а рабочая директория - /data (подключена как volume).

set -e

APP_DIR="/usr/src/app"

# Проверяем, что переданы хотя бы базовые аргументы
if [ "$#" -lt 4 ]; then
    echo "Usage: <input_video> <seg_model_name> <pose_model_name> <mask_image> [bg_mode] [--bg_image ...]"
    python3 "$APP_DIR/run_video_processing.py" --help
    exit 1
fi

# --- 1. Подготовка моделей ---
# Модели (2-й и 3-й аргументы) ищутся и/или скачиваются в директорию приложения.
SEG_MODEL_NAME=$(basename "$2")
POSE_MODEL_NAME=$(basename "$3")

SEG_MODEL_PATH="$APP_DIR/$SEG_MODEL_NAME"
POSE_MODEL_PATH="$APP_DIR/$POSE_MODEL_NAME"

# Скачиваем модель сегментации, если её нет
if [ ! -f "$SEG_MODEL_PATH" ]; then
    echo "Downloading segmentation model to $SEG_MODEL_PATH..."
    wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt" -O "$SEG_MODEL_PATH"
fi

# Скачиваем модель поз, если её нет
if [ ! -f "$POSE_MODEL_PATH" ]; then
    echo "Downloading pose model to $POSE_MODEL_PATH..."
    wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt" -O "$POSE_MODEL_PATH"
fi

echo "Docker setup complete. Models are ready. Starting video processing..."
echo "--------------------------------------------------------------------"

# --- 2. Запуск Python-скрипта ---
# Вызываем основной скрипт, подменяя имена моделей на полные пути внутри контейнера.
# Все остальные аргументы ($1, $4, $5...) передаются как есть.
# Они будут интерпретироваться относительно рабочей директории /data.
python3 "$APP_DIR/run_video_processing.py" "$1" "$SEG_MODEL_PATH" "$POSE_MODEL_PATH" "${@:4}"
