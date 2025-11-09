# Используем официальный образ Python
FROM python:3.9

# Устанавливаем системные зависимости, необходимые для OpenCV и FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Создаем директорию для нашего приложения
WORKDIR /usr/src/app

# Копируем файл с зависимостями и устанавливаем их
COPY python_video_processing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем скрипты в контейнер
COPY python_video_processing/run_video_processing.py .
COPY run_seg_face.sh .
RUN chmod +x run_seg_face.sh

# Устанавливаем рабочую директорию для данных.
# Все пути к файлам в команде 'docker run' будут относительно этой папки.
WORKDIR /data

# Устанавливаем точку входа. Эта команда будет выполняться при запуске контейнера.
ENTRYPOINT ["/usr/src/app/run_seg_face.sh"]

# Команда по умолчанию (если пользователь не передал аргументы)
CMD ["--help"]
