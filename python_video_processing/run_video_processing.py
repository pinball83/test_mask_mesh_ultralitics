
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import time
import subprocess
import os

# --- Утилиты и классы ---

class LandmarksEMA:
    """
    Экспоненциальное скользящее среднее для сглаживания ключевых точек.
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.last_landmarks = None

    def __call__(self, landmarks):
        if self.last_landmarks is None:
            self.last_landmarks = landmarks
            return landmarks

        smoothed_landmarks = self.alpha * landmarks + (1 - self.alpha) * self.last_landmarks
        self.last_landmarks = smoothed_landmarks
        return smoothed_landmarks

def select_device():
    """
    Автоматический выбор устройства для вычислений.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def process_background(frame, seg_mask, mode, bg_image=None):
    """
    Обработка фона: размытие или замена.
    """
    h, w = frame.shape[:2]

    # Убедимся, что маска бинарная и имеет правильные размеры
    if seg_mask.ndim == 3:
        seg_mask = seg_mask.squeeze(0) # Убираем лишнее измерение, если есть

    # Изменение размера маски до размеров кадра
    seg_mask_resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Создание 3-канальной маски для смешивания
    seg_mask_3ch = cv2.cvtColor((seg_mask_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Инвертированная маска для фона
    inv_mask = cv2.bitwise_not(seg_mask_3ch)

    # Применяем маску к человеку
    person = cv2.bitwise_and(frame, seg_mask_3ch)

    if mode == 'blur':
        # Размываем фон
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        background = cv2.bitwise_and(blurred_frame, inv_mask)
    elif mode == 'image' and bg_image is not None:
        # Заменяем фон изображением
        background = cv2.bitwise_and(bg_image, inv_mask)
    else: # По умолчанию или если что-то пошло не так
        background = cv2.bitwise_and(frame, inv_mask)

    return cv2.add(person, background)

def overlay_png_affine(frame, overlay_rgba, dst_tri):
    """
    Наложение PNG с использованием аффинной трансформации.
    """
    h, w = frame.shape[:2]
    overlay_h, overlay_w = overlay_rgba.shape[:2]

    # Определение исходных точек на PNG-шаблоне
    TEMPLATE_EYE_L = (int(overlay_w * 0.33), int(overlay_h * 0.36))
    TEMPLATE_EYE_R = (int(overlay_w * 0.67), int(overlay_h * 0.36))
    TEMPLATE_NOSE  = (int(overlay_w * 0.50), int(overlay_h * 0.46))
    src_tri = np.float32([TEMPLATE_EYE_L, TEMPLATE_EYE_R, TEMPLATE_NOSE])

    # Вычисление матрицы аффинного преобразования
    affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)

    # Применение трансформации к PNG. Размер (w, h) кадра важен,
    # чтобы трансформированное изображение оказалось в правильной позиции на холсте.
    warped_overlay = cv2.warpAffine(overlay_rgba, affine_matrix, (w, h))

    # Разделение наложенного изображения на RGB и альфа-канал
    overlay_rgb = warped_overlay[..., :3]
    alpha_mask = warped_overlay[..., 3]

    # Создание маски для смешивания
    mask = (alpha_mask.astype(np.float32) / 255.0)[..., None]

    # Альфа-смешивание
    result_frame = frame * (1 - mask) + overlay_rgb * mask

    return result_frame.astype(np.uint8)


# --- Основная функция ---

def main():
    parser = argparse.ArgumentParser(description="Video processing with background segmentation and face mask overlay.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("seg_model", help="Path to the YOLO segmentation model.")
    parser.add_argument("pose_model", help="Path to the YOLO pose model.")
    parser.add_argument("mask_image", help="Path to the PNG mask image with alpha channel.")
    parser.add_argument("bg_mode", choices=['blur', 'image'], help="Background processing mode: 'blur' or 'image'.")
    parser.add_argument("--bg_image", help="Path to the background image (required for 'image' mode).")

    args = parser.parse_args()

    if args.bg_mode == 'image' and not args.bg_image:
        print("Error: Background image path is required for 'image' mode.")
        return

    # --- 1. Инициализация ---
    start_time = time.time()
    DEVICE = select_device()
    print(f"Using device: {DEVICE}")

    # Загрузка моделей
    try:
        model_seg = YOLO(args.seg_model)
        model_pose = YOLO(args.pose_model)
        model_seg.to(DEVICE)
        model_pose.to(DEVICE)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Загрузка PNG маски
    try:
        mask_rgba = cv2.imread(args.mask_image, cv2.IMREAD_UNCHANGED)
        if mask_rgba.shape[2] != 4:
            raise ValueError("Mask image must have an alpha channel.")
        print(f"Mask image '{args.mask_image}' loaded ({mask_rgba.shape[1]}x{mask_rgba.shape[0]}).")
    except Exception as e:
        print(f"Error loading mask image: {e}")
        return

    # Загрузка фонового изображения, если нужно
    bg_image_resized = None
    if args.bg_mode == 'image':
        try:
            bg_image = cv2.imread(args.bg_image)
            print(f"Background image '{args.bg_image}' loaded.")
        except Exception as e:
            print(f"Error loading background image: {e}")
            return

    # Настройка видео
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error opening video file: {args.input_video}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Изменяем размер фонового изображения под кадр
    if bg_image_resized is None and args.bg_mode == 'image':
        bg_image_resized = cv2.resize(bg_image, (w, h))

    # Настройка выходного видео
    output_video_path = "output_temp.mp4"
    final_output_path = "output_masked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"Input video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames.")

    # Инициализация сглаживания
    landmarks_smoother = LandmarksEMA(alpha=0.6)
    frame_count = 0

    # --- 2. Основной цикл обработки ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        loop_start_time = time.time()

        # --- Сегментация фона ---
        # Класс 0 - 'person' в стандартных наборах данных COCO
        seg_results = model_seg(frame, classes=[0], verbose=False)

        final_frame = frame
        if seg_results[0].masks is not None and len(seg_results[0].masks.data) > 0:
            # Объединяем все маски людей в одну
            combined_mask = torch.any(seg_results[0].masks.data, dim=0).float()
            seg_mask_np = combined_mask.cpu().numpy()
            final_frame = process_background(frame, seg_mask_np, args.bg_mode, bg_image_resized)

        # --- Обнаружение и наложение маски на лицо ---
        pose_results = model_pose(frame, verbose=False)

        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
            # Берем первого обнаруженного человека
            keypoints = pose_results[0].keypoints.xy[0]

            # Получаем координаты ключевых точек
            # Индексы: 0-нос, 1-левый глаз, 2-правый глаз
            nose = keypoints[0].cpu().numpy()
            eye_l = keypoints[1].cpu().numpy()
            eye_r = keypoints[2].cpu().numpy()

            # Проверка на "зеркальность" (когда левый глаз правее правого)
            if eye_l[0] > eye_r[0]:
                eye_l, eye_r = eye_r, eye_l # Меняем местами

            # Проверка, что точки валидны (не нулевые)
            if np.all(nose > 0) and np.all(eye_l > 0) and np.all(eye_r > 0):
                # Сглаживание
                dst_tri_raw = np.float32([eye_l, eye_r, nose])
                dst_tri_smoothed = landmarks_smoother(dst_tri_raw)

                # Наложение
                final_frame = overlay_png_affine(final_frame, mask_rgba, dst_tri_smoothed)

        out.write(final_frame)
        frame_count += 1

        # Логирование прогресса
        if frame_count % 20 == 0:
            elapsed = time.time() - loop_start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"Processed frame {frame_count}/{total_frames} ({current_fps:.2f} FPS)")

    # --- 3. Постобработка ---
    print("Video processing finished. Releasing resources...")
    cap.release()
    out.release()

    # Слияние аудио с помощью ffmpeg
    print("Merging audio from original video...")
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-i', output_video_path,
        '-i', args.input_video,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0?', # '?' делает поток опциональным
        '-shortest',
        final_output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Final video saved to '{final_output_path}'")
        # Удаление временного файла
        os.remove(output_video_path)
    except subprocess.CalledProcessError as e:
        print("\n--- FFMPEG ERROR ---")
        print(f"FFmpeg command failed with exit code {e.returncode}")
        print("Command:", ' '.join(e.cmd))
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        print("--------------------\n")
        print(f"Temporary video without audio is available at: {output_video_path}")
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your PATH.")
        print(f"Temporary video without audio is available at: {output_video_path}")


    # --- 4. Финальная информация ---
    end_time = time.time()
    total_duration = end_time - start_time
    file_size = os.path.getsize(final_output_path) / (1024 * 1024)

    print("\n--- Summary ---")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Final file size: {file_size:.2f} MB")
    print("---------------")


if __name__ == "__main__":
    main()
