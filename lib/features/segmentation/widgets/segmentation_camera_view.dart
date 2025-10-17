import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

import '../controller/segmentation_controller.dart';
import 'segmentation_overlay.dart';

class SegmentationCameraView extends StatelessWidget {
  const SegmentationCameraView({super.key, required this.controller});

  final SegmentationController controller;

  @override
  Widget build(BuildContext context) {
    if (controller.isLoading) {
      return _LoadingState(
        message: controller.statusMessage,
        progress: controller.downloadProgress,
      );
    }

    if (controller.errorMessage != null) {
      return _ErrorState(
        message: controller.errorMessage!,
        onRetry: controller.refreshModel,
      );
    }

    final modelPath = controller.modelPath;
    if (modelPath == null) {
      return const _ErrorState(message: 'Segmentation model not available.');
    }

    controller.ensurePreferredCamera();

    return Stack(
      children: [
        Positioned.fill(
          child: YOLOView(
            key: ValueKey(modelPath),
            controller: controller.yoloController,
            modelPath: modelPath,
            task: YOLOTask.segment,
            streamingConfig: controller.streamingConfig,
            showOverlays: false,
            onResult: controller.onResults,
            onPerformanceMetrics: controller.onPerformance,
            onZoomChanged: controller.onZoomChanged,
          ),
        ),
        if (controller.showMasks)
          Positioned.fill(
            child: SegmentationOverlay(
              detections: controller.detections,
              maskThreshold: controller.maskThreshold,
              flipHorizontal: controller.flipMaskHorizontal,
              flipVertical: controller.flipMaskVertical,
            ),
          ),
        Positioned(
          top: 16,
          left: 16,
          child: _StatsBadge(
            fps: controller.fps,
            detections: controller.detections.length,
          ),
        ),
      ],
    );
  }
}

class _LoadingState extends StatelessWidget {
  const _LoadingState({required this.message, required this.progress});

  final String message;
  final double progress;

  @override
  Widget build(BuildContext context) {
    final percent = (progress * 100).clamp(0, 100).toStringAsFixed(0);
    return Center(
      child: Container(
        width: 260,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.7),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white10),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator.adaptive(),
            const SizedBox(height: 16),
            Text(
              message,
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              '$percent%',
              style: Theme.of(
                context,
              ).textTheme.labelLarge?.copyWith(color: Colors.tealAccent),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({required this.message, this.onRetry});

  final String message;
  final Future<void> Function()? onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.warning_amber_rounded,
              size: 72,
              color: Colors.orangeAccent,
            ),
            const SizedBox(height: 16),
            Text(
              message,
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            if (onRetry != null) ...[
              const SizedBox(height: 16),
              FilledButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ],
        ),
      ),
    );
  }
}

class _StatsBadge extends StatelessWidget {
  const _StatsBadge({required this.fps, required this.detections});

  final double fps;
  final int detections;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: DefaultTextStyle(
        style: Theme.of(context).textTheme.labelLarge!,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detections: $detections'),
            Text('FPS: ${fps.toStringAsFixed(1)}'),
          ],
        ),
      ),
    );
  }
}
