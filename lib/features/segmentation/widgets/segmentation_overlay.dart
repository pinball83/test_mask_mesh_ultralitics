import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class SegmentationOverlay extends StatelessWidget {
  const SegmentationOverlay({
    super.key,
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _SegmentationMaskPainter(
          detections: detections,
          maskThreshold: maskThreshold,
          flipHorizontal: flipHorizontal,
          flipVertical: flipVertical,
        ),
      ),
    );
  }
}

class _SegmentationMaskPainter extends CustomPainter {
  const _SegmentationMaskPainter({
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;

  @override
  void paint(Canvas canvas, Size size) {
    final overlayPaint = Paint()..blendMode = BlendMode.srcOver;
    canvas.save();
    canvas.clipRect(Offset.zero & size);

    final sourceSize = _estimateSourceSize();
    final sourceWidth = sourceSize?.width ?? size.width;
    final sourceHeight = sourceSize?.height ?? size.height;

    if (sourceWidth <= 0 || sourceHeight <= 0) {
      canvas.restore();
      return;
    }

    final scaleX = size.width / sourceWidth;
    final scaleY = size.height / sourceHeight;
    final scale = scaleX > scaleY ? scaleX : scaleY;
    final scaledWidth = sourceWidth * scale;
    final scaledHeight = sourceHeight * scale;
    final dx = (size.width - scaledWidth) / 2;
    final dy = (size.height - scaledHeight) / 2;

    for (final detection in detections) {
      final mask = detection.mask;
      final bounds = detection.boundingBox;
      if (mask == null || mask.isEmpty || bounds.isEmpty) continue;

      final maskHeight = mask.length;
      final maskWidth = mask.first.length;
      if (maskWidth == 0 || maskHeight == 0) continue;

      final scaledBounds = Rect.fromLTRB(
        dx + bounds.left * scale,
        dy + bounds.top * scale,
        dx + bounds.right * scale,
        dy + bounds.bottom * scale,
      );

      if (scaledBounds.width <= 0 || scaledBounds.height <= 0) continue;

      final cellWidth = scaledBounds.width / maskWidth;
      final cellHeight = scaledBounds.height / maskHeight;

      for (var y = 0; y < maskHeight; y++) {
        final row = mask[y];
        final mappedY = flipVertical ? (maskHeight - 1 - y) : y;
        final top = scaledBounds.top + mappedY * cellHeight;

        for (var x = 0; x < maskWidth; x++) {
          final value = row[x];
          if (value < maskThreshold) continue;

          final mappedX = flipHorizontal ? (maskWidth - 1 - x) : x;
          final left = scaledBounds.left + mappedX * cellWidth;

          final opacity = (value.clamp(0.0, 1.0) * 0.65).clamp(0.15, 0.7);
          overlayPaint.color = Colors.tealAccent.withValues(
            alpha: opacity.toDouble(),
          );

          canvas.drawRect(
            Rect.fromLTWH(left, top, cellWidth, cellHeight),
            overlayPaint,
          );
        }
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _SegmentationMaskPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.maskThreshold != maskThreshold ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }

  Size? _estimateSourceSize() {
    double? width;
    double? height;

    for (final detection in detections) {
      final normalized = detection.normalizedBox;
      final bounds = detection.boundingBox;

      final normalizedWidth = normalized.width;
      final normalizedHeight = normalized.height;

      if (width == null &&
          normalizedWidth > 0 &&
          bounds.width > 0 &&
          normalizedWidth.isFinite) {
        final candidate = bounds.width / normalizedWidth;
        if (candidate.isFinite && candidate > 0) {
          width = candidate;
        }
      }

      if (height == null &&
          normalizedHeight > 0 &&
          bounds.height > 0 &&
          normalizedHeight.isFinite) {
        final candidate = bounds.height / normalizedHeight;
        if (candidate.isFinite && candidate > 0) {
          height = candidate;
        }
      }

      if (width != null && height != null) {
        break;
      }
    }

    if (width != null && height != null) {
      return Size(width, height);
    }

    return null;
  }
}
