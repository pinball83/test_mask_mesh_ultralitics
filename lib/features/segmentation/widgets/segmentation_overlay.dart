import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class SegmentationOverlay extends StatelessWidget {
  const SegmentationOverlay({
    super.key,
    required this.detections,
    required this.maskThreshold,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;

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
        ),
      ),
    );
  }
}

class _SegmentationMaskPainter extends CustomPainter {
  _SegmentationMaskPainter({
    required this.detections,
    required this.maskThreshold,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;

  @override
  void paint(Canvas canvas, Size size) {
    final overlayPaint = Paint()..blendMode = BlendMode.srcOver;
    canvas.save();
    canvas.clipRect(Offset.zero & size);

    for (final detection in detections) {
      final mask = detection.mask;
      final bounds = detection.boundingBox;
      if (mask == null || mask.isEmpty || bounds.isEmpty) continue;

      final maskHeight = mask.length;
      final maskWidth = mask.first.length;
      if (maskWidth == 0 || maskHeight == 0) continue;

      final cellWidth = bounds.width / maskWidth;
      final cellHeight = bounds.height / maskHeight;

      for (var y = 0; y < maskHeight; y++) {
        final row = mask[y];
        for (var x = 0; x < maskWidth; x++) {
          final value = row[x];
          if (value < maskThreshold) continue;
          final opacity = (value.clamp(0.0, 1.0) * 0.65).clamp(0.15, 0.7);
          overlayPaint.color =
              Colors.tealAccent.withOpacity(opacity.toDouble());
          final dx = bounds.left + x * cellWidth;
          final dy = bounds.top + y * cellHeight;
          canvas.drawRect(
            Rect.fromLTWH(dx, dy, cellWidth, cellHeight),
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
        oldDelegate.maskThreshold != maskThreshold;
  }
}
