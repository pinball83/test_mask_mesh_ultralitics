import 'dart:math' as math;
import 'dart:ui';

import 'package:ultralytics_yolo/models/yolo_result.dart';

/// Encapsulates the mapping between YOLO's input tensor and the Flutter view.
///
/// The Ultralytics runtime reports detection boxes twice: once normalized
/// (0..1 inside the model tensor) and once in absolute pixels. By comparing the
/// two we can recover the original source resolution and the letterbox offsets
/// applied when stretching the camera feed to the current widget size. All
/// overlays should reuse this geometry to stay aligned with the preview on
/// both Android and iOS.
class DetectionViewGeometry {
  const DetectionViewGeometry({
    required this.sourceWidth,
    required this.sourceHeight,
    required this.scale,
    required this.dx,
    required this.dy,
  });

  final double sourceWidth;
  final double sourceHeight;
  final double scale;
  final double dx;
  final double dy;

  static DetectionViewGeometry? fromDetections(
    List<YOLOResult> detections,
    Size viewSize,
  ) {
    if (detections.isEmpty || viewSize.isEmpty) return null;

    double? inferredWidth;
    double? inferredHeight;

    for (final detection in detections) {
      final nb = detection.normalizedBox;
      final bb = detection.boundingBox;
      final normalizedWidth = (nb.right - nb.left).abs();
      final normalizedHeight = (nb.bottom - nb.top).abs();
      final boxWidth = bb.width.abs();
      final boxHeight = bb.height.abs();

      if (inferredWidth == null &&
          normalizedWidth > 0 &&
          normalizedWidth.isFinite &&
          boxWidth > 0) {
        inferredWidth = boxWidth / normalizedWidth;
      }

      if (inferredHeight == null &&
          normalizedHeight > 0 &&
          normalizedHeight.isFinite &&
          boxHeight > 0) {
        inferredHeight = boxHeight / normalizedHeight;
      }

      if (inferredWidth != null && inferredHeight != null) {
        break;
      }
    }

    if (inferredWidth == null ||
        inferredHeight == null ||
        inferredWidth <= 0 ||
        inferredHeight <= 0) {
      return null;
    }

    final scale = math.max(
      viewSize.width / inferredWidth,
      viewSize.height / inferredHeight,
    );
    final scaledW = inferredWidth * scale;
    final scaledH = inferredHeight * scale;
    final dx = (viewSize.width - scaledW) / 2.0;
    final dy = (viewSize.height - scaledH) / 2.0;

    return DetectionViewGeometry(
      sourceWidth: inferredWidth,
      sourceHeight: inferredHeight,
      scale: scale,
      dx: dx,
      dy: dy,
    );
  }

  Rect projectNormalizedRect(Rect normalized) {
    double projectX(double value) {
      final clamped = value.clamp(0.0, 1.0);
      return dx + clamped * sourceWidth * scale;
    }

    double projectY(double value) {
      final clamped = value.clamp(0.0, 1.0);
      return dy + clamped * sourceHeight * scale;
    }

    final left = projectX(normalized.left);
    final right = projectX(normalized.right);
    final top = projectY(normalized.top);
    final bottom = projectY(normalized.bottom);

    final resolvedLeft = math.min(left, right);
    final resolvedRight = math.max(left, right);
    final resolvedTop = math.min(top, bottom);
    final resolvedBottom = math.max(top, bottom);

    return Rect.fromLTRB(
      resolvedLeft,
      resolvedTop,
      resolvedRight,
      resolvedBottom,
    );
  }

  ({bool flipH, bool flipV}) detectMirroring({
    required Size viewSize,
    required YOLOResult? reference,
    ({bool flipH, bool flipV})? fallback,
  }) {
    final ref = reference;
    if (ref == null || ref.normalizedBox.isEmpty || ref.boundingBox.isEmpty) {
      return fallback ?? (flipH: false, flipV: false);
    }

    final predicted = projectNormalizedRect(ref.normalizedBox);
    final actual = ref.boundingBox;

    Rect mirrorH(Rect rect) {
      return Rect.fromLTRB(
        viewSize.width - rect.right,
        rect.top,
        viewSize.width - rect.left,
        rect.bottom,
      );
    }

    Rect mirrorV(Rect rect) {
      return Rect.fromLTRB(
        rect.left,
        viewSize.height - rect.bottom,
        rect.right,
        viewSize.height - rect.top,
      );
    }

    double distance(Rect a, Rect b) {
      return (a.left - b.left).abs() +
          (a.top - b.top).abs() +
          (a.right - b.right).abs() +
          (a.bottom - b.bottom).abs();
    }

    final mirroredH = mirrorH(predicted);
    final mirroredV = mirrorV(predicted);
    final mirroredHV = mirrorV(mirroredH);

    final base = distance(predicted, actual);
    final dH = distance(mirroredH, actual);
    final dV = distance(mirroredV, actual);
    final dHV = distance(mirroredHV, actual);

    const bias = 0.5;
    var best = base;
    var flipH = false;
    var flipV = false;

    if (dH + bias < best) {
      best = dH;
      flipH = true;
      flipV = false;
    }
    if (dV + bias < best) {
      best = dV;
      flipH = false;
      flipV = true;
    }
    if (dHV + bias < best) {
      flipH = true;
      flipV = true;
    }

    return (flipH: flipH, flipV: flipV);
  }
}
